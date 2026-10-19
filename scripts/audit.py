"""CPU-only verification of a staged experiment. Never imports torch or requests GPUs."""
import argparse
from collections import Counter
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import struct

from PIL import Image
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'eval-kit'))
from run import uid


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--require-ready', action='store_true')
    p.add_argument('--verify-downloads', action='store_true')
    args = p.parse_args()
    checks, problems = {}, []
    source = json.loads((ROOT / 'config/sources.json').read_text())
    extensions = json.loads((ROOT / 'config/cuda-extension-sources.json').read_text())
    for item in extensions.values():
        path = ROOT / item['path']
        if not path.is_file() or sha(path) != item['sha256']:
            problems.append('cuda_extension_source:' + item['path'])
    model = ROOT / 'models/Qwen3.5-4B'
    index = model / 'model.safetensors.index.json'
    if index.exists():
        index_data = json.loads(index.read_text())
        shards = set(index_data['weight_map'].values())
        checks['model_shards'] = {x: (model / x).is_file() for x in sorted(shards)}
        if not all(checks['model_shards'].values()): problems.append('model_weights_missing')
        else:
            tensors, total_bytes = {}, 0
            for name in sorted(shards):
                file = model / name
                with file.open('rb') as f:
                    header_size = struct.unpack('<Q', f.read(8))[0]
                    header = json.loads(f.read(header_size))
                for key, tensor in header.items():
                    if key == '__metadata__': continue
                    begin, end = tensor['data_offsets']
                    if not 0 <= begin <= end <= file.stat().st_size - header_size - 8:
                        problems.append('invalid_tensor_offsets:' + key)
                    tensors[key] = name
                    total_bytes += end - begin
            if tensors != index_data['weight_map']: problems.append('model_index_tensor_mismatch')
            if total_bytes != index_data['metadata']['total_size']: problems.append('model_tensor_byte_count')
            checks['model_structure'] = {'tensors': len(tensors), 'tensor_bytes': total_bytes}
    else: problems.append('model_index_missing')
    for name in ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'preprocessor_config.json']:
        if not (model / name).is_file(): problems.append('model_file_missing:' + name)
    train, validation = ROOT / 'data/ulvr/train.parquet', ROOT / 'data/ulvr/validation.json'
    original_hashes = set()
    if train.exists() and validation.exists():
        rows = pq.read_table(train).to_pylist()
        vals = json.loads(validation.read_text())
        val_ids = {x['sample_id'] for x in vals}
        val_q = {' '.join(x['query'].casefold().split()) for x in vals}
        val_images = {x['original_sha256'] for x in vals}
        image_paths = set()
        image_paths.update(x['images'][0] for x in vals)
        for row in rows:
            if row['reward_model']['ground_truth'] or row['extra_info']['answer']:
                problems.append('gold_leak_in_training')
            if row['extra_info']['sample_id'] in val_ids or ' '.join(row['extra_info']['question'].casefold().split()) in val_q:
                problems.append('train_validation_id_or_question_overlap')
            for field in ['images', 'bbox_images']:
                if len(row[field]) != 1: problems.append('wrong_image_count')
                image_paths.update(x['path'] for x in row[field])
            original_hashes.add(Path(row['images'][0]['path']).name.split('_')[0])
        if original_hashes & val_images: problems.append('train_validation_image_overlap')
        for path in sorted(image_paths):
            f = Path(path)
            if not f.is_file(): problems.append('missing_image:' + path); continue
            if sha(f) != f.name.split('_')[0]: problems.append('image_checksum:' + path)
            with Image.open(f) as image: image.verify()
        checks['training'] = {'n': len(rows), 'expected_n': 6240, 'updates': len(rows) // 96,
                              'validation_n': len(vals), 'unique_image_files_verified': len(image_paths),
                              'train_parquet_sha256': sha(train)}
        if len(rows) != 6240 or len(vals) != 2558: problems.append('training_validation_count')
        manifest = json.loads((ROOT / 'data/ulvr/manifest.json').read_text())
        if manifest['parquet_sha256'] != sha(train) or manifest['validation_sha256'] != sha(validation):
            problems.append('ulvr_manifest_checksum')
    else: problems.append('prepared_ulvr_missing')
    benchmark_counts, overlaps = {}, {}
    filenames = {'vstar': 'vstar.json', 'zoombench': 'zoombench.json', 'hrbench-4k': 'hr_bench_4k.json',
                 'hrbench-8k': 'hr_bench_8k.json', 'mme-realworld': 'MME_RealWorld.json', 'mme-realworld-cn': 'MME_RealWorld_CN.json'}
    hashes = {}
    manifests = ROOT / 'data/benchmarks/manifest.json'
    manifests = json.loads(manifests.read_text()) if manifests.is_file() else {}
    for bench, filename in filenames.items():
        file = ROOT / 'data/benchmarks' / filename
        if not file.is_file(): problems.append('benchmark_missing:' + bench); continue
        rows = json.loads(file.read_text())
        if bench not in manifests or manifests[bench]['sha256'] != sha(file):
            problems.append('benchmark_manifest_checksum:' + bench)
        benchmark_counts[bench] = len(rows)
        if len(rows) != source['benchmarks'][bench]: problems.append('benchmark_count:' + bench)
        ids = [uid(x, bench) for x in rows]
        if len(set(ids)) != len(ids): problems.append('duplicate_benchmark_ids:' + bench)
        seen = set()
        for row in rows:
            for path in row['images']:
                f = Path(path)
                if not f.is_file(): problems.append('benchmark_image_missing:' + path); continue
                if path not in hashes:
                    hashes[path] = sha(f)
                    with Image.open(f) as image: image.verify()
                seen.add(hashes[path])
        overlaps[bench] = len(seen & original_hashes)
        if overlaps[bench]: problems.append('train_benchmark_exact_image_overlap:' + bench)
    checks['benchmark_n'] = benchmark_counts
    checks['train_benchmark_exact_image_overlaps'] = overlaps
    (ROOT / 'reports/benchmark-image-sha256.json').write_text(json.dumps({
        str(Path(path).relative_to(ROOT)): checksum for path, checksum in hashes.items()
    }, indent=2))
    if args.verify_downloads:
        downloads = {}
        for file in (ROOT / 'reports').glob('download-*.jsonl'):
            for line in file.read_text().splitlines():
                record = json.loads(line); downloads[record['path']] = record
        for path, record in downloads.items():
            file = ROOT / path
            if not file.is_file() or file.stat().st_size != record['bytes'] or sha(file) != record['sha256']:
                problems.append('download_integrity:' + path)
        checks['download_files_verified'] = len(downloads)
    report = {'data_and_model_ready': not problems, 'checks': checks, 'problems': sorted(set(problems)),
              'training_started': False, 'scheduler_submitted': False,
              'runtime_status': 'GPU runtime and EMA-teacher checkpoint restore not validated; no job submitted',
              'judge_status': 'Qwen2.5-72B-Instruct endpoint still required for comparable final grading'}
    (ROOT / 'reports/readiness.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if args.require_ready and problems:
        sys.exit(1)


if __name__ == '__main__':
    main()
