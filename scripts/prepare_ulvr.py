"""Build the full 6240-example training run and complete bbox_crop validation set."""
import argparse
from collections import Counter
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random

from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
COLS = ['sample_id', 'question', 'answer', 'source_dataset', 'input_image', 'intermediate_image_1']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def normalized(question):
    return ' '.join(question.replace('<image>', '').casefold().split())


def read_rows(files, limit=None):
    count = 0
    for file in sorted(files):
        for batch in pq.ParquetFile(file).iter_batches(batch_size=32, columns=COLS):
            for row in batch.to_pylist():
                if limit is not None and count >= limit:
                    return
                row['_source_shard'] = file.name
                row['_source_row'] = count
                count += 1
                yield row


def image_bytes(obj):
    value = (obj or {}).get('bytes')
    if not value:
        raise ValueError('missing embedded image bytes')
    return value


def write_image(payload, label, directory, max_pixels=None):
    checksum = sha(payload)
    with Image.open(BytesIO(payload)) as im:
        size = im.size
        extension = {'JPEG': 'jpg', 'PNG': 'png', 'WEBP': 'webp'}.get(im.format, 'img')
        if max_pixels and im.width * im.height > max_pixels:
            raise ValueError('original_pixel_limit')
        im.verify()
    name = f'{checksum}_{label}.{extension}'
    dest = directory / name
    if not dest.exists():
        dest.write_bytes(payload)
    return str(dest.resolve()), checksum, list(size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate-n', type=int, default=8000)
    parser.add_argument('--n', type=int, default=6240)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    raw = ROOT / 'data/raw/ULVR/data'
    out = ROOT / 'data/ulvr'
    out.mkdir(exist_ok=True)
    images = out / 'images'
    images.mkdir(exist_ok=True)
    validation = []
    val_ids, val_q, val_images = set(), set(), set()
    for row in read_rows((raw / 'bbox_crop_val').glob('*.parquet')):
        path, checksum, size = write_image(image_bytes(row['input_image']), 'original', images)
        val_ids.add(row['sample_id']); val_q.add(normalized(row['question'])); val_images.add(checksum)
        validation.append({'sample_id': row['sample_id'], 'question_id': len(validation),
                           'images': [path], 'query': row['question'].replace('<image>', '').strip(),
                           'response': row['answer'], 'category': row['source_dataset'],
                           'original_sha256': checksum, 'image_size': size,
                           'source_shard': row['_source_shard'], 'source_row': row['_source_row']})
    if len(validation) != 2558:
        raise ValueError(f'Expected full 2558 validation rows, got {len(validation)}')
    (out / 'validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2))
    print('VALIDATION_READY', len(validation), flush=True)
    candidates = list(read_rows((raw / 'bbox_crop').glob('*.parquet'), args.candidate_n))
    if len(candidates) != args.candidate_n:
        raise ValueError(f'Only {len(candidates)} raw training candidates available; download more shards')
    random.Random(args.seed).shuffle(candidates)
    selected = []
    ids, questions = set(), set()
    skipped = Counter()
    for row in candidates:
        sid, q = row['sample_id'], normalized(row['question'])
        if sid in ids or q in questions:
            skipped['duplicate_id_or_question'] += 1; continue
        if sid in val_ids or q in val_q:
            skipped['validation_id_or_question_overlap'] += 1; continue
        if not 0 < len(row['question']) <= 1200:
            skipped['question_length'] += 1; continue
        try:
            original = image_bytes(row['input_image'])
            if sha(original) in val_images:
                skipped['validation_image_overlap'] += 1; continue
            original_path, original_hash, original_size = write_image(original, 'original', images, 6_000_000)
            crop_path, crop_hash, crop_size = write_image(image_bytes(row['intermediate_image_1']), 'crop', images)
        except (ValueError, OSError, Image.DecompressionBombError) as exc:
            skipped[str(exc)[:80]] += 1; continue
        selected.append({'sample_id': sid, 'question': row['question'].replace('<image>', '').strip(),
                         'source_dataset': row['source_dataset'], 'original': original_path, 'crop': crop_path,
                         'original_sha256': original_hash, 'crop_sha256': crop_hash,
                         'original_size': original_size, 'crop_size': crop_size,
                         'source_shard': row['_source_shard'], 'source_row': row['_source_row']})
        ids.add(sid); questions.add(q)
        if len(selected) % 500 == 0:
            print('SELECTED', len(selected), flush=True)
        if len(selected) == args.n:
            break
    if len(selected) != args.n:
        raise ValueError(f'Only {len(selected)} eligible; explicitly expand candidate-n and download more shards. Skips: {skipped}')
    records = [{'data_source': 'ulvr_bbox_crop_vopd',
                'prompt': [{'role': 'user', 'content': '<image>\n' + x['question']}],
                'images': [{'path': x['original']}], 'bbox_images': [{'path': x['crop']}],
                'ability': 'visual_question_answering', 'reward_model': {'style': 'none', 'ground_truth': ''},
                'extra_info': {'sample_id': x['sample_id'], 'question': x['question'], 'answer': '', 'source_dataset': x['source_dataset']}}
               for x in selected]
    parquet = out / 'train.parquet'
    pq.write_table(pa.Table.from_pylist(records), parquet)
    (out / 'samples.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in selected))
    sources = json.loads((ROOT / 'config/sources.json').read_text())
    manifest = {'dataset': sources['dataset'], 'train_n': len(selected), 'validation_n': len(validation),
                'candidate_n': args.candidate_n, 'seed': args.seed, 'batch_size': 96, 'updates_per_epoch': len(selected) // 96,
                'selection': 'seeded shuffle of first candidate_n pinned-Parquet train rows; documented filters',
                'filters': {'max_original_pixels': 6_000_000, 'max_question_chars': 1200},
                'skips': dict(skipped), 'train_source_counts': dict(Counter(x['source_dataset'] for x in selected)),
                'unique_training_original_images': len({x['original_sha256'] for x in selected}),
                'train_validation_id_overlap': len(ids & val_ids), 'train_validation_question_overlap': len(questions & val_q),
                'train_validation_original_bytes_overlap': len({x['original_sha256'] for x in selected} & val_images),
                'student_input': 'original image + question', 'teacher_input': 'crop image + same question (upstream image replacement)',
                'gold_in_training_parquet': False, 'parquet_sha256': sha(parquet.read_bytes()),
                'validation_sha256': sha((out / 'validation.json').read_bytes()),
                'note': 'ULVR adaptation; full validation untouched'}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print('TRAINING_DATA_READY', json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
