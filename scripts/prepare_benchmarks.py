"""Use the frozen upstream converters against already-downloaded immutable datasets."""
from collections import Counter
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--available-only', action='store_true', help='Convert completed snapshots while other downloads continue')
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('official_prepare', ROOT / 'eval-kit/vendor/eval/prepare_data.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sources = json.loads((ROOT / 'config/hub_sources.json').read_text())
    # The converter writes to its given local_dir. Return a symlink to the frozen local snapshot.
    def local_snapshot(repo_id, repo_type, local_dir):
        raw = ROOT / 'data/raw' / repo_id.replace('/', '--')
        if not raw.is_dir():
            raise FileNotFoundError(raw)
        dest = Path(local_dir)
        if not dest.exists():
            dest.symlink_to(raw, target_is_directory=True)
        if dest.resolve() != raw.resolve():
            raise ValueError(f'Unexpected raw directory {dest}')
        return str(raw)
    mod.snapshot_download = local_snapshot
    data = ROOT / 'data/benchmarks'
    steps = [
        ('vstar', 'vstar.json', 'lmms-lab/vstar-bench', lambda: mod.prepare_vstar(data)),
        ('zoombench', 'zoombench.json', 'inclusionAI/ZoomBench', lambda: mod.prepare_zoombench(data)),
        ('hrbench-4k', 'hr_bench_4k.json', 'DreamMr/HR-Bench', lambda: mod.prepare_hrbench(data, 'hrbench-4k')),
        ('hrbench-8k', 'hr_bench_8k.json', 'DreamMr/HR-Bench', lambda: mod.prepare_hrbench(data, 'hrbench-8k')),
        ('mme-realworld', 'MME_RealWorld.json', 'yifanzhang114/MME-RealWorld-Lmms-eval', lambda: mod.prepare_mme_realworld(data)),
        ('mme-realworld-cn', 'MME_RealWorld_CN.json', 'yifanzhang114/MME-RealWorld-CN-Lmms-eval', lambda: mod.prepare_mme_realworld_cn(data)),
    ]
    expected = json.loads((ROOT / 'config/sources.json').read_text())['benchmarks']
    manifest = {}
    for bench, filename, repo, prepare in steps:
        meta = json.loads((ROOT / 'source-evidence' / f'{repo.replace("/", "--")}.json').read_text())
        missing = [x['rfilename'] for x in meta['siblings'] if x['rfilename'].endswith('.parquet')
                   and not (ROOT / 'data/raw' / repo.replace('/', '--') / x['rfilename']).is_file()]
        if missing:
            if args.available_only:
                print('PENDING', bench, len(missing), 'raw files', flush=True)
                continue
            raise FileNotFoundError(f'{bench}: raw download incomplete: {missing}')
        path = data / filename
        if path.exists():
            rows = json.loads(path.read_text())
        else:
            print('CONVERTING', bench, flush=True)
            rows = prepare()
            if bench.startswith('mme-realworld'):
                # Preserve upstream prompts/answers, adding source category metadata only.
                import pyarrow.dataset as ds
                raw = ROOT / 'data/raw' / repo.replace('/', '--') / 'data'
                table = ds.dataset(str(raw), format='parquet').to_table(columns=['index', 'category', 'l2-category'])
                categories = {int(x['index']): x for x in table.to_pylist()}
                for row in rows:
                    item = categories[int(row['index'])]
                    row['category'] = item['category']
                    row['l2_category'] = item['l2-category']
            path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        if len(rows) != expected[bench]:
            raise ValueError(f'{bench}: {len(rows)} != expected {expected[bench]}')
        for row in rows:
            if not row['query'].strip() or not row['response'].strip() or not all(Path(x).is_file() for x in row['images']):
                raise ValueError(f'Invalid benchmark row in {bench}')
        source = {repo: sources[repo]['revision']}
        path.with_suffix('.provenance.json').write_text(json.dumps(source, indent=2))
        manifest[bench] = {'n': len(rows), 'json': str(path.relative_to(ROOT)),
                           'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'source': source,
                           'categories': dict(Counter(x.get('category', 'unknown') for x in rows))}
        (data / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
        print('READY', bench, len(rows), flush=True)


if __name__ == '__main__':
    main()
