"""Download immutable public experiment assets; CPU/network only, no job submission."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import requests

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download(task):
    repo, kind, revision, item, dest = task
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = item['size']
    expected_sha = (item.get('lfs') or {}).get('sha256')
    if dest.exists() and dest.stat().st_size == expected:
        sha = digest(dest)
        if not expected_sha or sha == expected_sha:
            return {'path': str(dest.relative_to(ROOT)), 'bytes': expected, 'sha256': sha, 'reused': True}
    partial = dest.with_name(dest.name + '.partial')
    if partial.exists() and partial.stat().st_size >= expected:
        partial_sha = digest(partial)
        if partial.stat().st_size == expected and (not expected_sha or partial_sha == expected_sha):
            partial.replace(dest)
            return {'path': str(dest.relative_to(ROOT)), 'bytes': expected, 'sha256': partial_sha, 'reused': True}
        partial.unlink()
    prefix = 'datasets/' if kind == 'datasets' else ''
    url = f'https://huggingface.co/{prefix}{repo}/resolve/{revision}/{item["rfilename"]}'
    for attempt in range(6):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {'Range': f'bytes={offset}-'} if offset else {}
            with requests.get(url, params={'download': 'true', 't': int(time.time())}, headers=headers,
                              stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                mode = 'ab' if offset and response.status_code == 206 else 'wb'
                with partial.open(mode) as out:
                    for block in response.iter_content(4 * 1024 * 1024):
                        if block:
                            out.write(block)
            if partial.stat().st_size != expected:
                raise ValueError(f'Wrong length: {partial.stat().st_size} vs {expected}')
            sha = digest(partial)
            if expected_sha and sha != expected_sha:
                partial.unlink()
                raise ValueError('Hub LFS SHA256 mismatch')
            partial.replace(dest)
            print('DOWNLOADED', repo, item['rfilename'], expected, flush=True)
            return {'path': str(dest.relative_to(ROOT)), 'bytes': expected, 'sha256': sha, 'reused': False}
        except Exception as exc:
            print('RETRY', repo, item['rfilename'], attempt + 1, type(exc).__name__, str(exc)[:150], flush=True)
            if attempt == 5:
                raise
            time.sleep(min(30, 2 ** attempt))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', choices=['ulvr', 'benchmarks', 'model'], required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--train-shards', type=int, default=8)
    args = parser.parse_args()
    sources = json.loads((ROOT / 'config/hub_sources.json').read_text())
    tasks = []
    for repo, source in sources.items():
        if args.group == 'ulvr' and repo != 'RuoliuYang/ULVR_v2_clean':
            continue
        if args.group == 'model' and source['kind'] != 'models':
            continue
        if args.group == 'benchmarks' and (source['kind'] != 'datasets' or 'ULVR' in repo):
            continue
        meta = json.loads((ROOT / 'source-evidence' / f'{repo.replace("/", "--")}.json').read_text())
        for item in meta['siblings']:
            name = item['rfilename']
            if args.group == 'ulvr':
                trains = sorted(x['rfilename'] for x in meta['siblings'] if x['rfilename'].startswith('data/bbox_crop/'))[:args.train_shards]
                include = name in trains or name.startswith('data/bbox_crop_val/') or name == 'README.md'
                dest = ROOT / 'data/raw/ULVR' / name
            elif args.group == 'benchmarks':
                include = name.endswith('.parquet') or name == 'README.md'
                dest = ROOT / 'data/raw' / repo.replace('/', '--') / name
            else:
                include = not name.startswith('.') and not name.endswith(('.md', '.png'))
                dest = ROOT / 'models/Qwen3.5-4B' / name
            if include:
                tasks.append((repo, source['kind'], source['revision'], item, dest))
    print('PLAN', args.group, len(tasks), sum(t[3]['size'] for t in tasks), flush=True)
    audit = ROOT / 'reports' / f'download-{args.group}.jsonl'
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download, task) for task in tasks]
        for future in as_completed(futures):
            result = future.result()
            with audit.open('a') as out:
                out.write(json.dumps(result) + '\n')
    print('COMPLETE', args.group, flush=True)


if __name__ == '__main__':
    main()
