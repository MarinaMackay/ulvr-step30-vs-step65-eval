"""Rebase prepared data after copying the package, without resampling or touching images."""
import argparse
import hashlib
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('--from-root', required=True)
args = p.parse_args()
old = args.from_root.rstrip('/') + '/'


def rewrite(value):
    if isinstance(value, str) and value.startswith(old):
        return str(ROOT) + '/' + value[len(old):]
    if isinstance(value, dict):
        return {k: rewrite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [rewrite(x) for x in value]
    return value


for path in (ROOT / 'data/benchmarks').iterdir():
    if path.is_symlink():
        target = str(path.readlink())
        if target.startswith(old):
            path.unlink()
            path.symlink_to(rewrite(target), target_is_directory=True)

for path in [ROOT / 'data/ulvr/validation.json', *list((ROOT / 'data/benchmarks').glob('*.json'))]:
    if path.is_file():
        path.write_text(json.dumps(rewrite(json.loads(path.read_text())), ensure_ascii=False, indent=2))
samples = ROOT / 'data/ulvr/samples.jsonl'
samples.write_text(''.join(json.dumps(rewrite(json.loads(line)), ensure_ascii=False) + '\n' for line in samples.read_text().splitlines()))
file = ROOT / 'data/ulvr/train.parquet'
table = pq.read_table(file)
pq.write_table(pa.Table.from_pylist(rewrite(table.to_pylist()), schema=table.schema), file)
manifest = ROOT / 'data/ulvr/manifest.json'
obj = json.loads(manifest.read_text())
obj['parquet_sha256'] = hashlib.sha256(file.read_bytes()).hexdigest()
obj['validation_sha256'] = hashlib.sha256((ROOT / 'data/ulvr/validation.json').read_bytes()).hexdigest()
manifest.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
manifest = ROOT / 'data/benchmarks/manifest.json'
if manifest.exists():
    obj = json.loads(manifest.read_text())
    for item in obj.values():
        item['sha256'] = hashlib.sha256((ROOT / item['json']).read_bytes()).hexdigest()
    manifest.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
print('Data paths rebased. Re-run scripts/audit.py before training or evaluation.')
