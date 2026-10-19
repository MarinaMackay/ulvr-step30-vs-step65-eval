"""Wait for this package's public downloads, then convert/audit on CPU. No training."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
deadline = time.monotonic() + 2400
while True:
    pending = []
    for repo, source in json.loads((ROOT / 'config/hub_sources.json').read_text()).items():
        if source['kind'] != 'datasets' or 'ULVR' in repo: continue
        meta = json.loads((ROOT / 'source-evidence' / f'{repo.replace("/", "--")}.json').read_text())
        for item in meta['siblings']:
            if item['rfilename'].endswith('.parquet'):
                path = ROOT / 'data/raw' / repo.replace('/', '--') / item['rfilename']
                if not path.is_file() or path.stat().st_size != item['size']:
                    pending.append(str(path.relative_to(ROOT)))
    (ROOT / 'reports/download-pending.json').write_text(json.dumps(pending, indent=2))
    print('PENDING_RAW_FILES', len(pending), flush=True)
    if not pending: break
    if time.monotonic() > deadline: raise TimeoutError('Downloads incomplete; inspect logs before resuming')
    time.sleep(30)
subprocess.run([sys.executable, '-u', str(ROOT / 'scripts/prepare_benchmarks.py')], check=True)
subprocess.run([sys.executable, '-u', str(ROOT / 'scripts/audit.py'), '--require-ready', '--verify-downloads'], check=True)
print('PREPARATION_VERIFIED: actual data/model/source are staged; no job submitted.', flush=True)
