"""Start the training run as a detached background process."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = 'ulvr_vopd_4b_seed42_20261018'
pidfile = ROOT / 'logs' / f'{RUN_ID}.pid'
if pidfile.exists():
    previous = int(pidfile.read_text())
    try:
        os.kill(previous, 0)
    except ProcessLookupError:
        pass
    else:
        status = Path(f'/proc/{previous}/stat')
        if not status.exists() or status.read_text().split()[2] != 'Z':
            raise RuntimeError(f'Training launcher {previous} already exists')
gpu = subprocess.check_output(['nvidia-smi', '--id=3', '--query-gpu=uuid,name,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip()
uuid, name, used, utilization = [x.strip() for x in gpu.split(',')]
if int(used) > 1024 or int(utilization) > 5:
    raise RuntimeError(f'Target GPU no longer idle; refusing to overlap another task: {gpu}')
logfile = ROOT / 'logs' / f'{RUN_ID}-launcher.log'
with logfile.open('a') as log:
    log.write('\nLAUNCH ATTEMPT ' + datetime.now(timezone.utc).isoformat() + '\n')
    log.flush()
    process = subprocess.Popen(['bash', str(ROOT / 'scripts/runtime.sh'), '--execute'],
                               cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
pidfile.write_text(str(process.pid) + '\n')
record = {'run_id': RUN_ID, 'hostname': socket.gethostname(), 'physical_gpu': 3, 'gpu_uuid': uuid,
          'gpu_name': name, 'launcher_pid': process.pid, 'created_utc': datetime.now(timezone.utc).isoformat(),
          'launcher_log': str(logfile), 'training_log': str(ROOT / 'runs' / RUN_ID / 'train.log'),
          'status': 'launching', 'scheduler_submitted': False, 'user_authorized': True}
(ROOT / 'reports/active-run.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
