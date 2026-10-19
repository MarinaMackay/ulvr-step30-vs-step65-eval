"""Exercise real upstream Bash argument expansion with python3 replaced by a CPU validator."""
import os
from pathlib import Path
import shlex
import subprocess
import sys
from train import command

ROOT = Path(__file__).resolve().parents[1]
stub = shlex.join([sys.executable, str(ROOT / 'scripts/capture_training_config.py')])
shell = 'python3() { ' + stub + ' "$@"; }; export -f python3; ' + shlex.join(command('ulvr_vopd_4b_seed42', 1))
subprocess.run(['bash', '-c', shell], cwd=ROOT / 'vendor/Vision-OPD', env=dict(os.environ, WORLD_SIZE='1'), check=True)
