"""Check the eventual runtime without launching workers or allocating CUDA tensors."""
import importlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
modules = ['torch', 'transformers', 'vllm', 'ray', 'datasets', 'flash_attn', 'causal_conv1d', 'fla']
report = {'python': sys.version, 'imports': {}, 'gpu_training_started': False}
for name in modules:
    try:
        module = importlib.import_module(name)
        report['imports'][name] = {'ok': True, 'version': getattr(module, '__version__', 'not exposed')}
    except Exception as exc:
        report['imports'][name] = {'ok': False, 'error': str(exc)}
result = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
report['pip_check'] = result.stdout + result.stderr
known = []
unknown = []
for line in report['pip_check'].splitlines():
    if not line.strip() or line.startswith('No broken requirements'):
        continue
    if line.lower().startswith('vllm ') and 'transformers' in line.lower():
        known.append(line)
    else:
        unknown.append(line)
report['known_upstream_metadata_override'] = known
report['unexpected_dependency_errors'] = unknown
report['runtime_imports_pass'] = all(x['ok'] for x in report['imports'].values()) and not unknown
(ROOT / 'reports/training-environment.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
sys.exit(0 if report['runtime_imports_pass'] else 1)
