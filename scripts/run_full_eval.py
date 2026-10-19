"""Sequential full evaluation of the trained checkpoints."""
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RUN = 'ulvr_vopd_4b_seed42_20261018'
STATUS = ROOT / 'reports/full-eval-status.json'
ENV = os.environ.copy()
ENV.update(CUDA_VISIBLE_DEVICES='3', CUDA_HOME=str(ROOT/'toolchains/cuda-12.8'),
           CC=str(ROOT/'toolchains/gcc13/bin/x86_64-conda-linux-gnu-gcc'),
           CXX=str(ROOT/'toolchains/gcc13/bin/x86_64-conda-linux-gnu-g++'),
           RUN_ID=RUN, API_BASE='http://127.0.0.1:18280/v1', WORKERS='2',
           OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', SIX_ONLY='1',
           TMPDIR='/tmp/ulvr-eval')
ENV['CUDAHOSTCXX'] = ENV['CXX']
ENV['EVAL_BENCHMARKS'] = 'vstar,zoombench,hrbench-4k,hrbench-8k'
ENV['PATH'] = ':'.join([str(ROOT/'.venv-train/bin'), str(ROOT/'toolchains/cuda-12.8/bin'),
                       str(ROOT/'toolchains/gcc13/bin'), ENV['PATH']])
Path(ENV['TMPDIR']).mkdir(exist_ok=True)

def status(phase, **kw):
    value=dict(phase=phase, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               run_id=RUN, physical_gpu=3, pid=os.getpid(), **kw)
    STATUS.write_text(json.dumps(value, indent=2)+'\n')
    print(value, flush=True)

def run(args):
    subprocess.run(args, cwd=ROOT, env=ENV, check=True)

def main():
    for step in (30, 65):
        dest=ROOT/f'runs/{RUN}/merged/step_{step}'
        if not list(dest.glob('*.safetensors')):
            status('merging', step=step)
            run(['bash', 'scripts/merge.sh', RUN, str(step)])
    for tag in ('step_30', 'step_65'):
        status('starting_server', tag=tag)
        with (ROOT/f'logs/eval-server-{tag}.log').open('a') as log:
            server=subprocess.Popen(['bash','scripts/serve.sh',tag,'--execute'],
                                    cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,
                                    start_new_session=True)
            try:
                for _ in range(1200):
                    if server.poll() is not None:
                        raise RuntimeError(f'{tag} server exited {server.returncode}')
                    try:
                        with urllib.request.urlopen(ENV['API_BASE']+'/models',timeout=3) as response:
                            ids={x['id'] for x in json.load(response)['data']}
                        if tag in ids:
                            break
                        raise RuntimeError(f'Unexpected model IDs: {ids}')
                    except (OSError, TimeoutError):
                        time.sleep(1)
                else:
                    raise TimeoutError(f'{tag} server startup timeout')
                status('inference', tag=tag, server_pid=server.pid,
                       benchmarks=4, benchmark_samples=2636, ulvr_validation_samples=0,
                       skipped=['mme-realworld','mme-realworld-cn'])
                ENV['OPENAI_MODEL_ID']=tag
                run(['bash','scripts/evaluate.sh',tag,'--infer-only'])
            finally:
                if server.poll() is None:
                    os.killpg(server.pid, signal.SIGTERM)
                    try:
                        server.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        os.killpg(server.pid, signal.SIGKILL)
                        server.wait()
    status('inference_complete_awaiting_judge', tags=['step_30','step_65'],
           required_judge='assistant semantic review (user authorized)')

if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        status('failed', error=str(exc))
        raise
