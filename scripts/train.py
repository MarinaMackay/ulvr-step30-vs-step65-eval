"""Print the complete command by default. --execute is an explicit, future GPU action."""
import argparse
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def command(run_id, gpus):
    run = ROOT / 'runs' / run_id
    return ['bash', str(ROOT / 'vendor/Vision-OPD/scripts/run_vision_opd.sh'),
            'trainer.nnodes=1', f'trainer.n_gpus_per_node={gpus}',
            'trainer.total_epochs=1', 'trainer.total_training_steps=65', 'trainer.save_freq=30',
            'trainer.max_actor_ckpt_to_keep=3', 'trainer.resume_mode=disable',
            f'trainer.experiment_name={run_id}', f'trainer.default_local_dir={run}/checkpoints',
            f'trainer.rollout_data_dir={run}/rollouts',
            'actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra]',
            f'actor_rollout_ref.model.path={ROOT}/models/Qwen3.5-4B',
            f'critic.model.path={ROOT}/models/Qwen3.5-4B',
            f'data.train_files=[{ROOT}/data/ulvr/train.parquet]',
            'data.train_batch_size=96', 'data.seed=42',
            'actor_rollout_ref.model.enable_gradient_checkpointing=True',
            'actor_rollout_ref.rollout.gpu_memory_utilization=0.5',
            'ray_kwargs.ray_init.num_cpus=16', '+ray_kwargs.ray_init.address=local',
            '+ray_kwargs.ray_init.num_gpus=1', '+ray_kwargs.ray_init.include_dashboard=False',
            '+ray_kwargs.ray_init.object_store_memory=17179869184']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-id', default='ulvr_vopd_4b_seed42')
    p.add_argument('--gpus', type=int, default=1)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--recheck-data', action='store_true', help='Repeat the full CPU data audit before launching')
    args = p.parse_args()
    if not args.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_id):
        p.error('Use a safe run-id')
    cmd = command(args.run_id, args.gpus)
    print(shlex.join(cmd))
    if not args.execute:
        print('PLAN ONLY: no training process, GPU allocation, or scheduler submission.')
        return
    if args.recheck_data:
        subprocess.run([sys.executable, str(ROOT / 'scripts/audit.py'), '--require-ready'], check=True)
    ready = json.loads((ROOT / 'reports/readiness.json').read_text())
    data_manifest = json.loads((ROOT / 'data/ulvr/manifest.json').read_text())
    if not ready['data_and_model_ready'] or ready['problems']:
        raise RuntimeError('Full data preparation/audit must pass before training')
    checksum = hashlib.sha256((ROOT / 'data/ulvr/train.parquet').read_bytes()).hexdigest()
    if checksum != data_manifest['parquet_sha256'] or checksum != ready['checks']['training']['train_parquet_sha256']:
        raise RuntimeError('Training parquet changed since audit')
    subprocess.run([sys.executable, str(ROOT / 'scripts/check_training_env.py')], check=True)
    model_index = json.loads((ROOT / 'models/Qwen3.5-4B/model.safetensors.index.json').read_text())
    # Three persistent full-model/optimizer checkpoints plus scratch headroom.
    minimum_free = model_index['metadata']['total_size'] * 18 + 20_000_000_000
    if shutil.disk_usage(ROOT).free < minimum_free:
        raise RuntimeError(f'Need at least {minimum_free / 1e9:.1f} GB extra free for retained full checkpoints; choose writable persistent storage before GPU launch.')
    run = ROOT / 'runs' / args.run_id
    if run.exists() and any(run.iterdir()):
        raise ValueError('Run directory already populated. Use a new run-id; automatic resume is disabled because upstream EMA-teacher restore is not verified.')
    run.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, WORLD_SIZE='1', WANDB_MODE='offline', TOKENIZERS_PARALLELISM='false',
               HF_HOME=str(ROOT / '.cache/huggingface'),
               TMPDIR=str(Path('/tmp') / ('ulvr-' + hashlib.sha256(args.run_id.encode()).hexdigest()[:10])))
    env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
    Path(env['TMPDIR']).mkdir(exist_ok=True)
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(), 'command': cmd,
                'sources': json.loads((ROOT / 'config/sources.json').read_text()),
                'data': data_manifest,
                'python': sys.executable, 'cuda_visible_devices': env.get('CUDA_VISIBLE_DEVICES'),
                'checkpoint_steps': [30, 60, 65], 'resume': 'disabled; EMA-teacher state restore not verified'}
    (run / 'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    with (run / 'train.log').open('w') as log:
        result = subprocess.run(cmd, cwd=ROOT / 'vendor/Vision-OPD', env=env, stdout=log, stderr=subprocess.STDOUT)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
