"""Used only by the CPU config validation harness, in place of training python3."""
import json
from pathlib import Path
import sys
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
args = sys.argv[1:]
assert args[:4] == ['-m', 'verl.trainer.main_ppo', '--config-name', 'vopd']
with initialize_config_dir(version_base=None, config_dir=str(ROOT / 'vendor/Vision-OPD/verl/trainer/config')):
    cfg = compose(config_name='vopd', overrides=args[4:])
    assert cfg.trainer.total_training_steps == 65
    assert cfg.trainer.save_freq == 30
    assert cfg.trainer.max_actor_ckpt_to_keep == 3
    assert cfg.actor_rollout_ref.actor.self_distillation.teacher_image_key == 'bbox_images'
    assert cfg.actor_rollout_ref.actor.self_distillation.distillation_topk == 100
    assert cfg.actor_rollout_ref.rollout.n == 8
    assert cfg.data.train_batch_size == 96
    assert cfg.actor_rollout_ref.actor.optim.lr == 2e-6
    assert cfg.actor_rollout_ref.actor.checkpoint.save_contents == ['model', 'optimizer', 'extra']
    assert Path(cfg.data.train_files[0]).is_file()
    assert str(ROOT / 'runs') in cfg.trainer.default_local_dir
    (ROOT / 'reports/training-config.resolved.yaml').write_text(OmegaConf.to_yaml(cfg))
    (ROOT / 'reports/training-config-validation.json').write_text(json.dumps({
        'hydra_compose_pass': True, 'training_launched': False, 'gpu_used': False,
        'overrides': args[4:], 'note': 'python3 was intercepted; no verl or torch imported'}, indent=2))
print('PASS: full upstream training CLI composed on CPU; no training was launched.')
