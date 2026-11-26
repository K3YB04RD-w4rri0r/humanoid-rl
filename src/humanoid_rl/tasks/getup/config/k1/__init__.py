# Copied from 66Leslie/booster-mjlab at 3c5b8e1 (src/mjlab_playground/tasks/getup/config/k1/__init__.py).
# Apache License 2.0, see src/humanoid_rl/tasks/getup/LICENSE.booster-mjlab. Lineage: mujocolab/mjlab_playground (Apache-2.0).
# Changed: imports point to this package; task ID renamed (was Mjlab-Getup-Flat-Booster-K1).
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import booster_k1_getup_env_cfg
from .rl_cfg import booster_k1_getup_ppo_runner_cfg

register_mjlab_task(
  task_id="Getup-Flat-K1",
  env_cfg=booster_k1_getup_env_cfg(),
  play_env_cfg=booster_k1_getup_env_cfg(play=True),
  rl_cfg=booster_k1_getup_ppo_runner_cfg(),
)
