# Copied from mujocolab/mjlab at d872bfd (src/mjlab/tasks/velocity/config/g1/__init__.py).
# Apache License 2.0, see src/humanoid_rl/tasks/velocity/LICENSE.mjlab. Changed: imports point to this package; task IDs renamed (were Mjlab-Velocity-{Rough,Flat}-Unitree-G1).
from mjlab.tasks.registry import register_mjlab_task
from humanoid_rl.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_g1_flat_env_cfg,
  unitree_g1_rough_env_cfg,
)
from .rl_cfg import unitree_g1_ppo_runner_cfg

register_mjlab_task(
  task_id="Velocity-Rough-G1",
  env_cfg=unitree_g1_rough_env_cfg(),
  play_env_cfg=unitree_g1_rough_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Velocity-Flat-G1",
  env_cfg=unitree_g1_flat_env_cfg(),
  play_env_cfg=unitree_g1_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
