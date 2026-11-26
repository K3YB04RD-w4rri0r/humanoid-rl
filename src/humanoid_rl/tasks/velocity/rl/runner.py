# Copied from mujocolab/mjlab at d872bfd (src/mjlab/tasks/velocity/rl/runner.py).
# Apache License 2.0, see src/humanoid_rl/tasks/velocity/LICENSE.mjlab. Changed: imports point to this package.
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper
  export_onnx_on_save = True
