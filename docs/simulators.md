## Simulators and frameworks

### Physics engines
- [MuJoCo](https://mujoco.readthedocs.io/en/stable/overview.html): CPU
- [MJX](https://mujoco.readthedocs.io/en/stable/mjx.html): MuJoCo on GPU, in JAX or Warp
- [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp): MuJoCo on NVIDIA GPUs
- [Isaac Sim](https://github.com/isaac-sim/IsaacSim): NVIDIA, needs an RTX GPU with 16 GB ([source](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html))
- [Isaac Gym](https://developer.nvidia.com/isaac-gym): NVIDIA, deprecated

### Training frameworks
- [MuJoCo Playground v0.2.0](https://github.com/google-deepmind/mujoco_playground/tree/v0.2.0): on MJX, T1 task uses Warp ([source](https://github.com/google-deepmind/mujoco_playground/blob/v0.2.0/mujoco_playground/_src/locomotion/t1/joystick.py#L97))
- [mjlab](https://github.com/mujocolab/mjlab): on MuJoCo Warp
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab): on Isaac Sim or Newton
- [Booster Gym](https://github.com/BoosterRobotics/booster_gym): on Isaac Gym

### RL libraries
- [Brax](https://github.com/google/brax): JAX, used by Playground ([source](https://github.com/google-deepmind/mujoco_playground/blob/v0.2.0/learning/train_jax_ppo.py))
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl): PyTorch, used by mjlab ([source](https://github.com/mujocolab/mjlab/blob/main/pyproject.toml))