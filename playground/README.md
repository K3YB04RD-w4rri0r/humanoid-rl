# Playground starter (MuJoCo Playground + JAX)

This folder is a self-contained project, separate from the Isaac Lab soccer code
at the root of the repository. It is the groundwork for a CentraleSupélec team in
the RoboCup Humanoid Soccer League.

Goal for 2026-2027: a reproducible training setup in MuJoCo Playground, then three
skills in simulation: walking, getting up, and kicking a ball. The target robot is
the Booster K1 (to be confirmed); we start from Playground's Booster T1 tasks.

## Setup

Requirements: Linux (or WSL2 on Windows), an NVIDIA GPU with a recent driver, and
[uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/K3YB04RD-w4rri0r/humanoid-rl.git
cd humanoid-rl/playground     # run every command below from this folder
uv sync --extra cuda          # GPU machine
# uv sync                     # CPU only, fine for reading code and tests
uv run python -c "import jax; print(jax.devices())"   # should list your GPUs
```

The first run downloads the robot models from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), so it
needs internet access.

### Why Playground 0.2.0 and JAX 0.9

MuJoCo Playground 1.0 (November 2026) is a rewrite on PyTorch and MuJoCo Warp. It
no longer ships the JAX locomotion tasks (Booster T1, Unitree G1 joystick, ...)
from the Playground paper, which are the baselines this project reproduces. So we
pin `playground==0.2.0`. Brax 0.14 fails on JAX 0.12, so JAX is pinned to 0.9.x.
Revisit these pins once the baselines are reproduced.

## Usage

```bash
# 1. Check the install end to end (tiny run; slow on CPU, about a minute of compile on GPU)
uv run python scripts/train.py --config configs/smoke_test.yaml --run-name smoke

# 2. Reproduce the Booster T1 walking baseline (200M steps, needs a GPU)
uv run python scripts/train.py --config configs/t1_flat.yaml

# 3. Watch the result
MUJOCO_GL=egl uv run python scripts/evaluate.py runs/<run-name> --vx 0.5

# Tests
uv run pytest
```

Each run writes to `runs/<run-name>/`: the config, the resolved PPO and
environment settings with the git commit (`run_info.json`), metrics per
evaluation (`metrics.jsonl`), a reward curve (`reward_curve.png`) and the trained
parameters (`params.pkl`). `runs/` is not committed.

Without a display (SSH, the DCE), use `--no-video` to print how far the robot
walked instead of rendering.

To train two seeds at once on a two-GPU machine:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py --config configs/t1_flat.yaml --seed 0 &
CUDA_VISIBLE_DEVICES=1 uv run python scripts/train.py --config configs/t1_flat.yaml --seed 1 &
```

## Layout

| Path | What |
| --- | --- |
| `configs/` | One YAML file per experiment: environment, seed, overrides of Playground's tuned PPO settings |
| `scripts/train.py` | Train PPO (Brax) on a Playground environment |
| `scripts/evaluate.py` | Roll out a trained policy with a speed command and save a video |
| `tests/` | Fast checks that the pinned stack works |
| `docs/learning-path.md` | Where to start if you are new to deep RL or JAX |

## Roadmap

| When | Phase | Gate |
| --- | --- | --- |
| Nov to Dec 2026 | Learn PPO and JAX, reproduce the T1 baseline, bring in the K1 model | K1 walks in simulation |
| Jan 2027 | Walking: follow speed commands, survive pushes | Robust walk |
| Feb 2027 | Getting up from front and back falls | Reliable get-up |
| Mar to Apr 2027 | Kicking a ball to a target, final demo and docs | Three-skill demo |
