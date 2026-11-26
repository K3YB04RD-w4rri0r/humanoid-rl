
## Setup

Needs [uv](https://docs.astral.sh/uv/getting-started/installation/). Training needs Linux and an NVIDIA GPU (driver 580+); running tasks without training works on Linux, macOS and Windows (WSL).

```bash
scripts/setup.sh          # K1 model
uv sync --extra cu130     # Linux + NVIDIA GPU
uv sync --extra cpu       # Linux, no NVIDIA GPU
uv sync                   # macOS
```

## Usage

```bash
uv run list-envs
uv run train Velocity-Flat-G1 --env.scene.num-envs 4096        # --gpu-ids "[1]" for the second GPU
uv run play Velocity-Flat-G1 --checkpoint-file latest          # latest trained policy
uv run play Velocity-Flat-G1 --agent random                    # check a task runs, no training needed
uv run train Getup-Flat-K1 --env.scene.num-envs 4096           # K1 getting up (from booster-mjlab)
```
