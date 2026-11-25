
## Setup

Needs NVIDIA driver 580.65.06+, Docker and the NVIDIA Container Toolkit.

```bash
scripts/setup_isaaclab.sh
docker run -it --gpus all -e ACCEPT_EULA=Y -v "$PWD":/workspace/humanoid-rl --name humanoid-rl --entrypoint bash nvcr.io/nvidia/isaac-lab:2.3.2
```

In the container:

```bash
cd /workspace/humanoid-rl/third_party/booster_train
pip install -e ../booster_assets -e source/booster_train
python scripts/csv_to_npz.py --headless --input_file=../booster_assets/motions/K1/k1_mj2_seg1.csv --input_fps=50 --output_file=../booster_assets/motions/K1/k1_mj2_seg1.npz
```

## Usage

```bash
docker start -ai humanoid-rl
```

Train:

```bash
cd /workspace/humanoid-rl/third_party/booster_train
python scripts/rsl_rl/train.py --task=Booster-K1-MJ_Dance_002-v0 --headless --device cuda:0
```

View the K1:

```bash
cd /workspace/humanoid-rl
python -m src.examples.view_k1 --video k1.mp4                # 5 s video of the K1 standing
python -m src.examples.view_k1 --video k1.mp4 --mode fall    # motors off
```
