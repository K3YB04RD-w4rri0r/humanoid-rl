
## Setup

#### Environment

Conda, Python 3.12:

```bash
conda create -n humanoid_env python=3.12
conda activate humanoid_env
python -m pip install -r requirements.txt    # ~1.6 GB
```

On the GPU workstation, install `jax[cuda12]==0.12.0.1` instead of `jax==0.12.0.1` (see `requirements.txt`).

#### Getting the K1

K1 model from [booster_assets](https://github.com/BoosterRobotics/booster_assets). The script downloads the model XML and its meshes into `src/data/assets/K1/`.

```bash
scripts/fetch_k1.sh
```


#### Basic scripts

```bash
python -m src.examples.view_k1                     # viewer, K1 standing (close the window to stop)
python -m src.examples.view_k1 --mode fall         # viewer, motors off
python -m src.examples.view_k1 --video k1.mp4      # no window, writes a 5 s video
```

