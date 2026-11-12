"""Train a PPO policy on a MuJoCo Playground environment.

Usage:
    uv run python scripts/train.py --config configs/t1_flat.yaml
    uv run python scripts/train.py --config configs/smoke_test.yaml --run-name smoke

Everything a run needs to be reproduced (config, resolved PPO settings, git
commit, package versions) is written to runs/<run-name>/ next to its metrics,
reward curve and trained parameters.
"""

import argparse
import datetime
import functools
import json
import subprocess
import time
from pathlib import Path

import jax
import yaml
from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import registry, wrapper
from mujoco_playground.config import locomotion_params

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=None, help="Overrides the config seed.")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    env_name = cfg["env_name"]
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    run_name = args.run_name or (
        f"{env_name}-s{seed}-{datetime.datetime.now():%Y%m%d-%H%M%S}"
    )
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Environment and PPO settings: Playground defaults, then our overrides.
    env_cfg = registry.get_default_config(env_name)
    for key, value in (cfg.get("env_overrides") or {}).items():
        env_cfg[key] = value
    ppo_params = locomotion_params.brax_ppo_config(env_name)
    for key, value in (cfg.get("ppo_overrides") or {}).items():
        ppo_params[key] = value

    env = registry.load(env_name, config=env_cfg)
    eval_env = registry.load(env_name, config=env_cfg)
    randomizer = registry.get_domain_randomizer(env_name)

    ppo_kwargs = dict(ppo_params)
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **ppo_kwargs.pop("network_factory")
    )

    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    (run_dir / "run_info.json").write_text(
        json.dumps(
            {
                "env_name": env_name,
                "seed": seed,
                "git_commit": git_commit(),
                "devices": [str(d) for d in jax.devices()],
                "jax": jax.__version__,
                "ppo_params": ppo_params.to_dict(),
                "env_config": env_cfg.to_dict(),
            },
            indent=2,
            default=str,
        )
    )

    steps, rewards, reward_stds = [], [], []
    metrics_file = (run_dir / "metrics.jsonl").open("w")
    start = time.time()

    def progress(num_steps, metrics):
        row = {"step": int(num_steps), "time_s": round(time.time() - start, 1)}
        row.update({k: float(v) for k, v in metrics.items()})
        metrics_file.write(json.dumps(row) + "\n")
        metrics_file.flush()
        if "eval/episode_reward" in metrics:
            steps.append(num_steps)
            rewards.append(float(metrics["eval/episode_reward"]))
            reward_stds.append(float(metrics.get("eval/episode_reward_std", 0.0)))
            print(
                f"[{row['time_s']:>7.1f}s] step {num_steps:>12,} "
                f"eval reward {rewards[-1]:8.2f} +/- {reward_stds[-1]:.2f}"
            )

    print(f"Training {env_name} (seed {seed}) on {jax.devices()}; writing to {run_dir}")
    make_inference_fn, params, _ = ppo.train(
        environment=env,
        eval_env=eval_env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        randomization_fn=randomizer,
        network_factory=network_factory,
        progress_fn=progress,
        seed=seed,
        **ppo_kwargs,
    )
    metrics_file.close()
    model.save_params(str(run_dir / "params.pkl"), params)

    if steps:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(steps, rewards, yerr=reward_stds, marker="o", capsize=3)
        ax.set_xlabel("environment steps")
        ax.set_ylabel("eval episode reward")
        ax.set_title(f"{env_name}, seed {seed}")
        fig.tight_layout()
        fig.savefig(run_dir / "reward_curve.png", dpi=120)

    print(f"Done in {time.time() - start:.0f}s. Params: {run_dir / 'params.pkl'}")


if __name__ == "__main__":
    main()
