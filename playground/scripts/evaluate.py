"""Roll out a trained policy and save a video.

Usage:
    uv run python scripts/evaluate.py runs/<run-name> [--seconds 10] [--vx 0.5]

Rendering needs an OpenGL backend. On a headless Linux machine or the DCE, set
MUJOCO_GL=egl (NVIDIA GPU) or MUJOCO_GL=osmesa (CPU) before running.
"""

import argparse
import functools
import json
from pathlib import Path

import jax
import jax.numpy as jp
import mediapy
import yaml
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground import registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--vx", type=float, default=0.5, help="Forward speed command (m/s).")
    parser.add_argument("--vy", type=float, default=0.0, help="Sideways speed command (m/s).")
    parser.add_argument("--yaw", type=float, default=0.0, help="Turn rate command (rad/s).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-video", action="store_true", help="Skip rendering (no OpenGL needed).")
    args = parser.parse_args()

    cfg = yaml.safe_load((args.run_dir / "config.yaml").read_text())
    info = json.loads((args.run_dir / "run_info.json").read_text())
    env_name = cfg["env_name"]

    env_cfg = registry.get_default_config(env_name)
    for key, value in (cfg.get("env_overrides") or {}).items():
        env_cfg[key] = value
    env = registry.load(env_name, config=env_cfg)

    # Rebuild the same networks the run was trained with.
    net_cfg = info["ppo_params"]["network_factory"]
    networks = ppo_networks.make_ppo_networks(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=running_statistics.normalize
        if info["ppo_params"].get("normalize_observations", True)
        else lambda x, _: x,
        **{k: tuple(v) if isinstance(v, list) else v for k, v in net_cfg.items()},
    )
    params = model.load_params(str(args.run_dir / "params.pkl"))
    policy = jax.jit(ppo_networks.make_inference_fn(networks)(params, deterministic=True))

    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    rng = jax.random.PRNGKey(args.seed)
    state = reset(rng)
    command = jp.array([args.vx, args.vy, args.yaw])

    trajectory = [state]
    n_steps = int(args.seconds / env.dt)
    for _ in range(n_steps):
        state.info["command"] = command
        rng, act_rng = jax.random.split(rng)
        action, _ = policy(state.obs, act_rng)
        state = step(state, action)
        trajectory.append(state)
        if state.done:
            print("Episode ended early (robot fell or left bounds).")
            break

    start, end = trajectory[0].data.qpos[:2], trajectory[-1].data.qpos[:2]
    seconds = (len(trajectory) - 1) * env.dt
    print(
        f"Ran {seconds:.1f}s; base moved {float(end[0] - start[0]):+.2f} m in x, "
        f"{float(end[1] - start[1]):+.2f} m in y (world frame)"
    )
    if args.no_video:
        return

    render_every = 2
    frames = env.render(trajectory[::render_every], height=480, width=640)
    out = args.run_dir / f"rollout_vx{args.vx}_vy{args.vy}_yaw{args.yaw}.mp4"
    mediapy.write_video(str(out), frames, fps=1.0 / env.dt / render_every)
    print(f"Saved {out} ({len(trajectory) - 1} steps)")


if __name__ == "__main__":
    main()
