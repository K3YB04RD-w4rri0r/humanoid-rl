"""Load the Booster K1 in MuJoCo and watch it stand (PD hold) or fall (motors off).

    python -m src.view_k1                      # viewer, holds a standing pose (close the window to stop)
    python -m src.view_k1 --mode fall          # viewer, motors off
    python -m src.view_k1 --video k1.mp4       # no window, writes a 5 s video
"""

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "data/assets/K1/K1_22dof.xml"

# PD gains for holding the zero pose (straight legs, arms out). Hand-picked, not Booster's.
KP_LEG, KD_LEG = 100.0, 2.0
KP_UPPER, KD_UPPER = 20.0, 0.5


def place_on_ground(model, data):
    """Lower the floating base until the robot just touches the floor."""
    ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    lo, hi = 0.0, 1.0
    for _ in range(30):
        data.qpos[2] = (lo + hi) / 2
        mujoco.mj_forward(model, data)
        touching = any(ground in (c.geom1, c.geom2) for c in data.contact[: data.ncon])
        lo, hi = (data.qpos[2], hi) if touching else (lo, data.qpos[2])
    data.qpos[2] = hi
    mujoco.mj_forward(model, data)


def track_trunk(cam, model):
    """Camera that follows the robot's trunk."""
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = model.body("trunk").id
    cam.distance, cam.elevation, cam.azimuth = 2.0, -15.0, 135.0


def pd_gains(model):
    """Leg joints get stiff gains, head and arms soft ones."""
    names = [model.actuator(i).name for i in range(model.nu)]
    leg = np.array([("hip" in n or "knee" in n or "ankle" in n) for n in names])
    return np.where(leg, KP_LEG, KP_UPPER), np.where(leg, KD_LEG, KD_UPPER)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["stand", "fall"], default="stand")
    p.add_argument("--seconds", type=float, default=5.0, help="video length (the viewer runs until closed)")
    p.add_argument("--video", help="write an .mp4 here instead of opening the viewer")
    args = p.parse_args()

    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    place_on_ground(model, data)

    # Each motor drives one hinge joint: qpos index = jnt_qposadr, qvel index = jnt_dofadr.
    joints = model.actuator_trnid[:, 0]
    qadr, vadr = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    q_target = data.qpos[qadr].copy()
    kp, kd = pd_gains(model)

    mass = model.body_subtreemass[0]
    print(f"K1 loaded: {model.nu} motors, nq={model.nq}, mass={mass:.1f} kg, dt={model.opt.timestep} s")

    def step():
        if args.mode == "stand":
            data.ctrl[:] = kp * (q_target - data.qpos[qadr]) - kd * data.qvel[vadr]
        mujoco.mj_step(model, data)

    z0 = data.qpos[2]
    n_steps = int(args.seconds / model.opt.timestep)

    if args.video:
        import imageio_ffmpeg
        import mediapy

        mediapy.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
        fps = 30
        renderer = mujoco.Renderer(model, height=480, width=640)
        cam = mujoco.MjvCamera()
        track_trunk(cam, model)
        frames = []
        for i in range(n_steps):
            step()
            if i % int(1 / (fps * model.opt.timestep)) == 0:
                renderer.update_scene(data, camera=cam)
                frames.append(renderer.render())
        mediapy.write_video(args.video, frames, fps=fps)
        print(f"wrote {args.video}")
    else:
        from mujoco import viewer as mj_viewer

        steps_per_frame = int(1 / (60 * model.opt.timestep))  # redraw at 60 Hz
        with mj_viewer.launch_passive(model, data) as viewer:
            track_trunk(viewer.cam, model)
            start = time.time()
            while viewer.is_running():
                for _ in range(steps_per_frame):
                    step()
                viewer.sync()
                time.sleep(max(0.0, data.time - (time.time() - start)))  # real time

    z1 = data.qpos[2]
    state = "standing" if z1 > 0.8 * z0 else "fallen"
    print(f"trunk height {z0:.2f} m -> {z1:.2f} m after {data.time:.1f} s: {state}")


if __name__ == "__main__":
    main()
