"""Load the Booster K1 in Isaac Lab and watch it stand (PD hold of the default pose) or fall (motors off).

Run inside the Isaac Lab container, from the repo root:

    python -m src.examples.view_k1 --headless                  # no window, 5 s, prints whether the K1 is still standing
    python -m src.examples.view_k1 --headless --mode fall      # motors off
    python -m src.examples.view_k1 --video k1.mp4              # no window, writes a 5 s video
    python -m src.examples.view_k1                             # Isaac Sim window, needs a display (close it to stop)
"""

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["stand", "fall"], default="stand")
parser.add_argument("--seconds", type=float, default=5.0, help="run length without a window (the window runs until closed)")
parser.add_argument("--video", help="write an .mp4 here instead of opening a window")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.headless = args.enable_cameras = True
simulation_app = AppLauncher(args).app

# Isaac Lab modules can only be imported once the app is running.
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from booster_train.assets.robots.booster_k1 import BOOSTER_K1_CFG  # noqa: E402

DT = 0.005  # physics step of booster_train's K1 tasks
FPS = 25
# PD gains for holding the pose. Hand-picked: Booster's (legs kp 18-60, arms kp 4) are too soft to stand without a policy.
# Booster's torque limits, torque-speed curve and command delay still apply.
KP_LEG, KD_LEG = 100.0, 2.0
KP_UPPER, KD_UPPER = 20.0, 0.5
CAM_OFFSET = (1.4, -1.4, 0.5)  # camera position relative to the trunk


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args.device))
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/ground", ground)
    light = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light.func("/World/light", light)
    robot = Articulation(BOOSTER_K1_CFG.replace(prim_path="/World/K1"))
    if args.video:
        from isaaclab.sensors import Camera, CameraCfg

        camera = Camera(
            CameraCfg(
                prim_path="/World/camera",
                height=480,
                width=640,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.1, 100.0)),
            )
        )
    sim.reset()

    # Start from booster_train's default pose (0.57 m up, arms down).
    root = robot.data.default_root_state.clone()
    robot.write_root_pose_to_sim(root[:, :7])
    robot.write_root_velocity_to_sim(root[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    robot.reset()
    robot.update(0.0)

    for group, actuator in robot.actuators.items():  # groups: legs, feet, arms, head
        leg = group in ("legs", "feet")
        stand = args.mode == "stand"
        actuator.stiffness[:] = (KP_LEG if leg else KP_UPPER) if stand else 0.0
        actuator.damping[:] = (KD_LEG if leg else KD_UPPER) if stand else 0.0

    trunk = robot.find_bodies("trunk")[0][0]
    mass = robot.data.default_mass.sum().item()
    print(f"K1 loaded: {robot.num_joints} joints, {robot.num_bodies} bodies, mass={mass:.1f} kg, dt={DT} s")

    offset = torch.tensor(CAM_OFFSET, device=sim.device)
    q_target = robot.data.default_joint_pos.clone()
    z0 = robot.data.body_pos_w[0, trunk, 2].item()
    n = 0  # physics steps taken

    def step(render):
        nonlocal n
        robot.set_joint_position_target(q_target)
        robot.write_data_to_sim()
        sim.step(render=render)
        robot.update(DT)
        n += 1

    def look_at_trunk():
        target = robot.data.body_pos_w[0, trunk]
        return target + offset, target

    steps_per_frame = round(1 / (FPS * DT))
    if args.video:
        import imageio

        frames = []
        for i in range(int(args.seconds / DT)):
            frame = i % steps_per_frame == 0
            if frame:
                eye, target = look_at_trunk()
                camera.set_world_poses_from_view(eye.unsqueeze(0), target.unsqueeze(0))
            step(render=frame)
            if frame:
                camera.update(DT * steps_per_frame)
                frames.append(camera.data.output["rgb"][0, ..., :3].cpu().numpy())
        imageio.mimwrite(args.video, frames, fps=FPS)
        print(f"wrote {args.video}")
    elif args.headless:
        for _ in range(int(args.seconds / DT)):
            step(render=False)
    else:
        start = time.time()
        while simulation_app.is_running():
            eye, target = look_at_trunk()
            sim.set_camera_view(eye.tolist(), target.tolist())
            for _ in range(steps_per_frame):
                step(render=False)
            sim.render()
            time.sleep(max(0.0, n * DT - (time.time() - start)))  # real time

    z1 = robot.data.body_pos_w[0, trunk, 2].item()
    state = "standing" if z1 > 0.8 * z0 else "fallen"
    print(f"trunk height {z0:.2f} m -> {z1:.2f} m after {n * DT:.1f} s: {state}")
    # Remove Isaac Lab's stop callback, as its environments do on close; otherwise simulation_app.close() hangs.
    sim.clear_instance()


if __name__ == "__main__":
    main()
    simulation_app.close()
