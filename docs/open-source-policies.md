# Open-source policies

## Trained weights released

- [booster_deploy](https://github.com/BoosterRobotics/booster_deploy) (Booster, Apache-2.0): walking policies for the K1, T1 and T2 (`k1_walk.pt`, `t1_walk.pt`, `t2_walk.pt`), plus K1 dance and fight motion-tracking policies. Runs them in MuJoCo and on the real robot. No get-up or kick.
- [booster_gym](https://github.com/BoosterRobotics/booster_gym) (Booster, Apache-2.0): one T1 policy, `deploy/models/T1.pt`; the repo covers walking only. Trains in Isaac Gym, with a MuJoCo test script.
- [Humanoid Kick, Zenodo archive](https://zenodo.org/records/21620490) (Tsinghua, MIT): code for the RoboCup 2025 T1 soccer paper. Vision-driven kicking on the T1, trained in Isaac Gym, with a MuJoCo play script. Includes one checkpoint, `code/logs/model.pth`, which the README does not describe.

## Code only

- [HoST](https://github.com/Sage-PhysicalAI/HoST) (MIT): getting up from many postures. Isaac Gym; Unitree G1, H1, H1-2 and HighTorque Pi. The README has a section on porting to other humanoids.
- [HumanUP](https://github.com/RunpeiDong/HumanUP) (Apache-2.0): getting up from face up and face down. Isaac Gym (legged_gym + rsl_rl); Unitree G1 only.
- [humanoid-soccer](https://github.com/Daffan/humanoid-soccer) (UT Austin, Apache-2.0): the four-stage T1 kicking pipeline from the striker paper, built on Booster Gym (Isaac Gym).
- [booster_mjlab](https://github.com/IntelligentRoboticsLab/booster_mjlab) (whIRLwind, no license file): K1 walking (with AMP) and motion tracking on mjlab. The only trained policy is the K1 walking network in the project page's browser demo (`website/static/demo/policy.bin`, a custom web format, not a PyTorch checkpoint).
- [booster_train](https://github.com/BoosterRobotics/booster_train) (Booster, Apache-2.0): BeyondMimic motion tracking for the K1, T1 and T2 in Isaac Lab. No get-up or kick task.

## Not open

- [robocup_demo](https://github.com/BoosterRobotics/robocup_demo) (Booster, Apache-2.0): full RoboCup match software for the T1 and K1 (vision, behavior trees, game controller). Getting up and the RL vision kick are calls to the robot firmware (`kGetUp`, `kVisualKick`), not policies in the repo. The vision kick is K1 only and needs firmware 1.5.2 or later. The model files in the repo are vision detectors.

## Summary

- No public get-up or kick policy for the K1.
- Public K1 weights cover walking only (`k1_walk.pt` in booster_deploy).
- Public kick weights exist for the T1 only (Zenodo archive).
- Public get-up code targets Unitree robots in Isaac Gym.
