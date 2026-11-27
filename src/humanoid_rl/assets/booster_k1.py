# Model: K1_22dof.xml from BoosterRobotics/booster_assets (BSD-3-Clause), fetched by scripts/setup.sh.
# Motor specs and gains: BoosterRobotics/booster_train at 83037fd (Apache-2.0),
# source/booster_train/booster_train/assets/robots/{booster_k1,actuator}.py.
# Layout adapted from 66Leslie/booster-mjlab at 3c5b8e1 (k1_constants.py), Apache-2.0, see LICENSE.booster-mjlab in this folder.
"""Booster K1 constants."""

import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

##
# MJCF and assets.
##

K1_XML = Path(__file__).resolve().parents[3] / "third_party/booster_assets/robots/K1/K1_22dof.xml"


def get_spec() -> mujoco.MjSpec:
  if not K1_XML.exists():
    raise FileNotFoundError(f"{K1_XML} not found. Run scripts/setup.sh.")
  spec = mujoco.MjSpec.from_file(str(K1_XML))

  # The file is a standalone scene: keep only the robot.
  for element in (*spec.actuators, *spec.lights, *spec.worldbody.geoms, *spec.materials, *spec.textures):
    spec.delete(element)
  # mjlab sets these from SimulationCfg and warns when an entity overrides them.
  default_option = mujoco.MjSpec().option
  for field in ("timestep", "tolerance", "impratio"):
    setattr(spec.option, field, getattr(default_option, field))

  # Contact settings of the booster-mjlab model the get-up task was tuned with. The file's
  # solref (1 ms) is below 2 * timestep, which MuJoCo would clamp anyway.
  for geom in spec.geoms:
    if geom.contype or geom.conaffinity:
      geom.condim = 3
      geom.priority = 1
      geom.solimp = (0.9, 0.99, 0.003, 0.5, 2.0)
      geom.solref = (0.015, 1.0)
  return spec


##
# Actuator config.
##


@dataclass(frozen=True)
class BoosterMotor:
  effort_limit: float
  """Peak torque (Nm)."""
  velocity_limit: float
  """Speed (rad/s) at which torque falls to zero."""
  knee_velocity: float
  """Speed (rad/s) up to which full torque is available."""
  armature: float


E6408 = BoosterMotor(68.0, 14.66, 1.88, 0.0478125)  # hip pitch
E4315 = BoosterMotor(43.0, 12.15, 8.38, 0.0339552)  # hip roll
E4310 = BoosterMotor(38.3, 17.59, 7.85, 0.0282528)  # hip yaw, ankles
E6416 = BoosterMotor(112.0, 12.57, 2.09, 0.095625)  # knee
R14 = BoosterMotor(14.0, 33.51, 5.24, 0.001)  # arms
HT4438 = BoosterMotor(6.0, 7.85, 10.47, 0.001)  # head


def _actuator(
  names: tuple[str, ...],
  motor: BoosterMotor,
  natural_freq: float,
  damping_ratio: float,
  armature_scale: float = 1.0,
) -> DcMotorActuatorCfg:
  armature = motor.armature * armature_scale
  omega = 2 * math.pi * natural_freq
  knee = min(motor.knee_velocity, motor.velocity_limit)
  return DcMotorActuatorCfg(
    target_names_expr=names,
    stiffness=armature * omega**2,
    damping=2 * damping_ratio * armature * omega,
    effort_limit=motor.effort_limit,
    # Booster's torque-speed curve is flat up to the knee, then linear to zero at velocity_limit.
    # mjlab's is linear from saturation_effort at rest, capped at effort_limit: this puts its corner at the knee.
    saturation_effort=motor.effort_limit * motor.velocity_limit / max(motor.velocity_limit - knee, 1e-6),
    velocity_limit=motor.velocity_limit,
    armature=armature,
    # Command delay of 2-8 physics steps (10-40 ms), sampled once per episode.
    delay_min_lag=2,
    delay_max_lag=8,
    delay_hold_prob=1.0,
  )


K1_ACTUATORS = (
  _actuator((".*_hip_pitch_joint",), E6408, 4.0, 1.5),
  _actuator((".*_hip_roll_joint",), E4315, 4.0, 1.5),
  _actuator((".*_hip_yaw_joint",), E4310, 4.0, 1.5),
  _actuator((".*_knee_pitch_joint",), E6416, 4.0, 1.0),
  # Parallel-linkage ankle, modelled as serial pitch and roll joints.
  _actuator((".*_ankle_pitch_joint",), E4310, 4.0, 1.5, armature_scale=1.4),
  _actuator((".*_ankle_roll_joint",), E4310, 4.0, 1.5, armature_scale=0.4),
  _actuator((".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_elbow_pitch_joint", ".*_elbow_yaw_joint"), R14, 10.0, 2.0),
  _actuator((".*head.*",), HT4438, 10.0, 2.0),
)

##
# Keyframes.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.543),
  joint_pos={
    "left_shoulder_roll_joint": -1.3,
    "left_elbow_yaw_joint": -0.4,
    "right_shoulder_roll_joint": 1.3,
    "right_elbow_yaw_joint": 0.4,
    ".*_hip_pitch_joint": -0.2,
    ".*_knee_pitch_joint": 0.4,
    ".*_ankle_pitch_joint": -0.2,
  },
  joint_vel={".*": 0.0},
)

##
# Final config.
##

K1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=K1_ACTUATORS,
  soft_joint_pos_limit_factor=0.9,
)


def get_k1_robot_cfg() -> EntityCfg:
  """Get a fresh K1 robot configuration instance."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    spec_fn=get_spec,
    articulation=K1_ARTICULATION,
  )


K1_ACTION_SCALE: dict[str, float] = {}
for a in K1_ARTICULATION.actuators:
  assert isinstance(a, DcMotorActuatorCfg)
  for n in a.target_names_expr:
    K1_ACTION_SCALE[n] = 0.25 * a.effort_limit / a.stiffness
