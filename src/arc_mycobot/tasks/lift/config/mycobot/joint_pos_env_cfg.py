# Copyright (c) 2026, arc-mycobot contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""myCobot 280 JN wiring for the cube-lift task.

The gripper is *actuated* here, which is the whole difference from the reach
task. See ``PARALLEL_FINGER_JOINTS`` and ``_PAD_INSET`` in
``assets/robots/mycobot_urdf.py`` for what that took: the vendor's five-joint
mimic cluster is replaced by two sliding fingers driven from one binary command,
and they get box grip pads because the vendor's collision meshes are too
asymmetric to close on anything. The rotating linkage it replaces held its pads
parallel but swept them 15.2 mm along the approach, which shoved the cube out of
the jaw and kept the task at 0% for four training runs.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

import arc_mycobot.tasks.lift.mdp as mdp
from arc_mycobot.assets.robots.mycobot_280 import (
    MYCOBOT_280_JN_LIFT_CFG,
    MYCOBOT_ARM_JOINTS,
    MYCOBOT_GRIPPER_COMMAND_JOINTS,
)
from arc_mycobot.assets.robots.mycobot_urdf import PARALLEL_CLOSED, PARALLEL_OPEN
from arc_mycobot.tasks.lift.config.mycobot.geometry import (
    CUBE_MASS,
    CUBE_SIZE,
    GOAL_POS_CENTRE,
    GOAL_POS_RANGE,
    JAW_OFFSET_IN_GRIPPER_BASE,
    LIFT_HEIGHT,
    SPAWN_POS_CENTRE,
    SPAWN_POS_RANGE,
)
from arc_mycobot.tasks.lift.lift_env_cfg import LiftEnvCfg

__all__ = ["MyCobotCubeLiftEnvCfg", "MyCobotCubeLiftEnvCfg_PLAY"]

JOINT_ACTION_SCALE = 0.5
"""Same 0.5 the reach task arrived at, and for the same reason.

Joint excursion does not shrink with arm size -- see the reach task's
``JOINT_ACTION_SCALE`` docstring for the measurement that established this and
for what going too small looks like in a training curve.
"""

REACH_STD = 0.05
"""Length scale of the "jaw near the cube" reward [m]. Upstream uses 0.1 m.

Halved because it is a length and this arm is a third of a Franka. At 0.1 the
term would still read as nearly maximal with the jaw a whole cube-width away.
"""

GOAL_STD_COARSE = 0.15
GOAL_STD_FINE = 0.03
"""Length scales of the goal-tracking pair [m]. Upstream: 0.3 and 0.05.

The coarse term has to carry signal across the whole goal region (half-diagonal
70 mm) and the fine term has to sharpen inside roughly a cube width.
"""

GRASP_STD = 0.04
"""Proximity kernel for the grasp bonus [m]. A shade wider than ``REACH_STD`` so
the bonus is already positive before the jaw is perfectly placed -- it is there
to be *found*."""

GRIPPER_CLOSED_BELOW = 0.0065
"""Finger travel **above** which the jaw counts as closed [m].

The left finger slides 0 (open) to 0.0131 (closed), so the midpoint splits them.
Note the sense flips with the parallel jaw: larger travel means more closed."""

GRIPPER_MARKER_SCALE = 0.05
"""Debug marker size [m]; the Isaac Lab default of 0.5 m dwarfs this robot."""


@configclass
class MyCobotCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = MYCOBOT_280_JN_LIFT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.env_spacing = 0.6

        # -- joints the policy sees and is penalised on: the six arm joints plus
        # the gripper *driver*. The three followers are slaved to it.
        observed = [*MYCOBOT_ARM_JOINTS, MYCOBOT_GRIPPER_COMMAND_JOINTS[0]]
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = observed
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = observed
        self.rewards.joint_vel.params["asset_cfg"].joint_names = observed

        # -- actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=MYCOBOT_ARM_JOINTS,
            scale=JOINT_ACTION_SCALE,
            use_default_offset=True,
        )
        # One binary command drives both sliding fingers, mirrored. Nothing
        # follows by constraint: the two are siblings off gripper_base, which is
        # precisely the case PhysX mimic cannot bind, so commanding both is what
        # makes the coupling exact. See MYCOBOT_GRIPPER_COMMAND_JOINTS.
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=MYCOBOT_GRIPPER_COMMAND_JOINTS,
            open_command_expr=dict(PARALLEL_OPEN),
            close_command_expr=dict(PARALLEL_CLOSED),
        )

        # -- the cube
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=SPAWN_POS_CENTRE, rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=sim_utils.CuboidCfg(
                size=(CUBE_SIZE,) * 3,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=1.0,
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=CUBE_MASS),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                # High on purpose. The pads are 24 x 20 mm of flat box and the
                # grip is a pinch with no form closure, so friction is what
                # actually holds the cube.
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.5, dynamic_friction=1.2, restitution=0.0
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.25, 0.1)),
            ),
        )
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-SPAWN_POS_RANGE[0], SPAWN_POS_RANGE[0]),
            "y": (-SPAWN_POS_RANGE[1], SPAWN_POS_RANGE[1]),
            "z": (0.0, 0.0),
        }

        # -- the point between the grip pads, not a link origin. See
        # JAW_OFFSET_IN_GRIPPER_BASE: measured, and constant across arm poses.
        marker = FRAME_MARKER_CFG.copy()
        marker.markers["frame"].scale = (GRIPPER_MARKER_SCALE,) * 3
        marker.prim_path = "/Visuals/EEFrame"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/joint1",
            debug_vis=False,
            visualizer_cfg=marker,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper_base",
                    name="jaw",
                    offset=OffsetCfg(pos=JAW_OFFSET_IN_GRIPPER_BASE),
                ),
            ],
        )

        # -- reward length scales and the lift threshold
        self.rewards.reaching_object.params["std"] = REACH_STD
        self.rewards.grasping_object.params["std"] = GRASP_STD
        self.rewards.grasping_object.params["closed_above"] = GRIPPER_CLOSED_BELOW
        self.rewards.grasping_object.params["robot_cfg"].joint_names = [MYCOBOT_GRIPPER_COMMAND_JOINTS[0]]
        self.rewards.lifting_object.params["minimal_height"] = LIFT_HEIGHT
        self.rewards.object_goal_tracking.params["std"] = GOAL_STD_COARSE
        self.rewards.object_goal_tracking.params["minimal_height"] = LIFT_HEIGHT
        self.rewards.object_goal_tracking_fine_grained.params["std"] = GOAL_STD_FINE
        self.rewards.object_goal_tracking_fine_grained.params["minimal_height"] = LIFT_HEIGHT

        # -- goal region
        self.commands.object_pose.body_name = "gripper_base"
        for axis, i in (("x", 0), ("y", 1), ("z", 2)):
            setattr(
                self.commands.object_pose.ranges,
                f"pos_{axis}",
                (GOAL_POS_CENTRE[i] - GOAL_POS_RANGE[i], GOAL_POS_CENTRE[i] + GOAL_POS_RANGE[i]),
            )
        goal_marker = self.commands.object_pose.goal_pose_visualizer_cfg.copy()
        goal_marker.markers["frame"].scale = (GRIPPER_MARKER_SCALE,) * 3
        self.commands.object_pose.goal_pose_visualizer_cfg = goal_marker
        current_marker = self.commands.object_pose.current_pose_visualizer_cfg.copy()
        current_marker.markers["frame"].scale = (GRIPPER_MARKER_SCALE,) * 3
        self.commands.object_pose.current_pose_visualizer_cfg = current_marker


@configclass
class MyCobotCubeLiftEnvCfg_PLAY(MyCobotCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 0.6
        self.observations.policy.enable_corruption = False
