# Copyright (c) 2026, arc-mycobot contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""myCobot 280 JetsonNano wiring for the position-reach task.

The numbers come from v1 (tag ``v1.0.0``), where they were derived from the arm's forward kinematics, except the goal
box's near face: ``scripts/reach_workspace_sweep.py`` moves it out of the region the arm reaches only through itself.
Rerun that sweep after changing the goal box, the URDF, or the collision meshes. Lengths (``TANH_STD``,
``SUCCESS_THRESHOLD``) are scaled to this arm's 0.28 m reach, not inherited from upstream's Franka and UR10 configs.
"""

import copy
import math

from isaaclab.utils import configclass, replace

from arc_mycobot.assets.elephant_robotics import MYCOBOT_280_JN_CFG, MYCOBOT_ARM_JOINTS, MYCOBOT_FLANGE_BODY

from ... import mdp
from ...reach_env_cfg import ReachEnvCfg

##
# Task constants
##

TARGET_POS_CENTRE = (0.165, 0.0, 0.20)
"""Goal-box centre in the robot root (``joint1``) frame [m].

v1's box was (0.16, 0, 0.20) +- (0.08, 0.12, 0.08): about 53% of the flange's 302 mm horizontal reach, inside the
dexterous region rather than at full extension, and 160 mm rather than 180 mm out because at 180 the far-top-side corner
is 9.3 mm out of reach. ``reach_workspace_sweep.py`` (1 cm grid, 256 IK starts per goal, 5 mm hull clearance) found
99.9% of that box reachable without self-collision; the 4 goals that are not sit at x = 0.08 m right above the base.
The near face therefore moves from x = 0.08 to 0.09 m, the largest box whose grid goals are all reachable (94% of v1's
volume).
"""

TARGET_POS_RANGE = (0.075, 0.12, 0.08)
"""Half-extent of the goal box [m]: x 0.09-0.24, y -0.12-0.12, z 0.12-0.28; the lower face is 120 mm above the mounting
plane."""

TARGET_ORIENTATION = (math.pi, 0.0, 0.0)
"""Goal ``(roll, pitch, yaw)`` [rad], pinned: orientation is neither observed nor rewarded; flange down reads well."""

TANH_STD = 0.05
"""Length scale of the fine-grained tracking reward [m], not upstream's 0.1 (12% of a Franka's reach)."""

SUCCESS_THRESHOLD = 0.02
"""Position error counted as on-target [m], about 7% of the reach and well above the +-0.5 mm repeatability.

In v1 a trained policy settled at 2.9-4.2 mm mean error with 93% of steps inside it, most misses just after a resample.
"""

JOINT_ACTION_SCALE = 0.5
"""Radians per unit of policy action: ``q_target = q_home + scale * action``.

Derived from the goal box: IK from the home posture needs a median 0.82 rad and a p95 1.47 rad on the worst joint, so
0.5 puts the p95 at about 3 sigma of the initial ``N(0, 1)`` action. v1 first used 0.15, reasoning that a small arm
needs small motions; joint excursion does not shrink with arm size. At 0.15 the run reached 39 mm and then regressed to
56 mm as exploration noise decayed (0.9% of steps within 20 mm); at 0.5 it reached 4.2 mm and 93% (300 iterations,
4096 envs).
"""

GOAL_MARKER_SCALE = 0.06
"""Goal-pose debug marker size [m]; ``FRAME_MARKER_CFG``'s 0.5 m frame is nearly twice this arm's reach."""


##
# Environment configuration
##


@configclass
class MyCobotReachEnvCfg(ReachEnvCfg):
    """myCobot 280 JN reach with joint-position control."""

    def __post_init__(self):
        super().__post_init__()

        # switch robot to the myCobot; 0.6 m between environments for a 280 mm arm and a 240 mm wide goal box
        self.scene.robot = replace(MYCOBOT_280_JN_CFG, prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.env_spacing = 0.6

        # joint selection, named once for every term that takes a joint list
        arm_joints = MYCOBOT_ARM_JOINTS
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = arm_joints
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = arm_joints
        self.events.reset_robot_joints.params["asset_cfg"].joint_names = arm_joints
        self.rewards.joint_vel.params["asset_cfg"].joint_names = arm_joints

        # tracked body: the tool flange
        for term in (
            self.rewards.end_effector_position_tracking,
            self.rewards.end_effector_position_tracking_fine_grained,
            self.rewards.end_effector_position_success,
        ):
            term.params["asset_cfg"].body_names = [MYCOBOT_FLANGE_BODY]
        self.rewards.end_effector_position_tracking_fine_grained.params["std"] = TANH_STD
        self.rewards.end_effector_position_success.params["threshold"] = SUCCESS_THRESHOLD

        # Joint-position offsets from home rather than a Cartesian controller: with 6 DOF and tight wrist limits a
        # task-space controller has no solution over part of the goal box, so the policy resolves the goal itself.
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=arm_joints,
            scale=JOINT_ACTION_SCALE,
            use_default_offset=True,
        )

        # goal poses
        self.commands.ee_pose.body_name = MYCOBOT_FLANGE_BODY
        self.commands.ee_pose.ranges.pos_x = (
            TARGET_POS_CENTRE[0] - TARGET_POS_RANGE[0],
            TARGET_POS_CENTRE[0] + TARGET_POS_RANGE[0],
        )
        self.commands.ee_pose.ranges.pos_y = (
            TARGET_POS_CENTRE[1] - TARGET_POS_RANGE[1],
            TARGET_POS_CENTRE[1] + TARGET_POS_RANGE[1],
        )
        self.commands.ee_pose.ranges.pos_z = (
            TARGET_POS_CENTRE[2] - TARGET_POS_RANGE[2],
            TARGET_POS_CENTRE[2] + TARGET_POS_RANGE[2],
        )
        self.commands.ee_pose.ranges.roll = (TARGET_ORIENTATION[0], TARGET_ORIENTATION[0])
        self.commands.ee_pose.ranges.pitch = (TARGET_ORIENTATION[1], TARGET_ORIENTATION[1])
        self.commands.ee_pose.ranges.yaw = (TARGET_ORIENTATION[2], TARGET_ORIENTATION[2])

        # Debug markers scaled to the robot. Deep-copied because the command cfg's default shares FRAME_MARKER_CFG's
        # nested `markers` dict; mutating it in place would resize every other task's frame markers.
        for attr in ("goal_pose_visualizer_cfg", "current_pose_visualizer_cfg"):
            marker = copy.deepcopy(getattr(self.commands.ee_pose, attr))
            marker.markers["frame"].scale = (GOAL_MARKER_SCALE,) * 3
            setattr(self.commands.ee_pose, attr, marker)
