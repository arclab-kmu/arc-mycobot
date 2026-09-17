# Copyright (c) 2026, arc-mycobot contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""myCobot 280 JetsonNano wiring for the position-reach task.

Every number here is either derived from the arm's forward kinematics, scaled
from the arm's reach, or marked ``TODO(unverified)``. The three groups worth
reading before changing anything:

**Goal box.** Not guessed and not copied from a larger arm. It was chosen by
sampling the flange's reachable workspace from the repaired URDF, then *verified*
by bounded IK: 600 goals drawn uniformly from the configured ranges all solve to
under 0.003 mm within the declared joint limits (``uv run workspace_sweep``).
That 100% matters more than it would on a bigger arm -- the myCobot's working
radius is 280 mm, so a box that overhangs the workspace by a few centimetres
overhangs it by a tenth of the whole reach, and the resulting tracking error
would be a property of the goal distribution that no amount of training removes.

**Length scales.** ``TANH_STD`` and ``SUCCESS_THRESHOLD`` are lengths, and
lengths do not transfer between arms. Upstream's reach config is written for a
Franka and a UR10 -- 0.85 m and 1.3 m of reach against this arm's 0.28 m -- so
both are scaled down rather than inherited. See each one.

**Action scale.** See :data:`JOINT_ACTION_SCALE`.
"""

import copy

from isaaclab.utils import configclass

import arc_mycobot.tasks.reach.mdp as mdp
from arc_mycobot.assets.robots.mycobot_280 import (
    MYCOBOT_280_JN_CFG,
    MYCOBOT_ARM_JOINTS,
    MYCOBOT_FLANGE_BODY,
)
from arc_mycobot.tasks.reach.config.mycobot.geometry import (
    SUCCESS_THRESHOLD,
    TANH_STD,
    TARGET_ORIENTATION,
    TARGET_POS_CENTRE,
    TARGET_POS_RANGE,
)
from arc_mycobot.tasks.reach.reach_env_cfg import ReachEnvCfg

__all__ = [
    "JOINT_ACTION_SCALE",
    "SUCCESS_THRESHOLD",
    "TANH_STD",
    "TARGET_ORIENTATION",
    "TARGET_POS_CENTRE",
    "TARGET_POS_RANGE",
    "MyCobotReachEnvCfg",
    "MyCobotReachEnvCfg_PLAY",
]

##
# Task constants
##

JOINT_ACTION_SCALE = 0.5
"""Scale on the policy action before it becomes a joint-position offset [rad].

``JointPositionAction`` with ``use_default_offset=True`` computes
``q_target = q_home + scale * action``, so ``scale`` sets how many radians one
unit of policy output is worth. The policy's output starts distributed as
``N(0, 1)`` per joint, which is what makes this constant matter: an action the
policy must routinely push to 10 sigma is one it learns slowly and badly.

**Derived from the goal box, not from upstream.** Solving IK for 400 goals drawn
from the box, seeded at the home posture -- the nearest solution, which is what a
policy actually needs -- the worst joint has to move a median of 0.82 rad and
1.47 rad at the 95th percentile. Dividing through:

===========  ===========================  ==========================
``scale``    required ``|action|`` (p95)  required ``|action|`` (max)
===========  ===========================  ==========================
0.15         9.8                          18.9
0.5          2.9                          5.7
0.8          1.8                          3.5
===========  ===========================  ==========================

0.5 puts the 95th percentile at about 3 sigma, which is reachable.

This started at 0.15, reasoned *down* from upstream's 0.5 on the grounds that
this arm is a third the size of a UR10 and so needs proportionally smaller
motions. That reasoning is wrong, and instructively so: the action is in **joint
space**, and joint excursion does not shrink with arm size. A small arm sweeps
its own workspace with the same joint angles a large one does -- it is the
*Cartesian* distance that scales, and the action term never sees one.

The cost was measured, at 4096 envs for 300 iterations, seed 0:

=========  ==================  =====================
``scale``  final mean error    steps within 20 mm
=========  ==================  =====================
0.15       39 mm, then worse   0.9%
0.5        4.2 mm              93%
=========  ==================  =====================

At 0.15 the run also *regressed* after about 130 iterations rather than
plateauing -- error climbed from 39 mm back to 56 mm as the policy's exploration
noise decayed and left it unable to express the actions it needed. That shape,
improvement followed by decay, is the signature to look for if this number is
ever changed again.
"""


GOAL_MARKER_SCALE = 0.06
"""Size of the goal-pose debug marker [m]. Visualization only -- no MDP effect.

Isaac Lab's ``FRAME_MARKER_CFG`` defaults to a 0.5 m frame, which is sized for
metre-scale robots and is nearly twice this arm's 280 mm reach. Left at the
default, the axis arrows bury the robot in any recorded video or viewport. 60 mm
reads against both the arm and the 160 x 240 x 160 mm goal box.
"""


##
# Environment configuration
##


@configclass
class MyCobotReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # -- robot
        self.scene.robot = MYCOBOT_280_JN_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # 0.6 m between environments. Upstream uses 2.5 m, sized for arms that
        # are metres across; this one is 280 mm and its goal box is 240 mm wide,
        # so 0.6 m keeps neighbours well clear while keeping 4096 environments
        # inside a viewport a human can read.
        self.scene.env_spacing = 0.6

        # -- joint selection. Named once and threaded through every term that
        # takes a joint list, so that adding a seventh DOF later is one edit.
        arm_joints = MYCOBOT_ARM_JOINTS
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = arm_joints
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = arm_joints
        self.events.reset_robot_joints.params["asset_cfg"].joint_names = arm_joints
        self.rewards.joint_vel.params["asset_cfg"].joint_names = arm_joints

        # -- tracked body: the tool flange. See MYCOBOT_FLANGE_BODY.
        for term in (
            self.rewards.end_effector_position_tracking,
            self.rewards.end_effector_position_tracking_fine_grained,
            self.rewards.end_effector_position_success,
        ):
            term.params["asset_cfg"].body_names = [MYCOBOT_FLANGE_BODY]
        self.rewards.end_effector_position_tracking_fine_grained.params["std"] = TANH_STD
        self.rewards.end_effector_position_success.params["threshold"] = SUCCESS_THRESHOLD

        # -- action: joint-position offsets from the home posture.
        # Joint space rather than an IK / task-space controller, for the reason
        # that makes this arm different from a Franka: the myCobot's 6 DOF give
        # it no redundancy and its wrist limits are tight, so a Cartesian
        # controller spends a real fraction of the goal box with no solution to
        # track. The goal position enters as an observation instead and the
        # policy resolves it.
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=arm_joints,
            scale=JOINT_ACTION_SCALE,
            use_default_offset=True,
        )

        # -- goal poses
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
        # Degenerate on purpose -- orientation is not part of this task.
        self.commands.ee_pose.ranges.roll = (TARGET_ORIENTATION[0], TARGET_ORIENTATION[0])
        self.commands.ee_pose.ranges.pitch = (TARGET_ORIENTATION[1], TARGET_ORIENTATION[1])
        self.commands.ee_pose.ranges.yaw = (TARGET_ORIENTATION[2], TARGET_ORIENTATION[2])

        # -- debug markers, scaled to the robot. See GOAL_MARKER_SCALE.
        # Deep-copied because FRAME_MARKER_CFG is a module-level singleton and
        # the command cfg's default is `FRAME_MARKER_CFG.replace(prim_path=...)`,
        # which does not copy the nested `markers` dict -- mutating it in place
        # would resize the frame markers of every other task in the process.
        for attr in ("goal_pose_visualizer_cfg", "current_pose_visualizer_cfg"):
            marker = copy.deepcopy(getattr(self.commands.ee_pose, attr))
            marker.markers["frame"].scale = (GOAL_MARKER_SCALE,) * 3
            setattr(self.commands.ee_pose, attr, marker)


@configclass
class MyCobotReachEnvCfg_PLAY(MyCobotReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 0.6
        # disable randomization for play
        self.observations.policy.enable_corruption = False
