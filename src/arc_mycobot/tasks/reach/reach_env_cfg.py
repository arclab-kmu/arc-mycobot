# Copyright (c) 2026, arc-mycobot contributors.
#
# SPDX-License-Identifier: Apache-2.0
#
# Derived from
#   isaaclab_tasks/manager_based/manipulation/reach/reach_env_cfg.py
# in Isaac Lab 2.3.2. Divergences from upstream are few, deliberate and
# commented in place; the list is in the ReachEnvCfg docstring below.

"""Robot-agnostic base for end-effector **position** reach.

The robot, the tracked body and every number that depends on the arm's size are
left ``MISSING`` here and filled in by ``config/<robot>``. That split is what
makes the numbers in the myCobot config auditable: anything robot-specific is in
one file, next to the measurement it came from.
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import arc_mycobot.tasks.reach.mdp as mdp

##
# Scene definition
##


@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # The ground is 1.05 m below the arm's mounting plane and is *cosmetic*.
    # Upstream does the same, and the reason is worth stating: this task has no
    # contact model worth the name -- self-collision is off and the arm's
    # collision geometry is convex hulls off its visual meshes -- so a floor at
    # the mounting plane would be a collision surface the policy could exploit
    # or get stuck on rather than a constraint it respects. Clearance is handled
    # where it belongs instead: the goal box is defined to sit entirely above
    # the mounting plane (see TARGET_POS_CENTRE in the robot config).
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # robots
    robot: ArticulationCfg = MISSING

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        # 3 s per goal against a 12 s episode: four goals per episode, so the
        # policy is trained on the transition between goals as much as on
        # holding one.
        resampling_time_range=(3.0, 3.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=MISSING,
            pos_y=MISSING,
            pos_z=MISSING,
            # Orientation is sampled but never observed and never rewarded --
            # this is a position-reach task. The robot config pins all three to
            # a single value so the debug marker points somewhere sensible; see
            # RewardsCfg below for why orientation is out of scope and what to
            # change to bring it back.
            roll=MISSING,
            pitch=MISSING,
            yaw=MISSING,
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group.

        The term order below **is** the observation layout. For the myCobot's six
        joints it comes to 6 + 6 + 3 + 6 = 21 numbers.

        Every joint term names its joints explicitly rather than taking the
        framework default of "every joint in the articulation". The default
        happens to select the same six here -- the gripper is welded, so the arm
        joints are the only DOFs -- but that is a fact about the current URDF
        repair, not a contract, and a future actuated gripper would silently
        widen the observation instead of failing.
        """

        # observation terms (order preserved)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=MISSING)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=MISSING)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        # Position only, three numbers -- not the pose command's seven. See
        # arc_mycobot.tasks.reach.mdp.observations.
        target_position = ObsTerm(func=mdp.position_command, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING),
            # An *offset* from the home posture, not upstream's scale of it.
            # Upstream's `reset_joints_by_scale` with a (0.5, 1.5) range
            # multiplies each joint angle, which is meaningless for a joint whose
            # home value is 0.0 -- and three of the myCobot's six are exactly
            # that, so scaling would leave half the arm starting from an
            # identical pose every episode. An offset randomizes all six.
            # +-0.2 rad is about 11 deg per joint.
            "position_range": (-0.2, 0.2),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP.

    **Position only.** Upstream's ``end_effector_orientation_tracking`` is not
    here. The scope of this task is end-effector position, and on a 6-DOF arm
    with a 280 mm reach, asking for full pose tracking over a workspace-sized
    goal box is a materially harder problem: a position the arm reaches easily
    can be one it can only reach in a single wrist configuration, so the
    orientation term would spend most of its gradient on goals that are
    position-feasible and pose-infeasible.

    To turn it back on: add an ``end_effector_orientation_tracking`` term using
    ``isaaclab_tasks...reach.mdp.orientation_command_error``, import it in
    ``mdp/__init__.py``, and replace the pinned orientation ranges in the robot
    config with a range that a reachability sweep says the arm can actually hold
    across the goal box. Do not do the first two without the third.

    The two tracking terms are upstream's, with upstream's shape and this
    robot's scale:

    * the L2 term supplies a gradient that does not saturate, so it still pulls
      from across the workspace;
    * the tanh term is what actually closes the last centimetres, and its ``std``
      is the length scale over which it is sharp -- which is why it is set in the
      robot config, not here.
    """

    # -- task terms
    end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose"},
    )
    end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "std": MISSING,
            "command_name": "ee_pose",
        },
    )
    # The success criterion. Small weight on purpose -- it is a step function
    # with no gradient, read as an instrument rather than trained against. See
    # arc_mycobot.tasks.reach.mdp.rewards.position_command_success.
    end_effector_position_success = RewTerm(
        func=mdp.position_command_success,
        weight=0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "threshold": MISSING,
            "command_name": "ee_pose",
        },
    )

    # -- action penalty
    # Both start near zero and are raised by the curriculum below: penalising
    # motion before the policy can reach anything just teaches it to stand still.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=MISSING)},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP.

    Time-out only, and that is the whole design. The goal resamples several times
    within an episode, so there is no terminal state to reach: the task is to
    track, not to arrive. Success is measured by
    ``RewardsCfg.end_effector_position_success`` instead -- see that term's
    docstring for why making it a termination would break training rather than
    tighten it.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP. Weights and step counts are upstream's."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500}
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 4500}
    )


##
# Environment configuration
##


@configclass
class ReachEnvCfg(ManagerBasedRLEnvCfg):
    """End-effector position tracking, robot-agnostic.

    Changes from Isaac Lab 2.3.2's ``manipulation/reach``:

    * **position only** -- the orientation tracking reward is dropped and the
      goal observation is the 3-vector rather than the 7-vector pose (see
      :class:`RewardsCfg`);
    * a **success** reward term, which upstream has no equivalent of;
    * ``reset_joints_by_offset`` instead of ``reset_joints_by_scale``, because
      scaling does not randomize a joint whose home angle is zero (see
      :class:`EventCfg`);
    * joint terms name their joints explicitly instead of defaulting to the
      whole articulation;
    * no table asset and no teleop devices -- neither is used by this task, and
      the table in particular is a 0.55 m-offset prop sized for a Franka, which
      on a 280 mm arm sits through the workspace;
    * 120 Hz physics rather than upstream's 60, giving a 60 Hz policy.
    """

    # Scene settings
    scene: ReachSceneCfg = ReachSceneCfg(num_envs=4096, env_spacing=1.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # 1/120 s physics with decimation 2 -> a 60 Hz policy. Upstream's reach
        # runs 60 Hz physics with the same decimation, i.e. a 30 Hz policy; this
        # arm's joints are light and its gains comparatively stiff, so the finer
        # step keeps the implicit PD loop well inside its stability margin.
        self.decimation = 2
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0
        # Framed on the goal box rather than the whole scene: this arm is 280 mm
        # across, and upstream's (3.5, 3.5, 3.5) camera leaves it a few pixels
        # tall in recorded video. (0.7, 0.7, 0.5) was the first try and was still
        # too wide -- the arm filled about a quarter of the frame height in a
        # 720p recording. These numbers were set by looking at that recording.
        self.viewer.eye = (0.45, 0.42, 0.34)
        self.viewer.lookat = (0.13, 0.0, 0.17)
