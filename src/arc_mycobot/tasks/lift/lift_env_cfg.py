# Copyright (c) 2026, arc-mycobot contributors.
#
# SPDX-License-Identifier: Apache-2.0
#
# Derived from
#   isaaclab_tasks/manager_based/manipulation/lift/lift_env_cfg.py
# in Isaac Lab 2.3.2. Divergences are listed in the LiftEnvCfg docstring.

"""Robot-agnostic base for picking up a cube and carrying it to a goal."""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.utils import configclass

import arc_mycobot.tasks.lift.mdp as mdp

##
# Scene definition
##


@configclass
class LiftSceneCfg(InteractiveSceneCfg):
    """Robot, cube and the surface they sit on."""

    robot: ArticulationCfg = MISSING
    # The point between the grip pads. Set by the robot config.
    ee_frame: FrameTransformerCfg = MISSING
    object: RigidObjectCfg = MISSING

    # The ground is the **work surface** here, at the robot's own mounting plane
    # -- unlike the reach task, where it is cosmetic and sits 1.05 m below. The
    # cube rests on it, so it has to be where the cube is.
    #
    # Upstream instead puts a SeattleLabTable at x = 0.5 and the plane 1.05 m
    # down. That table is sized and placed for a Franka; on a 280 mm arm it
    # would stand through the middle of the workspace.
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Where the cube should end up."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=MISSING,
            pos_y=MISSING,
            pos_z=MISSING,
            # Pinned, and never observed or rewarded: this task scores where the
            # cube ends up, not how it is oriented when it gets there.
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    """Six arm joints plus one binary open/close."""

    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group.

        For the myCobot this is 7 + 7 + 3 + 3 + 7 = 27 numbers.

        The joint terms name their joints, and the list is the six arm joints
        **plus the gripper driver only**. The three follower gripper joints are
        slaved to the driver by the action term, so they carry no information the
        driver does not already give -- including them would widen the
        observation by six for nothing.
        """

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=MISSING)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=MISSING)},
        )
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        # Position only -- see arc_mycobot.tasks.reach.mdp.observations.
        target_position = ObsTerm(func=mdp.position_command, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": MISSING,
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP. Upstream's shape, this robot's length scales.

    The four task terms form the usual ladder for a pick task -- get the jaw to
    the cube, get the cube off the ground, then get it to the goal, coarse then
    fine. The weights are upstream's; only the ``std`` values, which are lengths,
    are rescaled for a 280 mm arm.
    """

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": MISSING}, weight=1.0)

    # The rung upstream's ladder is missing -- see mdp/rewards.py. Weight 4 sits
    # between the approach term (1) and the lift term (15): enough to be found,
    # not enough to be worth stopping at.
    grasping_object = RewTerm(
        func=mdp.grasping_object,
        weight=4.0,
        params={"std": MISSING, "closed_above": MISSING, "robot_cfg": SceneEntityCfg("robot", joint_names=MISSING)},
    )

    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": MISSING}, weight=15.0)

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": MISSING, "minimal_height": MISSING, "command_name": "object_pose"},
        weight=16.0,
    )

    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": MISSING, "minimal_height": MISSING, "command_name": "object_pose"},
        weight=5.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=MISSING)},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # The cube has left the work surface downward -- through the ground, which is
    # reachable here because the arm carries no collision geometry but the cube
    # does. Ending the episode is right: nothing recoverable follows.
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP.

    Upstream's weights, but **three times upstream's delay**, and that is
    measured rather than taste. The penalties jump by a factor of 1000 when they
    fire, and at ``num_steps = 10000`` -- iteration 417 at 24 steps per
    environment -- that lands while this task is still learning to grasp:

    ===================  ==================  ======================
    run                  lifted at iter 400  mean reward at iter 450
    ===================  ==================  ======================
    earlier gripper      50.6%               54.8 (dipped, recovered)
    earlier gripper      37.5%               10.2 (dipped, recovered)
    this gripper         9.2%                **-11.5** (never recovered)
    ===================  ==================  ======================

    A policy that already lifts half the time has enough task reward to absorb
    the penalty and then benefits from it. One that lifts a tenth of the time
    does not: the cheapest way to stop paying is to stop moving, and it learns
    that instead. 30000 steps puts the transition at about iteration 1250, once
    the grasp is solid.
    """

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 30000}
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 30000}
    )


##
# Environment configuration
##


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Pick up a cube and carry it to a goal position.

    Changes from Isaac Lab 2.3.2's ``manipulation/lift``:

    * the ground is the work surface rather than cosmetic, and there is no table
      prop (see :class:`LiftSceneCfg`);
    * the goal observation is the 3-vector position, not the 7-vector pose;
    * joint terms name their joints -- upstream can default to the whole
      articulation because its follower joints are two symmetric fingers; this
      gripper has three followers slaved to a driver;
    * every ``std`` and height threshold is a length and is set per robot.
    """

    scene: LiftSceneCfg = LiftSceneCfg(num_envs=4096, env_spacing=1.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 0.01  # 100 Hz physics -> a 50 Hz policy
        self.sim.render_interval = self.decimation

        # Upstream's contact-solver settings, kept verbatim: this task is
        # contact-rich in exactly the way they were tuned for.
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        # Framed on the cube, not the scene. See the reach task for why the
        # default camera is useless on an arm this size. Pulled back from the
        # reach task's numbers because this scene's ground is at the mounting
        # plane rather than 1.05 m down, so the same eye point crops the arm.
        self.viewer.eye = (0.58, 0.52, 0.44)
        self.viewer.lookat = (0.13, 0.0, 0.10)
