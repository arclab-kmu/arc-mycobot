# Copyright (c) 2026, arc-mycobot contributors.
#
# SPDX-License-Identifier: Apache-2.0
#
# Derived from isaaclab_tasks/core/reach/reach_env_cfg.py in Isaac Lab 3.0.0 (BSD-3-Clause). The divergences are
# listed in the ReachEnvCfg docstring.

"""Robot-agnostic base for end-effector **position** reach.

The robot, the tracked body, and every number that depends on the arm's size are ``MISSING`` here and set by
``config/<robot>``, so everything robot-specific sits in one file next to the measurement it came from.
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
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.visualizers import VisualizerCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab_tasks.utils import PresetCfg

from . import mdp

##
# Physics backend presets
##


@configclass
class ReachPhysicsCfg(PresetCfg):
    """Physics backend presets; upstream reach's values."""

    isaacsim_physx: PhysxCfg = PhysxCfg(bounce_threshold_velocity=0.2)
    ovphysx: OvPhysxCfg = OvPhysxCfg()
    physx: PhysxAutoCfg = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)

    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=100,
            nconmax=20,
            cone="pyramidal",
            integrator="implicitfast",
            impratio=1,
            update_data_interval=2,
        ),
        num_substeps=2,
        debug_mode=False,
        use_cuda_graph=True,
    )
    default: NewtonCfg = newton_mjwarp


##
# Scene definition
##


@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # 1.05 m below the mounting plane and cosmetic, as upstream. The arm collides through convex hulls, so a floor at
    # the mounting plane would be a surface to exploit or get stuck on. The goal box sits entirely above the mounting
    # plane instead (TARGET_POS_CENTRE in the robot config).
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # robots
    robot: ArticulationCfg = MISSING

    # Every robot body, for the self-contact penalty; nothing else in the scene is within the arm's reach.
    self_contacts = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=2)

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
        # 3 s per goal in a 12 s episode: four goals per episode, so moving between goals is trained as much as holding.
        resampling_time_range=(3.0, 3.0),
        debug_vis=True,
        # Orientation is sampled but never observed or rewarded; the robot config pins it to one value so the debug
        # marker points somewhere sensible.
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=MISSING,
            pos_y=MISSING,
            pos_z=MISSING,
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

        The term order is the observation layout; for six joints it is 6 + 6 + 3 + 6 = 21 numbers. Joint terms name
        their joints so a future actuated joint fails loudly instead of silently widening the observation.
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
        # Position only: three numbers, not the pose command's seven.
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
            # An offset from the home posture, not upstream's scale of it: scaling leaves a joint whose home angle is
            # zero (three of the myCobot's six) at the same angle every episode. +-0.2 rad is about 11 deg per joint.
            "position_range": (-0.2, 0.2),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP.

    **Position only.** On a 6-DOF arm with a 280 mm reach, full pose tracking over a workspace-sized goal box is a much
    harder problem: a position reachable in only one wrist configuration would spend the orientation term's gradient on
    goals that are position-feasible and pose-infeasible. To restore it, add ``orientation_command_error`` and replace
    the pinned orientation ranges with ones a reachability sweep says the arm can hold across the goal box.

    The L2 term gives a gradient that does not saturate across the workspace; the tanh term closes the last
    centimetres, and its ``std`` is a length set in the robot config.
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
    # The success criterion; small weight on purpose. See mdp.position_command_success.
    end_effector_position_success = RewTerm(
        func=mdp.position_command_success,
        weight=0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "threshold": MISSING,
            "command_name": "ee_pose",
        },
    )

    # -- self-contact. Contacts are soft enough that a policy pushing at the torque limit sinks one link 20+ mm into
    # another, so self-collision alone does not stop it. Each touching body costs more than the task terms can pay.
    self_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={"sensor_cfg": SceneEntityCfg("self_contacts", body_names=".*"), "threshold": 0.1},
    )

    # -- action penalty: start near zero and rise with the curriculum, so the policy does not learn to stand still.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=MISSING)},
    )


@configclass
class TerminationsCfg:
    """Time-out only: the goal resamples within an episode, so there is no terminal state to reach."""

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

    Changes from Isaac Lab 3.0.0's ``core/reach`` (all carried over from v1):

    * **position only**: no orientation reward, no ``action_magnitude`` term, and a 3-vector goal observation;
    * a success *reward* term instead of upstream's success termination;
    * ``reset_joints_by_offset`` instead of ``reset_joints_by_scale``;
    * joint terms name their joints explicitly;
    * no table asset: upstream's is sized for a Franka and would sit through a 280 mm arm's workspace;
    * decimation 2 at 120 Hz physics, a 60 Hz policy instead of upstream's 30 Hz;
    * new in v2: a contact sensor on every robot body and a ``self_contact`` penalty, for robots spawned with
      self-collision on.
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
        # general settings
        self.decimation = 2
        self.episode_length_s = 12.0
        # simulation settings
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physics = ReachPhysicsCfg()
        # visualizer settings: framed on the goal box, since upstream's (3.5, 3.5, 3.5) leaves a 280 mm arm a few
        # pixels tall in recorded video.
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(0.45, 0.42, 0.34), lookat=(0.13, 0.0, 0.17))
