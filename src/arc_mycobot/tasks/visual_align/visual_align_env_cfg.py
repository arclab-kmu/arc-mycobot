"""Robot-independent scene and MDP for RGB box centering."""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    ActionTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

import arc_mycobot.tasks.visual_align.mdp as mdp


@configclass
class VisualAlignSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING
    target: RigidObjectCfg = MISSING
    camera: TiledCameraCfg = MISSING
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    arm_action: ActionTermCfg = MISSING


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        box = ObservationTermCfg(func=mdp.detected_box)
        joint_pos = ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=MISSING)},
        )
        joint_vel = ObservationTermCfg(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=MISSING)},
        )
        actions = ObservationTermCfg(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_robot_joints = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING),
            "position_range": (-0.05, 0.05),
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_wrist = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )
    hold_wrist_target = EventTermCfg(
        func=mdp.hold_default_joint_target,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=MISSING)},
    )
    reset_hand_box_cache = EventTermCfg(func=mdp.clear_hand_box_cache, mode="reset")
    reset_target = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("target"),
            "pose_range": {"x": (0.0, 0.0), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
            "velocity_range": {},
        },
    )
    move_target = EventTermCfg(
        func=mdp.MoveTargetInImagePlane,
        mode="interval",
        interval_range_s=(1.0 / 30.0, 1.0 / 30.0),
        params={"amplitude_yz": (0.025, 0.025), "frequency_hz": (0.10, 0.13)},
    )


@configclass
class RewardsCfg:
    visible = RewardTermCfg(func=mdp.box_visible, weight=0.2)
    centering = RewardTermCfg(func=mdp.box_centering, weight=1.0, params={"std": 0.25})
    success = RewardTermCfg(func=mdp.centered_success, weight=0.05, params={"threshold": 0.1})
    action_rate = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.0001)
    joint_vel = RewardTermCfg(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=MISSING)},
    )


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)


@configclass
class VisualAlignEnvCfg(ManagerBasedRLEnvCfg):
    scene: VisualAlignSceneCfg = VisualAlignSceneCfg(num_envs=64, env_spacing=0.7)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.episode_length_s = 4.0
        self.viewer.eye, self.viewer.lookat = (0.72, 0.62, 0.67), (0.18, -0.06, 0.29)
