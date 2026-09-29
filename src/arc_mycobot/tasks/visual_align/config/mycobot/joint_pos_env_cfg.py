"""myCobot camera, marker or hand target, and joint action for visual alignment."""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

import arc_mycobot.tasks.visual_align.mdp as mdp
from arc_mycobot.assets.robots.mycobot_280 import (
    MYCOBOT_280_JN_LIFT_CFG,
    MYCOBOT_ARM_JOINTS,
)
from arc_mycobot.assets.targets import HAND_PHOTO_USD
from arc_mycobot.tasks.visual_align.visual_align_env_cfg import VisualAlignEnvCfg


@configclass
class MyCobotVisualAlignEnvCfg(VisualAlignEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = MYCOBOT_280_JN_LIFT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Isolate visual control from the uncalibrated gravity/servo model.
        self.scene.robot.spawn.rigid_props.disable_gravity = True
        # J6 alone removes the camera's 45-degree image roll.
        home = {name: 0.0 for name in MYCOBOT_ARM_JOINTS}
        home["joint6output_to_joint6"] = -math.pi / 4.0
        self.scene.robot.init_state.joint_pos = home
        self.scene.camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/camera_link/Camera",
            offset=TiledCameraCfg.OffsetCfg(convention="ros"),
            data_types=["rgb"],
            update_latest_camera_pose=True,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0,
                horizontal_aperture=20.955,
                # The target is 0.32 m away; reject red targets in cloned
                # neighbouring environments (spacing 0.7 m).
                clipping_range=(0.02, 0.6),
            ),
            width=128,
            height=128,
        )
        self.scene.target = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Target",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.380, -0.0646, 0.3946), rot=(1.0, 0.0, 0.0, 0.0)),
            spawn=sim_utils.CuboidCfg(
                size=(0.04, 0.04, 0.04),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        )
        for term in (
            self.observations.policy.joint_pos,
            self.observations.policy.joint_vel,
            self.events.reset_robot_joints,
            self.rewards.joint_vel,
        ):
            term.params["asset_cfg"].joint_names = MYCOBOT_ARM_JOINTS
        self.events.reset_robot_joints.params["asset_cfg"].joint_names = MYCOBOT_ARM_JOINTS[:-1]
        self.events.reset_wrist.params["asset_cfg"].joint_names = MYCOBOT_ARM_JOINTS[-1:]
        self.events.hold_wrist_target.params["asset_cfg"].joint_names = MYCOBOT_ARM_JOINTS[-1:]
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=MYCOBOT_ARM_JOINTS[:-1],
            scale=0.25,
            use_default_offset=True,
        )


@configclass
class MyCobotVisualAlignEnvCfg_PLAY(MyCobotVisualAlignEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8


@configclass
class MyCobotYoloHandEnvCfg(MyCobotVisualAlignEnvCfg):
    """The same wrist servo task with a textured hand target and live YOLO boxes."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.target = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Target",
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.380, -0.0646, 0.3946)),
            spawn=sim_utils.UsdFileCfg(
                usd_path=HAND_PHOTO_USD,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
            ),
        )
        self.scene.camera.width = 256
        self.scene.camera.height = 256
        self.observations.policy.box.func = mdp.detected_hand_box
        for term in (self.rewards.visible, self.rewards.centering, self.rewards.success):
            term.params["box_func"] = mdp.detected_hand_box


@configclass
class MyCobotYoloHandEnvCfg_PLAY(MyCobotYoloHandEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
