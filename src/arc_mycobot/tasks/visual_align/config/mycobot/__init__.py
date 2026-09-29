"""Gym registration for the myCobot visual alignment task."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Visual-Align-MyCobot280JN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:MyCobotVisualAlignEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyCobotVisualAlignPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Visual-Align-MyCobot280JN-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:MyCobotVisualAlignEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyCobotVisualAlignPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Visual-Align-YOLO-Hand-MyCobot280JN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:MyCobotYoloHandEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyCobotYoloHandPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Visual-Align-YOLO-Hand-MyCobot280JN-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:MyCobotYoloHandEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyCobotYoloHandPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Visual-Align-YOLO-Hand-DR-MyCobot280JN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:MyCobotYoloHandDomainRandEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyCobotYoloHandDomainRandPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Visual-Align-YOLO-Hand-DR-MyCobot280JN-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:MyCobotYoloHandDomainRandEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyCobotYoloHandDomainRandPPORunnerCfg",
    },
)
