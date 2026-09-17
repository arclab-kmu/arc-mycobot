# Copyright (c) 2026, arc-mycobot contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Gym registration for the myCobot 280 JetsonNano reach task.

Registration is CPU-safe: the entry points are *strings*, so nothing from
``isaaclab`` -- and in particular nothing that repairs the vendor URDF or starts
Isaac Sim -- is imported until an environment is actually constructed. That is
what lets ``uv run list_envs`` and the test suite import this package without a
GPU.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Reach-MyCobot280JN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:MyCobotReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyCobotReachPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Reach-MyCobot280JN-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:MyCobotReachEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyCobotReachPPORunnerCfg",
    },
)
