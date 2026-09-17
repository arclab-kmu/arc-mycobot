# Copyright (c) 2026, arc-mycobot contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Gym registration for the myCobot cube-lift task. String entry points only."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Lift-Cube-MyCobot280JN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:MyCobotCubeLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyCobotCubeLiftPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Lift-Cube-MyCobot280JN-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:MyCobotCubeLiftEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyCobotCubeLiftPPORunnerCfg",
    },
)
