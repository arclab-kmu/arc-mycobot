# Copyright (c) 2026, arc-mycobot contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""myCobot 280 JetsonNano reach environment."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="ARC-Reach-MyCobot280JN",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:MyCobotReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyCobotReachPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
