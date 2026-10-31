# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the project's task registrations."""

import gymnasium as gym

import arc_mycobot.tasks  # noqa: F401


def test_task_registrations():
    """The project tasks must expose valid environment and agent entry points."""
    expected = {
        "ARC-Reach-MyCobot280JN": {
            "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
            "env_cfg_entry_point": "arc_mycobot.tasks.reach.config.mycobot.joint_pos_env_cfg:MyCobotReachEnvCfg",
            "rsl_rl_cfg_entry_point": (
                "arc_mycobot.tasks.reach.config.mycobot.agents.rsl_rl_ppo_cfg:MyCobotReachPPORunnerCfg"
            ),
            "default_agent": "rsl_rl",
        },
    }

    registered = {
        task_id
        for task_id, spec in gym.registry.items()
        if str(spec.kwargs.get("env_cfg_entry_point", "")).startswith("arc_mycobot.tasks")
    }
    assert registered == set(expected)
    for task_id, expected_values in expected.items():
        spec = gym.spec(task_id)
        assert spec.entry_point == expected_values["entry_point"]
        for key in ("env_cfg_entry_point", "rsl_rl_cfg_entry_point", "default_agent"):
            assert spec.kwargs[key] == expected_values[key]
