# Copyright (c) 2026, arc-mycobot contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Observation terms specific to the position-reach task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def position_command(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The goal *position* only, in the robot's root frame.

    ``generated_commands`` returns the whole pose command, three position and four quaternion components. This task
    does not track orientation and pins the goal orientation to one value, so the quaternion would be four inputs that
    never vary. This term keeps the first three.

    Returns:
        ``(num_envs, 3)`` goal position [m] in the robot root frame.
    """
    return env.command_manager.get_command(command_name)[:, :3]
