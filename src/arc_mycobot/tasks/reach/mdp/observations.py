"""Observation terms specific to the position-reach task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["position_command"]


def position_command(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The goal *position* only, in the robot's root frame.

    The framework's :func:`~isaaclab.envs.mdp.observations.generated_commands`
    returns the whole ``UniformPoseCommand``, which is seven numbers: three of
    position and a four-component quaternion. This task does not track
    orientation (see :mod:`arc_mycobot.tasks.reach.reach_env_cfg`), so its goal
    orientation range is a single fixed value -- and feeding that constant
    quaternion to the policy would add four inputs that never vary and never
    mean anything. This term takes the first three components instead.

    Returns:
        ``(num_envs, 3)`` goal position [m] in the robot root frame.
    """
    return env.command_manager.get_command(command_name)[:, :3]
