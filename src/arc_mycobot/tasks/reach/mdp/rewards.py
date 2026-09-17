"""Reward terms specific to the position-reach task.

The two *tracking* terms this task trains against are not here -- they are
imported unchanged from Isaac Lab's own reach package (see :mod:`.` for why).
What is here is the success indicator, which the framework has no equivalent of.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["position_command_success"]


def position_command_success(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """1.0 while the tracked body is within ``threshold`` of the goal, else 0.0.

    This is the task's **success criterion**, and it is deliberately wired as a
    reward term rather than as a termination.

    As a reward term it is logged per-episode by the framework, so
    ``Episode_Reward/end_effector_position_success`` reads directly as the
    fraction of control steps spent on target -- the one number that says
    whether the policy is working, visible from the first hundred iterations
    without a separate evaluation pass.

    As a *termination* it would be actively harmful. The goal resamples every few
    seconds within an episode, so ending the episode on first touch would train
    the policy to arrive and stop caring, when what is wanted is arriving and
    holding. Non-time-out terminations also do not bootstrap, which on a task
    whose dominant term is a negative distance penalty makes ending the episode
    early a reward in itself.

    Its weight should stay small: it is a step function, so it carries no
    gradient toward the goal and cannot do the shaping that
    ``position_command_error_tanh`` does. It is a tie-breaker and an instrument,
    not a training signal.

    Args:
        command_name: Name of the pose command term supplying the goal.
        asset_cfg: The robot, with ``body_names`` naming the single tracked body.
        threshold: Position error at or below which the step counts as on-target [m].
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, command[:, :3])
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore[index]
    return (torch.norm(curr_pos_w - des_pos_w, dim=1) <= threshold).float()
