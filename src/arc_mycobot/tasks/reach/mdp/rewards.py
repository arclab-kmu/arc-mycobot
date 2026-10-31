# Copyright (c) 2026, arc-mycobot contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Reward terms specific to the position-reach task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp import position_command_error
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def position_command_success(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """1.0 while the tracked body is within ``threshold`` [m] of the goal, else 0.0.

    The task's success criterion, wired as a reward term rather than as upstream 3.0's ``pose_command_success``
    termination. As a reward it is logged per episode, so ``Episode_Reward/end_effector_position_success`` reads as
    the fraction of control steps on target. As a termination it would end the episode on first touch while the goal
    resamples every few seconds, training the policy to arrive and stop caring, and with a negative distance penalty
    dominating, ending early would itself pay. Keep the weight small: it is a step function with no gradient.
    """
    return (position_command_error(env, command_name, asset_cfg) <= threshold).float()
