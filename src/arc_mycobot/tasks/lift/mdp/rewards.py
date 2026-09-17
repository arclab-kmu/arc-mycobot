# Copyright (c) 2026, arc-mycobot contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""The one reward term this task needs that Isaac Lab's lift package does not have."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["grasping_object"]


def grasping_object(
    env: ManagerBasedRLEnv,
    std: float,
    closed_above: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward closing the jaw *while* it is around the object.

    **Why this exists.** Upstream's lift ladder is reach -> lift -> carry, and it
    has no rung for closing the gripper: ``object_is_lifted`` only pays once the
    object is already off the ground, which needs a successful grasp first. That
    works on a Franka, whose jaw is wide enough that a policy flailing near the
    cube closes on it by accident often enough to discover the rung.

    It does not work here, and the failure was measured rather than guessed.
    Trained without this term, the policy learned the approach beautifully --
    jaw 4-7 mm from the cube, held for the whole episode -- and then hovered.
    Probing its actions across 8 environments and 250 steps, the gripper
    command was positive at *every* sample (+1.9 to +2.7), and the action term
    closes on negative values. It never once shut the jaw in 1500 iterations, so
    it never saw a lift, so nothing ever pushed the command negative.

    This term bridges that gap: it pays for being closed *and* near, so the
    gradient toward closing exists before any lift has ever happened. Once lifts
    start, ``object_is_lifted`` (weight 15) and the goal terms dominate it.

    Args:
        std: Length scale of the proximity kernel [m].
        closed_above: Finger travel above which the jaw counts as closed [m].
            The sliding finger runs 0 open to 0.0131 closed.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    distance = torch.norm(obj.data.root_pos_w - ee_frame.data.target_pos_w[..., 0, :], dim=1)
    near = 1 - torch.tanh(distance / std)
    driver = robot.data.joint_pos[:, robot_cfg.joint_ids[0]]  # type: ignore[index]
    closed = (driver > closed_above).float()
    return near * closed
