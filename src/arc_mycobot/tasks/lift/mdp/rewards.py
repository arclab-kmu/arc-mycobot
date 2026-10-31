# Copyright (c) 2026, arc-mycobot contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Reward terms this task needs that Isaac Lab's lift package does not have."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import quat_apply

from arc_mycobot.kinematics.link_geometry import link_hull_points

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["LinkFloorClearance", "grasping_object"]


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


class LinkFloorClearance(ManagerTermBase):
    """Penalty for link hulls coming closer to the table than their own margin.

    Returns ``sum(max(margin_i - lowest_z_i, 0)) / scale`` over the configured
    links, so with ``scale`` equal to a 5 mm margin a link resting on the table
    costs 1. The wrist, gripper body and camera flange carry colliders and
    keep a clearance margin: touching the table is contact a real camera should
    never make. The finger links have no colliders and margin 0, so only
    sinking below the table is penalized. Heights are measured in each
    environment's frame, whose z = 0 is the work surface the robot stands on.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        names = list(asset_cfg.body_names)
        if len(cfg.params["margins"]) != len(names):
            raise ValueError("one clearance margin per link is required")
        self.margins = torch.tensor(cfg.params["margins"], dtype=torch.float32, device=env.device)
        geometry = link_hull_points(names, cfg.params["urdf_path"])
        count = max(len(points) for points in geometry.values())
        self.points = torch.zeros((len(names), count, 3), device=env.device)
        self.valid = torch.zeros((len(names), count), dtype=torch.bool, device=env.device)
        for index, name in enumerate(names):
            points = torch.as_tensor(geometry[name], dtype=torch.float32, device=env.device)
            self.points[index, : len(points)] = points
            self.valid[index, : len(points)] = True

    def lowest(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        """``(num_envs, links)`` lowest hull height above the work surface [m]."""
        robot: Articulation = env.scene[asset_cfg.name]
        position = robot.data.body_pos_w[:, asset_cfg.body_ids] - env.scene.env_origins[:, None, :]
        rotation = robot.data.body_quat_w[:, asset_cfg.body_ids]
        count = self.points.shape[1]
        world = quat_apply(
            rotation[:, :, None, :].expand(-1, -1, count, -1), self.points[None].expand(env.num_envs, -1, -1, -1)
        )
        heights = world[..., 2] + position[:, :, None, 2]
        return torch.where(self.valid[None], heights, torch.inf).amin(dim=2)

    def __call__(
        self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, urdf_path: str, margins: list[float], scale: float
    ) -> torch.Tensor:
        return (self.margins - self.lowest(env, asset_cfg)).clamp(min=0.0).sum(dim=1) / scale
