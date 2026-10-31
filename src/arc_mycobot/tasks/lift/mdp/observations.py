"""Cube box seen by the fixed external camera, as a real detector would report it."""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply, subtract_frame_transforms

from arc_mycobot.vision.camera_calibration import external_camera_calibration
from arc_mycobot.vision.camera_projection import FixedCameraProjector

MAX_AGE_S = 1.0


class ExternalCubeBox(ManagerTermBase):
    """Last seen cube box in the external camera, a fresh-detection flag, and its age.

    Output per environment: ``(cx, cy, w, h, detected, age)``. The box keeps the
    last detection while the cube is missed (the cube does not move before it
    is grasped), and ``age`` counts seconds since that detection, capped at 1.
    The box is the projection of the cube's eight corners through the packaged
    camera calibration, not a render, so it ignores occlusion by the arm. Each
    episode draws its own detector error levels: centre and size jitter and a
    per-step miss probability.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.projector = FixedCameraProjector(external_camera_calibration(), device=self.device)
        half = 0.5 * float(cfg.params["cube_size"])
        signs = torch.tensor(
            [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)], device=self.device
        )
        self.corners = signs * half
        self.box = torch.zeros((self.num_envs, 4), device=self.device)
        self.age = torch.full((self.num_envs,), MAX_AGE_S, device=self.device)
        self.center_std = torch.zeros((self.num_envs, 1), device=self.device)
        self.size_std = torch.zeros_like(self.center_std)
        self.miss_probability = torch.zeros(self.num_envs, device=self.device)
        self.reset()

    def reset(self, env_ids=None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        params = self.cfg.params
        self.box[ids] = 0.0
        self.age[ids] = MAX_AGE_S
        self.center_std[ids] = torch.rand_like(self.center_std[ids]) * params["center_std_max"]
        self.size_std[ids] = torch.rand_like(self.size_std[ids]) * params["size_std_max"]
        self.miss_probability[ids] = torch.rand_like(self.miss_probability[ids]) * params["miss_max"]

    def projected_box(self, env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg, robot_cfg: SceneEntityCfg):
        """Noise-free box of the cube's eight corners and whether it is in view; no state change."""
        robot = env.scene[robot_cfg.name]
        cube = env.scene[object_cfg.name]
        position, orientation = subtract_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w, cube.data.root_pos_w, cube.data.root_quat_w
        )
        corners = position[:, None, :] + quat_apply(
            orientation[:, None, :].expand(-1, self.corners.shape[0], -1),
            self.corners[None].expand(self.num_envs, -1, -1),
        )
        return self.projector.box(corners)

    def latch(self, box: torch.Tensor, detected: torch.Tensor, step_dt: float) -> torch.Tensor:
        """Keep the last detected box and count seconds since it was seen."""
        self.box[detected] = box[detected]
        self.age = torch.where(detected, torch.zeros_like(self.age), (self.age + step_dt).clamp(max=MAX_AGE_S))
        return torch.cat((self.box, detected.float()[:, None], self.age[:, None]), dim=1)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        cube_size: float,
        center_std_max: float,
        size_std_max: float,
        miss_max: float,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        box, visible = self.projected_box(env, object_cfg, robot_cfg)
        noisy = box.clone()
        noisy[:, :2] = (box[:, :2] + torch.randn_like(box[:, :2]) * self.center_std).clamp(-1.0, 1.0)
        noisy[:, 2:] = (box[:, 2:] * (1.0 + torch.randn_like(box[:, 2:]) * self.size_std)).clamp(0.0, 2.0)
        detected = visible & (torch.rand(self.num_envs, device=self.device) >= self.miss_probability)
        return self.latch(noisy, detected, env.step_dt)
