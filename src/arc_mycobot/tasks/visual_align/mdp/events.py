"""Move the visual target slowly across the camera image plane."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg


class MoveTargetInImagePlane(ManagerTermBase):
    """Kinematic YZ motion with random phase and no jump at episode reset."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._anchor_yz = torch.zeros((self.num_envs, 2), device=self.device)
        self._phase = torch.zeros((self.num_envs, 2), device=self.device)

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        target = self._env.scene["target"]
        self._anchor_yz[env_ids] = target.data.root_pos_w[env_ids, 1:3]
        self._phase[env_ids] = 2.0 * torch.pi * torch.rand_like(self._phase[env_ids])

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        amplitude_yz: tuple[float, float] = (0.025, 0.025),
        frequency_hz: tuple[float, float] = (0.10, 0.13),
    ) -> None:
        target = env.scene["target"]
        t = env.episode_length_buf[env_ids, None] * env.step_dt
        omega = 2.0 * torch.pi * torch.tensor(frequency_hz, device=self.device)
        amplitude = torch.tensor(amplitude_yz, device=self.device)
        yz = self._anchor_yz[env_ids] + amplitude * (
            torch.sin(t * omega + self._phase[env_ids]) - torch.sin(self._phase[env_ids])
        )
        pose = target.data.root_pose_w[env_ids].clone()
        pose[:, 1:3] = yz
        target.write_root_pose_to_sim(pose, env_ids=env_ids)


def hold_default_joint_target(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
) -> None:
    """Keep a joint's actuator target at its configured default after reset."""
    robot = env.scene[asset_cfg.name]
    target = robot.data.default_joint_pos[env_ids[:, None], asset_cfg.joint_ids].clone()
    robot.set_joint_position_target(target, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)
