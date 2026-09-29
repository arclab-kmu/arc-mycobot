"""Visible-target and image-centering rewards."""

from collections.abc import Callable

import torch
from isaaclab.envs import ManagerBasedRLEnv

from .observations import detected_box


def box_visible(env: ManagerBasedRLEnv, box_func: Callable = detected_box) -> torch.Tensor:
    """Reward keeping the configured target visible."""
    return box_func(env)[:, 4]


def box_centering(env: ManagerBasedRLEnv, std: float = 0.25, box_func: Callable = detected_box) -> torch.Tensor:
    """Dense reward for reducing image-plane box-centre error."""
    box = box_func(env)
    error_sq = torch.sum(torch.square(box[:, :2]), dim=1)
    return box[:, 4] * torch.exp(-error_sq / (std * std))


def centered_success(env: ManagerBasedRLEnv, threshold: float = 0.1, box_func: Callable = detected_box) -> torch.Tensor:
    """Diagnostic fraction with visible centre within threshold."""
    box = box_func(env)
    return box[:, 4] * (torch.linalg.vector_norm(box[:, :2], dim=1) < threshold)
