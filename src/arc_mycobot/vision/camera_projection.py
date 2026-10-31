"""Batched pinhole projection of robot-frame points into a calibrated fixed camera.

Torch only, no Isaac import, so the box a policy sees in simulation can be
checked on the CPU against OpenCV's projection of the same calibration.
"""

from __future__ import annotations

import torch

from .camera_calibration import CameraCalibration, matrix_from_quaternion_wxyz

__all__ = ["FixedCameraProjector"]


class FixedCameraProjector:
    """Map points in the robot base frame to normalized image boxes.

    Boxes use the hand-policy contract on the raw, unrotated stream:
    ``(cx, cy, w, h)`` with ``cx = (x1 + x2) / width - 1`` and ``w = (x2 - x1) / width``.
    """

    def __init__(self, calibration: CameraCalibration, device: torch.device | str = "cpu", near_m: float = 0.02):
        self.width = calibration.width
        self.height = calibration.height
        self.near_m = near_m
        fx, _, cx, _, fy, cy = calibration.intrinsic_matrix[:6]
        self.focal = torch.tensor((fx, fy), dtype=torch.float32, device=device)
        self.principal = torch.tensor((cx, cy), dtype=torch.float32, device=device)
        axes = matrix_from_quaternion_wxyz(calibration.quaternion_wxyz)
        self.base_to_camera = torch.tensor(axes.T, dtype=torch.float32, device=device)
        self.position = torch.tensor(calibration.position_m, dtype=torch.float32, device=device)

    def pixels(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(..., 3)`` base-frame points to ``(..., 2)`` pixels and ``(...)`` depths."""
        camera = (points - self.position) @ self.base_to_camera.T
        depth = camera[..., 2]
        safe = depth.clamp(min=self.near_m)
        return camera[..., :2] / safe[..., None] * self.focal + self.principal, depth

    def box(self, points: torch.Tensor, min_visible: float = 0.3) -> tuple[torch.Tensor, torch.Tensor]:
        """Box around each ``(N, K, 3)`` point set, clipped to the image.

        A set counts as visible when every point is in front of the lens and at
        least ``min_visible`` of its unclipped box area lies inside the image.
        """
        pixels, depth = self.pixels(points)
        low = pixels.amin(dim=1)
        high = pixels.amax(dim=1)
        size = torch.tensor((self.width, self.height), dtype=pixels.dtype, device=pixels.device)
        clipped_low = torch.minimum(torch.maximum(low, torch.zeros_like(low)), size)
        clipped_high = torch.minimum(torch.maximum(high, torch.zeros_like(high)), size)
        area = (high - low).prod(dim=1).clamp(min=1e-6)
        inside = (clipped_high - clipped_low).prod(dim=1) / area
        visible = (depth > self.near_m).all(dim=1) & (inside >= min_visible)
        boxes = torch.cat(((clipped_low + clipped_high) / size - 1.0, (clipped_high - clipped_low) / size), dim=1)
        return torch.where(visible[:, None], boxes, torch.zeros_like(boxes)), visible
