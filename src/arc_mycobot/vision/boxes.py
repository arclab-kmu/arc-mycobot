"""RGB marker detector and box normalization shared with future YOLO input."""

from __future__ import annotations

import torch

__all__ = ["box_features", "red_cube_box"]


def box_features(xyxy: torch.Tensor, valid: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Map pixel boxes to (cx, cy, width, height, detected) observations."""
    x1, y1, x2, y2 = xyxy.unbind(dim=-1)
    result = torch.stack(
        (
            (x1 + x2) / width - 1.0,
            (y1 + y2) / height - 1.0,
            (x2 - x1) / width,
            (y2 - y1) / height,
            valid.to(xyxy.dtype),
        ),
        dim=-1,
    )
    return torch.where(valid[:, None], result, torch.zeros_like(result))


def red_cube_box(rgb: torch.Tensor) -> torch.Tensor:
    """Find a red cube's tight box in batched RGB images without depth."""
    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError(f"expected (N,H,W,3) RGB images, got {tuple(rgb.shape)}")
    _, height, width, _ = rgb.shape
    red = rgb[..., 0].to(torch.int16)
    green = rgb[..., 1].to(torch.int16)
    blue = rgb[..., 2].to(torch.int16)
    mask = (red > 100) & (red > 2 * green) & (red > 2 * blue)
    valid = mask.flatten(1).sum(dim=1) >= 4
    xs = torch.arange(width, device=rgb.device).view(1, 1, width)
    ys = torch.arange(height, device=rgb.device).view(1, height, 1)
    xyxy = torch.stack(
        (
            torch.where(mask, xs, width).amin(dim=(1, 2)),
            torch.where(mask, ys, height).amin(dim=(1, 2)),
            torch.where(mask, xs + 1, 0).amax(dim=(1, 2)),
            torch.where(mask, ys + 1, 0).amax(dim=(1, 2)),
        ),
        dim=-1,
    ).float()
    return box_features(xyxy, valid, width, height)
