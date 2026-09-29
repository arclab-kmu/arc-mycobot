"""Perturb detector boxes while preserving the five-value deployment contract."""

from __future__ import annotations

import torch


def corrupt_box_features(
    boxes: torch.Tensor,
    center_std: torch.Tensor,
    size_std: torch.Tensor,
    dropout_probability: torch.Tensor,
    false_positive_probability: torch.Tensor,
) -> torch.Tensor:
    """Apply per-environment detector errors to (cx, cy, w, h, detected)."""
    if boxes.ndim != 2 or boxes.shape[1] != 5:
        raise ValueError(f"expected (N, 5) boxes, got {tuple(boxes.shape)}")
    for name, values in (
        ("center_std", center_std),
        ("size_std", size_std),
        ("dropout_probability", dropout_probability),
        ("false_positive_probability", false_positive_probability),
    ):
        if values.shape != (boxes.shape[0], 1):
            raise ValueError(f"{name} must have shape (N, 1), got {tuple(values.shape)}")

    out = boxes.clone()
    detected = boxes[:, 4] > 0.5
    out[:, :2] = (boxes[:, :2] + torch.randn_like(boxes[:, :2]) * center_std).clamp(-1.0, 1.0)
    out[:, 2:4] = (boxes[:, 2:4] * (1.0 + torch.randn_like(boxes[:, 2:4]) * size_std)).clamp(0.0, 1.0)
    dropped = detected & (torch.rand(boxes.shape[0], device=boxes.device) < dropout_probability[:, 0])
    out[~detected | dropped] = 0.0

    hallucinated = ~detected & (torch.rand(boxes.shape[0], device=boxes.device) < false_positive_probability[:, 0])
    count = int(hallucinated.sum())
    if count:
        out[hallucinated, :2] = torch.rand((count, 2), device=boxes.device) * 1.4 - 0.7
        out[hallucinated, 2:4] = torch.rand((count, 2), device=boxes.device) * 0.25 + 0.05
        out[hallucinated, 4] = 1.0
    return out
