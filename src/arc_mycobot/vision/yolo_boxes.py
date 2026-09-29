"""Adapt Ultralytics detection results to the visual-alignment box contract.

The model and weights are supplied by the caller. This module does not run
YOLO or add it as a training dependency; the first task uses a red marker.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from .boxes import box_features

__all__ = ["ultralytics_box_features"]


def ultralytics_box_features(
    results: Sequence[Any],
    *,
    class_id: int,
    width: int,
    height: int,
    min_confidence: float = 0.4,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Choose the highest-confidence target-class box in each RGB image.

    Ultralytics Results expose boxes.xyxy, boxes.conf and boxes.cls. A missing
    target yields the same all-zero feature vector as the red marker detector.
    """
    xyxy = torch.zeros((len(results), 4), device=device)
    valid = torch.zeros(len(results), dtype=torch.bool, device=device)
    for i, result in enumerate(results):
        boxes = result.boxes
        if boxes is None:
            continue
        confidence = boxes.conf.to(device)
        classes = boxes.cls.to(device)
        matching = (classes.to(torch.int64) == class_id) & (confidence >= min_confidence)
        if bool(matching.any()):
            best = torch.argmax(torch.where(matching, confidence, -torch.inf))
            xyxy[i] = boxes.xyxy[best].to(device)
            valid[i] = True
    return box_features(xyxy, valid, width, height)
