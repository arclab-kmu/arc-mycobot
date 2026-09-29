"""Frozen Open Images YOLOv8 hand detector for wrist-camera observations.

The model receives RGB frames from TiledCamera. Only class 267 (Human hand)
is allowed through prediction; PPO receives the resulting five box features.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .yolo_boxes import ultralytics_box_features

HAND_CLASS_ID = 267
MODEL_NAME = "yolov8n-oiv7.pt"
INFERENCE_SIZE = 320
MIN_CONFIDENCE = 0.05

_model: Any = None


def _model_path() -> Path:
    override = os.environ.get("ARC_MYCOBOT_YOLO_WEIGHTS")
    if override:
        return Path(override).expanduser().resolve()
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_root / "arc-mycobot" / "weights" / MODEL_NAME


def _get_model() -> Any:
    global _model
    if _model is None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install the optional YOLO dependency: uv sync --extra yolo") from exc

        path = _model_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        _model = YOLO(str(path))
        if _model.names.get(HAND_CLASS_ID) != "Human hand":
            raise ValueError(f"{path} does not provide Open Images class {HAND_CLASS_ID}: Human hand")
    return _model


@torch.inference_mode()
def detect_human_hands(rgb: torch.Tensor) -> torch.Tensor:
    """Map batched uint8 RGB (N,H,W,3) to (cx,cy,w,h,detected)."""
    if rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError(f"expected (N,H,W,3) RGB images, got {tuple(rgb.shape)}")

    # Ultralytics accepts normalized BCHW tensors. Upsampling matches its
    # 320-pixel prediction path and preserves hand detection at camera 256px.
    chw = rgb.permute(0, 3, 1, 2).float().div_(255.0)
    images = F.interpolate(chw, size=(INFERENCE_SIZE, INFERENCE_SIZE), mode="bilinear", align_corners=False)
    results = _get_model().predict(
        source=images,
        classes=[HAND_CLASS_ID],
        conf=MIN_CONFIDENCE,
        imgsz=INFERENCE_SIZE,
        max_det=1,
        device=str(rgb.device),
        verbose=False,
    )
    return ultralytics_box_features(
        results,
        class_id=HAND_CLASS_ID,
        width=INFERENCE_SIZE,
        height=INFERENCE_SIZE,
        min_confidence=MIN_CONFIDENCE,
        device=rgb.device,
    )
