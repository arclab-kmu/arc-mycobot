"""Isaac-free, Python 3.8-compatible YOLO + PPO inference for the Jetson."""

from __future__ import annotations

import hashlib
import inspect
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

HF_REPO = "arclab-kmu/arc-mycobot-yolo-hand-policy"
HF_REVISION = "main"
CHECKPOINT_NAME = "arc-mycobot-yolo-hand-policy.pt"
CHECKPOINT_SHA256 = "b38c11966cba9985add2e46c0d3877479b50206b9593d81f2fdf28adc18a7a56"
YOLO_NAME = "yolov8n-oiv7.pt"
HAND_CLASS = 267
IMAGE_SIZE = 320
MIN_CONFIDENCE = 0.05
HOME_DEG = (0.0, 0.0, 0.0, 0.0, 0.0, -45.0)
ACTION_SCALE_RAD = 0.25


@dataclass(frozen=True)
class Prediction:
    box: tuple[float, ...]
    action: tuple[float, ...]
    target_deg: tuple[float, ...]


def resolve_checkpoint(local_path=None, repo=HF_REPO, revision=HF_REVISION, filename=CHECKPOINT_NAME):
    """Download one pinned Hub file, or use a local copy."""
    if local_path is not None:
        path = Path(local_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(str(path))
        return path
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo, filename=filename, revision=revision))


def verify_checkpoint(path, expected_sha256=CHECKPOINT_SHA256):
    """Hash before torch.load, including on older Torch without weights_only."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"checkpoint SHA-256 mismatch: expected {expected_sha256}, got {actual}")


def build_actor(checkpoint_path, device="cpu"):
    """Load the exact 22 -> 128 -> 128 -> 5 RSL-RL deterministic actor."""
    verify_checkpoint(checkpoint_path)
    if "weights_only" in inspect.signature(torch.load).parameters:
        state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)["model_state_dict"]
    else:
        state = torch.load(str(checkpoint_path), map_location="cpu")["model_state_dict"]
    actor = nn.Sequential(nn.Linear(22, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 5))
    weights = {name[6:]: value for name, value in state.items() if name.startswith("actor.")}
    actor.load_state_dict(weights, strict=True)
    return actor.to(device).eval()


def box_from_results(results):
    """Same normalized (cx, cy, w, h, detected) contract as training."""
    boxes = results[0].boxes
    if boxes is None or len(boxes.xyxy) == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    matching = (boxes.cls.int() == HAND_CLASS) & (boxes.conf >= MIN_CONFIDENCE)
    if not bool(matching.any()):
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    best = torch.argmax(torch.where(matching, boxes.conf, torch.tensor(float("-inf"), device=boxes.conf.device)))
    x1, y1, x2, y2 = [float(value) for value in boxes.xyxy[best].tolist()]
    return (
        (x1 + x2) / IMAGE_SIZE - 1.0,
        (y1 + y2) / IMAGE_SIZE - 1.0,
        (x2 - x1) / IMAGE_SIZE,
        (y2 - y1) / IMAGE_SIZE,
        1.0,
    )


def default_yolo_weights():
    override = os.environ.get("ARC_MYCOBOT_YOLO_WEIGHTS")
    if override:
        return override
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    path = cache_root / "arc-mycobot" / "weights" / YOLO_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


class HandPolicyRuntime:
    def __init__(self, checkpoint_path, device="cpu", yolo_weights=None):
        from ultralytics import YOLO

        self.device = str(device)
        self.actor = build_actor(checkpoint_path, self.device)
        self.detector = YOLO(yolo_weights or default_yolo_weights())
        if self.detector.names.get(HAND_CLASS) != "Human hand":
            raise ValueError("YOLO weights lack Open Images class 267 (Human hand)")

    def warmup(self):
        """Pay the first YOLO/CUDA inference cost before opening the robot."""
        self.predict(
            torch.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=torch.uint8),
            HOME_DEG,
            [0.0] * 6,
            [0.0] * 5,
        )

    @torch.no_grad()
    def predict(self, rgb, angles_deg, velocities_deg_s, previous_action):
        """Run one uint8 HWC RGB frame; angles and velocities are J1..J6."""
        if not isinstance(rgb, torch.Tensor) or rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != torch.uint8:
            raise ValueError("expected one HWC uint8 RGB torch tensor")
        if len(angles_deg) != 6 or len(velocities_deg_s) != 6 or len(previous_action) != 5:
            raise ValueError("expected six angles, six velocities, and five previous actions")
        frame = rgb.unsqueeze(0).to(self.device).permute(0, 3, 1, 2).float().div(255.0)
        frame = F.interpolate(frame, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        detections = self.detector.predict(
            source=frame,
            classes=[HAND_CLASS],
            conf=MIN_CONFIDENCE,
            imgsz=IMAGE_SIZE,
            max_det=1,
            device=self.device,
            verbose=False,
        )
        box = box_from_results(detections)
        angles_rel = [math.radians(float(q) - home) for q, home in zip(angles_deg, HOME_DEG)]
        velocities = [math.radians(float(v)) for v in velocities_deg_s]
        obs = torch.tensor(
            list(box) + angles_rel + velocities + list(previous_action), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        if not bool(torch.isfinite(obs).all()):
            raise ValueError("non-finite observation")
        action = tuple(float(value) for value in self.actor(obs)[0].cpu().tolist())
        target = tuple(math.degrees(ACTION_SCALE_RAD * value) for value in action) + (HOME_DEG[-1],)
        return Prediction(box, action, target)
