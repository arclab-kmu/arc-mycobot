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
FINETUNED_MIN_CONFIDENCE = 0.25
RAW_DETECTION_CONFIDENCE = 0.05
LOWEST_CONFIDENCE = 0.01
BOX_FILTER_ALPHA = 0.4
HOME_DEG = (0.0, 0.0, 0.0, 0.0, 0.0, -45.0)
ACTION_SCALE_RAD = 0.25


@dataclass(frozen=True)
class Prediction:
    box: tuple[float, ...]
    action: tuple[float, ...]
    target_deg: tuple[float, ...]
    confidence: float = 0.0
    policy_box: tuple[float, ...] = ()


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


def _selected_hand_index(boxes, hand_class, min_confidence, reference_box=None):
    matching = (boxes.cls.int() == hand_class) & (boxes.conf >= min_confidence)
    candidates = torch.nonzero(matching).flatten()
    if len(candidates) == 0:
        return None
    if reference_box is None or reference_box[4] != 1.0:
        return candidates[torch.argmax(boxes.conf[candidates])]
    rects = boxes.xyxy[candidates]
    centers = torch.stack(
        ((rects[:, 0] + rects[:, 2]) / IMAGE_SIZE - 1.0, (rects[:, 1] + rects[:, 3]) / IMAGE_SIZE - 1.0), dim=1
    )
    reference = torch.tensor(reference_box[:2], dtype=centers.dtype, device=centers.device)
    return candidates[torch.argmin(torch.sum((centers - reference) ** 2, dim=1))]


def box_from_results(results, hand_class=HAND_CLASS, min_confidence=MIN_CONFIDENCE, reference_box=None):
    """Same normalized (cx, cy, w, h, detected) contract as training."""
    boxes = results[0].boxes
    if boxes is None or len(boxes.xyxy) == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    best = _selected_hand_index(boxes, hand_class, min_confidence, reference_box)
    if best is None:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    x1, y1, x2, y2 = [float(value) for value in boxes.xyxy[best].tolist()]
    return (
        (x1 + x2) / IMAGE_SIZE - 1.0,
        (y1 + y2) / IMAGE_SIZE - 1.0,
        (x2 - x1) / IMAGE_SIZE,
        (y2 - y1) / IMAGE_SIZE,
        1.0,
    )


def hand_confidence_from_results(results, hand_class, min_confidence=MIN_CONFIDENCE, reference_box=None):
    boxes = results[0].boxes
    if boxes is None or len(boxes.xyxy) == 0:
        return 0.0
    matching = boxes.cls.int() == hand_class
    if not bool(matching.any()):
        return 0.0
    selected = _selected_hand_index(boxes, hand_class, min_confidence, reference_box)
    if selected is not None:
        return float(boxes.conf[selected].item())
    return float(boxes.conf[matching].max().item())


def default_yolo_weights():
    override = os.environ.get("ARC_MYCOBOT_YOLO_WEIGHTS")
    if override:
        return override
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    path = cache_root / "arc-mycobot" / "weights" / YOLO_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


class HandPolicyRuntime:
    def __init__(self, checkpoint_path, device="cpu", yolo_weights=None, yolo_confidence=None):
        from ultralytics import YOLO

        self.device = str(device)
        self.actor = build_actor(checkpoint_path, self.device)
        self.detector = YOLO(yolo_weights or default_yolo_weights())
        hand_classes = [key for key, name in self.detector.names.items() if name.lower() == "human hand"]
        if len(hand_classes) != 1:
            raise ValueError("YOLO weights must have exactly one Human hand class")
        self.hand_class = hand_classes[0]
        default_confidence = FINETUNED_MIN_CONFIDENCE if self.hand_class == 0 else MIN_CONFIDENCE
        self.min_confidence = default_confidence if yolo_confidence is None else float(yolo_confidence)
        if not LOWEST_CONFIDENCE <= self.min_confidence <= 1.0:
            raise ValueError("YOLO confidence must be between 0.01 and 1.0")
        self._filtered_box = None

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
            classes=[self.hand_class],
            conf=min(RAW_DETECTION_CONFIDENCE, self.min_confidence),
            imgsz=IMAGE_SIZE,
            max_det=5,
            device=self.device,
            verbose=False,
        )
        box = box_from_results(
            detections,
            hand_class=self.hand_class,
            min_confidence=self.min_confidence,
            reference_box=self._filtered_box,
        )
        confidence = hand_confidence_from_results(detections, self.hand_class, self.min_confidence, self._filtered_box)
        if box[4] == 1.0:
            if self._filtered_box is None:
                self._filtered_box = box
            else:
                self._filtered_box = tuple(
                    (1.0 - BOX_FILTER_ALPHA) * previous + BOX_FILTER_ALPHA * current
                    for previous, current in zip(self._filtered_box[:4], box[:4])
                ) + (1.0,)
        else:
            self._filtered_box = None
        policy_box = self._filtered_box if self._filtered_box is not None else (0.0, 0.0, 0.0, 0.0, 0.0)
        angles_rel = [math.radians(float(q) - home) for q, home in zip(angles_deg, HOME_DEG)]
        velocities = [math.radians(float(v)) for v in velocities_deg_s]
        obs = torch.tensor(
            list(policy_box) + angles_rel + velocities + list(previous_action), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        if not bool(torch.isfinite(obs).all()):
            raise ValueError("non-finite observation")
        action = tuple(float(value) for value in self.actor(obs)[0].cpu().tolist())
        target = tuple(math.degrees(ACTION_SCALE_RAD * value) for value in action) + (HOME_DEG[-1],)
        return Prediction(box, action, target, confidence, policy_box)
