"""Fine-tune a one-class YOLO hand detector from the Hand Keypoints pose dataset.

Run on a CUDA workstation, not on the Jetson Nano. Keep the downloaded dataset
and resulting checkpoint outside Git; see deploy/README.md.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path


def prepare_detection_dataset(pose_dataset: Path, detection_dataset: Path) -> Path:
    """Keep pose images; convert each keypoint row to its first five YOLO box fields."""
    pose_dataset = pose_dataset.expanduser().resolve()
    detection_dataset = detection_dataset.expanduser().resolve()
    for split in ("train", "val"):
        image_dir = pose_dataset / "images" / split
        label_dir = pose_dataset / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"missing pose dataset split: {split}")
        target_images = detection_dataset / "images" / split
        if target_images.is_symlink():
            raise ValueError(f"remove image directory symlink before converting: {target_images}")
        target_images.mkdir(parents=True, exist_ok=True)
        images = sorted(image_dir.glob("*.jpg"))
        for source in images:
            target = target_images / source.name
            if not target.exists():
                os.link(source, target)
            elif not target.samefile(source):
                raise ValueError(f"image hardlink points to another dataset: {target}")
        target_labels = detection_dataset / "labels" / split
        target_labels.mkdir(parents=True, exist_ok=True)
        labels = sorted(label_dir.glob("*.txt"))
        if len(labels) != len(images):
            raise ValueError(f"image/label count differs for {split}")
        for source in labels:
            if not (image_dir / f"{source.stem}.jpg").is_file():
                raise ValueError(f"pose label has no matching image: {source}")
            rows = []
            for row in source.read_text().splitlines():
                fields = row.split()
                if len(fields) < 5 or fields[0] != "0":
                    raise ValueError(f"invalid hand pose label: {source}")
                cx, cy, width, height = [float(value) for value in fields[1:5]]
                x1, x2 = max(0.0, cx - width / 2), min(1.0, cx + width / 2)
                y1, y2 = max(0.0, cy - height / 2), min(1.0, cy + height / 2)
                if not all(math.isfinite(value) for value in (cx, cy, width, height)) or x2 <= x1 or y2 <= y1:
                    raise ValueError(f"invalid hand box: {source}")
                rows.append(f"0 {(x1 + x2) / 2:.8f} {(y1 + y2) / 2:.8f} {x2 - x1:.8f} {y2 - y1:.8f}")
            if not rows:
                raise ValueError(f"empty hand pose label: {source}")
            (target_labels / source.name).write_text("\n".join(rows) + "\n")
    yaml = detection_dataset / "data.yaml"
    yaml.write_text(f"path: {detection_dataset}\ntrain: images/train\nval: images/val\nnames:\n  0: Human hand\n")
    return yaml


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-dataset", type=Path, required=True, help="extracted Ultralytics hand-keypoints zip")
    parser.add_argument("--dataset-output", type=Path, required=True, help="converted labels and image hardlinks")
    parser.add_argument("--base-weights", type=Path, required=True, help="pretrained yolov8n-oiv7.pt")
    parser.add_argument("--project", type=Path, required=True, help="directory for training runs")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.workers < 0:
        parser.error("epochs and batch must be positive; workers cannot be negative")
    yaml = prepare_detection_dataset(args.pose_dataset, args.dataset_output)
    from ultralytics import YOLO

    YOLO(str(args.base_weights)).train(
        data=str(yaml),
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        imgsz=320,
        device=args.device,
        project=str(args.project),
        name="hand-detect",
        exist_ok=True,
        seed=0,
        patience=5,
        plots=False,
        verbose=False,
    )


if __name__ == "__main__":
    cli()
