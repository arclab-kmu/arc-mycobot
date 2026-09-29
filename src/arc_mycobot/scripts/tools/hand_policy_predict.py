"""One-frame YOLO + trained actor preview. Does not command the robot."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from arc_mycobot.vision.hand_policy import DEFAULT_HF_FILENAME, HandAlignPolicy, resolve_checkpoint


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="local model_299.pt or exported checkpoint")
    source.add_argument("--hf-repo", help="Hugging Face model repo, e.g. username/mycobot-hand-policy")
    parser.add_argument("--hf-file", default=DEFAULT_HF_FILENAME, help="checkpoint filename in the model repo")
    parser.add_argument("--hf-revision", help="optional tag or commit hash")
    frame = parser.add_mutually_exclusive_group(required=True)
    frame.add_argument("--image", type=Path, help="single RGB camera frame")
    frame.add_argument("--camera-index", type=int, help="capture one frame from a local camera")
    parser.add_argument("--angles-deg", type=float, nargs=6, required=True, metavar="J", help="measured J1..J6 angles")
    parser.add_argument("--velocities-deg-s", type=float, nargs=6, default=[0.0] * 6, metavar="V")
    parser.add_argument("--previous-action", type=float, nargs=5, default=[0.0] * 5, metavar="A")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    checkpoint = resolve_checkpoint(
        checkpoint=args.checkpoint,
        hf_repo=args.hf_repo,
        hf_filename=args.hf_file,
        hf_revision=args.hf_revision,
    )
    policy = HandAlignPolicy(checkpoint, device=args.device)
    if args.image is not None:
        with Image.open(args.image) as image:
            rgb = torch.from_numpy(np.asarray(image.convert("RGB")).copy())
    else:
        import cv2

        camera = cv2.VideoCapture(args.camera_index)
        try:
            ok, bgr = camera.read()
        finally:
            camera.release()
        if not ok:
            raise RuntimeError(f"could not read camera index {args.camera_index}")
        rgb = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy())
    prediction = policy.predict(
        rgb,
        joint_angles_rad=[math.radians(x) for x in args.angles_deg],
        joint_velocities_rad_s=[math.radians(x) for x in args.velocities_deg_s],
        previous_action=args.previous_action,
    )
    print(
        json.dumps(
            {
                "human_hand_box_cx_cy_w_h_detected": prediction.box,
                "policy_action_j1_to_j5": prediction.action,
                "simulation_target_angles_deg_j1_to_j6": [math.degrees(x) for x in prediction.sim_target_angles_rad],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    cli()
