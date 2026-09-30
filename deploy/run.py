"""Preview or run the hand-centering policy on a Jetson Nano myCobot 280."""

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from deploy.arm import ArmConfig, ArmSession, require_pymycobot
from deploy.controller import Limits, UnsafeTarget, check_start, plan_target
from deploy.preview import BrowserPreview
from deploy.runtime import CHECKPOINT_NAME, HF_REPO, HF_REVISION, HandPolicyRuntime, resolve_checkpoint

PERIOD_S = 0.2
MAX_OBSERVATION_AGE_S = 1.0
MAX_FEEDBACK_LATENCY_S = 0.5
MAX_VELOCITY_DEG_S = 120.0
COMMAND_SPEED = 10
HAND_REACQUIRE_FRAMES = 2


def rgb_frame(bgr, rotate=0, flip_x=False, flip_y=False):
    """Center crop to the square camera geometry used in training."""
    if bgr is None or bgr.ndim != 3 or bgr.shape[-1] != 3:
        raise ValueError("camera returned no BGR color frame")
    height, width = bgr.shape[:2]
    side = min(height, width)
    top, left = (height - side) // 2, (width - side) // 2
    image = bgr[top : top + side, left : left + side]
    if rotate:
        image = np.rot90(image, k=rotate // 90)
    if flip_x:
        image = image[:, ::-1]
    if flip_y:
        image = image[::-1]
    return torch.from_numpy(np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))


def joint_velocities(current, previous, dt):
    if previous is None:
        return [0.0] * 6
    if dt <= 0.0:
        raise UnsafeTarget("non-positive joint feedback interval")
    speeds = [(float(now) - float(before)) / dt for now, before in zip(current, previous)]
    if any(not math.isfinite(speed) or abs(speed) > MAX_VELOCITY_DEG_S for speed in speeds):
        raise UnsafeTarget("joint feedback velocity is implausible")
    return speeds


def run_offline(args, runtime):
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(str(args.image))
    prediction = runtime.predict(
        rgb_frame(image, args.rotate, args.flip_x, args.flip_y), args.angles_deg, [0.0] * 6, [0.0] * 5
    )
    print(
        json.dumps(
            {"box": prediction.box, "policy_action": prediction.action, "simulation_target_deg": prediction.target_deg},
            indent=2,
        )
    )


def run_camera(args, runtime, arm_config_type=ArmConfig, arm_session_type=ArmSession, camera_factory=None):
    warmup_started = time.monotonic()
    runtime.warmup()
    print(json.dumps({"warmup_s": round(time.monotonic() - warmup_started, 3)}), flush=True)
    camera = cv2.VideoCapture(args.camera, cv2.CAP_V4L2) if camera_factory is None else camera_factory(args.camera)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"cannot open V4L2 camera {args.camera}")
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    config = arm_config_type(
        read_only=not args.execute,
        port=args.port,
        baud=args.baud,
        speed=COMMAND_SPEED,
        max_feedback_age_s=MAX_FEEDBACK_LATENCY_S,
        max_observation_age_s=MAX_OBSERVATION_AGE_S,
        max_command_delta_deg=Limits().max_step_degrees + 0.01,
    )
    last_angles = None
    last_feedback_at = None
    previous_action = [0.0] * 5
    hand_streak = 0
    preview = BrowserPreview(args.web_preview_port) if args.web_preview_port is not None else None
    try:
        if preview is not None:
            print(f"YOLO browser preview: {preview.start()}", flush=True)
        with arm_session_type(config) as arm:
            try:
                for index in range(args.max_frames or 2**63):
                    cycle_start = time.monotonic()
                    status = arm.read_status()
                    if args.execute and (status is None or not status.is_normal):
                        raise UnsafeTarget(
                            f"arm status is not normal: {'unavailable' if status is None else status.describe()}"
                        )
                    capture_started = time.monotonic()
                    ok, bgr = camera.read()
                    if not ok:
                        raise RuntimeError("camera read failed")
                    frame_at = time.monotonic()
                    reading = arm.read_joints(retry_s=0.2)
                    feedback_at = time.monotonic()
                    if args.execute and last_angles is None:
                        check_start(reading.angles)
                    dt = 0.0 if last_feedback_at is None else feedback_at - last_feedback_at
                    velocities = joint_velocities(reading.angles, last_angles, dt)
                    last_angles, last_feedback_at = list(reading.angles), feedback_at
                    rgb = rgb_frame(bgr, args.rotate, args.flip_x, args.flip_y)
                    inference_started = time.monotonic()
                    prediction = runtime.predict(rgb, reading.angles, velocities, previous_action)
                    age = time.monotonic() - min(feedback_at, frame_at)
                    camera_read_s = frame_at - capture_started
                    inference_s = time.monotonic() - inference_started
                    target = None
                    hand_detected = prediction.box[4] == 1.0
                    hand_streak = hand_streak + 1 if hand_detected else 0
                    hold_reason = None
                    if args.execute and not hand_detected:
                        hold_reason = "Human hand not detected"
                        arm.halt(hold_reason)
                        previous_action = [0.0] * 5
                    elif args.execute and hand_streak < HAND_REACQUIRE_FRAMES:
                        hold_reason = "Waiting for consecutive hand detections"
                    if args.execute and hand_detected:
                        if age > MAX_OBSERVATION_AGE_S:
                            raise UnsafeTarget(
                                f"observation is stale ({age:.3f} s; "
                                f"camera read {camera_read_s:.3f} s, inference {inference_s:.3f} s); "
                                "no command sent; check read-only preview timing"
                            )
                        if hand_streak >= HAND_REACQUIRE_FRAMES:
                            target = plan_target(prediction, reading.angles)
                            arm.gate.open("fresh hand detection and joint feedback")
                            try:
                                arm.command_joints(target, measured=reading)
                            finally:
                                arm.gate.close("one policy command sent")
                            previous_action = list(prediction.action)
                    motion_state = (
                        "preview"
                        if not args.execute
                        else "waiting_for_hand"
                        if not hand_detected
                        else "acquiring_hand"
                        if hold_reason
                        else "tracking"
                    )
                    if preview is not None:
                        preview.publish(rgb, prediction, motion_state.upper(), age)
                    print(
                        json.dumps(
                            {
                                "frame": index,
                                "box": prediction.box,
                                "angles_deg": reading.angles,
                                "policy_action": prediction.action,
                                "sent_target_deg": target,
                                "motion_state": motion_state,
                                "hold_reason": hold_reason,
                                "observation_age_s": round(age, 3),
                                "camera_read_s": round(camera_read_s, 3),
                                "inference_s": round(inference_s, 3),
                                "status": None if status is None else status.describe(),
                            }
                        ),
                        flush=True,
                    )
                    remaining = PERIOD_S - (time.monotonic() - cycle_start)
                    if remaining > 0:
                        time.sleep(remaining)
            except KeyboardInterrupt:
                print("Stopped by operator.", flush=True)
            finally:
                if args.execute:
                    arm.halt("deployment ended")
    finally:
        camera.release()
        if preview is not None:
            preview.close()


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="enable bounded physical joint commands")
    parser.add_argument("--image", type=Path, help="single offline image; never connects to the arm")
    parser.add_argument("--camera", default="/dev/video0", help="USB camera device on the Jetson")
    parser.add_argument("--port", default="/dev/ttyTHS1")
    parser.add_argument("--baud", type=int, default=1000000)
    parser.add_argument("--device", default="cpu", help="Torch/YOLO device, e.g. cpu or cuda:0")
    parser.add_argument("--checkpoint", type=Path, help="local checkpoint; otherwise fetch pinned Hugging Face file")
    parser.add_argument("--hf-repo", default=HF_REPO)
    parser.add_argument("--hf-revision", default=HF_REVISION)
    parser.add_argument("--hf-file", default=CHECKPOINT_NAME)
    parser.add_argument("--yolo-weights", help="local YOLO weights; default is XDG cache")
    parser.add_argument(
        "--angles-deg", type=float, nargs=6, default=[0, 0, 0, 0, 0, -45], help="offline-image pose only"
    )
    parser.add_argument("--max-frames", type=int, help="stop after this many camera frames")
    parser.add_argument("--web-preview-port", type=int, help="show YOLO boxes in a local browser on this port")
    parser.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=0)
    parser.add_argument("--flip-x", action="store_true")
    parser.add_argument("--flip-y", action="store_true")
    args = parser.parse_args()
    if args.image is not None and args.execute:
        parser.error("--image and --execute cannot be combined")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if args.web_preview_port is not None and not 1 <= args.web_preview_port <= 65535:
        parser.error("--web-preview-port must be between 1 and 65535")
    if args.image is not None and args.web_preview_port is not None:
        parser.error("--web-preview-port requires the live camera")
    try:
        if args.image is None:
            require_pymycobot()
    except RuntimeError as exc:
        parser.exit(2, f"{exc}\n")
    checkpoint = resolve_checkpoint(args.checkpoint, args.hf_repo, args.hf_revision, args.hf_file)
    runtime = HandPolicyRuntime(checkpoint, device=args.device, yolo_weights=args.yolo_weights)
    if args.image is not None:
        run_offline(args, runtime)
    else:
        run_camera(args, runtime)


if __name__ == "__main__":
    cli()
