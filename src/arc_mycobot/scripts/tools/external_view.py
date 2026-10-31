"""Render the calibrated external camera, optionally blended with a real frame.

Run with the Isaac Lab Python environment. Without ``--checkpoint`` the arm holds
``--joint_angles_deg`` (J1..J6 in degrees, the same order and units as
``pymycobot.get_angles``). With a checkpoint the PPO policy runs for ``--steps``
and ``--video`` records every external frame. The external camera never feeds
the policy.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Isaac-Visual-Align-YOLO-Hand-DR-MyCobot280JN-View-v0")
parser.add_argument("--calibration", type=str, default=None, help="camera JSON; default is the packaged one")
parser.add_argument("--joint_angles_deg", type=float, nargs=6, default=[0.0, 0.0, 0.0, 0.0, 0.0, -45.0])
parser.add_argument("--checkpoint", type=str, default=None, help="RSL-RL checkpoint; omit to hold the joint angles")
parser.add_argument("--steps", type=int, default=150, help="policy steps with --checkpoint (30 per second)")
parser.add_argument("--image", type=str, default="external_view.png", help="last external frame")
parser.add_argument("--video", type=str, default=None, help="MP4 of every policy step")
parser.add_argument("--overlay", type=str, default=None, help="real frame from the same camera and resolution")
parser.add_argument("--alpha", type=float, default=0.5, help="simulation weight in --overlay")
parser.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=0, help="counter-clockwise, output only")
parser.add_argument("--seed", type=int, default=20261006)
args = parser.parse_args()
if args.calibration:
    os.environ["ARC_MYCOBOT_EXTERNAL_CAMERA"] = str(Path(args.calibration).expanduser().resolve())
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import json
import math

import cv2
import gymnasium as gym
import numpy as np
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner

import arc_mycobot.tasks  # noqa: F401
from arc_mycobot.assets.robots.mycobot_280 import MYCOBOT_ARM_JOINTS
from arc_mycobot.tasks.visual_align.config.mycobot.joint_pos_env_cfg import external_camera_cfg
from arc_mycobot.vision.camera_calibration import external_camera_calibration


def external_frame(env) -> np.ndarray:
    rgb = env.scene["external_camera"].data.output["rgb"][0, ..., :3]
    return cv2.cvtColor(rgb.cpu().numpy(), cv2.COLOR_RGB2BGR)


def for_output(frame: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(frame, k=args.rotate // 90))


def hold_joint_angles(env, wrapped) -> None:
    robot = env.scene["robot"]
    joint_ids, _ = robot.find_joints(MYCOBOT_ARM_JOINTS, preserve_order=True)
    angles = torch.tensor([math.radians(value) for value in args.joint_angles_deg], device=env.device)[None]
    robot.write_joint_state_to_sim(angles, torch.zeros_like(angles), joint_ids=joint_ids)
    # Visual align commands J1..J5 and holds J6 by event; lift commands all six plus a binary jaw.
    commanded = env.action_manager.get_term("arm_action").action_dim
    robot.set_joint_position_target(angles[:, commanded:], joint_ids=joint_ids[commanded:])
    default = robot.data.default_joint_pos[:, joint_ids[:commanded]]
    action = torch.ones((1, env.action_manager.total_action_dim), device=env.device)  # any binary jaw: open
    action[:, :commanded] = (angles[:, :commanded] - default) / env.cfg.actions.arm_action.scale
    for _ in range(10):
        wrapped.step(action)


def main() -> None:
    calibration = external_camera_calibration()
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
    if getattr(env_cfg.scene, "external_camera", None) is None:
        env_cfg.scene.external_camera = external_camera_cfg(calibration)
    env_cfg.seed = args.seed
    env = gym.make(args.task, cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env)
    print(
        "EXTERNAL_CAMERA "
        + json.dumps(
            {
                "calibrated": calibration.calibrated,
                "position_m": calibration.position_m,
                "horizontal_fov_deg": round(calibration.horizontal_fov_deg, 2),
                "size": [calibration.width, calibration.height],
            }
        ),
        flush=True,
    )
    if not calibration.calibrated:
        print("WARNING: uncalibrated external camera estimate; see deploy/calibrate_camera.py", flush=True)

    obs, _ = wrapped.reset()
    writer = None
    if args.checkpoint:
        agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device="cuda:0")
        if args.video:
            size = for_output(external_frame(env.unwrapped)).shape[1::-1]
            fps = round(1.0 / env.unwrapped.step_dt)
            writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        for _ in range(args.steps):
            with torch.inference_mode():
                obs, _, _, _ = wrapped.step(policy(obs))
            if writer is not None:
                writer.write(for_output(external_frame(env.unwrapped)))
        if writer is not None:
            writer.release()
    else:
        hold_joint_angles(env.unwrapped, wrapped)

    frame = external_frame(env.unwrapped)
    if args.overlay:
        real = cv2.imread(args.overlay, cv2.IMREAD_COLOR)
        if real is None:
            raise FileNotFoundError(args.overlay)
        if real.shape[:2] != frame.shape[:2]:
            raise ValueError(
                f"--overlay is {real.shape[1]}x{real.shape[0]}, calibration is {frame.shape[1]}x{frame.shape[0]}"
            )
        if any(calibration.distortion):
            matrix = np.asarray(calibration.intrinsic_matrix).reshape(3, 3)
            real = cv2.undistort(real, matrix, np.asarray(calibration.distortion))
        image = Path(args.image)
        cv2.imwrite(str(image.with_name(f"{image.stem}_sim{image.suffix}")), for_output(frame))
        frame = cv2.addWeighted(real, 1.0 - args.alpha, frame, args.alpha, 0.0)
    cv2.imwrite(args.image, for_output(frame))
    print(f"EXTERNAL_VIEW image={args.image}" + (f" video={args.video}" if writer is not None else ""), flush=True)
    wrapped.close()


def cli() -> int:
    from arc_mycobot.scripts._entrypoint import run_and_exit

    run_and_exit(main, simulation_app)


if __name__ == "__main__":
    cli()
