"""Evaluate the external-camera lift policy with a rendered, color-thresholded cube box.

Training projects the cube through the camera calibration and so never sees
the arm occlude it. Here the external camera is rendered and the policy box
comes from an orange-pixel mask, the same detector a real camera would run.
``--box projected`` keeps the training box for comparison. Run with the Isaac
Lab Python environment.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-ExtCam-MyCobot280JN-Play-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--box", choices=("rendered", "projected"), default="rendered")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=750, help="policy steps at 50 Hz; one episode is 250")
parser.add_argument("--seed", type=int, default=20261006)
parser.add_argument("--videos", type=str, default=None, help="directory for env 0's overview, external and wrist MP4s")
args = parser.parse_args()
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import json
import shutil
import subprocess
from pathlib import Path

import cv2
import gymnasium as gym
import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner

import arc_mycobot.tasks  # noqa: F401
from arc_mycobot.tasks.lift.config.mycobot.geometry import LIFT_HEIGHT
from arc_mycobot.tasks.lift.mdp.observations import ExternalCubeBox
from arc_mycobot.tasks.visual_align.config.mycobot.joint_pos_env_cfg import external_camera_cfg
from arc_mycobot.vision.camera_calibration import external_camera_calibration

MIN_PIXELS = 4
OVERVIEW_EYE = (0.62, 0.42, 0.36)
OVERVIEW_TARGET = (0.16, 0.0, 0.10)


def look_at_quaternion_wxyz(eye, target):
    """ROS/OpenCV optical axes (x right, y down, z forward) looking from ``eye`` at ``target``."""
    from scipy.spatial.transform import Rotation

    forward = np.subtract(target, eye) / np.linalg.norm(np.subtract(target, eye))
    right = np.cross(forward, (0.0, 0.0, 1.0))
    right /= np.linalg.norm(right)
    x, y, z, w = Rotation.from_matrix(np.stack((right, np.cross(forward, right), forward), axis=1)).as_quat()
    return (float(w), float(x), float(y), float(z))


def recording_cameras(env_cfg) -> None:
    """Wrist camera as in visual-align training, plus a fixed overview; neither feeds the policy."""
    env_cfg.scene.wrist_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/camera_link/Camera",
        offset=TiledCameraCfg.OffsetCfg(convention="ros"),
        data_types=["rgb"],
        update_latest_camera_pose=True,
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, horizontal_aperture=20.955, clipping_range=(0.01, 1.5)),
        width=480,
        height=480,
    )
    env_cfg.scene.overview_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/OverviewCamera",
        offset=TiledCameraCfg.OffsetCfg(
            pos=OVERVIEW_EYE, rot=look_at_quaternion_wxyz(OVERVIEW_EYE, OVERVIEW_TARGET), convention="ros"
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=14.0, horizontal_aperture=20.955, clipping_range=(0.01, 3.0)),
        width=640,
        height=480,
    )


class Recorder:
    """Per-view MP4 writers, re-encoded to H.264 when ffmpeg is available."""

    def __init__(self, directory: str, fps: int):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.writers: dict[str, tuple[cv2.VideoWriter, Path]] = {}

    def write(self, name: str, frame: np.ndarray) -> None:
        if name not in self.writers:
            path = self.directory / f"{name}.mp4v.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, frame.shape[1::-1])
            self.writers[name] = (writer, path)
        self.writers[name][0].write(np.ascontiguousarray(frame))

    def close(self) -> list[Path]:
        outputs = []
        for name, (writer, path) in self.writers.items():
            writer.release()
            output = self.directory / f"{name}.mp4"
            if shutil.which("ffmpeg"):
                command = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-c:v", "libx264"]
                command += ["-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(output)]
                subprocess.run(command, check=True)
                path.unlink()
            else:
                path.rename(output)
            outputs.append(output)
        return outputs


def camera_bgr(scene, name: str) -> np.ndarray:
    return cv2.cvtColor(scene[name].data.output["rgb"][0, ..., :3].cpu().numpy(), cv2.COLOR_RGB2BGR)


def label(frame: np.ndarray, text: str) -> np.ndarray:
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(frame, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def orange_boxes(rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalized boxes, detection flags, and pixel counts of the orange cube in ``(N, H, W, 3)`` uint8."""
    red, green, blue = (rgb[..., channel].to(torch.int16) for channel in range(3))
    # The lit cube renders near (228, 172, 114); the white arm, grey ground and sky have R - G <= 0.
    mask = (red > 100) & (red - green > 25) & (green - blue > 15)
    count = mask.flatten(1).sum(dim=1)
    height, width = mask.shape[1:]
    columns = mask.any(dim=1)
    rows = mask.any(dim=2)
    xs = torch.arange(width, device=rgb.device)
    ys = torch.arange(height, device=rgb.device)
    x1 = torch.where(columns, xs, width).amin(dim=1)
    x2 = torch.where(columns, xs + 1, 0).amax(dim=1)
    y1 = torch.where(rows, ys, height).amin(dim=1)
    y2 = torch.where(rows, ys + 1, 0).amax(dim=1)
    detected = count >= MIN_PIXELS
    boxes = torch.stack(((x1 + x2) / width - 1.0, (y1 + y2) / height - 1.0, (x2 - x1) / width, (y2 - y1) / height), 1)
    return torch.where(detected[:, None], boxes.float(), torch.zeros_like(boxes, dtype=torch.float32)), detected, count


class RenderedCubeBox(ExternalCubeBox):
    """Same latching contract as training, but the box comes from the rendered camera."""

    stats: dict[str, list] = {"detected": [], "pixels": [], "projected_visible": [], "center_error": []}

    def __call__(
        self,
        env,
        cube_size,
        center_std_max,
        size_std_max,
        miss_max,
        object_cfg=SceneEntityCfg("object"),
        robot_cfg=SceneEntityCfg("robot"),
    ):
        projected, projected_visible = self.projected_box(env, object_cfg, robot_cfg)
        rgb = env.scene["external_camera"].data.output["rgb"][..., :3]
        box, detected, count = orange_boxes(rgb)
        both = detected & projected_visible
        RenderedCubeBox.stats["detected"].append(detected.float().mean().item())
        RenderedCubeBox.stats["pixels"].append(count.float().mean().item())
        RenderedCubeBox.stats["projected_visible"].append(projected_visible.float().mean().item())
        if both.any():
            error = torch.linalg.vector_norm(box[both, :2] - projected[both, :2], dim=1)
            RenderedCubeBox.stats["center_error"].append(error.mean().item())
        if args.box == "projected":
            return self.latch(projected, projected_visible, env.step_dt)
        return self.latch(box, detected, env.step_dt)


def main() -> None:
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env_cfg.scene.env_spacing = 4.0  # keep neighbouring cubes outside the 1.5 m far clip
    env_cfg.scene.external_camera = external_camera_cfg(external_camera_calibration())
    env_cfg.commands.object_pose.debug_vis = False  # the goal marker has a red axis
    env_cfg.observations.policy.object_box.func = RenderedCubeBox
    if args.videos:
        recording_cameras(env_cfg)
    env = gym.make(args.task, cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device="cuda:0")
    cube = env.unwrapped.scene["object"]
    goal = env.unwrapped.command_manager.get_command("object_pose")
    recorder = Recorder(args.videos, round(1.0 / env.unwrapped.step_dt)) if args.videos else None
    lifted, near_goal = [], []
    obs, _ = wrapped.reset()
    for step in range(args.steps):
        with torch.inference_mode():
            obs, _, _, _ = wrapped.step(policy(obs))
        position = cube.data.root_pos_w - env.unwrapped.scene.env_origins
        lifted.append((position[:, 2] > LIFT_HEIGHT).float().mean().item())
        goal_position = goal[:, :3]
        robot = env.unwrapped.scene["robot"]
        cube_in_root = position - (robot.data.root_pos_w - env.unwrapped.scene.env_origins)
        near_goal.append((torch.linalg.vector_norm(cube_in_root - goal_position, dim=1) < 0.03).float().mean().item())
        if recorder is not None:
            scene = env.unwrapped.scene
            t = step * env.unwrapped.step_dt
            height = position[0, 2].item()
            distance = torch.linalg.vector_norm(cube_in_root[0] - goal_position[0]).item()
            overview = label(
                camera_bgr(scene, "overview_camera"),
                f"overview  t={t:4.1f}s  cube z={height * 1000:4.0f} mm  goal {distance * 1000:4.0f} mm",
            )
            external = camera_bgr(scene, "external_camera")
            cx, cy, w, h, detected = obs["policy"][0, 14:19].tolist()
            color = (0, 255, 0) if detected > 0.5 else (0, 0, 255)
            x1, y1 = int((cx + 1 - w) * 320), int((cy + 1 - h) * 240)
            x2, y2 = int((cx + 1 + w) * 320), int((cy + 1 + h) * 240)
            cv2.rectangle(external, (x1, y1), (x2, y2), color, 2)
            external = label(
                cv2.rotate(external, cv2.ROTATE_90_CLOCKWISE),
                "camera 0 (external): " + ("detected" if detected > 0.5 else "last box"),
            )
            wrist = label(camera_bgr(scene, "wrist_camera"), "camera 1 (wrist)")
            recorder.write("overview", overview)
            recorder.write("external", external)
            recorder.write("wrist", wrist)
            side = cv2.resize(external, (360, 480))
            recorder.write("combined", np.hstack((overview, side, wrist)))
    if recorder is not None:
        for path in recorder.close():
            print(f"VIDEO {path}", flush=True)
    stats = RenderedCubeBox.stats
    metrics = {
        "checkpoint": args.checkpoint,
        "box": args.box,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "lifted_fraction": sum(lifted) / len(lifted),
        "within_3cm_of_goal_fraction": sum(near_goal) / len(near_goal),
        "rendered_detection_rate": sum(stats["detected"]) / len(stats["detected"]),
        "projected_visible_rate": sum(stats["projected_visible"]) / len(stats["projected_visible"]),
        "mean_orange_pixels": sum(stats["pixels"]) / len(stats["pixels"]),
        "mean_center_error_norm": sum(stats["center_error"]) / max(len(stats["center_error"]), 1),
    }
    print("METRICS " + json.dumps(metrics), flush=True)
    wrapped.close()


def cli() -> int:
    from arc_mycobot.scripts._entrypoint import run_and_exit

    run_and_exit(main, simulation_app)


if __name__ == "__main__":
    cli()
