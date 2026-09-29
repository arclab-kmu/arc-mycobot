"""Camera-only target features for the actor."""

import torch
from isaaclab.envs import ManagerBasedRLEnv

from arc_mycobot.vision.boxes import red_cube_box


def detected_box(env: ManagerBasedRLEnv, sensor_name: str = "camera") -> torch.Tensor:
    """Five RGB-derived box features; no depth or target world pose."""
    return red_cube_box(env.scene[sensor_name].data.output["rgb"])


def detected_hand_box(env: ManagerBasedRLEnv, sensor_name: str = "camera") -> torch.Tensor:
    """Run the frozen Open Images YOLO detector on each wrist RGB frame."""
    from arc_mycobot.vision.hand_yolo import detect_human_hands

    step = env.common_step_counter
    cache = getattr(env, "_hand_yolo_box_cache", None)
    if cache is not None and cache[0] == step:
        return cache[1]
    rgb = env.scene[sensor_name].data.output["rgb"]
    boxes = detect_human_hands(rgb)
    env._hand_yolo_box_cache = (step, boxes)
    return boxes


def clear_hand_box_cache(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Discard pre-reset detections before the next policy observation."""
    if hasattr(env, "_hand_yolo_box_cache"):
        del env._hand_yolo_box_cache
