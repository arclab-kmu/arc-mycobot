"""Camera-only target features for the actor."""

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase

from arc_mycobot.vision.box_randomization import corrupt_box_features
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


class RandomizedHandBox(ManagerTermBase):
    """Episode-varying YOLO jitter, missed detections, and rare false boxes."""

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.center_std = torch.zeros((self.num_envs, 1), device=self.device)
        self.size_std = torch.zeros_like(self.center_std)
        self.dropout_probability = torch.zeros_like(self.center_std)
        self.false_positive_probability = torch.zeros_like(self.center_std)
        self.reset()

    def reset(self, env_ids=None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        limits = self.cfg.params
        for values, name in (
            (self.center_std, "center_std_max"),
            (self.size_std, "size_std_max"),
            (self.dropout_probability, "dropout_max"),
            (self.false_positive_probability, "false_positive_max"),
        ):
            values[ids] = torch.rand_like(values[ids]) * limits[name]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        center_std_max: float,
        size_std_max: float,
        dropout_max: float,
        false_positive_max: float,
    ) -> torch.Tensor:
        return corrupt_box_features(
            detected_hand_box(env),
            self.center_std,
            self.size_std,
            self.dropout_probability,
            self.false_positive_probability,
        )


def clear_hand_box_cache(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Discard pre-reset detections before the next policy observation."""
    if hasattr(env, "_hand_yolo_box_cache"):
        del env._hand_yolo_box_cache
