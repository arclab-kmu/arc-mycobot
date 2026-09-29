"""Run the trained hand-box alignment actor without starting Isaac Sim."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .hand_yolo import detect_human_hands

HOME_RAD = (0.0, 0.0, 0.0, 0.0, 0.0, -math.pi / 4)
ACTION_SCALE_RAD = 0.25
DEFAULT_HF_FILENAME = "arc-mycobot-yolo-hand-policy.pt"


@dataclass(frozen=True)
class HandPolicyPrediction:
    """One-frame detector output and deterministic PPO actor output."""

    box: tuple[float, float, float, float, float]
    action: tuple[float, float, float, float, float]
    sim_target_angles_rad: tuple[float, float, float, float, float, float]


def resolve_checkpoint(
    *,
    checkpoint: str | Path | None = None,
    hf_repo: str | None = None,
    hf_filename: str = DEFAULT_HF_FILENAME,
    hf_revision: str | None = None,
) -> Path:
    """Use a local RSL-RL checkpoint or fetch one from a Hugging Face model repo."""
    if (checkpoint is None) == (hf_repo is None):
        raise ValueError("provide exactly one of checkpoint or hf_repo")
    if checkpoint is not None:
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    assert hf_repo is not None
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=hf_repo, filename=hf_filename, revision=hf_revision))


class HandAlignPolicy:
    """YOLO Human hand box -> 22-value observation -> five joint actions.

    All joint inputs are in J1..J6 order and radians. The caller supplies the
    measured velocities and last *executed* policy action, if any. This class
    calculates outputs only; it never sends commands to the robot.
    """

    def __init__(self, checkpoint: str | Path, *, device: str = "cpu") -> None:
        self.device = torch.device(device)
        state = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        model_state = state["model_state_dict"]
        actor = nn.Sequential(
            nn.Linear(22, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 5),
        )
        actor_weights = {
            key.removeprefix("actor."): value for key, value in model_state.items() if key.startswith("actor.")
        }
        actor.load_state_dict(actor_weights)
        self.actor = actor.to(self.device).eval()

    @torch.inference_mode()
    def predict(
        self,
        rgb: torch.Tensor,
        *,
        joint_angles_rad: Sequence[float],
        joint_velocities_rad_s: Sequence[float],
        previous_action: Sequence[float],
    ) -> HandPolicyPrediction:
        """Predict for one HWC uint8 RGB frame using the training observation order."""
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != torch.uint8:
            raise ValueError("rgb must be a single HWC uint8 RGB frame")
        if len(joint_angles_rad) != 6 or len(joint_velocities_rad_s) != 6 or len(previous_action) != 5:
            raise ValueError("expected 6 joint angles, 6 velocities, and 5 previous actions")
        box = detect_human_hands(rgb.unsqueeze(0).to(self.device))[0]
        angles = torch.as_tensor(joint_angles_rad, dtype=torch.float32, device=self.device)
        velocities = torch.as_tensor(joint_velocities_rad_s, dtype=torch.float32, device=self.device)
        previous = torch.as_tensor(previous_action, dtype=torch.float32, device=self.device)
        home = torch.as_tensor(HOME_RAD, dtype=torch.float32, device=self.device)
        observation = torch.cat((box, angles - home, velocities, previous)).unsqueeze(0)
        if not bool(torch.isfinite(observation).all()):
            raise ValueError("observation contains non-finite values")
        action = self.actor(observation)[0]
        sim_target = torch.cat((ACTION_SCALE_RAD * action, home[-1:]))
        return HandPolicyPrediction(
            box=tuple(box.cpu().tolist()),
            action=tuple(action.cpu().tolist()),
            sim_target_angles_rad=tuple(sim_target.cpu().tolist()),
        )
