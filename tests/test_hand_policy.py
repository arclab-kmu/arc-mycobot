"""Check the offline hand-policy observation and Hub checkpoint contract."""

import math

import pytest
import torch
from torch import nn

from arc_mycobot.vision import hand_policy


def test_hand_policy_uses_training_observation_order(tmp_path, monkeypatch):
    actor = nn.Sequential(nn.Linear(22, 128), nn.ELU(), nn.Linear(128, 128), nn.ELU(), nn.Linear(128, 5))
    checkpoint = tmp_path / "policy.pt"
    torch.save({"model_state_dict": {f"actor.{key}": value for key, value in actor.state_dict().items()}}, checkpoint)
    policy = hand_policy.HandAlignPolicy(checkpoint)
    box = torch.tensor([[0.1, -0.2, 0.3, 0.4, 1.0]])
    monkeypatch.setattr(hand_policy, "detect_human_hands", lambda rgb: box.to(rgb.device))
    observed = []
    policy.actor.register_forward_pre_hook(lambda _module, inputs: observed.append(inputs[0].clone()))

    prediction = policy.predict(
        torch.zeros((32, 32, 3), dtype=torch.uint8),
        joint_angles_rad=[0.1, 0.2, 0.3, 0.4, 0.5, -math.pi / 4],
        joint_velocities_rad_s=[0.01] * 6,
        previous_action=[-0.1] * 5,
    )

    expected = torch.tensor([0.1, -0.2, 0.3, 0.4, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.0, *([0.01] * 6), *([-0.1] * 5)])
    torch.testing.assert_close(observed[0][0], expected)
    torch.testing.assert_close(torch.tensor(prediction.action), actor(expected).detach())
    assert prediction.sim_target_angles_rad[-1] == pytest.approx(-math.pi / 4)
    assert prediction.sim_target_angles_rad[:5] == pytest.approx([0.25 * x for x in prediction.action])


def test_resolve_checkpoint_from_hub(tmp_path, monkeypatch):
    checkpoint = tmp_path / "policy.pt"
    checkpoint.touch()
    received = {}

    def fake_download(**kwargs):
        received.update(kwargs)
        return str(checkpoint)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    assert (
        hand_policy.resolve_checkpoint(hf_repo="owner/model", hf_filename="policy.pt", hf_revision="abc123")
        == checkpoint
    )
    assert received == {"repo_id": "owner/model", "filename": "policy.pt", "revision": "abc123"}
    with pytest.raises(ValueError, match="exactly one"):
        hand_policy.resolve_checkpoint()
