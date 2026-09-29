"""RGB marker and detector-box features share the same policy contract."""

from types import SimpleNamespace

import pytest
import torch

from arc_mycobot.vision.box_randomization import corrupt_box_features
from arc_mycobot.vision.boxes import box_features, red_cube_box
from arc_mycobot.vision.yolo_boxes import ultralytics_box_features


def test_red_cube_box_and_missing_detection():
    rgb = torch.zeros((2, 16, 20, 3), dtype=torch.uint8)
    rgb[0, 4:8, 8:12, 0] = 240
    rgb[0, 4:8, 8:12, 1] = 20
    box = red_cube_box(rgb)
    assert box[0].tolist() == pytest.approx([-0.0, -0.25, 0.2, 0.25, 1.0])
    assert box[1].tolist() == [0.0] * 5


def test_xyxy_detector_input_uses_the_same_normalization():
    xyxy = torch.tensor([[8.0, 4.0, 12.0, 8.0], [0.0, 0.0, 0.0, 0.0]])
    valid = torch.tensor([True, False])
    features = box_features(xyxy, valid, width=20, height=16)
    assert features[0].tolist() == pytest.approx([0.0, -0.25, 0.2, 0.25, 1.0])
    assert features[1].tolist() == [0.0] * 5


def test_yolo_adapter_selects_target_class_and_handles_missing():
    detections = SimpleNamespace(
        xyxy=torch.tensor([[0.0, 0.0, 4.0, 4.0], [8.0, 4.0, 12.0, 8.0]]),
        conf=torch.tensor([0.99, 0.8]),
        cls=torch.tensor([1.0, 2.0]),
    )
    results = [SimpleNamespace(boxes=detections), SimpleNamespace(boxes=None)]
    features = ultralytics_box_features(results, class_id=2, width=20, height=16)
    assert features[0].tolist() == pytest.approx([0.0, -0.25, 0.2, 0.25, 1.0])
    assert features[1].tolist() == [0.0] * 5


def test_corrupt_boxes_preserves_missing_contract_and_bounds():
    torch.manual_seed(12)
    boxes = torch.tensor([[0.95, -0.95, 0.9, 0.9, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
    one = torch.ones((2, 1))
    zero = torch.zeros((2, 1))
    output = corrupt_box_features(boxes, one, one, zero, zero)
    assert output.shape == (2, 5)
    assert output[1].tolist() == [0.0] * 5
    assert bool((output[0, :2].abs() <= 1.0).all())
    assert bool(((output[0, 2:4] >= 0.0) & (output[0, 2:4] <= 1.0)).all())
    assert output[0, 4] == 1.0


def test_corrupt_boxes_can_drop_and_hallucinate():
    boxes = torch.tensor([[0.1, 0.2, 0.3, 0.4, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
    zero = torch.zeros((2, 1))
    one = torch.ones((2, 1))
    output = corrupt_box_features(boxes, zero, zero, one, one)
    assert output[0].tolist() == [0.0] * 5
    assert output[1, 4] == 1.0
    assert bool((output[1, 2:4] > 0.0).all())
