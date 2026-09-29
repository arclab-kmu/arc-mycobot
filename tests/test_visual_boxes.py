"""RGB marker and detector-box features share the same policy contract."""

from types import SimpleNamespace

import pytest
import torch

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
