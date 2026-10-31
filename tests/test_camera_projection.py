"""The simulated external-camera cube box matches OpenCV and covers the lift workspace."""

import itertools

import cv2
import numpy as np
import pytest
import torch

from arc_mycobot.tasks.lift.config.mycobot.geometry import CUBE_SIZE, SPAWN_POS_CENTRE, SPAWN_POS_RANGE
from arc_mycobot.vision.camera_calibration import external_camera_calibration, matrix_from_quaternion_wxyz
from arc_mycobot.vision.camera_projection import FixedCameraProjector


def cube_corners(center):
    half = CUBE_SIZE / 2
    return np.array([np.add(center, signs) for signs in itertools.product((-half, half), repeat=3)])


def test_projection_matches_opencv():
    calibration = external_camera_calibration()
    projector = FixedCameraProjector(calibration)
    rng = np.random.default_rng(0)
    points = np.column_stack((rng.uniform(0.1, 0.3, 50), rng.uniform(-0.1, 0.1, 50), rng.uniform(0.0, 0.3, 50)))
    rotation = matrix_from_quaternion_wxyz(calibration.quaternion_wxyz).T
    translation = -rotation @ np.asarray(calibration.position_m)
    matrix = np.asarray(calibration.intrinsic_matrix).reshape(3, 3)
    expected, _ = cv2.projectPoints(points, cv2.Rodrigues(rotation)[0], translation, matrix, None)
    pixels, depth = projector.pixels(torch.tensor(points, dtype=torch.float32))
    assert (depth > 0).all()
    assert pixels.numpy() == pytest.approx(expected[:, 0], abs=1e-2)


def test_box_contract_and_visibility():
    projector = FixedCameraProjector(external_camera_calibration())
    corners = torch.tensor(cube_corners(SPAWN_POS_CENTRE), dtype=torch.float32)[None]
    pixels, _ = projector.pixels(corners[0])
    box, visible = projector.box(corners)
    assert visible.item()
    x1, y1 = pixels.amin(0).tolist()
    x2, y2 = pixels.amax(0).tolist()
    expected = ((x1 + x2) / 640 - 1, (y1 + y2) / 480 - 1, (x2 - x1) / 640, (y2 - y1) / 480)
    assert box[0].tolist() == pytest.approx(expected, abs=1e-5)
    behind = torch.tensor(cube_corners(projector.position.numpy() - [0.0, 0.3, 0.0]), dtype=torch.float32)[None]
    box, visible = projector.box(behind)
    assert not visible.item() and box.abs().sum() == 0


def test_every_spawn_corner_of_the_lift_region_is_in_view():
    """If a new calibration fails this, the camera no longer sees where the cube starts."""
    projector = FixedCameraProjector(external_camera_calibration())
    centers = [
        np.add(SPAWN_POS_CENTRE, (sx * SPAWN_POS_RANGE[0], sy * SPAWN_POS_RANGE[1], 0.0))
        for sx, sy in itertools.product((-1, 1), repeat=2)
    ]
    corners = torch.tensor(np.stack([cube_corners(center) for center in centers]), dtype=torch.float32)
    box, visible = projector.box(corners, min_visible=1.0)
    assert visible.all()
    assert (box[:, 2:] > 0.01).all()
