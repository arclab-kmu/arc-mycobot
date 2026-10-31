"""External-camera calibration on synthetic checkerboard images; no camera or GPU."""

import json
import math

import cv2
import numpy as np
import pytest
from deploy.calibrate_camera import (
    board_points,
    calibrate_intrinsics,
    calibration_json,
    camera_pose_from_board,
    draw_base_axes,
    find_corners,
    parse_board,
    quaternion_wxyz,
)

from arc_mycobot.vision.camera_calibration import (
    external_camera_calibration,
    load_camera_calibration,
    matrix_from_quaternion_wxyz,
)

BOARD = (9, 6)
SQUARE_M = 0.02
WIDTH, HEIGHT = 640, 480
MATRIX = np.array([[420.0, 0.0, 318.0], [0.0, 421.0, 243.0], [0.0, 0.0, 1.0]])
LENS = {
    "width": WIDTH,
    "height": HEIGHT,
    "intrinsic_matrix": MATRIX.ravel().tolist(),
    "distortion": [0.0] * 5,
    "rms_px": 0.0,
    "images_used": ["synthetic"] * 12,
}


def look_at(position, target, roll_deg=0.0):
    """Camera axes (OpenCV x right, y down, z forward) in the base frame."""
    forward = np.asarray(target, float) - np.asarray(position, float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    axes = np.stack((right, np.cross(forward, right), forward), axis=1)
    roll, _ = cv2.Rodrigues(forward * math.radians(roll_deg))
    return roll @ axes


def render_board(rotation, translation, flip=False, pixels_per_square=40, margin=40):
    """Warp a printed 10x7-square board (9x6 inner corners) into the camera image."""
    columns, rows = BOARD
    texture = np.full(
        ((rows + 1) * pixels_per_square + 2 * margin, (columns + 1) * pixels_per_square + 2 * margin), 255, np.uint8
    )
    for row in range(rows + 1):
        for column in range(columns + 1):
            if (row + column) % 2 == 0:
                y, x = margin + row * pixels_per_square, margin + column * pixels_per_square
                texture[y : y + pixels_per_square, x : x + pixels_per_square] = 0
    if flip:
        texture = np.ascontiguousarray(texture[::-1, ::-1])
    scale = SQUARE_M / pixels_per_square
    texture_to_board = np.array(
        [
            [scale, 0.0, -(margin + (columns + 1) * pixels_per_square / 2) * scale],
            [0.0, scale, -(margin + (rows + 1) * pixels_per_square / 2) * scale],
            [0.0, 0.0, 1.0],
        ]
    )
    board_to_image = MATRIX @ np.column_stack((rotation[:, 0], rotation[:, 1], translation))
    homography = np.asarray(board_to_image @ texture_to_board, dtype=np.float64)
    image = cv2.warpPerspective(texture, homography, (WIDTH, HEIGHT), borderValue=(128.0,))
    return cv2.cvtColor(cv2.GaussianBlur(image, (3, 3), 0), cv2.COLOR_GRAY2BGR)


def table_scene(camera_position, camera_axes, center_xy):
    """Board flat on the table, long side along +x; returns base->camera and board->camera."""
    base_to_camera = camera_axes.T
    board_to_base = np.eye(3)  # board x = base x, board y = base y, board z = up
    rotation = base_to_camera @ board_to_base
    translation = base_to_camera @ (np.array([center_xy[0], center_xy[1], 0.0]) - np.asarray(camera_position))
    return rotation, translation


CAMERA_POSITION = np.array([0.04, -0.42, 0.30])
CAMERA_AXES = look_at(CAMERA_POSITION, (0.12, -0.10, 0.08), roll_deg=-90.0)


@pytest.mark.parametrize("flip", [False, True])
def test_table_board_recovers_camera_pose_for_either_corner_order(flip):
    rotation, translation = table_scene(CAMERA_POSITION, CAMERA_AXES, (0.14, -0.16))
    image = render_board(rotation, translation, flip=flip)
    corners = find_corners(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), BOARD)
    assert corners is not None
    position, axes, reprojection = camera_pose_from_board(corners, LENS, BOARD, SQUARE_M, (0.14, -0.16), 0.0)
    assert reprojection < 0.5
    assert np.linalg.norm(position - CAMERA_POSITION) < 0.005
    angle = math.degrees(math.acos(np.clip((np.trace(axes.T @ CAMERA_AXES) - 1) / 2, -1, 1)))
    assert angle < 0.5
    check = draw_base_axes(image, LENS, position, axes)
    assert check.shape == image.shape and not np.array_equal(check, image)


def test_wrong_board_center_is_rejected_instead_of_guessing():
    rotation, translation = table_scene(CAMERA_POSITION, CAMERA_AXES, (0.14, -0.16))
    corners = find_corners(cv2.cvtColor(render_board(rotation, translation), cv2.COLOR_BGR2GRAY), BOARD)
    with pytest.raises(ValueError, match="board orientations"):
        camera_pose_from_board(corners, LENS, BOARD, SQUARE_M, (3.0, 3.0), 0.0)


def test_intrinsics_from_synthetic_views():
    images = []
    tilts = [(rx, ry, rz) for rx in (-25.0, 25.0) for ry in (-25.0, 0.0, 25.0) for rz in (0.0, 15.0)]
    for index, tilt in enumerate(tilts):
        rotation, _ = cv2.Rodrigues(np.radians(tilt))
        translation = np.array([0.02 * (index % 3 - 1), 0.015 * (index % 2), 0.32 + 0.01 * (index % 4)])
        images.append((f"view{index}", render_board(rotation, translation)))
    lens = calibrate_intrinsics(images, BOARD, SQUARE_M)
    fx, _, cx, _, fy, cy = lens["intrinsic_matrix"][:6]
    assert (lens["width"], lens["height"]) == (WIDTH, HEIGHT)
    assert fx == pytest.approx(420.0, rel=0.02) and fy == pytest.approx(421.0, rel=0.02)
    assert abs(cx - 318.0) < 6 and abs(cy - 243.0) < 6
    assert lens["rms_px"] < 0.5


def test_written_json_loads_in_isaac_free_loader(tmp_path):
    payload = calibration_json(LENS, CAMERA_POSITION, CAMERA_AXES, "synthetic")
    path = tmp_path / "camera.json"
    path.write_text(json.dumps(payload))
    calibration = load_camera_calibration(path)
    assert calibration.calibrated
    assert calibration.position_m == pytest.approx(tuple(CAMERA_POSITION))
    assert matrix_from_quaternion_wxyz(calibration.quaternion_wxyz) == pytest.approx(CAMERA_AXES, abs=1e-5)
    assert calibration.horizontal_fov_deg == pytest.approx(math.degrees(2 * math.atan(320 / 420)))


def test_quaternion_round_trip_covers_every_branch():
    rng = np.random.default_rng(0)
    rotations = [np.eye(3), np.diag([1.0, -1.0, -1.0]), np.diag([-1.0, 1.0, -1.0]), np.diag([-1.0, -1.0, 1.0])]
    rotations += [cv2.Rodrigues(rng.normal(size=3))[0] for _ in range(20)]
    for rotation in rotations:
        quaternion = quaternion_wxyz(rotation)
        assert quaternion[0] >= 0
        assert matrix_from_quaternion_wxyz(quaternion) == pytest.approx(rotation, abs=1e-9)


def test_board_parser_and_points():
    assert parse_board("9x6") == (9, 6)
    for bad in ("6x6", "9", "2x5"):
        with pytest.raises(Exception, match="board"):
            parse_board(bad)
    points = board_points((9, 6), 0.02)
    assert points.shape == (54, 3)
    assert points.mean(axis=0) == pytest.approx((0.0, 0.0, 0.0))
    assert points[1, 0] - points[0, 0] == pytest.approx(0.02)


def test_packaged_external_camera_and_override(tmp_path, monkeypatch):
    packaged = external_camera_calibration()
    assert not packaged.calibrated
    assert (packaged.width, packaged.height) == (640, 480)
    custom = tmp_path / "camera.json"
    custom.write_text(json.dumps(calibration_json(LENS, CAMERA_POSITION, CAMERA_AXES, "measured")))
    monkeypatch.setenv("ARC_MYCOBOT_EXTERNAL_CAMERA", str(custom))
    assert external_camera_calibration().calibrated


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("quaternion_wxyz", [1.0, 0.0, 0.0, 0.5], "normalized"),
        ("intrinsic_matrix", [400.0, 1.0, 320.0, 0.0, 400.0, 240.0, 0.0, 0.0, 1.0], "intrinsic_matrix"),
        ("intrinsic_matrix", [400.0, 0.0, 900.0, 0.0, 400.0, 240.0, 0.0, 0.0, 1.0], "principal point"),
        ("position_m", [0.0, 0.0], "position_m"),
    ],
)
def test_loader_rejects_malformed_calibration(tmp_path, field, value, message):
    payload = calibration_json(LENS, CAMERA_POSITION, CAMERA_AXES, "bad")
    payload[field] = value
    path = tmp_path / "camera.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=message):
        load_camera_calibration(path)
