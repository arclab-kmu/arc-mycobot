"""Pinhole calibration for a camera fixed relative to the robot base.

The JSON written by ``deploy/calibrate_camera.py`` on the Jetson is read here
without importing Isaac Lab. ``position_m`` and ``quaternion_wxyz`` place the
camera's OpenCV optical frame (x right, y down, z forward) in the robot base
frame, which Isaac Lab calls the ``"ros"`` camera convention.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from arc_mycobot.assets.cameras import EXTERNAL_CAMERA_JSON

__all__ = [
    "CameraCalibration",
    "external_camera_calibration",
    "load_camera_calibration",
    "matrix_from_quaternion_wxyz",
]

EXTERNAL_CAMERA_ENV = "ARC_MYCOBOT_EXTERNAL_CAMERA"


@dataclass(frozen=True)
class CameraCalibration:
    width: int
    height: int
    intrinsic_matrix: tuple[float, ...]
    """Row-major ``[fx, 0, cx, 0, fy, cy, 0, 0, 1]`` in pixels."""
    distortion: tuple[float, ...]
    """OpenCV coefficients; the simulator renders an undistorted image."""
    position_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    calibrated: bool
    """False for a hand-estimated pose that has not been measured."""
    note: str = ""

    @property
    def horizontal_fov_deg(self) -> float:
        return math.degrees(2.0 * math.atan(self.width / (2.0 * self.intrinsic_matrix[0])))

    def look_at(self, distance_m: float = 0.5) -> tuple[float, float, float]:
        """A point on the optical axis, for viewers that take an eye/target pair."""
        x, y, z = np.asarray(self.position_m) + distance_m * matrix_from_quaternion_wxyz(self.quaternion_wxyz)[:, 2]
        return float(x), float(y), float(z)


def matrix_from_quaternion_wxyz(quaternion: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(value) for value in quaternion)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def load_camera_calibration(path: str | Path) -> CameraCalibration:
    data = json.loads(Path(path).read_text())
    position = [float(value) for value in data["position_m"]]
    quaternion = [float(value) for value in data["quaternion_wxyz"]]
    if len(position) != 3 or len(quaternion) != 4:
        raise ValueError(f"{path}: position_m needs 3 values and quaternion_wxyz 4")
    calibration = CameraCalibration(
        width=int(data["width"]),
        height=int(data["height"]),
        intrinsic_matrix=tuple(float(value) for value in data["intrinsic_matrix"]),
        distortion=tuple(float(value) for value in data.get("distortion", ())),
        position_m=(position[0], position[1], position[2]),
        quaternion_wxyz=(quaternion[0], quaternion[1], quaternion[2], quaternion[3]),
        calibrated=bool(data["calibrated"]),
        note=str(data.get("note", "")),
    )
    fx, skew, cx, zero_a, fy, cy, zero_b, zero_c, one = calibration.intrinsic_matrix
    if calibration.width <= 0 or calibration.height <= 0:
        raise ValueError(f"{path}: image size must be positive")
    if fx <= 0 or fy <= 0 or skew != 0 or (zero_a, zero_b, zero_c, one) != (0, 0, 0, 1):
        raise ValueError(f"{path}: intrinsic_matrix must be [fx, 0, cx, 0, fy, cy, 0, 0, 1]")
    if not (0 <= cx <= calibration.width and 0 <= cy <= calibration.height):
        raise ValueError(f"{path}: principal point lies outside the image")
    if abs(float(np.linalg.norm(calibration.quaternion_wxyz)) - 1.0) > 1e-3:
        raise ValueError(f"{path}: quaternion_wxyz is not normalized")
    return calibration


def external_camera_calibration() -> CameraCalibration:
    """The packaged external camera, or the file named by ``ARC_MYCOBOT_EXTERNAL_CAMERA``."""
    return load_camera_calibration(os.environ.get(EXTERNAL_CAMERA_ENV) or EXTERNAL_CAMERA_JSON)
