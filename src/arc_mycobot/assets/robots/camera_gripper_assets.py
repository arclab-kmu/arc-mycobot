"""Packaged myCobot camera/gripper URDF paths; no vendor checkout required."""

from pathlib import Path

URDF_DIR = Path(__file__).resolve().parent / "urdf"
CAMERA_GRIPPER_URDF = URDF_DIR / "mycobot_280_jn_camera_gripper.urdf"
CAMERA_GRIPPER_PARALLEL_URDF = URDF_DIR / "mycobot_280_jn_camera_gripper_parallel.urdf"

__all__ = ["CAMERA_GRIPPER_URDF", "CAMERA_GRIPPER_PARALLEL_URDF"]
