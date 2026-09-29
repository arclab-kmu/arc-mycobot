"""Forward kinematics straight off the packaged URDF. NumPy only, no Isaac import.

This exists so the reach task's goal box can be *derived* rather than guessed.
A goal distribution that reaches outside the arm's workspace puts a floor under
the achievable tracking error that no amount of training removes, and the
myCobot 280's 280 mm reach leaves very little room to be sloppy about it -- an
error that would be a rounding detail on a 1.3 m UR10e is a quarter of this
arm's workspace.

It reads the same file Isaac Lab imports (:data:`~arc_mycobot.assets.robots.camera_gripper_assets.CAMERA_GRIPPER_URDF`),
so the chain here and the chain in simulation cannot drift apart. Only the
joint types the myCobot uses are implemented -- ``revolute``, ``prismatic`` and
``fixed`` -- and anything else raises rather than being silently treated as
fixed.

Convention: URDF's own. Each joint applies its ``<origin>`` as a fixed
translation plus extrinsic-XYZ (roll-pitch-yaw) rotation, then either rotates by
the joint angle about ``<axis>`` (revolute) or translates along it (prismatic),
with the axis expressed in the child frame.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from arc_mycobot.assets.robots.camera_gripper_assets import CAMERA_GRIPPER_URDF

__all__ = ["Chain", "Joint", "load_chain", "rpy_to_matrix"]


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF's fixed-axis roll-pitch-yaw to a 3x3 rotation matrix (``Rz @ Ry @ Rx``)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


@dataclass(frozen=True)
class Joint:
    """One URDF joint, reduced to what forward kinematics needs."""

    name: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rot: np.ndarray
    axis: np.ndarray | None
    """``None`` for a fixed joint."""
    limit: tuple[float, float] | None
    """``(lower, upper)`` in rad or m; ``None`` for a fixed joint."""
    prismatic: bool = False
    """Slides along :attr:`axis` instead of rotating about it."""

    @property
    def is_actuated(self) -> bool:
        return self.axis is not None


@dataclass(frozen=True)
class Chain:
    """A serial kinematic chain resolved from the repaired URDF."""

    joints: tuple[Joint, ...]
    """Every joint from the root to the last link, fixed ones included, in order."""

    @property
    def actuated(self) -> tuple[Joint, ...]:
        return tuple(joint for joint in self.joints if joint.is_actuated)

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(joint.name for joint in self.actuated)

    @property
    def limits(self) -> np.ndarray:
        """``(n, 2)`` array of ``(lower, upper)`` for the actuated joints [rad]."""
        return np.array([joint.limit for joint in self.actuated], dtype=float)

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pose of the chain's tip in the root frame.

        Args:
            q: Actuated joint values -- rad for revolute, m for prismatic -- in
                :attr:`joint_names` order.

        Returns:
            ``(position, rotation)`` -- a ``(3,)`` translation [m] and a
            ``(3, 3)`` rotation matrix.
        """
        q = np.asarray(q, dtype=float)
        if q.shape != (len(self.actuated),):
            raise ValueError(f"expected {len(self.actuated)} joint angles, got {q.shape}")

        position = np.zeros(3)
        rotation = np.eye(3)
        index = 0
        for joint in self.joints:
            position = position + rotation @ joint.origin_xyz
            rotation = rotation @ joint.origin_rot
            if joint.is_actuated:
                if joint.prismatic:
                    position = position + rotation @ (joint.axis * q[index])  # type: ignore[operator]
                else:
                    rotation = rotation @ _axis_angle(joint.axis, q[index])  # type: ignore[arg-type]
                index += 1
        return position, rotation


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' rotation about a unit ``axis``."""
    axis = axis / np.linalg.norm(axis)
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def _parse_joint(element: ET.Element) -> Joint:
    joint_type = element.attrib["type"]
    if joint_type not in ("revolute", "continuous", "prismatic", "fixed"):
        raise NotImplementedError(
            f"joint {element.attrib['name']!r} is {joint_type!r}; this FK implements revolute, prismatic and fixed only"
        )
    origin = element.find("origin")
    xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
    rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)

    axis = None
    limit = None
    if joint_type != "fixed":
        axis_element = element.find("axis")
        axis = np.fromstring(axis_element.attrib["xyz"], sep=" ") if axis_element is not None else np.array([1.0, 0, 0])
        limit_element = element.find("limit")
        if limit_element is None:
            raise ValueError(f"joint {element.attrib['name']!r} is {joint_type} but declares no <limit>")
        limit = (float(limit_element.attrib["lower"]), float(limit_element.attrib["upper"]))

    parent_element = element.find("parent")
    child_element = element.find("child")
    if parent_element is None or child_element is None:
        raise ValueError(f"joint {element.attrib['name']!r} has no parent or child link")

    return Joint(
        name=element.attrib["name"],
        parent=parent_element.attrib["link"],
        child=child_element.attrib["link"],
        origin_xyz=xyz,
        origin_rot=rpy_to_matrix(*rpy),
        axis=axis,
        limit=limit,
        prismatic=joint_type == "prismatic",
    )


def load_chain(tip_link: str, urdf_path: Path | None = None) -> Chain:
    """Resolve the chain from the articulation root down to ``tip_link``.

    Args:
        tip_link: The link whose pose :meth:`Chain.fk` returns.
        urdf_path: Defaults to the packaged camera/gripper URDF.

    Raises:
        ValueError: ``tip_link`` is unknown, or the chain to it is not serial.
    """
    path = urdf_path or CAMERA_GRIPPER_URDF
    robot = ET.parse(path).getroot()
    by_child: dict[str, ET.Element] = {}
    for joint in robot.findall("joint"):
        child = joint.find("child")
        if child is None:
            raise ValueError(f"joint {joint.attrib['name']!r} has no child link")
        by_child[child.attrib["link"]] = joint

    if tip_link not in by_child and robot.find(f"link[@name='{tip_link}']") is None:
        known = sorted(link.attrib["name"] for link in robot.findall("link"))
        raise ValueError(f"unknown link {tip_link!r}; the URDF defines {known}")

    reversed_chain: list[Joint] = []
    link = tip_link
    while link in by_child:
        joint = _parse_joint(by_child[link])
        reversed_chain.append(joint)
        link = joint.parent
        if len(reversed_chain) > len(by_child):  # pragma: no cover - a cycle in the URDF
            raise ValueError(f"cycle in the kinematic chain above {tip_link!r}")

    return Chain(joints=tuple(reversed(reversed_chain)))
