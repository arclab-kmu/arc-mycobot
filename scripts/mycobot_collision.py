# Copyright (c) 2026, arc-mycobot contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Simulator-free forward kinematics and self-collision checks for the packaged myCobot URDF.

Shared by ``reach_workspace_sweep.py`` and ``reach_self_collision_check.py``. It reads the same URDF the importer
converts and uses its ``*_hull.stl`` collision meshes, which the importer turns into convex hulls
(``collision_type="Convex Hull"``).

Pairs are filtered the way the Newton + MuJoCo Warp backend filters them with self-collision on: links welded together
by fixed joints form one group, and a pair is skipped when both links are in the same group or their groups are joined
by one joint. Every other pair can collide.

Overlap is measured as the depth of hull sample points inside the other hull, expanded by ``margin``. For a convex
hull the distance to the nearest face plane is the exact depth of a point. Sampling makes this a lower bound, and so
does the vertex subsampling below (``max_vertices``).
"""

from __future__ import annotations

import itertools
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "src" / "arc_mycobot" / "assets" / "data" / "mycobot_280_jn"
URDF_PATH = PACKAGE_DIR / "urdf" / "mycobot_280_jn_camera_gripper.urdf"
"""The reach URDF (fixed gripper); same file as ``MYCOBOT_280_JN_URDF_PATH`` without importing Isaac Lab."""
ARM_JOINTS = [
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
    "joint6output_to_joint6",
]
"""Same order as ``MYCOBOT_ARM_JOINTS``."""
ROOT_LINK = "joint1"
FLANGE_LINK = "joint6_flange"


def _attr(element: ET.Element, name: str) -> str:
    value = element.get(name)
    if value is None:
        raise ValueError(f"<{element.tag}> has no '{name}' attribute")
    return value


def _child(element: ET.Element, path: str) -> ET.Element:
    found = element.find(path)
    if found is None:
        raise ValueError(f"<{element.tag} name='{element.get('name')}'> has no <{path}>")
    return found


def _floats(text: str) -> list[float]:
    return [float(v) for v in text.split()]


def _origin(element: ET.Element | None) -> np.ndarray:
    transform = np.eye(4)
    if element is None:
        return transform
    transform[:3, :3] = Rotation.from_euler("xyz", _floats(element.get("rpy", "0 0 0"))).as_matrix()
    transform[:3, 3] = _floats(element.get("xyz", "0 0 0"))
    return transform


@dataclass
class _Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    limits: tuple[float, float]


class MyCobotGeometry:
    """Forward kinematics and collision hulls of the packaged URDF, batched on ``device``."""

    def __init__(
        self,
        urdf_path: Path = URDF_PATH,
        device: str = "cuda",
        max_vertices: int = 96,
        surface_samples: int = 96,
    ):
        self.device = device
        root = ET.parse(urdf_path).getroot()
        joints = []
        for element in root.findall("joint"):
            kind = _attr(element, "type")
            limit = element.find("limit")
            axis = element.find("axis")
            joints.append(
                _Joint(
                    name=_attr(element, "name"),
                    kind=kind,
                    parent=_attr(_child(element, "parent"), "link"),
                    child=_attr(_child(element, "child"), "link"),
                    origin=_origin(element.find("origin")),
                    axis=np.array(_floats(_attr(axis, "xyz"))) if axis is not None else np.zeros(3),
                    limits=(float(_attr(limit, "lower")), float(_attr(limit, "upper")))
                    if limit is not None and kind != "fixed"
                    else (0.0, 0.0),
                )
            )
        # topological order from the root
        self.joints: list[_Joint] = []
        known = {ROOT_LINK}
        while len(self.joints) < len(joints):
            for joint in joints:
                if joint.parent in known and joint.child not in known:
                    self.joints.append(joint)
                    known.add(joint.child)
        self.links = [ROOT_LINK] + [joint.child for joint in self.joints]
        by_name = {joint.name: joint for joint in self.joints}
        self.arm_joints = [by_name[name] for name in ARM_JOINTS]
        self.lower = torch.tensor([j.limits[0] for j in self.arm_joints], device=device)
        self.upper = torch.tensor([j.limits[1] for j in self.arm_joints], device=device)
        self._origins = {j.name: torch.tensor(j.origin, dtype=torch.float32, device=device) for j in self.joints}
        self._axes = {j.name: torch.tensor(j.axis, dtype=torch.float32, device=device) for j in self.joints}

        # weld groups: a link belongs to the group of its nearest ancestor reached through a non-fixed joint
        parent_joint = {j.child: j for j in self.joints}
        self.group = {}
        for link in self.links:
            node = link
            while node in parent_joint and parent_joint[node].kind == "fixed":
                node = parent_joint[node].parent
            self.group[link] = node
        group_parent = {g: self.group[parent_joint[g].parent] for g in set(self.group.values()) if g in parent_joint}

        # collision hulls in their link frames
        self.hulls: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        rng = np.random.default_rng(0)
        for element in root.findall("link"):
            collision = element.find("collision")
            if collision is None:
                continue
            mesh_path = urdf_path.parent / _attr(_child(collision, "geometry/mesh"), "filename")
            loaded = trimesh.load(mesh_path, force="mesh")
            assert isinstance(loaded, trimesh.Trimesh)
            mesh = loaded.convex_hull
            mesh.apply_transform(_origin(collision.find("origin")))
            if len(mesh.vertices) > max_vertices:
                keep = rng.choice(len(mesh.vertices), max_vertices, replace=False)
                mesh = trimesh.convex.convex_hull(mesh.vertices[keep])
            points = np.vstack([mesh.vertices, trimesh.sample.sample_surface(mesh, surface_samples, seed=0)[0]])
            normals = mesh.face_normals
            offsets = np.einsum("fi,fi->f", normals, mesh.triangles[:, 0])
            points_t, normals_t, offsets_t = (
                torch.tensor(a, dtype=torch.float32, device=device) for a in (points, normals, offsets)
            )
            self.hulls[_attr(element, "name")] = (points_t, normals_t, offsets_t)
        self.pairs = [
            (a, b)
            for a, b in itertools.combinations(self.hulls, 2)
            if self.group[a] != self.group[b]
            and group_parent.get(self.group[a]) != self.group[b]
            and group_parent.get(self.group[b]) != self.group[a]
        ]

    def forward_kinematics(self, q: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Link poses ``[B, 4, 4]`` in the root frame, and the joint frames (pre-rotation) of the arm joints."""
        batch = q.shape[0]
        eye = torch.eye(4, device=self.device).expand(batch, 4, 4)
        poses = {ROOT_LINK: eye}
        frames = {}
        angles = {joint.name: q[:, i] for i, joint in enumerate(self.arm_joints)}
        for joint in self.joints:
            frame = poses[joint.parent] @ self._origins[joint.name]
            if joint.kind == "fixed":
                poses[joint.child] = frame
                continue
            frames[joint.name] = frame
            angle = angles.get(joint.name, torch.zeros(batch, device=self.device))
            rotation = torch.eye(4, device=self.device).repeat(batch, 1, 1)
            rotation[:, :3, :3] = _axis_angle(self._axes[joint.name], angle)
            poses[joint.child] = frame @ rotation
        return poses, frames

    def flange_jacobian(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Flange position ``[B, 3]`` and its position Jacobian ``[B, 3, 6]`` in the root frame."""
        poses, frames = self.forward_kinematics(q)
        position = poses[FLANGE_LINK][:, :3, 3]
        columns = []
        for joint in self.arm_joints:
            frame = frames[joint.name]
            axis = frame[:, :3, :3] @ self._axes[joint.name]
            columns.append(torch.cross(axis, position - frame[:, :3, 3], dim=-1))
        return position, torch.stack(columns, -1)

    def solve_position_ik(
        self, goals: torch.Tensor, starts: torch.Tensor, iterations: int = 300, damping: float = 0.02
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Damped least squares from each start toward its goal, clamped to the joint limits.

        Returns the joint vectors ``[B, 6]`` and the final position errors ``[B]`` [m].
        """
        q = starts.clone()
        eye = torch.eye(3, device=self.device)
        for _ in range(iterations):
            position, jacobian = self.flange_jacobian(q)
            error = goals - position
            jjt = jacobian @ jacobian.transpose(1, 2) + damping**2 * eye
            step = jacobian.transpose(1, 2) @ torch.linalg.solve(jjt, error.unsqueeze(-1))
            q = torch.clamp(q + step.squeeze(-1), self.lower, self.upper)
        position, _ = self.flange_jacobian(q)
        return q, (goals - position).norm(dim=-1)

    def pair_depths(self, poses: dict[str, torch.Tensor], margin: float = 0.0, chunk: int = 4096) -> torch.Tensor:
        """Per-pair overlap depth ``[num_pairs, B]`` [m], with each hull expanded by ``margin``; 0 when apart."""
        batch = next(iter(poses.values())).shape[0]
        out = torch.zeros(len(self.pairs), batch, device=self.device)
        for k, (a, b) in enumerate(self.pairs):
            for start in range(0, batch, chunk):
                sl = slice(start, start + chunk)
                out[k, sl] = torch.maximum(
                    self._depth_into(a, b, poses, sl, margin), self._depth_into(b, a, poses, sl, margin)
                )
        return out

    def _depth_into(self, a: str, b: str, poses: dict[str, torch.Tensor], sl: slice, margin: float) -> torch.Tensor:
        points, _, _ = self.hulls[a]
        _, normals, offsets = self.hulls[b]
        relative = torch.linalg.inv(poses[b][sl]) @ poses[a][sl]
        local = points @ relative[:, :3, :3].transpose(1, 2) + relative[:, None, :3, 3]
        depth = (offsets + margin - local @ normals.T).amin(-1)
        return depth.amax(-1).clamp_min(0.0)

    def lowest_point(self, poses: dict[str, torch.Tensor]) -> torch.Tensor:
        """Lowest hull point ``[B]`` [m] of every link outside the root's weld group, in the root frame."""
        lowest = []
        for link, (points, _, _) in self.hulls.items():
            if self.group[link] == ROOT_LINK:
                continue
            pose = poses[link]
            world = points @ pose[:, :3, :3].transpose(1, 2) + pose[:, None, :3, 3]
            lowest.append(world[..., 2].amin(-1))
        return torch.stack(lowest).amin(0)


def _axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    x, y, z = axis / axis.norm()
    c, s = torch.cos(angle), torch.sin(angle)
    t = 1 - c
    return torch.stack(
        [
            torch.stack([c + x * x * t, x * y * t - z * s, x * z * t + y * s], -1),
            torch.stack([y * x * t + z * s, c + y * y * t, y * z * t - x * s], -1),
            torch.stack([z * x * t - y * s, z * y * t + x * s, c + z * z * t], -1),
        ],
        -2,
    )


def poses_from_sim(positions: np.ndarray, quaternions_xyzw: np.ndarray, device: str = "cuda") -> torch.Tensor:
    """Homogeneous transforms ``[..., 4, 4]`` from simulator body positions and XYZW quaternions."""
    shape = positions.shape[:-1]
    transform = np.zeros((*shape, 4, 4), dtype=np.float32)
    transform[..., :3, :3] = Rotation.from_quat(quaternions_xyzw.reshape(-1, 4)).as_matrix().reshape(*shape, 3, 3)
    transform[..., :3, 3] = positions
    transform[..., 3, 3] = 1.0
    return torch.tensor(transform, device=device)
