"""Collision-hull points of URDF links in their own link frames. NumPy only, no Isaac import.

The lift reward that keeps the camera and wrist off the table needs each link's
lowest point, which for a convex hull is always one of its vertices. Links with
a collision mesh use it; the finger links, which have none, use the packaged
convex hull of their visual mesh. Keeping
only the vertices that are extreme along a few hundred directions makes the
per-step check cheap while staying within a fraction of a millimetre.
"""

from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from arc_mycobot.assets.robots.camera_gripper_assets import CAMERA_GRIPPER_PARALLEL_URDF

from .urdf_fk import rpy_to_matrix

__all__ = ["link_hull_points", "stl_vertices"]


def stl_vertices(path: str | Path) -> np.ndarray:
    """Unique vertices of an ASCII or binary STL file."""
    data = Path(path).read_bytes()
    if data[:5].lower() == b"solid" and b"facet" in data[:300]:
        lines = data.decode().splitlines()
        vertices = np.array(
            [[float(v) for v in line.split()[1:4]] for line in lines if line.strip().startswith("vertex")]
        )
    else:
        count = struct.unpack("<I", data[80:84])[0]
        record = np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
        vertices = np.frombuffer(data[84 : 84 + count * 50], dtype=record)["vertices"].reshape(-1, 3).astype(float)
    return np.unique(vertices, axis=0)


def _directions(count: int) -> np.ndarray:
    index = np.arange(count) + 0.5
    polar = np.arccos(1.0 - 2.0 * index / count)
    azimuth = np.pi * (1.0 + 5.0**0.5) * index
    return np.column_stack((np.cos(azimuth) * np.sin(polar), np.sin(azimuth) * np.sin(polar), np.cos(polar)))


def link_hull_points(
    link_names: Sequence[str], urdf_path: str | Path = CAMERA_GRIPPER_PARALLEL_URDF, directions: int = 400
) -> dict[str, np.ndarray]:
    """Extreme hull vertices of each link, expressed in that link's frame."""
    root = ET.parse(urdf_path).getroot()
    points = {}
    for name in link_names:
        element = root.find(f"link[@name='{name}']/collision")
        mesh = None if element is None else element.find("geometry/mesh")
        if mesh is not None:
            path = Path(urdf_path).parent / mesh.attrib["filename"]
        else:
            element = root.find(f"link[@name='{name}']/visual")
            mesh = None if element is None else element.find("geometry/mesh")
            if mesh is None:
                raise ValueError(f"{name} has neither a collision nor a visual mesh in {urdf_path}")
            path = Path(urdf_path).parent / mesh.attrib["filename"]
            hull = path.with_name(f"{path.stem}_hull.stl")
            path = hull if hull.is_file() else path
        vertices = stl_vertices(path)
        extreme = np.unique((vertices @ _directions(directions).T).argmax(axis=0))
        origin = element.find("origin")
        xyz = np.array([float(v) for v in origin.attrib["xyz"].split()]) if origin is not None else np.zeros(3)
        rpy = [float(v) for v in origin.attrib["rpy"].split()] if origin is not None else [0.0, 0.0, 0.0]
        points[name] = vertices[extreme] @ rpy_to_matrix(*rpy).T + xyz
    return points
