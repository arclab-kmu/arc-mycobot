# Copyright (c) 2026, arc-mycobot contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Pin the two URDFs used by the runtime without requiring the vendor checkout."""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from arc_mycobot.assets.robots.camera_gripper_assets import CAMERA_GRIPPER_PARALLEL_URDF, CAMERA_GRIPPER_URDF
from arc_mycobot.assets.robots.mycobot_urdf import ARM_JOINTS, LIFT_HOME_POSE, LIFT_START_POSE, PARALLEL_FINGER_JOINTS
from arc_mycobot.kinematics.urdf_fk import load_chain
from arc_mycobot.tasks.lift.config.mycobot.geometry import JAW_OFFSET_IN_GRIPPER_BASE, SPAWN_POS_CENTRE


@pytest.mark.parametrize("path", [CAMERA_GRIPPER_URDF, CAMERA_GRIPPER_PARALLEL_URDF])
def test_packaged_meshes_and_camera_mount(path: Path):
    root = ET.parse(path).getroot()
    meshes = [mesh.attrib["filename"] for mesh in root.findall(".//mesh")]
    assert meshes
    assert all(name.startswith("../meshes/") and (path.parent / name).is_file() for name in meshes)
    assert root.findall(".//mimic") == []
    assert [joint.attrib["name"] for joint in root.findall("joint") if joint.attrib["type"] == "revolute"] == list(
        ARM_JOINTS
    )
    for link in ("camera_link", "camera_flange", "tool_mount", "tcp"):
        assert root.find(f"link[@name='{link}']") is not None
    mount = root.find("joint[@name='joint6_flange_to_tool_mount']/origin")
    assert mount is not None
    assert float(mount.attrib["rpy"].split()[2]) == pytest.approx(math.pi / 4, abs=1e-5)


@pytest.mark.parametrize("path", [CAMERA_GRIPPER_URDF, CAMERA_GRIPPER_PARALLEL_URDF])
def test_packaged_mass_matches_component_specs(path: Path):
    """SKU 4010100018 arm + adaptive gripper + camera-flange nominal masses."""
    root = ET.parse(path).getroot()
    masses = {}
    for link in root.findall("link"):
        mass = link.find("inertial/mass")
        if mass is not None:
            masses[link.attrib["name"]] = float(mass.attrib["value"])
    arm = sum(mass for name, mass in masses.items() if name.startswith("joint"))
    gripper = sum(mass for name, mass in masses.items() if name.startswith("gripper"))
    assert arm == pytest.approx(1.030, abs=1e-6)
    assert gripper == pytest.approx(0.110, abs=1e-6)
    assert masses["camera_flange"] == pytest.approx(0.060, abs=1e-6)
    # PhysX requires nonzero inertial frames when fixed joints are retained for
    # camera attachment. Their total numerical contribution is only 0.4 g.
    for frame in ("tool_mount", "camera_mount", "camera_link", "tcp"):
        assert masses[frame] == pytest.approx(0.0001, abs=1e-8)
    assert len(masses) == len(root.findall("link"))
    assert sum(masses.values()) == pytest.approx(1.2004, abs=1e-6)
    inertia = root.find("link[@name='camera_flange']/inertial/inertia")
    assert inertia is not None
    assert all(float(inertia.attrib[axis]) > 0 for axis in ("ixx", "iyy", "izz"))


def test_reach_and_lift_have_distinct_jaws():
    reach = ET.parse(CAMERA_GRIPPER_URDF).getroot()
    lift = ET.parse(CAMERA_GRIPPER_PARALLEL_URDF).getroot()
    assert all(
        j.attrib["type"] == "fixed" for j in reach.findall("joint") if j.attrib["name"] in PARALLEL_FINGER_JOINTS
    )
    assert [j.attrib["name"] for j in lift.findall("joint") if j.attrib["type"] == "prismatic"] == list(
        PARALLEL_FINGER_JOINTS
    )
    colliders = [link.attrib["name"] for link in lift.findall("link") if link.find("collision") is not None]
    assert colliders == ["joint6", "joint6_flange", "gripper_base", "gripper_left3", "gripper_right3", "camera_flange"]
    for pad in ("gripper_left3", "gripper_right3"):
        assert lift.find(f"link[@name='{pad}']/collision/geometry/box") is not None


def test_lift_hull_colliders_follow_their_visual_meshes():
    """Same frame as the visual mesh, so the hull sits where the part is drawn."""
    lift = ET.parse(CAMERA_GRIPPER_PARALLEL_URDF).getroot()
    for name in ("joint6", "joint6_flange", "gripper_base", "camera_flange"):
        link = lift.find(f"link[@name='{name}']")
        visual, collision = link.find("visual"), link.find("collision")
        assert collision.find("origin").attrib == visual.find("origin").attrib
        visual_mesh = visual.find("geometry/mesh").attrib["filename"]
        hull_mesh = collision.find("geometry/mesh").attrib["filename"]
        assert hull_mesh == visual_mesh.replace(".stl", "_hull.stl")
        assert (CAMERA_GRIPPER_PARALLEL_URDF.parent / hull_mesh).is_file()


def test_cpu_fk_reads_the_reach_asset():
    camera_chain = load_chain("camera_link")
    assert camera_chain.joint_names == ARM_JOINTS
    assert load_chain("joint6_flange").joint_names == ARM_JOINTS


def test_lift_home_jaw_faces_the_cube():
    chain = load_chain("gripper_base", CAMERA_GRIPPER_PARALLEL_URDF)
    angles = np.array([LIFT_HOME_POSE[name] for name in ARM_JOINTS])
    pos, rot = chain.fk(angles)
    jaw = pos + rot @ np.array(JAW_OFFSET_IN_GRIPPER_BASE)
    closing_axis = rot[:, 0]
    assert jaw[:2] == pytest.approx(SPAWN_POS_CENTRE[:2], abs=0.001)
    assert abs(closing_axis[1]) > 0.95  # the parallel jaw is symmetric along its closing line
    assert abs(closing_axis[2]) < 0.06
    camera, _ = load_chain("camera_link", CAMERA_GRIPPER_PARALLEL_URDF).fk(angles)
    assert camera[2] > jaw[2] + 0.03, "the camera must sit above the jaw, not between it and the table"


def test_lift_starts_from_the_physical_straight_pose():
    angles = np.array([LIFT_START_POSE[name] for name in ARM_JOINTS])
    assert angles == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, -math.pi / 4])
    home_j6 = LIFT_HOME_POSE["joint6output_to_joint6"]
    assert abs(home_j6 - angles[5]) < math.radians(3)
