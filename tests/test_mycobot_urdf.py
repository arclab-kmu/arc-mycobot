# Copyright (c) 2026, arc-mycobot contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""The URDF repair is the part of this project most likely to break silently.

It reads a file in a *sibling checkout* that nothing here controls. A vendor
change can leave the output well-formed and importable while quietly moving a
joint limit, dropping an inertial, or resurrecting a mimic -- none of which
raises, and all of which change what gets trained. These tests pin the
properties Isaac Lab's importer and the reach task actually depend on.

They need no GPU and no Isaac Sim.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from arc_mycobot.assets.robots.mycobot_urdf import (
    ARM_VELOCITY_LIMIT,
    SOURCE_URDF,
    build_mycobot_urdf,
)

pytestmark = pytest.mark.skipif(not SOURCE_URDF.is_file(), reason=f"mycobot_ros2 checkout not found at {SOURCE_URDF}")

# Written out here rather than imported from mycobot_urdf on purpose. This is
# the order the observation and action layout depend on, so the test has to
# assert it against an independent statement of what it should be -- importing
# the module's own tuple would make the assertion tautological.
EXPECTED_ARM_JOINTS = (
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
    "joint6output_to_joint6",
)


@pytest.fixture(scope="module")
def robot() -> ET.Element:
    return ET.parse(build_mycobot_urdf(force=True)).getroot()


def test_source_urdf_is_still_malformed():
    """The vendor file is not well-formed XML, which is why the repair exists.

    If this ever fails, upstream has fixed the stray quote in
    ``joint2_to_joint1``'s limit. That is good news, and it means ``_repair_text``
    should be re-read rather than left to no-op silently.
    """
    with pytest.raises(ET.ParseError):
        ET.parse(SOURCE_URDF)


def test_repaired_urdf_parses(robot):
    assert robot.tag == "robot"
    assert robot.attrib["name"] == "mycobot_280_jn_adaptive_gripper"


def test_no_xacro_elements_survive(robot):
    """urdfdom has no xacro support; a surviving element is a parse failure."""
    assert [child.tag for child in robot if "xacro" in child.tag] == []


def test_six_actuated_joints_in_order(robot):
    """Exactly six DOF, in base-to-flange order.

    The order is the observation and action layout, so it is a contract, not an
    incidental property of how the file happens to be written.
    """
    revolute = [j.attrib["name"] for j in robot.findall("joint") if j.attrib["type"] == "revolute"]
    assert tuple(revolute) == EXPECTED_ARM_JOINTS


def test_gripper_is_welded_and_no_mimic_survives(robot):
    """The reach task welds the cluster, so no <mimic> may survive: one naming a
    joint that has become fixed makes the importer raise a bare IndexError."""
    assert robot.findall(".//mimic") == []
    gripper = [j for j in robot.findall("joint") if "gripper" in j.attrib["name"]]
    assert gripper, "the gripper joints vanished from the vendor URDF"
    assert all(j.attrib["type"] == "fixed" for j in gripper)


def test_every_joint_has_a_usable_velocity_limit(robot):
    """The vendor writes velocity="0", which imports as a joint that cannot move."""
    for joint in robot.findall("joint"):
        limit = joint.find("limit")
        if limit is None:
            assert joint.attrib["type"] == "fixed"
            continue
        assert float(limit.attrib["velocity"]) == pytest.approx(ARM_VELOCITY_LIMIT)


def test_joint2_to_joint1_limit_survived_the_text_repair(robot):
    """The repaired limit must be the value the vendor meant, not a truncation.

    ``lower = "-2.932"1`` is repaired to ``-2.9321``, matching the symmetric
    ``upper = "2.9321"``. Dropping the stray digit instead would silently narrow
    the joint by 0.1 mrad -- harmless, but it would mean the repair guessed.
    """
    limit = robot.find("joint[@name='joint2_to_joint1']/limit")
    assert float(limit.attrib["lower"]) == pytest.approx(-2.9321)
    assert float(limit.attrib["upper"]) == pytest.approx(2.9321)


def test_every_link_has_positive_mass_and_inertia(robot):
    """A massless link in a PhysX articulation is a hard failure, and the vendor
    URDF has no <inertial> at all."""
    links = robot.findall("link")
    assert len(links) == 14
    for link in links:
        inertial = link.find("inertial")
        assert inertial is not None, f"{link.attrib['name']} has no <inertial>"
        assert float(inertial.find("mass").attrib["value"]) > 0.0
        inertia = inertial.find("inertia")
        for axis in ("ixx", "iyy", "izz"):
            assert float(inertia.attrib[axis]) > 0.0


def test_total_mass_matches_the_published_figure(robot):
    """0.85 kg arm + ~0.11 kg gripper. Guards against a typo in _LINK_MASSES."""
    total = sum(float(link.find("inertial/mass").attrib["value"]) for link in robot.findall("link"))
    assert total == pytest.approx(0.968, abs=0.05)


def test_all_mesh_paths_are_absolute_and_exist(robot):
    """urdfdom resolves package:// only under ROS, which this project does not use."""
    meshes = [mesh.attrib["filename"] for mesh in robot.iter("mesh")]
    assert meshes, "the URDF lost all of its mesh references"
    for filename in meshes:
        assert not filename.startswith("package://")
        assert filename.startswith("/")


def test_root_link_is_joint1(robot):
    """The articulation root, and the frame every goal position is expressed in."""
    children = {joint.find("child").attrib["link"] for joint in robot.findall("joint")}
    links = {link.attrib["name"] for link in robot.findall("link")}
    assert links - children == {"joint1"}


def test_rebuild_is_cached():
    """A second call must not rewrite the file: it is imported at module scope by
    mycobot_280.py, so an unconditional rebuild would cost a parse on every
    import of anything in this package."""
    first = build_mycobot_urdf(force=True)
    stamp = first.stat().st_mtime_ns
    assert build_mycobot_urdf() == first
    assert first.stat().st_mtime_ns == stamp
