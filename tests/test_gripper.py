# Copyright (c) 2026, arc-mycobot contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""The actuated-gripper URDF is the lift task's foundation, and it is fragile.

It drops the vendor's <mimic> coupling and rebuilds it in the action term, and it
replaces the fingertip collision meshes with box pads. Both are silent if they
break: a lost multiplier or a mis-signed pad offset still produces a valid URDF
and an articulation that imports fine, and the only symptom is that nothing can
be picked up. These tests pin the invariants.

CPU only, no Isaac Sim.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from arc_mycobot.assets.robots.mycobot_urdf import (
    ACTUATED_GRIPPER_JOINTS,
    ARM_JOINTS,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    MIMIC_COMMANDED_JOINTS,
    SOURCE_URDF,
    build_mycobot_urdf,
)
from arc_mycobot.kinematics.urdf_fk import load_chain
from arc_mycobot.tasks.lift.config.mycobot.geometry import CUBE_SIZE, JAW_OFFSET_IN_GRIPPER_BASE

pytestmark = pytest.mark.skipif(not SOURCE_URDF.is_file(), reason="mycobot_ros2 checkout not found")


@pytest.fixture(scope="module")
def actuated():
    return build_mycobot_urdf("actuated", force=True)


@pytest.fixture(scope="module")
def robot(actuated):
    return ET.parse(actuated).getroot()


def test_welded_and_actuated_are_different_files():
    assert build_mycobot_urdf("welded") != build_mycobot_urdf("actuated")


def test_gripper_mode_is_validated():
    with pytest.raises(ValueError):
        build_mycobot_urdf("half-open")  # type: ignore[arg-type]


def test_actuated_has_ten_dof(robot):
    revolute = [j.attrib["name"] for j in robot.findall("joint") if j.attrib["type"] == "revolute"]
    assert len(revolute) == 10
    assert set(ACTUATED_GRIPPER_JOINTS) <= set(revolute)


def test_dangling_parallel_links_stay_welded(robot):
    """gripper_left2/right2 are the open end of a four-bar URDF cannot close."""
    for name in ("gripper_base_to_gripper_left2", "gripper_base_to_gripper_right2"):
        joint = robot.find(f"joint[@name='{name}']")
        assert joint is not None and joint.attrib["type"] == "fixed"


def test_no_mimic_survives_in_either_mode(robot):
    """In this mode the coupling lives in the action term, not in <mimic>."""
    assert robot.findall(".//mimic") == []
    welded = ET.parse(build_mycobot_urdf("welded")).getroot()
    assert welded.findall(".//mimic") == []


def test_open_and_close_targets_respect_joint_limits(robot):
    for targets in (GRIPPER_OPEN, GRIPPER_CLOSED):
        for name, value in targets.items():
            limit = robot.find(f"joint[@name='{name}']/limit")
            assert limit is not None, f"{name} is not movable"
            lower, upper = float(limit.attrib["lower"]), float(limit.attrib["upper"])
            assert lower <= value <= upper, f"{name}={value} outside [{lower}, {upper}]"


def test_open_and_close_use_the_vendor_multipliers():
    """Driver-relative ratios must stay +-1.0, or the fingers stop being parallel."""
    driver = ACTUATED_GRIPPER_JOINTS[0]
    for targets in (GRIPPER_OPEN, GRIPPER_CLOSED):
        ratios = {n: targets[n] / targets[driver] for n in ACTUATED_GRIPPER_JOINTS}
        assert ratios == pytest.approx(
            {
                "gripper_controller": 1.0,
                "gripper_left3_to_gripper_left1": -1.0,
                "gripper_base_to_gripper_right3": -1.0,
                "gripper_right3_to_gripper_right1": 1.0,
            }
        )


def test_only_the_fingertips_carry_collision(robot):
    """Everything else on the gripper is stripped -- see _PAD_INSET."""
    with_collision = {link.attrib["name"] for link in robot.findall("link") if link.find("collision") is not None}
    assert with_collision == {"gripper_left1", "gripper_right1"}


def test_grip_pads_are_boxes_and_mirrored(robot):
    offsets = {}
    for name in ("gripper_left1", "gripper_right1"):
        collision = robot.find(f"link[@name='{name}']/collision")
        assert collision.find("geometry/box") is not None, f"{name} pad is not a box"
        offsets[name] = float(collision.find("origin").attrib["xyz"].split()[0])
    assert offsets["gripper_left1"] == pytest.approx(-offsets["gripper_right1"])
    assert offsets["gripper_left1"] > 0


def _jaw_separation(urdf, targets):
    left, right = load_chain("gripper_left1", urdf), load_chain("gripper_right1", urdf)
    home = dict.fromkeys(ARM_JOINTS, 0.0)

    def q(chain):
        return np.array([home[n] if n in ARM_JOINTS else targets[n] for n in chain.joint_names])

    return float(np.linalg.norm(left.fk(q(left))[0] - right.fk(q(right))[0]))


def _pad_gap(urdf, targets):
    """Free space between the two grip pads, across the closing axis [m].

    The quantity that decides what fits, and *not* the distance between the
    fingertip origins: the fingers splay as well as close, so the 3-D origin
    separation overstates the closing-axis projection by about 12 mm. Sizing the
    pads off the 3-D number once left the jaw 12 mm too tight.
    """
    root = ET.parse(urdf).getroot()
    base = load_chain("gripper_base", urdf)
    home = dict.fromkeys(ARM_JOINTS, 0.0)

    def q(chain):
        return np.array([home[n] if n in ARM_JOINTS else targets[n] for n in chain.joint_names])

    def corners(name):
        collision = root.find(f"link[@name='{name}']/collision")
        origin = np.fromstring(collision.find("origin").attrib["xyz"], sep=" ")
        half = np.fromstring(collision.find("geometry/box").attrib["size"], sep=" ") / 2
        local = np.array([origin + half * s for s in np.ndindex(2, 2, 2)] + [origin - half])
        local = np.array([origin + half * (np.array(s) * 2 - 1) for s in np.ndindex(2, 2, 2)])
        chain = load_chain(name, urdf)
        pos, rot = chain.fk(q(chain))
        world = (rot @ local.T).T + pos
        base_pos, base_rot = base.fk(q(base))
        return (base_rot.T @ (world - base_pos).T).T

    left, right = corners("gripper_left1"), corners("gripper_right1")
    return float(right[:, 0].min() - left[:, 0].max())


def test_closing_actually_narrows_the_jaw(actuated):
    """Sign check. If the driver's closed value had the wrong sign the jaw would
    open wider on 'close' and every grasp would silently fail."""
    opened = _jaw_separation(actuated, GRIPPER_OPEN)
    closed = _jaw_separation(actuated, GRIPPER_CLOSED)
    assert closed < opened
    assert opened - closed > 0.02, "jaw travel under 20 mm leaves nothing to grip with"


def test_cube_fits_between_the_pads(mimic_urdf):
    """The cube must pass between the open pads and be squeezed by the closed ones."""
    open_gap = _pad_gap(mimic_urdf, GRIPPER_OPEN)
    closed_gap = _pad_gap(mimic_urdf, GRIPPER_CLOSED)
    assert closed_gap < CUBE_SIZE < open_gap, f"cube {CUBE_SIZE} outside jaw range [{closed_gap}, {open_gap}]"


def test_grasp_point_lies_in_the_band_the_pads_sweep(mimic_urdf):
    """``JAW_OFFSET_IN_GRIPPER_BASE`` must stay inside the pads for the *whole* stroke.

    The pads translate along the approach as the knuckles rotate -- about 15 mm
    between open and closed -- so there is only a band they cover in both poses.
    A grasp point outside it means the object is only between the pads for part
    of the closing motion and gets shoved along for the rest, which is what
    happened when the point was set from the fingertip origins instead.
    """
    root = ET.parse(mimic_urdf).getroot()
    base = load_chain("gripper_base", mimic_urdf)
    home = dict.fromkeys(ARM_JOINTS, 0.0)

    def span(targets):
        def q(chain):
            return np.array([home[n] if n in ARM_JOINTS else targets[n] for n in chain.joint_names])

        collision = root.find("link[@name='gripper_left1']/collision")
        origin = np.fromstring(collision.find("origin").attrib["xyz"], sep=" ")
        half = np.fromstring(collision.find("geometry/box").attrib["size"], sep=" ") / 2
        local = np.array([origin + half * (np.array(s) * 2 - 1) for s in np.ndindex(2, 2, 2)])
        chain = load_chain("gripper_left1", mimic_urdf)
        pos, rot = chain.fk(q(chain))
        base_pos, base_rot = base.fk(q(base))
        in_base = (base_rot.T @ (((rot @ local.T).T + pos) - base_pos).T).T
        return in_base[:, 1].min(), in_base[:, 1].max()

    open_lo, open_hi = span(GRIPPER_OPEN)
    closed_lo, closed_hi = span(GRIPPER_CLOSED)
    lo, hi = max(open_lo, closed_lo), min(open_hi, closed_hi)
    assert lo < hi, "the pads share no common band between open and closed"
    assert lo <= JAW_OFFSET_IN_GRIPPER_BASE[1] <= hi, (
        f"grasp point {JAW_OFFSET_IN_GRIPPER_BASE[1]:.4f} outside the swept band [{lo:.4f}, {hi:.4f}]"
    )


# --------------------------------------------------------------------------
# gripper="mimic": what the lift task actually uses
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mimic_urdf():
    return build_mycobot_urdf("mimic", force=True)


@pytest.fixture(scope="module")
def mimic(mimic_urdf):
    return ET.parse(mimic_urdf).getroot()


def test_mimic_mode_keeps_four_gripper_dof(mimic):
    revolute = [j.attrib["name"] for j in mimic.findall("joint") if j.attrib["type"] == "revolute"]
    gripper = [n for n in revolute if "gripper" in n]
    assert len(revolute) == 10
    assert set(gripper) == set(ACTUATED_GRIPPER_JOINTS)


def test_every_surviving_mimic_names_an_adjacent_joint(mimic):
    """The reason this mode exists.

    PhysX articulation mimic joints bind only parent-child adjacent joints: one
    joint's child link must be the other's parent link. A ``<mimic>`` naming a
    sibling imports as a mimic prim with an *empty* reference and silently
    constrains nothing, which is what the vendor's own tags do -- four of their
    five name the driver from a sibling branch.
    """
    parent = {j.attrib["name"]: j.find("parent").attrib["link"] for j in mimic.findall("joint")}
    child = {j.attrib["name"]: j.find("child").attrib["link"] for j in mimic.findall("joint")}
    found = 0
    for joint in mimic.findall("joint"):
        tag = joint.find("mimic")
        if tag is None:
            continue
        found += 1
        name, reference = joint.attrib["name"], tag.attrib["joint"]
        assert reference in parent, f"{name} mimics undefined joint {reference}"
        adjacent = child[reference] == parent[name] or child[name] == parent[reference]
        assert adjacent, f"{name} mimics {reference}, which is not parent-child adjacent"
    assert found == 2, f"expected two mimic couplings, found {found}"


def test_mimic_couplings_preserve_the_vendor_ratio(mimic):
    """Re-targeting must not change the physical relationship.

    The vendor writes ``right1 = +1.0 * driver``; since ``right3 = -1.0 * driver``,
    ``right1 = -1.0 * right3`` is the same constraint expressed against an
    adjacent joint.
    """
    expected = {
        "gripper_left3_to_gripper_left1": ("gripper_controller", -1.0),
        "gripper_right3_to_gripper_right1": ("gripper_base_to_gripper_right3", -1.0),
    }
    for name, (reference, multiplier) in expected.items():
        tag = mimic.find(f"joint[@name='{name}']/mimic")
        assert tag is not None, f"{name} lost its coupling"
        assert tag.attrib["joint"] == reference
        assert float(tag.attrib["multiplier"]) == pytest.approx(multiplier)


def test_commanded_joints_carry_no_mimic(mimic):
    """A commanded joint with a <mimic> would fight its own action term."""
    for name in MIMIC_COMMANDED_JOINTS:
        assert mimic.find(f"joint[@name='{name}']/mimic") is None


def test_commanded_joints_are_mirrored():
    driver, mirror = MIMIC_COMMANDED_JOINTS
    assert GRIPPER_OPEN[mirror] == pytest.approx(-GRIPPER_OPEN[driver])
    assert GRIPPER_CLOSED[mirror] == pytest.approx(-GRIPPER_CLOSED[driver])


def test_jaw_travel_matches_the_published_range(mimic_urdf):
    """Elephant Robotics quote a **20-45 mm** gripping range for this gripper.

    The pads are sized to reproduce it: 19.4 mm closed, 45.6 mm open. This is the
    one number in the gripper model with a datasheet behind it, so it is the one
    worth pinning.
    """
    assert _pad_gap(mimic_urdf, GRIPPER_CLOSED) == pytest.approx(0.0194, abs=0.002)
    assert _pad_gap(mimic_urdf, GRIPPER_OPEN) == pytest.approx(0.0456, abs=0.002)


def test_fingertips_are_boxes_in_both_roles(mimic):
    """Collision must equal visual on the pads, or the object is pinched by
    geometry nobody can see. That mismatch is what made an earlier version look
    like the cube was being crushed into the gripper housing."""
    for name in ("gripper_left1", "gripper_right1"):
        link = mimic.find(f"link[@name='{name}']")
        for role in ("visual", "collision"):
            elements = link.findall(role)
            assert len(elements) == 1, f"{name} has {len(elements)} {role} elements"
            assert elements[0].find("geometry/box") is not None, f"{name} {role} is not a box"
        vis, col = link.find("visual"), link.find("collision")
        assert vis.find("origin").attrib["xyz"] == col.find("origin").attrib["xyz"]
        assert vis.find("geometry/box").attrib["size"] == col.find("geometry/box").attrib["size"]


def test_grasp_point_clears_the_gripper_body(mimic_urdf):
    """A cube at the grasp point must not intersect ``gripper_base``.

    The first version put the grasp point at the midpoint of the fingertip
    origins, 16.8 mm beyond the body -- a 32 mm cube there overlapped the housing
    by 13 mm and was wedged rather than gripped.
    """
    root = ET.parse(mimic_urdf).getroot()
    origin = np.fromstring(root.find("link[@name='gripper_base']/visual/origin").attrib["xyz"], sep=" ")
    # the body's own extent along the approach axis, from its mesh origin
    body_reach = abs(origin[1]) + 0.014
    assert JAW_OFFSET_IN_GRIPPER_BASE[1] - CUBE_SIZE / 2 > body_reach, "grasp point is inside the gripper body"
