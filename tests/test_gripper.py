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
    PARALLEL_CLOSED,
    PARALLEL_FINGER_JOINTS,
    PARALLEL_OPEN,
    PARALLEL_TRAVEL,
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


MIMIC_PADS = ("gripper_left1", "gripper_right1")
"""Pad links in ``gripper="mimic"`` mode -- the vendor fingertips, re-boxed."""

PARALLEL_PADS = ("gripper_left3", "gripper_right3")
"""Pad links in ``gripper="parallel"`` mode -- the two sliding fingers."""


def _pad_corners(urdf, targets, pad):
    """The eight corners of one grip pad, expressed in ``gripper_base`` [m]."""
    root = ET.parse(urdf).getroot()
    base = load_chain("gripper_base", urdf)
    home = dict.fromkeys(ARM_JOINTS, 0.0)

    def q(chain):
        return np.array([home[n] if n in ARM_JOINTS else targets[n] for n in chain.joint_names])

    collision = root.find(f"link[@name='{pad}']/collision")
    origin = np.fromstring(collision.find("origin").attrib["xyz"], sep=" ")
    half = np.fromstring(collision.find("geometry/box").attrib["size"], sep=" ") / 2
    local = np.array([origin + half * (np.array(s) * 2 - 1) for s in np.ndindex(2, 2, 2)])
    chain = load_chain(pad, urdf)
    pos, rot = chain.fk(q(chain))
    base_pos, base_rot = base.fk(q(base))
    return (base_rot.T @ (((rot @ local.T).T + pos) - base_pos).T).T


def _pad_gap(urdf, targets, pads=MIMIC_PADS):
    """Free space between the two grip pads, across the closing axis [m].

    The quantity that decides what fits, and *not* the distance between the
    fingertip origins: the fingers splay as well as close, so the 3-D origin
    separation overstates the closing-axis projection by about 12 mm. Sizing the
    pads off the 3-D number once left the jaw 12 mm too tight.
    """
    first, second = (_pad_corners(urdf, targets, pad) for pad in pads)
    lo, hi = (first, second) if first[:, 0].mean() < second[:, 0].mean() else (second, first)
    return float(hi[:, 0].min() - lo[:, 0].max())


def _approach_span(urdf, targets, pads=MIMIC_PADS):
    """Extent of both pads along the approach axis, in ``gripper_base`` [m]."""
    corners = np.vstack([_pad_corners(urdf, targets, pad) for pad in pads])
    return float(corners[:, 1].min()), float(corners[:, 1].max())


def test_closing_actually_narrows_the_jaw(actuated):
    """Sign check. If the driver's closed value had the wrong sign the jaw would
    open wider on 'close' and every grasp would silently fail."""
    opened = _jaw_separation(actuated, GRIPPER_OPEN)
    closed = _jaw_separation(actuated, GRIPPER_CLOSED)
    assert closed < opened
    assert opened - closed > 0.02, "jaw travel under 20 mm leaves nothing to grip with"


# --------------------------------------------------------------------------
# gripper="mimic": the rotating linkage, kept buildable but not shipped
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


# --------------------------------------------------------------------------
# gripper="parallel": what the lift task actually ships
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def parallel_urdf():
    return build_mycobot_urdf("parallel", force=True)


@pytest.fixture(scope="module")
def parallel(parallel_urdf):
    return ET.parse(parallel_urdf).getroot()


def test_parallel_fingers_slide_by_the_declared_travel(parallel):
    """Both fingers are prismatic, with the limits :data:`PARALLEL_TRAVEL` declares.

    Revolute here would be the vendor's rotating linkage back again, and that is
    the defect that made the task unlearnable for four successive runs.
    """
    for name, (lower, upper) in PARALLEL_TRAVEL.items():
        joint = parallel.find(f"joint[@name='{name}']")
        assert joint is not None, f"{name} is missing from the parallel build"
        assert joint.attrib["type"] == "prismatic", f"{name} is {joint.attrib['type']}, not prismatic"
        limit = joint.find("limit")
        assert float(limit.attrib["lower"]) == pytest.approx(lower)
        assert float(limit.attrib["upper"]) == pytest.approx(upper)


def test_parallel_build_carries_no_mimic(parallel):
    """There is no ``<mimic>`` here and none is possible.

    PhysX binds mimic joints only between parent-child *adjacent* joints, and the
    two sliding fingers are siblings off ``gripper_base``. Both are commanded
    directly instead, so the coupling is exact with no follower to drift. The
    consequence for the asset config: ``convert_mimic_joints_to_normal_joints``
    is inert on this build -- it has nothing to convert.
    """
    assert parallel.findall("joint/mimic") == []


def test_parallel_fingers_are_mirrored():
    driver, mirror = PARALLEL_FINGER_JOINTS
    assert PARALLEL_OPEN[mirror] == pytest.approx(-PARALLEL_OPEN[driver])
    assert PARALLEL_CLOSED[mirror] == pytest.approx(-PARALLEL_CLOSED[driver])


def test_parallel_jaw_travel_matches_the_published_range(parallel_urdf):
    """Elephant Robotics quote a **20-45 mm** gripping range for this gripper.

    The pads are sized to reproduce it: 19.4 mm closed, 45.6 mm open. This is the
    one number in the gripper model with a datasheet behind it, so it is pinned
    against the build the lift task actually loads.
    """
    assert _pad_gap(parallel_urdf, PARALLEL_CLOSED, PARALLEL_PADS) == pytest.approx(0.0194, abs=0.002)
    assert _pad_gap(parallel_urdf, PARALLEL_OPEN, PARALLEL_PADS) == pytest.approx(0.0456, abs=0.002)


def test_parallel_pads_do_not_sweep_along_the_approach(parallel_urdf):
    """The regression test for the defect that cost four training runs.

    A rotating linkage holds its pads *parallel* but not *stationary*: they rode
    the knuckle's arc 15.2 mm forward between open and closed and shoved the cube
    14-21 mm out of the jaw, invariant to where the pads were mounted. Sliding
    fingers have no arc, so the approach-axis extent must be identical in both
    poses -- not merely close.
    """
    open_lo, open_hi = _approach_span(parallel_urdf, PARALLEL_OPEN, PARALLEL_PADS)
    closed_lo, closed_hi = _approach_span(parallel_urdf, PARALLEL_CLOSED, PARALLEL_PADS)
    assert open_lo == pytest.approx(closed_lo, abs=1e-9), "pads slid along the approach while closing"
    assert open_hi == pytest.approx(closed_hi, abs=1e-9), "pads slid along the approach while closing"


def test_parallel_cube_fits_between_the_pads(parallel_urdf):
    """The cube must pass between the open pads and be squeezed by the closed ones."""
    open_gap = _pad_gap(parallel_urdf, PARALLEL_OPEN, PARALLEL_PADS)
    closed_gap = _pad_gap(parallel_urdf, PARALLEL_CLOSED, PARALLEL_PADS)
    assert closed_gap < CUBE_SIZE < open_gap, f"cube {CUBE_SIZE} outside jaw range [{closed_gap}, {open_gap}]"


def test_parallel_grasp_point_lies_between_the_pads(parallel_urdf):
    """``JAW_OFFSET_IN_GRIPPER_BASE`` is where the task drives the cube to sit.

    Because the pads do not sweep, the band is simply their extent -- but the
    grasp point still has to be inside it, and it was not in the first version:
    taken from the fingertip origins it landed 16.8 mm out, inside a housing that
    reaches 13.9 mm, and the cube was wedged rather than pinched.
    """
    lo, hi = _approach_span(parallel_urdf, PARALLEL_CLOSED, PARALLEL_PADS)
    assert lo <= JAW_OFFSET_IN_GRIPPER_BASE[1] <= hi, (
        f"grasp point {JAW_OFFSET_IN_GRIPPER_BASE[1]:.4f} outside the pads [{lo:.4f}, {hi:.4f}]"
    )


def test_parallel_grasp_point_clears_the_gripper_body(parallel):
    """A cube at the grasp point must not intersect ``gripper_base``."""
    origin = np.fromstring(parallel.find("link[@name='gripper_base']/visual/origin").attrib["xyz"], sep=" ")
    # the body's own extent along the approach axis, from its mesh origin
    body_reach = abs(origin[1]) + 0.014
    assert JAW_OFFSET_IN_GRIPPER_BASE[1] - CUBE_SIZE / 2 > body_reach, "grasp point is inside the gripper body"


def test_parallel_pads_are_boxes_in_both_roles(parallel):
    """Collision must equal visual on the pads, or the object is pinched by
    geometry nobody can see. That mismatch is what made an earlier version look
    like the cube was being crushed into the gripper housing."""
    for name in PARALLEL_PADS:
        link = parallel.find(f"link[@name='{name}']")
        for role in ("visual", "collision"):
            elements = link.findall(role)
            assert len(elements) == 1, f"{name} has {len(elements)} {role} elements"
            assert elements[0].find("geometry/box") is not None, f"{name} {role} is not a box"
        vis, col = link.find("visual"), link.find("collision")
        assert vis.find("origin").attrib["xyz"] == col.find("origin").attrib["xyz"]
        assert vis.find("geometry/box").attrib["size"] == col.find("geometry/box").attrib["size"]


def test_parallel_robot_carries_exactly_two_colliders(parallel):
    """The lift config's defining simplification, pinned so it cannot drift back.

    ``collision_from_visuals`` is off for this asset: with it on, the arm carries
    convex hulls off its decorative visual meshes, and at the low poses a grasp
    needs they press into the ground -- 0.19 rad of joint error while merely
    standing still, roughly 30 mm at the jaw, larger than the cube. The cost is
    that the arm's links can pass through the ground and through each other.
    """
    colliders = {link.attrib["name"] for link in parallel.findall("link") if link.find("collision") is not None}
    assert colliders == set(PARALLEL_PADS)
