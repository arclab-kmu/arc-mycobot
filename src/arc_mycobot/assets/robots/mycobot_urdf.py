"""Repair the vendor myCobot 280 JetsonNano URDF into one Isaac Lab can import.

Why this module exists
----------------------
``mycobot_ros2`` ships the JetsonNano-with-adaptive-gripper description as

    mycobot_description/urdf/mycobot_280_jn/mycobot_280_jn_adaptive_gripper.urdf

and that file cannot be handed to Isaac Lab's
:class:`~isaaclab.sim.converters.UrdfConverterCfg` as it stands. It carries four
independent defects, every one of which was confirmed against the checkout
rather than assumed:

1. **It is not well-formed XML.** The ``joint2_to_joint1`` limit is written
   ``lower = "-2.932"1 upper = "2.9321"`` -- a digit outside the closing quote.
   ``xml.etree`` stops at *line 89, column 45*; urdfdom stops in the same place
   with a less readable message. Nothing else can happen until this is repaired
   at the text level, which is why :func:`_repair_text` runs before any parse.

2. **Every joint declares ``velocity="0"``.** All thirteen of them. A zero
   velocity limit is not "unlimited" -- the importer writes it straight through
   to the PhysX joint, and the articulation then cannot move. Replaced with
   :data:`ARM_VELOCITY_LIMIT` / :data:`GRIPPER_VELOCITY_LIMIT`.

3. **There is not a single ``<inertial>`` element,** and the six *arm* links
   have no ``<collision>`` either (the seven gripper links do). Links with
   neither are massless, and ``UrdfConverterCfg.link_density`` defaults to
   ``0.0``, so the import would produce a zero-mass articulation. Handled in two
   places: this module writes explicit ``<inertial>`` blocks (see
   :data:`_LINK_MASSES` for where the numbers come from), and
   :mod:`.mycobot_280` pairs that with ``collision_from_visuals=True``.

4. **The gripper is a five-joint ``<mimic>`` cluster.** ``gripper_controller``
   drives four followers at multiplier +-1.0, and a fifth mimics through another
   link. What happens to it depends on the ``gripper`` mode: welded for the reach
   task, which never grasps (:data:`GRIPPER_FIXED_AT`); replaced by a sliding
   parallel jaw for the lift task (:data:`PARALLEL_FINGER_JOINTS`). Two further
   modes, ``"actuated"`` and ``"mimic"``, are kept as the measured steps between
   those two -- see :data:`PARALLEL_FINGER_JOINTS` for why neither survived.

It also carries an ``<?xml version="1.1"?>`` declaration and one stray
``<xacro:property>`` in a file with a ``.urdf`` extension; both are normalized
away.

The vendor checkout is **never modified**. The repaired URDF is written to
``<repo>/generated/`` (git-ignored) and rebuilt whenever the source is newer.

Mesh references
---------------
Meshes come through as ``package://mycobot_description/urdf/...`` URIs. Isaac
Lab runs urdfdom, which resolves ``package://`` only under an ament/ROS
environment -- and this repository deliberately needs no ROS. So the prefix is
rewritten to an absolute path into the local checkout and every resulting path
is checked to exist before the file is written.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal

__all__ = [
    "ACTUATED_GRIPPER_JOINTS",
    "ARM_JOINTS",
    "MIMIC_COMMANDED_JOINTS",
    "PARALLEL_CLOSED",
    "PARALLEL_FINGER_JOINTS",
    "PARALLEL_OPEN",
    "PARALLEL_TRAVEL",
    "ARM_VELOCITY_LIMIT",
    "FLANGE_BODY",
    "GENERATED_DIR",
    "GRIPPER_FIXED_AT",
    "GRIPPER_OPEN",
    "GRIPPER_CLOSED",
    "GRIPPER_VELOCITY_LIMIT",
    "HOME_POSE",
    "MYCOBOT_ROS2_DIR",
    "ROOT_LINK",
    "SOURCE_URDF",
    "build_mycobot_urdf",
    "validate_references",
]

_REPO_ROOT = Path(__file__).resolve().parents[4]
"""Repository root (``.../arc-mycobot``), i.e. four levels above this file."""

MYCOBOT_ROS2_DIR = Path(os.environ.get("MYCOBOT_ROS2_DIR", _REPO_ROOT.parent / "mycobot_ros2")).resolve()
"""The sibling ``mycobot_ros2`` checkout. Override with ``$MYCOBOT_ROS2_DIR``."""

SOURCE_URDF = (
    MYCOBOT_ROS2_DIR / "mycobot_description" / "urdf" / "mycobot_280_jn" / "mycobot_280_jn_adaptive_gripper.urdf"
)
"""The vendor description: myCobot 280 JetsonNano + adaptive gripper."""

GENERATED_DIR = _REPO_ROOT / "generated"
"""Where the repaired URDF lands. Git-ignored: it is derived, not authored."""

_PACKAGE_PREFIX = "package://mycobot_description/"
_MESH_ROOT = MYCOBOT_ROS2_DIR / "mycobot_description"
_XACRO_NS = "http://www.ros.org/wiki/xacro"

ARM_VELOCITY_LIMIT = 2.0944
"""Per-joint velocity limit written over the vendor's ``velocity="0"`` [rad/s].

120 deg/s, the maximum joint speed on Elephant Robotics' myCobot 280 spec sheet.

.. warning::
    ``TODO(unverified)``: the spec sheet quotes one figure for the whole arm and
    does not break it down per joint, and it was not measured on this unit. It is
    used here as a *ceiling* -- the PD gains in :mod:`.mycobot_280`, not this
    number, shape the actual response -- so it being loose costs little. Replace
    it with a per-joint measurement before any sim-to-real transfer.
"""

GRIPPER_VELOCITY_LIMIT = 1.0
"""Velocity limit for the welded gripper joints [rad/s]. Inert -- see :data:`GRIPPER_FIXED_AT`."""

GRIPPER_FIXED_AT = 0.0
"""Angle the gripper is welded at in ``gripper="welded"`` mode [rad]: neutral, open.

The reach task never closes the fingers, so welding them puts the gripper's mass
and geometry where they belong and costs nothing. 0.0 rad is inside every joint's
limits (the followers span -0.8..0.8, the driver -0.74..0.15), so the weld
introduces no pre-load. The lift task uses ``gripper="parallel"`` instead.
"""

# -- the gripper linkage ------------------------------------------------------
# Per side the vendor builds a four-bar:
#
#     gripper_base --+-- gripper_left3  (gripper_controller, the DRIVER)
#                    |     \__ gripper_left1   (fingertip, mimic x-1.0)
#                    +-- gripper_left2  (parallel link, mimic x+1.0)
#
# and its mirror on the right. The fingertip's x-1.0 coupling is what keeps the
# pad parallel to itself as the knuckle swings -- that is the whole point of the
# mechanism, and it is why the fingertips stay actuated rather than welded.
#
# `gripper_left2` / `gripper_right2` are the *second* bar of each four-bar. A
# real four-bar closes the loop back onto the fingertip; URDF is a tree and
# cannot express that, so in the vendor file those two links dangle -- they are
# driven at one end and attached to nothing at the other. They contribute
# geometry and mass and no kinematics, so they are welded in both modes.
_GRIPPER_ALWAYS_WELDED = (
    "gripper_base_to_gripper_left2",
    "gripper_base_to_gripper_right2",
)

# -- how the mimic mode wires the linkage --------------------------------------
# PhysX articulation mimic joints only bind a joint to a **parent-child adjacent**
# joint -- one joint's child link must be the other's parent link. Measured, not
# assumed: importing the vendor's five <mimic> tags verbatim produced five
# PhysxMimicJointAPI prims of which exactly one, `gripper_left3_to_gripper_left1`,
# got a reference target. That one is the only parent-child pair; the other four
# name `gripper_controller` from a *sibling* branch. The four silently ended up
# unconstrained, and the right finger then swung to +1.10 rad against a 0.7 rad
# limit.
#
# So the couplings are re-expressed against adjacent joints where that is
# possible, and carried by the action term where it is not:
#
#   gripper_controller              driver, commanded
#   gripper_left3_to_gripper_left1  mimic of the driver      (adjacent: shares gripper_left3)
#   gripper_base_to_gripper_right3  commanded, mirrored      (sibling of the driver)
#   gripper_right3_to_gripper_right1 mimic of right3         (adjacent: shares gripper_right3)
#
# The vendor writes right1 as x+1.0 of the *driver*; since right3 is x-1.0 of the
# driver, x-1.0 of right3 is the identical constraint, and it is adjacent.
# The fingertips -- the pads that touch the object -- are therefore held parallel
# by a real hardware-style constraint rather than by an open-loop command.
_MIMIC_DRIVER = "gripper_controller"
_MIMIC_RETARGET = {
    "gripper_left3_to_gripper_left1": ("gripper_controller", -1.0),
    "gripper_right3_to_gripper_right1": ("gripper_base_to_gripper_right3", -1.0),
    "gripper_base_to_gripper_right3": None,
}

# -- the parallel jaw ----------------------------------------------------------
# `gripper="parallel"` replaces the vendor's knuckle linkage with two fingers
# that **slide**, which is what the datasheet actually advertises: Elephant
# Robotics describe this gripper as doing "parallel gripping" over a 20-45 mm
# range.
#
# Why the linkage had to go. In the rotating model the pads do stay parallel --
# that is what the fingertip's x-1.0 coupling buys -- but their *position* still
# rides the knuckle's arc, and measured in the gripper's own frame that arc
# carries them **15.2 mm forward along the approach** between open and closed.
# The sweep is a property of the mechanism, not of where the pads are mounted:
# moving the pad 10, 24 or 34 mm out along the finger leaves it at 15.2 mm every
# time. Its consequence is a vice that shoves what it is closing on -- a scripted
# grasp displaced the cube 14-21 mm -- and it puts the usable object size in a
# narrow band (34-42 mm here) where contact happens early enough that little
# sweep is left. Objects smaller than that are pushed out before the pads meet.
#
# Sliding fingers have no arc, so the sweep is zero by construction and any
# object inside the jaw range is gripped the same way.
#
# What is given up: the vendor's four-bar as a *shape*. What is kept: the flange,
# the mount, the gripper body, the jaw range, and the two-finger behaviour the
# hardware is sold for.
PARALLEL_FINGER_JOINTS = ("gripper_controller", "gripper_base_to_gripper_right3")
"""The two sliding fingers. Both are commanded; the pair is mirrored.

There is no ``<mimic>`` here and none is possible: PhysX binds mimic joints only
between parent-child *adjacent* joints (see :data:`_MIMIC_RETARGET`), and these
two are siblings off ``gripper_base``. It costs nothing -- a pair of
position-commanded joints driven from one binary action is already exactly
coupled, with no follower to drift.
"""

PARALLEL_TRAVEL = {
    "gripper_controller": (0.0, 0.0131),
    "gripper_base_to_gripper_right3": (-0.0131, 0.0),
}
"""Slide range per finger [m]. 13.1 mm each, 26.2 mm of jaw travel.

Chosen so the pads reproduce the rotating model's measured range exactly --
45.6 mm open, 19.4 mm closed -- which brackets the published 20-45 mm.
"""

PARALLEL_OPEN = {"gripper_controller": 0.0, "gripper_base_to_gripper_right3": 0.0}
PARALLEL_CLOSED = {"gripper_controller": 0.0131, "gripper_base_to_gripper_right3": -0.0131}
"""Finger targets for the open and closed jaw [m]."""

MIMIC_COMMANDED_JOINTS = ("gripper_controller", "gripper_base_to_gripper_right3")
"""The two joints a policy commands in ``gripper="mimic"`` mode.

Mirrored (the right knuckle is x-1.0 of the driver). Their fingertips follow by
PhysX mimic constraint, so the action space stays one binary command.
"""

ACTUATED_GRIPPER_JOINTS = (
    "gripper_controller",
    "gripper_left3_to_gripper_left1",
    "gripper_base_to_gripper_right3",
    "gripper_right3_to_gripper_right1",
)
"""The four gripper DOFs in ``gripper="actuated"`` mode, driver first.

**Superseded by** ``gripper="mimic"``, which is what the lift task uses. This
mode drops the ``<mimic>`` elements and reproduces the coupling open-loop from
the action term, driving all four joints from one binary command at the vendor's
multipliers. That holds only as long as nothing pushes a finger off its commanded
angle -- under contact it is an approximation, and a measurably worse one: with
these pads a centred cube was displaced 40 mm on closing, where the mimic mode
moves it 0.3 mm.

Kept because it is the fallback if PhysX mimic joints are ever unavailable, and
because the comparison is the reason the mimic mode exists.
"""

# Vendor multipliers relative to `gripper_controller`, read off the <mimic> tags.
_GRIPPER_MULTIPLIER = {
    "gripper_controller": 1.0,
    "gripper_left3_to_gripper_left1": -1.0,
    "gripper_base_to_gripper_right3": -1.0,
    "gripper_right3_to_gripper_right1": 1.0,
}

_DRIVER_OPEN = 0.15
_DRIVER_CLOSED = -0.5
"""Driver angles for the open and closed pose [rad].

+0.15 is the driver's own upper limit. -0.5 rather than its -0.74 lower limit
because the fingertips are coupled at x-1.0 and their range is only +-0.5: at
-0.74 the commanded fingertip angle would sit 0.24 rad outside its joint limit,
and the drive would fight the limit for the whole grasp.

Which of the two actually closes the jaw is measured, not assumed -- see
``scripts/tools/gripper_check.py``.
"""

# -- grip pads ----------------------------------------------------------------
# In `gripper="actuated"` mode the two fingertips get a **box** collider each,
# replacing the vendor's collision mesh, and every other gripper link loses its
# collider entirely.
#
# Why. The vendor's fingertip meshes are decorative and wildly asymmetric --
# `gripper_left1` spans 59 mm along its long axis, `gripper_right1` only 13 mm --
# and their convex hulls sit at very different distances from the jaw centre:
# measured at a side-grasp pose, the left pad's inner face is 13.6 mm from the
# centreline and the right's is 37.9 mm. A "jaw" like that does not close on
# anything symmetrically; a cube placed at the centre is struck by one pad and
# missed by the other, which is exactly what a scripted grasp did before this
# existed.
#
# The two fingertip *link origins* are symmetric (41.1 mm either side of the jaw
# centre when open), so pads defined in the link frames are symmetric by
# construction. Local axes, measured with forward kinematics:
#   gripper_left1  : inward = +x, approach = +y, world-up = -z
#   gripper_right1 : inward = -x, approach = +y, world-up = -z
_PAD_INSET = 0.0123
"""Distance from the fingertip link origin to the pad's gripping face [m].

Sets the jaw range. The number that matters is the fingertip separation
**projected onto the closing axis**, not the 3D distance between the origins:
the fingers splay as well as close, so the two differ. Measured in the
``gripper_base`` frame, the origins are 70.2 mm apart across the jaw when open
and 44.0 mm when closed (the 3D distances are 82.2 and 56.0 -- using those was an
error that made the jaw 12 mm too tight). So:

    open   gap = 70.2 - 2 * 12.3 = 45.6 mm
    closed gap = 44.0 - 2 * 12.3 = 19.4 mm

which is Elephant Robotics' published **20-45 mm gripping range** to within a
millimetre. The jaw is sized to the datasheet, not to the vendor's mesh.
"""

_PAD_SIZE = (0.006, 0.026, 0.022)
"""Pad box in fingertip-link axes: thickness across the jaw, length along the
approach direction, height. About 26 x 22 mm of face, matching the real pad."""

_PAD_FORWARD = 0.024
"""How far the pad sits *ahead* of the fingertip origin, along the approach [m].

Without this the pads straddle a point only 16.8 mm beyond the gripper body, and
a 32 mm object centred there intersects ``gripper_base``. Measured: the base
occupies y in [-13.9, 13.9] mm in its own frame, so a 32 mm cube at the bare
fingertip midpoint overlaps it by about 13 mm. That is exactly the failure that
made an earlier version look like the cube was being crushed into the gripper
body rather than pinched -- it was. 24 mm puts the pocket clear of the body.
"""

GRIPPER_OPEN = {name: _DRIVER_OPEN * m for name, m in _GRIPPER_MULTIPLIER.items()}
"""Per-joint targets for the open jaw [rad]. Feed straight to the action term."""

GRIPPER_CLOSED = {name: _DRIVER_CLOSED * m for name, m in _GRIPPER_MULTIPLIER.items()}
"""Per-joint targets for the closed jaw [rad]. Feed straight to the action term."""

ARM_JOINTS = (
    "joint2_to_joint1",
    "joint3_to_joint2",
    "joint4_to_joint3",
    "joint5_to_joint4",
    "joint6_to_joint5",
    "joint6output_to_joint6",
)
"""The six actuated revolute joints, in URDF (base-to-flange) order.

This order is the observation and action layout of the trained policy, and it is
also the order Elephant Robotics' ``pymycobot`` API returns angles in. Listed
rather than matched by regex so that the ordering is stated here instead of
being whatever Isaac Lab's internal sort produces.
"""

ROOT_LINK = "joint1"
"""The articulation root, and the frame every goal position is expressed in.

The vendor names its *links* ``joint1`` .. ``joint6``. That is confusing and it
is theirs: renaming here would put this repository and the ``mycobot_ros2``
checkout into permanent disagreement. Links are ``jointN``; joints are
``jointN_to_jointM``.
"""

FLANGE_BODY = "joint6_flange"
"""The body the reach task tracks: the tool flange. Not a tool centre point.

The adaptive gripper's links are welded (see :data:`GRIPPER_FIXED_AT`) and
merged into this body, so the flange carries the gripper's mass but the tracked
*point* is the flange origin -- about 34 mm behind where the gripper body begins
and further still from where its fingers close. A TCP would be added as an
offset here and in the goal box together, since both come from the same forward
kinematics and must move together.
"""

HOME_POSE = {
    "joint2_to_joint1": 0.0,
    "joint3_to_joint2": -0.5,
    "joint4_to_joint3": -1.0,
    "joint5_to_joint4": -0.6,
    "joint6_to_joint5": 0.0,
    "joint6output_to_joint6": 0.0,
}
"""Reset posture [rad]: elbow bent, flange forward and clear of the mount plane.

Chosen with the forward kinematics in :mod:`arc_mycobot.kinematics.urdf_fk`, not
by eye. It puts the flange at ``(0.189, -0.065, 0.184)`` in the root frame, which
is **inside** the goal box, so every episode starts somewhere the goals also
live rather than opening with a gross traverse across the workspace before any
tracking begins. ``tests/test_reach_geometry.py`` checks that it stays inside.

The -65 mm of ``y`` is not a mistake: at zero wrist angles the flange sits off
the arm's centreline by the wrist's own offset.

It lives here, beside the joint names, rather than in :mod:`.mycobot_280`,
because it is a statement about the URDF's joints and a CPU-only test has to be
able to read it without starting Omniverse.
"""

_GRIPPER_JOINTS = (
    "gripper_controller",
    "gripper_base_to_gripper_left2",
    "gripper_left3_to_gripper_left1",
    "gripper_base_to_gripper_right3",
    "gripper_base_to_gripper_right2",
    "gripper_right3_to_gripper_right1",
)
"""The mimic cluster: one driver plus five followers. All welded."""

# -- link masses --------------------------------------------------------------
# The vendor URDF has no <inertial> at all, so these are *authored here*, not
# imported. Sources and method:
#
#   * The myCobot 280 weighs 850 g in total and its payload is 250 g
#     (Elephant Robotics spec sheet); the adaptive gripper adds about 110 g.
#   * That 850 g is distributed across the six arm links by the usual taper for
#     a serial arm -- heavier near the base where the larger servos and the
#     Jetson carrier sit, lighter toward the wrist -- normalized to sum to 850 g.
#   * The gripper's 110 g is split 70 g on the body and 8 g on each of the five
#     moving fingers. Because the cluster is welded (see GRIPPER_FIXED_AT) and
#     `merge_fixed_joints` folds them into the flange, only the *total* and the
#     combined centre of mass actually reach PhysX.
#
# TODO(unverified): the split between links is derived from the published total,
# not weighed. It is the right order of magnitude and it sums to the right
# number, which is what a position-reach task with implicit PD actuators needs;
# it is not good enough for torque-level sim-to-real.
_LINK_MASSES = {
    "joint1": 0.240,
    "joint2": 0.190,
    "joint3": 0.150,
    "joint4": 0.110,
    "joint5": 0.090,
    "joint6": 0.050,
    "joint6_flange": 0.020,
    "gripper_base": 0.070,
    "gripper_left1": 0.008,
    "gripper_left2": 0.008,
    "gripper_left3": 0.008,
    "gripper_right1": 0.008,
    "gripper_right2": 0.008,
    "gripper_right3": 0.008,
}

_INERTIA_RADIUS = 0.03
"""Radius of gyration used to turn each mass into an inertia tensor [m].

A solid-sphere approximation, ``I = 2/5 * m * r^2``, isotropic. 30 mm is roughly
the myCobot's link cross-section. An isotropic tensor cannot be wrong in the way
a wrongly-*oriented* anisotropic one can, it is unconditionally positive-definite
and it satisfies the triangle inequality, so PhysX accepts it for every link.

TODO(unverified): replace with tensors computed from the .dae meshes (trimesh
can do it) if this project ever grows past position reach.
"""


def _repair_text(raw: str) -> str:
    """Fix the defects that must be repaired *before* the file can be parsed.

    Args:
        raw: The vendor URDF, verbatim.

    Returns:
        Well-formed XML text, still carrying the semantic defects that
        :func:`build_mycobot_urdf` repairs on the parsed tree.

    Raises:
        ValueError: The malformed limit this function knows how to repair is no
            longer present *and* the text still fails to parse -- i.e. the vendor
            file changed and this repair needs revisiting.
    """
    # Defect 1: a digit outside the closing quote in joint2_to_joint1's limit.
    # Matched by shape rather than by literal text so that a vendor fix to the
    # *value* does not silently turn this into a no-op that fails later.
    repaired, n_limit = re.subn(r'lower\s*=\s*"(-?[\d.]+)"(\d)\s', r'lower = "\1\2" ', raw)

    # An XML 1.1 declaration: expat implements 1.0 only. Nothing in this file
    # uses an XML 1.1 feature, so the declaration is simply wrong.
    repaired = repaired.replace('<?xml version="1.1"?>', '<?xml version="1.0"?>')

    try:
        ET.fromstring(repaired)
    except ET.ParseError as error:
        raise ValueError(
            f"{SOURCE_URDF} is still not well-formed after {n_limit} known text repair(s): {error}.\n"
            "The vendor file has changed in a way arc_mycobot.assets.robots.mycobot_urdf does not know about."
        ) from error
    return repaired


def _rewrite_mesh_paths(xml_text: str) -> str:
    """Replace ``package://`` mesh URIs with absolute paths into the local checkout.

    Raises:
        ValueError: A ``package://`` URI survived the rewrite.
        FileNotFoundError: A rewritten path does not point at a real file.
    """
    replaced = xml_text.replace(_PACKAGE_PREFIX, f"{_MESH_ROOT}/")
    leftover = re.findall(r'filename="(package://[^"]*)"', replaced)
    if leftover:
        raise ValueError(f"unresolved package:// mesh URIs: {sorted(set(leftover))}")
    missing = [ref for ref in re.findall(r'filename="([^"]*)"', replaced) if not Path(ref).is_file()]
    if missing:
        raise FileNotFoundError(f"URDF references {len(missing)} missing mesh file(s): {missing[:5]}")
    return replaced


def _add_inertial(link: ET.Element, mass: float) -> None:
    """Give ``link`` an ``<inertial>`` block. See :data:`_LINK_MASSES`."""
    inertia = round(0.4 * mass * _INERTIA_RADIUS**2, 9)
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": f"{mass:.6f}"})
    ET.SubElement(
        inertial,
        "inertia",
        {"ixx": f"{inertia}", "ixy": "0", "ixz": "0", "iyy": f"{inertia}", "iyz": "0", "izz": f"{inertia}"},
    )


def _fit_grip_pads(robot: ET.Element) -> None:
    """Replace the two fingertips -- **visual and collision** -- with box pads.

    Why the vendor geometry is not used, in the order the problems were found:

    * the two fingertip meshes are not mirror images. ``gripper_left1`` spans
      59 mm along its long axis and ``gripper_right1`` 13 mm, and expressed in
      the gripper's own frame they *overlap in the closing axis* at both the
      open and the closed pose. Two jaw faces cannot overlap; whatever those
      meshes are, they are not a matched pair of pads.
    * their ``<collision>`` is the same mesh as their ``<visual>``, so there is
      no separate, cleaner collision model to fall back on.
    * a symmetric collision box bolted onto asymmetric visuals is worse than
      either: the object is then pinched by geometry the viewer cannot see, and
      visibly passes through geometry that does not collide.

    So the fingertips become boxes in both roles, sized to the published 20-45 mm
    gripping range (see :data:`_PAD_INSET`), and every other gripper link keeps
    its visual and loses its collider. What you see is what grips.

    The arm, the flange and the gripper body are untouched.
    """
    inward = {"gripper_left1": 1.0, "gripper_right1": -1.0}
    for link in robot.findall("link"):
        name = link.attrib["name"]
        if not name.startswith("gripper"):
            continue
        for collision in link.findall("collision"):
            link.remove(collision)
        if name not in inward:
            continue
        for visual in link.findall("visual"):
            link.remove(visual)
        offset = f"{inward[name] * (_PAD_INSET + _PAD_SIZE[0] / 2):.6f} {_PAD_FORWARD:.6f} 0"
        size = " ".join(f"{v:.6f}" for v in _PAD_SIZE)
        for role in ("visual", "collision"):
            element = ET.SubElement(link, role)
            ET.SubElement(element, "origin", {"xyz": offset, "rpy": "0 0 0"})
            geometry = ET.SubElement(element, "geometry")
            ET.SubElement(geometry, "box", {"size": size})


_PARALLEL_PAD_X = 0.0138
"""Pad centre offset from its finger link origin, across the jaw [m].

Puts the pad faces 22.8 mm either side of the jaw axis with the fingers home,
i.e. a 45.6 mm opening, and 9.7 mm with them fully closed, i.e. 19.4 mm.
"""

_PARALLEL_PAD_Y = 0.0434
"""Pad centre offset along the approach [m], so the pocket lands 48.4 mm ahead of
``gripper_base`` -- clear of the housing, which reaches 13.9 mm."""


def _fit_parallel_pads(robot: ET.Element) -> None:
    """Mount the grip pads on the two sliding fingers; strip every other collider."""
    side = {"gripper_left3": -1.0, "gripper_right3": 1.0}
    for link in robot.findall("link"):
        name = link.attrib["name"]
        if not name.startswith("gripper"):
            continue
        for role in ("visual", "collision"):
            for element in link.findall(role):
                if name in side or name in ("gripper_left1", "gripper_right1") or role == "collision":
                    link.remove(element)
        if name not in side:
            continue
        offset = f"{side[name] * _PARALLEL_PAD_X:.6f} {_PARALLEL_PAD_Y:.6f} 0"
        size = " ".join(f"{v:.6f}" for v in _PAD_SIZE)
        for role in ("visual", "collision"):
            element = ET.SubElement(link, role)
            ET.SubElement(element, "origin", {"xyz": offset, "rpy": "0 0 0"})
            geometry = ET.SubElement(element, "geometry")
            ET.SubElement(geometry, "box", {"size": size})


def validate_references(robot: ET.Element) -> None:
    """Fail loudly on any name the URDF mentions but does not define.

    Isaac Sim's URDF importer resolves parent/child/mimic names in C++ maps and
    reports a missing one as a bare ``IndexError: map::at``, with no indication
    of which name or which element. Catching it here costs one pass over the
    tree and turns that into a readable message.

    Raises:
        ValueError: A dangling reference, or a root link other than ``joint1``.
    """
    links = {link.attrib["name"] for link in robot.findall("link")}
    joints = {joint.attrib["name"] for joint in robot.findall("joint")}
    dangling: list[str] = []
    children: set[str] = set()

    for joint in robot.findall("joint"):
        name = joint.attrib.get("name", "<unnamed>")
        for role in ("parent", "child"):
            element = joint.find(role)
            if element is None:
                dangling.append(f"joint {name!r} has no <{role}>")
                continue
            link = element.attrib.get("link")
            if link not in links:
                dangling.append(f"joint {name!r} {role} link {link!r} is not defined")
            if role == "child" and link is not None:
                children.add(link)
        for mimic in joint.findall("mimic"):
            if mimic.attrib.get("joint") not in joints:
                dangling.append(f"joint {name!r} mimics {mimic.attrib.get('joint')!r}, which is not defined")

    # exactly one link is never a child: the articulation root
    roots = links - children
    if roots != {"joint1"}:
        dangling.append(f"expected 'joint1' to be the only root link, found {sorted(roots)}")

    for link in links:
        if link not in _LINK_MASSES:
            dangling.append(f"link {link!r} has no mass in _LINK_MASSES; the vendor URDF gained a link")

    if dangling:
        raise ValueError("repaired URDF has unresolved references:\n  " + "\n  ".join(dangling))


def build_mycobot_urdf(
    gripper: Literal["welded", "actuated", "mimic", "parallel"] = "welded", force: bool = False
) -> Path:
    """Write (or reuse) the repaired myCobot 280 JN URDF and return its path.

    Args:
        gripper: ``"welded"`` fixes the whole gripper open and leaves the
            articulation with the six arm DOFs -- what the reach task wants.
            ``"actuated"`` keeps the four joints in
            :data:`ACTUATED_GRIPPER_JOINTS` movable so the jaw can close, for
            the lift task. Each mode writes its own file.
        force: Rebuild even when the cached output is newer than the source.

    Returns:
        Path to a plain, well-formed URDF with absolute mesh paths, real
        velocity limits, an ``<inertial>`` on every link and the gripper mimic
        cluster welded at :data:`GRIPPER_FIXED_AT`.

    Raises:
        FileNotFoundError: The ``mycobot_ros2`` checkout or a mesh is missing.
        ValueError: The vendor file changed in a way this module does not handle.
    """
    if not SOURCE_URDF.is_file():
        raise FileNotFoundError(
            f"myCobot URDF not found at {SOURCE_URDF}.\nClone the robot descriptions next to this"
            " repository:\n    git clone --depth 1 https://github.com/elephantrobotics/mycobot_ros2.git"
            f" {MYCOBOT_ROS2_DIR}\nor point $MYCOBOT_ROS2_DIR at an existing checkout."
        )

    if gripper not in ("welded", "actuated", "mimic", "parallel"):
        raise ValueError(f"gripper must be 'welded', 'actuated', 'mimic' or 'parallel', got {gripper!r}")
    suffix = {"welded": "", "actuated": "_actuated", "mimic": "_mimic", "parallel": "_parallel"}[gripper]
    out_path = GENERATED_DIR / f"mycobot_280_jn_adaptive_gripper{suffix}.urdf"
    if not force and out_path.is_file() and out_path.stat().st_mtime >= SOURCE_URDF.stat().st_mtime:
        return out_path

    robot = ET.fromstring(_repair_text(SOURCE_URDF.read_text(encoding="utf-8")))
    robot.attrib["name"] = "mycobot_280_jn_adaptive_gripper"
    # The vendor names the robot "firefighter" and declares the xacro namespace
    # for a single <xacro:property> it never reads. Both are noise to urdfdom.
    robot.attrib.pop(f"{{{_XACRO_NS}}}version", None)
    for stray in [child for child in robot if child.tag.startswith(f"{{{_XACRO_NS}}}")]:
        robot.remove(stray)

    for link in robot.findall("link"):
        name = link.attrib["name"]
        if name not in _LINK_MASSES:
            raise ValueError(f"link {name!r} has no entry in _LINK_MASSES; the vendor URDF gained a link")
        _add_inertial(link, _LINK_MASSES[name])

    for joint in robot.findall("joint"):
        name = joint.attrib["name"]
        limit = joint.find("limit")

        if name in _GRIPPER_JOINTS:
            if gripper == "parallel":
                for mimic in joint.findall("mimic"):
                    joint.remove(mimic)
                if name in PARALLEL_FINGER_JOINTS:
                    joint.attrib["type"] = "prismatic"
                    axis = joint.find("axis")
                    if axis is None:
                        axis = ET.SubElement(joint, "axis")
                    axis.attrib["xyz"] = "1 0 0"
                    lower, upper = PARALLEL_TRAVEL[name]
                    if limit is None:
                        limit = ET.SubElement(joint, "limit")
                    limit.attrib.update(
                        lower=f"{lower}", upper=f"{upper}", effort="30.0", velocity=f"{GRIPPER_VELOCITY_LIMIT}"
                    )
                    continue
                joint.attrib["type"] = "fixed"
                for tag in ("axis", "limit"):
                    element = joint.find(tag)
                    if element is not None:
                        joint.remove(element)
                continue

            if gripper == "mimic":
                if name in _GRIPPER_ALWAYS_WELDED:
                    for mimic in joint.findall("mimic"):
                        joint.remove(mimic)
                    joint.attrib["type"] = "fixed"
                    for tag in ("axis", "limit"):
                        element = joint.find(tag)
                        if element is not None:
                            joint.remove(element)
                    continue
                if limit is not None and float(limit.attrib.get("velocity", 0.0)) == 0.0:
                    limit.attrib["velocity"] = f"{GRIPPER_VELOCITY_LIMIT}"
                retarget = _MIMIC_RETARGET.get(name, "keep")
                if retarget is None or name == _MIMIC_DRIVER:
                    # Commanded by the action term, so any <mimic> it still
                    # carries would bind to a sibling and silently do nothing.
                    for mimic in joint.findall("mimic"):
                        joint.remove(mimic)
                elif retarget != "keep":
                    reference, multiplier = retarget
                    mimic = joint.find("mimic")
                    if mimic is None:
                        mimic = ET.SubElement(joint, "mimic")
                    mimic.attrib.update(joint=reference, multiplier=f"{multiplier}", offset="0")
                continue

            # A <mimic> naming a joint that has since become fixed makes the
            # importer raise a bare IndexError, so these go first in both of the
            # other modes.
            for mimic in joint.findall("mimic"):
                joint.remove(mimic)

            keep_movable = gripper == "actuated" and name not in _GRIPPER_ALWAYS_WELDED
            if keep_movable:
                # Independent revolute joint; the coupling moves into the action
                # term. It still needs a non-zero velocity limit like every other
                # joint in this file.
                if limit is not None and float(limit.attrib.get("velocity", 0.0)) == 0.0:
                    limit.attrib["velocity"] = f"{GRIPPER_VELOCITY_LIMIT}"
                continue

            joint.attrib["type"] = "fixed"
            # urdfdom ignores <axis>/<limit> on a fixed joint, but leaving them
            # invites a future reader to think the joint still moves.
            for tag in ("axis", "limit"):
                element = joint.find(tag)
                if element is not None:
                    joint.remove(element)
            continue

        if limit is None:
            if joint.attrib["type"] != "fixed":
                raise ValueError(f"joint {name!r} is {joint.attrib['type']} but declares no <limit>")
            continue

        # Defect 2: velocity="0" would import as a PhysX joint that cannot move.
        if float(limit.attrib.get("velocity", 0.0)) == 0.0:
            limit.attrib["velocity"] = f"{ARM_VELOCITY_LIMIT}"
        # The vendor writes effort="1000" on a 250 g-payload arm. Left alone
        # here and overridden per-actuator in mycobot_280.py, so that this file
        # stays a faithful repair and the modelling choice lives with the model.

    if gripper in ("actuated", "mimic"):
        _fit_grip_pads(robot)
    elif gripper == "parallel":
        _fit_parallel_pads(robot)

    validate_references(robot)
    for expected in (*ARM_JOINTS, *_GRIPPER_JOINTS):
        if robot.find(f"joint[@name='{expected}']") is None:
            raise ValueError(f"repaired URDF is missing the expected joint {expected!r}")

    ET.indent(robot, space="    ")
    xml_text = (
        "<?xml version='1.0' encoding='utf-8'?>\n"
        "<!-- GENERATED by arc_mycobot.assets.robots.mycobot_urdf - do not edit, do not commit.\n"
        f"     Source: {SOURCE_URDF}\n"
        "     Repairs: malformed <limit> quote; XML 1.1 declaration; stray <xacro:property>;\n"
        f"     velocity=0 -> {ARM_VELOCITY_LIMIT} rad/s; <inertial> added to every link;\n"
        f"     gripper={gripper} ("
        + (
            f"whole mimic cluster welded at {GRIPPER_FIXED_AT} rad"
            if gripper == "welded"
            else f"mimic dropped, {len(ACTUATED_GRIPPER_JOINTS)} joints left movable"
        )
        + "). See the module docstring. -->\n"
        + ET.tostring(robot, encoding="unicode")
        + "\n"
    )
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_rewrite_mesh_paths(xml_text), encoding="utf-8")
    return out_path


if __name__ == "__main__":  # pragma: no cover - convenience for eyeballing the output
    for mode in ("welded", "actuated", "mimic"):
        print(mode, "->", build_mycobot_urdf(mode, force=True))
