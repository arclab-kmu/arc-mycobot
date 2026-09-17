# Copyright (c) 2026, arc-mycobot contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Geometric constants for the cube-lift task. **No Isaac Lab import.**

Same rule as the reach task's ``geometry.py``: numbers a CPU-only tool has to be
able to check live here, so checking them costs milliseconds instead of an
Omniverse Kit bootstrap.
"""

from __future__ import annotations

__all__ = [
    "CUBE_MASS",
    "CUBE_SIZE",
    "GOAL_POS_CENTRE",
    "GOAL_POS_RANGE",
    "JAW_OFFSET_IN_GRIPPER_BASE",
    "LIFT_HEIGHT",
    "SPAWN_POS_CENTRE",
    "SPAWN_POS_RANGE",
]

CUBE_SIZE = 0.025
"""Cube edge [m]. Sized to the jaw, which is the binding constraint.

The pads open to 45.6 mm and close to 19.4 mm (see ``_PAD_INSET`` in
``assets/robots/mycobot_urdf.py``), matching the published 20-45 mm range.
25 mm, and it can be this small **because the fingers slide**. The earlier
rotating linkage carried its pads 15 mm forward as they closed, which shoved a
small object out of the jaw before the pads met -- with it, 30 mm and under were
never caught and only 34-42 mm worked. A sliding jaw has no such sweep, so size
is bounded only by the 19.4-45.6 mm opening.
"""

CUBE_MASS = 0.02
"""Cube mass [kg]. 20 g, well inside the myCobot 280's 250 g rated payload."""

SPAWN_POS_CENTRE = (0.22, 0.0, CUBE_SIZE / 2 + 0.001)
"""Where the cube starts, in the robot's root frame [m]. Resting on the ground.

220 mm, not the 170 mm this started at. The grasp point sits 48.4 mm out from
``gripper_base`` (see :data:`JAW_OFFSET_IN_GRIPPER_BASE`), which makes the
gripper effectively that much longer, and the arm cannot fold tightly enough to
put the jaw closer than about 200 mm with a horizontal approach -- swept, the
residual falls linearly with x and hits zero at 200 mm. Everything from 200 mm
out is reachable across the full +-40 mm of y.
"""

SPAWN_POS_RANGE = (0.015, 0.025, 0.0)
"""Half-extent of the spawn randomization [m]: +-15 mm in x, +-25 mm in y.

Every point solves exactly for a horizontal side grasp (swept on a grid out to
+-20/40 mm, all reachable), so this is not a reachability limit -- it is a
*discovery* limit. With the corrected jaw the grasp pocket is narrow and the pads
sweep 15 mm forward as they close, so the pose that works is specific; widening
the spawn multiplies the variety a policy has to find it in before the action
penalties arrive. Measured at +-20/40 mm, the policy was still at 9% lifted by
iteration 400, against 38-51% for the earlier, more forgiving gripper.

Widen it again once the task trains reliably.
"""

LIFT_HEIGHT = 0.06
"""Cube height counted as "lifted" [m].

The cube's centre rests at 17 mm, so this is about 43 mm of clear daylight --
comfortably more than the settling jitter of a cube being squeezed, and low
enough to be reachable early in training when the reward still has to bootstrap.
"""

GOAL_POS_CENTRE = (0.21, 0.0, 0.12)
"""Centre of the goal region for the lifted cube, in the root frame [m]."""

GOAL_POS_RANGE = (0.03, 0.05, 0.04)
"""Half-extent of the goal region [m]: the cube must be carried, not just raised."""

JAW_OFFSET_IN_GRIPPER_BASE = (0.0, 0.0484, 0.0)
"""Grasp point expressed in the ``gripper_base`` link frame [m].

**Not** the midpoint of the two fingertip link origins. That was the first
version and it was wrong by 32 mm: the origin midpoint sits only 16.8 mm beyond
the gripper body, so a 32 mm object centred there overlaps ``gripper_base``
(which reaches y = 13.9 mm) by about 13 mm. An object "grasped" there was being
wedged into the gripper's body, not pinched by its pads -- visible in a close-up
render as the cube sunk into the white housing.

This is instead the middle of the volume the pads sweep. The pads run
y in [27.8, 53.8] mm when open and [43.0, 69.0] mm when closed -- they translate
15 mm forward as the knuckles rotate -- so the band they cover for the *whole*
closing motion is [43.0, 53.8] and its centre is 48.4 mm. An object there stays
between the pads from first contact to full grip, and clears the gripper body by
18.5 mm.

Constant across arm poses, like the old value: it is a fixed offset in a link
frame. ``tests/test_gripper.py`` checks that.
"""
