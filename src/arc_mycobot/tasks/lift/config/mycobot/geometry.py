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

SPAWN_POS_CENTRE = (0.237, 0.0, CUBE_SIZE / 2 + 0.001)
"""Where the cube starts, in the robot's root frame [m]. Resting on the ground.

The camera flange adds 17 mm along the gripper approach to the former 220 mm
spawn centre. At the adjusted lift home pose the packaged model's jaw centre is
at (237, 0, 89) mm; the cube starts directly below it.
"""

SPAWN_POS_RANGE = (0.015, 0.025, 0.0)
"""Half-extent of the spawn randomization [m]: +-15 mm in x, +-25 mm in y.

This range belonged to the earlier no-camera model. The new jaw centre is
aligned with the shifted spawn centre at home, but grasp performance over the
full range has not yet been re-measured.

Widen it again once the task trains reliably.
"""

LIFT_HEIGHT = 0.06
"""Cube height counted as "lifted" [m].

The cube's centre rests at 13.5 mm, so this is about 46.5 mm of clear daylight --
comfortably more than the settling jitter of a cube being squeezed, and low
enough to be reachable early in training when the reward still has to bootstrap.
"""

GOAL_POS_CENTRE = (0.227, 0.0, 0.12)
"""Goal centre shifted 17 mm with the camera flange to retain the earlier carry displacement [m]."""

GOAL_POS_RANGE = (0.03, 0.05, 0.04)
"""Half-extent of the goal region [m]: the cube must be carried, not just raised."""

JAW_OFFSET_IN_GRIPPER_BASE = (0.0, 0.0484, 0.0)
"""Grasp point expressed in the ``gripper_base`` link frame [m].

The parallel pads have centre y = 43.4 mm and extend 13 mm in each direction,
so 48.4 mm lies inside their contact length. Their y position stays constant
while the fingers slide across the jaw. The offset remains tied to the
``gripper_base`` frame even when the camera flange moves that frame outward.

Constant across arm poses, like the old value: it is a fixed offset in a link
frame. ``tests/test_gripper.py`` checks that.
"""
