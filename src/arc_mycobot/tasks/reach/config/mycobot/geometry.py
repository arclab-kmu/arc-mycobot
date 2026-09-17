# Copyright (c) 2026, arc-mycobot contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""The task's geometric constants, with **no Isaac Lab import**.

Everything here is a length or an angle that describes *where the task happens*,
as opposed to how it is wired. They live in their own module for one concrete
reason: ``scripts/tools/workspace_sweep.py`` checks them against the arm's
forward kinematics, and that check has to run on a CPU in a few seconds. Reading
them out of ``joint_pos_env_cfg`` would drag in ``isaaclab``, which starts the
Omniverse Kit bootstrap -- tens of seconds, a GPU, and an EULA prompt -- to read
six floats.

So the rule is: a number the sweep validates lives here; a number that only means
something to a manager config lives in :mod:`.joint_pos_env_cfg`. Both are
re-exported from there, so the task config still reads as one piece.
"""

from __future__ import annotations

import math

__all__ = [
    "SUCCESS_THRESHOLD",
    "TANH_STD",
    "TARGET_ORIENTATION",
    "TARGET_POS_CENTRE",
    "TARGET_POS_RANGE",
]

TARGET_POS_CENTRE = (0.16, 0.0, 0.20)
"""Goal-box centre in the robot's root (``joint1``) frame [m].

In front of the arm and 200 mm above its mounting plane. The flange's reachable
workspace runs to a horizontal radius of 302 mm and a height of 447 mm, so this
sits at about 53% of the horizontal reach -- inside the dexterous region rather
than out at the boundary where the arm is at full extension and the Jacobian is
near-singular.

160 mm and not the 180 mm this started at. At 180 the box's far-top-side
*corner* (0.26, -0.12, 0.28) is 9.3 mm out of reach, even though 300 goals drawn
uniformly from the same box all solved -- uniform sampling essentially never
visits a corner of a 3-box. Moving the centre 20 mm toward the base brings all
eight corners inside the workspace without shrinking the box, which is the
better trade: the box's size is what makes the task non-trivial, its position is
free. ``tests/test_reach_geometry.py`` checks the corners explicitly for exactly
this reason.
"""

TARGET_POS_RANGE = (0.08, 0.12, 0.08)
"""Half-extent of the goal box [m]: 160 x 240 x 160 mm about the centre.

The lower ``z`` face sits at 120 mm above the mounting plane, which is the
clearance this task has instead of a floor constraint (see ``ReachSceneCfg``).

Sized to be the largest box whose eight corners all remain reachable, once
``TARGET_POS_CENTRE`` is placed to allow it. Checked at (0.07, 0.10, 0.07) and
(0.06, 0.09, 0.06) as well; both also pass, and the largest was taken because a
box the policy can cross is what makes the task more than a fixed-point regress.

Verify with ``uv run workspace_sweep`` after changing either this or the centre.
"""

TARGET_ORIENTATION = (math.pi, 0.0, 0.0)
"""Goal orientation, ``(roll, pitch, yaw)`` [rad], **pinned to a single value**.

This is a position-reach task: the orientation is neither observed nor rewarded
(see ``RewardsCfg`` in :mod:`arc_mycobot.tasks.reach.reach_env_cfg`). The command
term still needs an orientation because ``UniformPoseCommand`` generates poses,
and this one -- flange pointing down -- makes the debug marker readable in the
viewport. Pinning it rather than sampling it keeps the sampled goals identical
between runs in the one respect that would otherwise vary invisibly.
"""

TANH_STD = 0.05
"""Length scale of the fine-grained tracking reward [m]. **Not** upstream's 0.1.

``position_command_error_tanh`` returns ``1 - tanh(distance / std)``, so ``std``
is the distance at which the term has given up most of its value -- it is what
makes the reward sharp near the goal. Upstream's 0.1 m is 12% of a Franka's
reach; the same *fraction* of this arm's 0.28 m is about 0.035 m. 0.05 m is
chosen a little looser than that so the term still carries signal from the far
side of the goal box, whose half-diagonal is 0.166 m.
"""

SUCCESS_THRESHOLD = 0.02
"""Position error counted as on-target [m]. See ``position_command_success``.

20 mm, about 7% of the arm's reach. Chosen against what the *hardware* could
support rather than what looks impressive: Elephant Robotics quote +-0.5 mm
repeatability for this arm, and repeatability is not accuracy, so a threshold an
order of magnitude above it leaves room for the modelling error that the
``TODO(unverified)`` actuator numbers in
:mod:`arc_mycobot.assets.robots.mycobot_280` guarantee.

It is an instrument, not a target, and it is now calibrated: a policy trained
with the committed configuration settles at 2.9-4.2 mm mean error and spends 93%
of its control steps inside this 20 mm threshold, with most of the remaining 7%
falling just after a goal resamples. So 20 mm currently measures "has arrived at
all" rather than "how well". Tighten it toward 5 mm if the interesting question
becomes settling quality -- but note that changing it changes the reward, so it
is not a free measurement.
"""
