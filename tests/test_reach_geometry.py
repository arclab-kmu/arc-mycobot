# Copyright (c) 2026, arc-mycobot contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""The goal box has to stay inside the arm's workspace, and the home posture has
to stay inside the goal box.

Both are properties of numbers in two different files -- the joint limits in the
repaired URDF and the constants in ``config/mycobot/geometry.py`` -- and neither
file knows about the other. A change to either can break the pair without
breaking anything that raises, and the symptom would be a training run that
plateaus at a tracking error nobody can explain.

CPU only: forward kinematics is numpy, and the goal box lives in an
Isaac-free module on purpose (see ``geometry.py``).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import least_squares

from arc_mycobot.assets.robots.mycobot_urdf import ARM_JOINTS, FLANGE_BODY, HOME_POSE, SOURCE_URDF
from arc_mycobot.kinematics.urdf_fk import load_chain
from arc_mycobot.tasks.reach.config.mycobot.geometry import (
    SUCCESS_THRESHOLD,
    TANH_STD,
    TARGET_POS_CENTRE,
    TARGET_POS_RANGE,
)

pytestmark = pytest.mark.skipif(not SOURCE_URDF.is_file(), reason=f"mycobot_ros2 checkout not found at {SOURCE_URDF}")

# Both come from mycobot_urdf, which imports no isaaclab -- that is why this
# whole file runs on CPU in under a second. The robot config re-exports them
# under MYCOBOT_* names rather than redeclaring them, so there is one source.
HOME = np.array([HOME_POSE[name] for name in ARM_JOINTS])


@pytest.fixture(scope="module")
def chain():
    return load_chain(FLANGE_BODY)


def test_chain_has_six_actuated_joints(chain):
    assert len(chain.actuated) == 6


def test_home_pose_is_inside_the_goal_box(chain):
    """Every episode should start where the goals are.

    A home posture outside the box makes each episode open with a gross traverse
    across the workspace before any tracking happens, which spends part of every
    goal window on approach rather than on settling.
    """
    position, _ = chain.fk(HOME)
    centre, half = np.array(TARGET_POS_CENTRE), np.array(TARGET_POS_RANGE)
    assert np.all(np.abs(position - centre) <= half), f"home flange {position} is outside the goal box"


def test_home_pose_covers_every_arm_joint():
    """A missing joint would silently default to 0.0 in the articulation config.

    There is no longer a separate copy of this posture to drift from: the robot
    config re-exports ``HOME_POSE`` rather than redeclaring it. What can still go
    wrong is the dict and the joint list disagreeing, which this catches.
    """
    assert set(HOME_POSE) == set(ARM_JOINTS)


def test_goal_box_clears_the_mounting_plane():
    """The scene has no floor at the mount plane, so clearance is the box's job."""
    lowest = TARGET_POS_CENTRE[2] - TARGET_POS_RANGE[2]
    assert lowest > 0.05, f"goal box reaches down to z={lowest:.3f} m"


def test_goal_box_corners_are_reachable(chain):
    """The eight corners are the extreme case; if they solve, the interior does.

    Tolerance is 1 mm, an order of magnitude below SUCCESS_THRESHOLD, so a goal
    that passes here cannot be what limits a trained policy.
    """
    lower, upper = chain.limits[:, 0], chain.limits[:, 1]
    rng = np.random.default_rng(0)
    centre, half = np.array(TARGET_POS_CENTRE), np.array(TARGET_POS_RANGE)

    for signs in np.ndindex(2, 2, 2):
        corner = centre + half * (np.array(signs) * 2 - 1)
        best = np.inf
        for _ in range(12):
            result = least_squares(
                lambda q, target=corner: chain.fk(q)[0] - target,
                rng.uniform(lower, upper),
                bounds=(lower, upper),
                xtol=1e-13,
                ftol=1e-13,
            )
            best = min(best, float(np.linalg.norm(result.fun)))
            if best < 1e-4:
                break
        assert best < 1e-3, f"goal-box corner {np.round(corner, 3).tolist()} is {best * 1000:.2f} mm out of reach"


def test_length_scales_are_ordered():
    """success threshold < tanh std < box half-diagonal.

    Each inequality is a separate way the task can be quietly misconfigured:
    a success threshold above the reward's length scale would report success
    where the reward has already saturated, and a tanh std above the box's own
    size would make the fine-grained term flat everywhere the goals are.
    """
    half_diagonal = float(np.linalg.norm(TARGET_POS_RANGE))
    assert SUCCESS_THRESHOLD < TANH_STD < half_diagonal
