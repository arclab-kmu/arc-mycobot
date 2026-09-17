# Copyright (c) 2026, arc-mycobot contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify that the configured goal box is inside the arm's reachable workspace.

This is the check that produced ``TARGET_POS_CENTRE`` and ``TARGET_POS_RANGE``
in the task config, and it exists so those numbers can be *re-derived* rather
than trusted. Run it after changing the goal box, the tracked body, the joint
limits or the URDF repair.

It launches no simulator and imports nothing from Isaac Lab: forward kinematics
comes from :mod:`arc_mycobot.kinematics.urdf_fk`, reading the same repaired URDF
that Isaac Lab converts, and the inverse problem is a bounded least-squares
solve over the declared joint limits. So it runs on CPU in a few seconds and can
gate a change before an eight-hour training run discovers the same thing slowly.

What "reachable" means here: there exists a joint vector *inside the declared
limits* whose forward kinematics puts the tracked body within
:data:`REACH_TOLERANCE` of the goal. It says nothing about whether the arm can
get there from its home posture without self-collision, and nothing about
orientation -- this task does not track orientation.

    uv run workspace_sweep
    uv run workspace_sweep --samples 2000 --seed 7
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from scipy.optimize import least_squares

# mycobot_urdf and geometry are both Isaac-free on purpose; importing the
# flange name from arc_mycobot.assets.robots.mycobot_280 instead would build an
# ArticulationCfg and start the Omniverse bootstrap this script exists to avoid.
from arc_mycobot.assets.robots.mycobot_urdf import FLANGE_BODY
from arc_mycobot.kinematics.urdf_fk import Chain, load_chain
from arc_mycobot.tasks.reach.config.mycobot.geometry import TARGET_POS_CENTRE, TARGET_POS_RANGE

REACH_TOLERANCE = 1e-3
"""A goal counts as reached when IK closes to within this distance [m]: 1 mm.

An order of magnitude below the task's own 20 mm success threshold, so a goal
that passes here cannot be the thing limiting a trained policy's error.
"""

_IK_RESTARTS = 8
"""Random restarts per goal. The myCobot has no redundancy but it does have
multiple branches (elbow up/down, wrist flipped), and a single seed lands in a
local minimum often enough to produce false failures."""


def solve_ik(chain: Chain, target: np.ndarray, seed: int) -> tuple[float, np.ndarray]:
    """Bounded least-squares IK for a position target.

    ``seed`` is per-goal rather than drawn from a generator shared across the
    sweep, and that is not a detail. With a shared generator each goal gets
    whatever state the previous goals left behind, so one unlucky goal can
    consume eight poor starting points and be reported unreachable while the
    identical goal passes when the sweep is run with a different ``--samples``.
    That exact false failure showed up while this box was being sized. Seeding
    per goal makes every result reproducible on its own.

    Returns:
        ``(residual, q)`` -- the best position error [m] and the joint vector
        achieving it, which is guaranteed to lie inside the declared limits.
    """
    rng = np.random.default_rng(seed)
    lower, upper = chain.limits[:, 0], chain.limits[:, 1]
    best = (np.inf, lower)
    for _ in range(_IK_RESTARTS):
        result = least_squares(
            lambda q: chain.fk(q)[0] - target,
            rng.uniform(lower, upper),
            bounds=(lower, upper),
            xtol=1e-13,
            ftol=1e-13,
        )
        residual = float(np.linalg.norm(result.fun))
        if residual < best[0]:
            best = (residual, result.x)
        if residual < REACH_TOLERANCE * 0.1:
            break
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=600, help="Goals drawn from the configured box.")
    parser.add_argument("--seed", type=int, default=1, help="RNG seed, so a reported number can be reproduced.")
    parser.add_argument("--body", type=str, default=FLANGE_BODY, help="Tracked link.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    chain = load_chain(args.body)
    lower, upper = chain.limits[:, 0], chain.limits[:, 1]

    print(f"chain to {args.body!r}: {len(chain.actuated)} actuated joints")
    for joint in chain.actuated:
        print(f"  {joint.name:24s} limits [{joint.limit[0]:+.4f}, {joint.limit[1]:+.4f}] rad")

    # -- the workspace the arm actually has, for context
    envelope = np.array([chain.fk(q)[0] for q in rng.uniform(lower, upper, size=(20000, len(chain.actuated)))])
    horizontal = np.linalg.norm(envelope[:, :2], axis=1)
    print(
        f"\nreachable envelope: horizontal radius <= {horizontal.max():.4f} m, "
        f"z in [{envelope[:, 2].min():+.4f}, {envelope[:, 2].max():+.4f}] m"
    )

    # -- the box the task actually samples
    centre, half = np.array(TARGET_POS_CENTRE), np.array(TARGET_POS_RANGE)
    print(
        f"\ngoal box: centre {tuple(centre)} half-extent {tuple(half)}\n"
        f"  x [{centre[0] - half[0]:+.3f}, {centre[0] + half[0]:+.3f}]  "
        f"y [{centre[1] - half[1]:+.3f}, {centre[1] + half[1]:+.3f}]  "
        f"z [{centre[2] - half[2]:+.3f}, {centre[2] + half[2]:+.3f}] m"
    )

    # The eight corners first, explicitly. Uniform samples essentially never
    # visit a corner of a 3-box -- the box this task shipped with at one point
    # passed 300 uniform goals while its far-top-side corner was 9.3 mm out of
    # reach -- and the corners are where the arm is most extended.
    corners = np.array([centre + half * (np.array(signs) * 2 - 1) for signs in np.ndindex(2, 2, 2)])
    corner_residuals = np.array([solve_ik(chain, corner, seed)[0] for seed, corner in enumerate(corners)])
    print(f"\n8 corners: worst {corner_residuals.max() * 1000:.4f} mm")
    for corner, residual in zip(corners, corner_residuals, strict=True):
        flag = "" if residual <= REACH_TOLERANCE else "   <-- OUT OF REACH"
        print(f"  {np.round(corner, 3).tolist()}  {residual * 1000:8.4f} mm{flag}")

    goals = centre + rng.uniform(-half, half, size=(args.samples, 3))
    residuals = np.empty(args.samples)
    solutions = np.empty((args.samples, len(chain.actuated)))
    for index, goal in enumerate(goals):
        residuals[index], solutions[index] = solve_ik(chain, goal, args.seed * 100_003 + index)

    reached = residuals <= REACH_TOLERANCE
    corners_reached = corner_residuals <= REACH_TOLERANCE
    print(
        f"\n{args.samples} goals, {_IK_RESTARTS} IK restarts each:\n"
        f"  reachable (<= {REACH_TOLERANCE * 1000:.1f} mm): {reached.mean() * 100:.1f}%\n"
        f"  residual: median {np.median(residuals) * 1000:.4f} mm, max {residuals.max() * 1000:.4f} mm"
    )

    # How close the solutions sit to a hard joint limit. Not a pass/fail -- these
    # are the residuals of *one* IK branch, and another branch may sit further
    # in -- but a box whose solutions all pin a joint is a box worth shrinking.
    margin = np.minimum(solutions - lower, upper - solutions).min(axis=1)
    print(f"  joint-limit margin of the returned solutions: median {np.median(margin):.3f} rad")

    if not (reached.all() and corners_reached.all()):
        worst = goals[np.argmax(residuals)]
        print(
            f"\nFAIL: {(~corners_reached).sum()} corner(s) and {(~reached).sum()} sampled goal(s)"
            " are outside the reachable workspace.\n"
            f"  worst sampled: {np.round(worst, 4).tolist()} at {residuals.max() * 1000:.2f} mm\n"
            "  Shrink TARGET_POS_RANGE or move TARGET_POS_CENTRE in"
            " arc_mycobot/tasks/reach/config/mycobot/joint_pos_env_cfg.py.",
            file=sys.stderr,
        )
        return 1

    print("\nOK: every sampled goal is reachable within the declared joint limits.")
    return 0


def cli() -> int:
    """Console-script entry point. No simulator, so no ``run_and_exit`` dance."""
    return sys.exit(main())


if __name__ == "__main__":
    cli()
