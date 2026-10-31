# Copyright (c) 2026, arc-mycobot contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Find the reach goal box: the largest axis-aligned box whose goals the arm reaches without self-collision.

A goal is *feasible* when some joint vector inside the URDF limits puts the flange origin within ``--tolerance`` of it
and keeps every collision-checked link pair (see ``mycobot_collision.py``) at least ``--margin`` apart, with no hull
below the mounting plane. Damped least-squares IK runs from ``--starts`` random starts (plus the home posture) per goal.
The search grid spans v1's goal box, (0.16, 0, 0.20) +- (0.08, 0.12, 0.08) m, and the result is the
largest-volume box of grid goals that are all feasible.

It needs no simulator, and it does not check that the arm can get from one goal to the next without contact. The
simulator enforces that once self-collision is on. ``reach_self_collision_check.py`` measures it for a trained policy.

    uv run python scripts/reach_workspace_sweep.py
    uv run python scripts/reach_workspace_sweep.py --margin 0.003 --starts 512
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from mycobot_collision import ARM_JOINTS, MyCobotGeometry

V1_CENTRE = (0.16, 0.0, 0.20)
V1_RANGE = (0.08, 0.12, 0.08)
HOME_POSE = (0.0, -0.5, -1.0, -0.6, 0.0, 0.0)
"""``MYCOBOT_HOME_POSE`` in ``ARM_JOINTS`` order."""
RESET_OFFSET = 0.2
"""Half-width [rad] of the reach task's uniform reset offset around home."""


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--margin", type=float, default=0.005, help="Required clearance between link hulls [m].")
    parser.add_argument("--tolerance", type=float, default=0.001, help="IK position tolerance [m].")
    parser.add_argument("--starts", type=int, default=256, help="Random IK starts per goal.")
    parser.add_argument("--step", type=float, default=0.01, help="Grid spacing [m].")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    geometry = MyCobotGeometry(device=args.device)
    print(f"checked link pairs: {len(geometry.pairs)}")

    # home posture and the reset distribution
    home = torch.tensor([HOME_POSE], device=args.device)
    resets = home + (torch.rand(4096, 6, device=args.device) * 2 - 1) * RESET_OFFSET
    resets = torch.clamp(resets, geometry.lower, geometry.upper)
    for name, q in (("home", home), (f"reset (home +- {RESET_OFFSET} rad)", resets)):
        poses, _ = geometry.forward_kinematics(q)
        depth = geometry.pair_depths(poses).amax(0)
        clearance = geometry.pair_depths(poses, margin=args.margin).amax(0)
        print(
            f"{name}: overlap > 0 in {100 * (depth > 0).float().mean():.2f}% (max {1e3 * depth.max():.1f} mm), "
            f"closer than the margin in {100 * (clearance > 0).float().mean():.2f}%"
        )

    axes = [
        np.round(np.arange(c - r, c + r + args.step / 2, args.step), 6)
        for c, r in zip(V1_CENTRE, V1_RANGE, strict=True)
    ]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    goals = torch.tensor(grid, dtype=torch.float32, device=args.device)
    print(f"grid: {' x '.join(str(len(a)) for a in axes)} = {len(grid)} goals, {args.starts + 1} IK starts each")

    feasible = torch.zeros(len(grid), dtype=torch.bool, device=args.device)
    reachable = torch.zeros_like(feasible)
    batch = 64 * 1024 // (args.starts + 1)
    for begin in range(0, len(grid), batch):
        block = goals[begin : begin + batch]
        n = block.shape[0]
        starts = geometry.lower + torch.rand(n, args.starts, 6, device=args.device) * (geometry.upper - geometry.lower)
        starts = torch.cat([home.expand(n, 1, 6), starts], 1).reshape(-1, 6)
        q, error = geometry.solve_position_ik(block.repeat_interleave(args.starts + 1, 0), starts)
        poses, _ = geometry.forward_kinematics(q)
        clear = (geometry.pair_depths(poses, margin=args.margin).amax(0) == 0) & (geometry.lowest_point(poses) >= 0)
        converged = (error < args.tolerance).reshape(n, -1)
        reachable[begin : begin + n] = converged.any(1)
        feasible[begin : begin + n] = (converged & clear.reshape(n, -1)).any(1)

    shape = tuple(len(a) for a in axes)
    ok = feasible.reshape(shape).cpu().numpy()
    reach = reachable.reshape(shape).cpu().numpy()
    print(f"\nv1 box: reachable {100 * reach.mean():.1f}%, reachable without self-collision {100 * ok.mean():.1f}%")
    for name, axis, dim in (("x", axes[0], 0), ("z", axes[2], 2)):
        other = tuple(i for i in range(3) if i != dim)
        rows = ", ".join(f"{v:.2f}: {100 * f:.0f}%" for v, f in zip(axis, ok.mean(other), strict=True))
        print(f"feasible share by goal {name}: {rows}")

    box = _largest_box(ok)
    lo = [axes[d][box[d][0]] for d in range(3)]
    hi = [axes[d][box[d][1]] for d in range(3)]
    centre = [round(float(a + b) / 2, 3) for a, b in zip(lo, hi, strict=True)]
    half = [round(float(b - a) / 2, 3) for a, b in zip(lo, hi, strict=True)]
    bounds = ", ".join(f"{name} [{a:.2f}, {b:.2f}]" for name, a, b in zip("xyz", lo, hi, strict=True))
    print(f"\nlargest all-feasible box: {bounds}")
    print(f"TARGET_POS_CENTRE = {tuple(centre)}")
    print(f"TARGET_POS_RANGE = {tuple(half)}")
    volume, v1_volume = np.prod([2 * h for h in half]), np.prod([2 * r for r in V1_RANGE])
    print(f"volume {1e3 * volume:.2f} L ({100 * volume / v1_volume:.0f}% of v1)")
    print(f"joint order: {ARM_JOINTS}")


def _largest_box(ok: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Index ranges (inclusive) of the largest-volume all-true box, by 3D prefix sums."""
    sx, sy, sz = ok.shape
    prefix = np.zeros((sx + 1, sy + 1, sz + 1), dtype=np.int64)
    prefix[1:, 1:, 1:] = ok.astype(np.int64).cumsum(0).cumsum(1).cumsum(2)
    y0, y1 = np.triu_indices(sy)
    z0, z1 = np.triu_indices(sz)
    best, best_box = -1, ((0, 0), (0, 0), (0, 0))
    for x0 in range(sx):
        for x1 in range(x0, sx):
            # count of true cells in [x0..x1] x [y0..y1] x [z0..z1] for every (y, z) range pair
            slab = prefix[x1 + 1] - prefix[x0]
            total = (
                slab[y1[:, None] + 1, z1[None] + 1]
                - slab[y0[:, None], z1[None] + 1]
                - slab[y1[:, None] + 1, z0[None]]
                + slab[y0[:, None], z0[None]]
            )
            cells = (x1 - x0 + 1) * (y1 - y0 + 1)[:, None] * (z1 - z0 + 1)[None]
            volume = np.where(total == cells, (x1 - x0) * (y1 - y0)[:, None] * (z1 - z0)[None], -1)
            k = np.unravel_index(np.argmax(volume), volume.shape)
            if volume[k] > best:
                best = volume[k]
                best_box = ((x0, x1), (int(y0[k[0]]), int(y1[k[0]])), (int(z0[k[1]]), int(z1[k[1]])))
    return best_box


if __name__ == "__main__":
    main()
