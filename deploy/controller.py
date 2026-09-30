"""Pure joint-target checks before handing a command to the serial session."""

import math
from dataclasses import dataclass


class UnsafeTarget(ValueError):
    pass


@dataclass(frozen=True)
class Limits:
    start_degrees: float = 10.0
    envelope_degrees: float = 30.0
    wrist_tolerance_degrees: float = 3.0
    max_step_degrees: float = 1.0


class TargetSmoother:
    """Filter policy goals while keeping every commanded joint inside the envelope."""

    def __init__(self, alpha=0.35, limits=Limits()):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("target filter alpha must be in (0, 1]")
        self.alpha = alpha
        self.limits = limits
        self._goal = None

    def reset(self):
        self._goal = None

    def update(self, target_deg, measured_deg):
        if len(target_deg) != 6 or len(measured_deg) != 6:
            raise UnsafeTarget("six joints are required")
        if not all(math.isfinite(float(value)) for value in list(target_deg) + list(measured_deg)):
            raise UnsafeTarget("non-finite joint angle")
        if abs(float(target_deg[5]) + 45.0) > 1e-4:
            raise UnsafeTarget("policy attempted to move J6")
        projected = [
            max(-self.limits.envelope_degrees, min(self.limits.envelope_degrees, float(value)))
            for value in target_deg[:5]
        ]
        clipped = any(value != float(raw) for value, raw in zip(projected, target_deg[:5]))
        if self._goal is None:
            self._goal = [float(value) for value in measured_deg[:5]]
        self._goal = [previous + self.alpha * (goal - previous) for previous, goal in zip(self._goal, projected)]
        return tuple(self._goal) + (-45.0,), clipped


def check_start(angles_deg, limits=Limits()):
    if len(angles_deg) != 6 or not all(math.isfinite(float(x)) for x in angles_deg):
        raise UnsafeTarget("six finite joint angles are required")
    if any(abs(float(x)) > limits.start_degrees for x in angles_deg[:5]):
        raise UnsafeTarget("J1-J5 must start near the trained zero pose")
    if abs(float(angles_deg[5]) + 45.0) > limits.wrist_tolerance_degrees:
        raise UnsafeTarget("J6 must already be near -45 degrees")


def plan_target(prediction, measured_deg, limits=Limits()):
    """Absolute policy target, bounded around home and stepped from feedback."""
    if prediction.box[4] != 1.0:
        raise UnsafeTarget("Human hand not detected")
    if len(measured_deg) != 6 or len(prediction.target_deg) != 6:
        raise UnsafeTarget("six joints are required")
    if not all(math.isfinite(float(x)) for x in list(measured_deg) + list(prediction.target_deg)):
        raise UnsafeTarget("non-finite joint angle")
    if any(abs(float(x)) > limits.envelope_degrees for x in measured_deg[:5]):
        raise UnsafeTarget("measured J1-J5 outside deployment envelope")
    if abs(float(measured_deg[5]) + 45.0) > limits.wrist_tolerance_degrees:
        raise UnsafeTarget("measured J6 left its -45 degree pose")
    if abs(float(measured_deg[5]) + 45.0) > 1.0:
        raise UnsafeTarget("J6 is too far from -45 degrees for one safe command")
    if any(abs(float(x)) > limits.envelope_degrees for x in prediction.target_deg[:5]):
        raise UnsafeTarget("policy target outside deployment envelope")
    if abs(float(prediction.target_deg[5]) + 45.0) > 1e-4:
        raise UnsafeTarget("policy attempted to move J6")
    arm_target = [
        float(current) + max(-limits.max_step_degrees, min(limits.max_step_degrees, float(target) - float(current)))
        for current, target in zip(measured_deg[:5], prediction.target_deg[:5])
    ]
    return arm_target + [-45.0]
