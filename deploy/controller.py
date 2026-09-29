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
    if any(abs(float(x)) > limits.envelope_degrees for x in prediction.target_deg[:5]):
        raise UnsafeTarget("policy target outside deployment envelope")
    if abs(float(prediction.target_deg[5]) + 45.0) > 1e-4:
        raise UnsafeTarget("policy attempted to move J6")
    return [
        float(current) + max(-limits.max_step_degrees, min(limits.max_step_degrees, float(target) - float(current)))
        for current, target in zip(measured_deg, prediction.target_deg)
    ]
