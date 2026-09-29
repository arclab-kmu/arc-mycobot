"""Direct pymycobot serial session for the Jetson deployment."""

from __future__ import annotations

import fcntl
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Optional

from deploy.controller import Limits, UnsafeTarget

LOCK_PATH = "/tmp/mycobot_lock"
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArmConfig:
    read_only: bool
    port: str = "/dev/ttyTHS1"
    baud: int = 1000000
    speed: int = 10
    max_feedback_age_s: float = 0.5
    max_observation_age_s: float = 1.0
    max_command_delta_deg: float = 1.01

    def __post_init__(self):
        if not self.port or self.baud <= 0 or not 1 <= self.speed <= 100:
            raise ValueError("invalid arm serial settings or command speed")


@dataclass(frozen=True)
class JointReading:
    angles: tuple[float, ...]
    observed_at: float
    latency_s: float


@dataclass(frozen=True)
class StatusReading:
    error_code: Optional[int]  # noqa: UP045 - Jetson Python 3.8
    power_on: Optional[bool]  # noqa: UP045 - Jetson Python 3.8

    @property
    def is_normal(self):
        return self.error_code == 0 and self.power_on is True

    def describe(self):
        return f"error {self.error_code}, power {self.power_on}"


class CommandGate:
    def __init__(self):
        self.opened = False

    def open(self, _reason):
        self.opened = True

    def close(self, _reason):
        self.opened = False

    def require_open(self):
        if not self.opened:
            raise UnsafeTarget("command gate is closed")


def _angles(raw):
    if not isinstance(raw, (list, tuple)) or len(raw) != 6:
        return None
    try:
        values = tuple(float(value) for value in raw)
    except (TypeError, ValueError):
        return None
    if any(not math.isfinite(value) or value == -1.0 for value in values):
        return None
    return values


def _reply_int(raw):
    return raw if type(raw) is int and raw >= 0 else None


def require_pymycobot():
    try:
        from pymycobot import MyCobot280
    except ImportError as exc:
        raise RuntimeError("pymycobot is missing; run: python -m pip install -r deploy/requirements.txt") from exc
    return MyCobot280


class ArmSession:
    """Own the serial port for one run; reject commands outside the measured envelope."""

    def __init__(self, config, robot_factory=None, lock_path=LOCK_PATH):
        self.config = config
        self.gate = CommandGate()
        self._robot_factory = robot_factory
        self._lock_path = lock_path
        self._lock_fd = None
        self._robot = None
        self._last_joints = None
        self._halted = False

    def __enter__(self):
        fd = os.open(self._lock_path, os.O_RDWR | os.O_CREAT, 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            raise RuntimeError(f"serial port lock is busy: {self._lock_path}") from exc
        self._lock_fd = fd
        try:
            robot_factory = self._robot_factory
            if robot_factory is None:
                robot_factory = require_pymycobot()
            self._robot = robot_factory(self.config.port, self.config.baud)
            time.sleep(0.05)
            if not self.config.read_only:
                self._robot.set_fresh_mode(1)
                if self._robot.get_fresh_mode() != 1:
                    raise UnsafeTarget("could not verify fresh command mode")
            return self
        except Exception:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, _exc_type, _exc, _tb):
        if not self.config.read_only:
            self.halt("session ended")
        self.gate.close("session ended")
        robot, self._robot = self._robot, None
        try:
            if robot is not None:
                robot.close()
        except Exception as exc:
            log.warning("close() failed: %s", exc)
        finally:
            if self._lock_fd is not None:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                os.close(self._lock_fd)
                self._lock_fd = None

    def _require_robot(self):
        if self._robot is None:
            raise RuntimeError("arm serial session is not open")
        return self._robot

    def read_joints(self, retry_s=0.2):
        robot = self._require_robot()
        deadline = time.monotonic() + max(0.0, retry_s)
        while True:
            started = time.monotonic()
            raw = robot.get_angles()
            observed_at = time.monotonic()
            latency = observed_at - started
            if latency > self.config.max_feedback_age_s:
                raise UnsafeTarget(f"get_angles() took {latency:.3f} s; feedback is stale")
            angles = _angles(raw)
            if angles is not None:
                reading = JointReading(angles, observed_at, latency)
                self._last_joints = reading
                return reading
            if observed_at >= deadline:
                raise UnsafeTarget("no valid six-joint feedback from get_angles()")
            time.sleep(0.02)

    def read_status(self):
        robot = self._require_robot()
        try:
            error = _reply_int(robot.get_error_information())
            power = _reply_int(robot.is_power_on())
        except Exception as exc:
            log.warning("robot status read failed: %s", exc)
            return None
        return StatusReading(error, None if power not in (0, 1) else bool(power))

    def command_joints(self, target, measured):
        robot = self._require_robot()
        self.gate.require_open()
        if self.config.read_only:
            raise UnsafeTarget("preview cannot send joint commands")
        if measured is not self._last_joints:
            raise UnsafeTarget("command needs the latest joint feedback")
        if time.monotonic() - measured.observed_at > self.config.max_observation_age_s:
            raise UnsafeTarget("joint feedback expired before command")
        if len(target) != 6:
            raise UnsafeTarget("command needs six joint targets")
        values = tuple(float(value) for value in target)
        if any(not math.isfinite(value) for value in values):
            raise UnsafeTarget("joint target is not finite")
        limits = Limits()
        if any(abs(actual) > limits.envelope_degrees for actual in measured.angles[:5]):
            raise UnsafeTarget("measured joints outside deployment envelope")
        if abs(measured.angles[5] + 45.0) > limits.wrist_tolerance_degrees:
            raise UnsafeTarget("measured J6 left its -45 degree pose")
        if any(abs(value) > limits.envelope_degrees for value in values[:5]):
            raise UnsafeTarget("joint target outside deployment envelope")
        if abs(values[5] + 45.0) > 1e-4:
            raise UnsafeTarget("J6 target must stay at -45 degrees")
        if any(
            abs(value - actual) > self.config.max_command_delta_deg for value, actual in zip(values, measured.angles)
        ):
            raise UnsafeTarget("joint command exceeds feedback step limit")
        robot.send_angles(list(values), self.config.speed, _async=True)

    def halt(self, reason="halt"):
        self.gate.close(reason)
        if self._halted or self.config.read_only or self._robot is None:
            return
        self._halted = True
        try:
            self._robot.stop()
        except Exception as exc:
            log.warning("stop() failed: %s", exc)
