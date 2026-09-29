"""Real-arm deployment guards using fake camera and serial robot."""

import ast
import builtins
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from deploy.arm import ArmConfig, ArmSession, require_pymycobot
from deploy.controller import UnsafeTarget, check_start, plan_target
from deploy.run import cli, rgb_frame, run_camera
from deploy.runtime import Prediction, box_from_results, verify_checkpoint


class FakeCamera:
    def __init__(self, _source):
        self.released = False

    def isOpened(self):
        return True

    def set(self, _key, _value):
        return True

    def read(self):
        return True, np.zeros((4, 6, 3), dtype=np.uint8)

    def release(self):
        self.released = True


class FakeGate:
    def __init__(self):
        self.opened = False

    def open(self, _reason):
        self.opened = True

    def close(self, _reason):
        self.opened = False


class FakeArm:
    last = None

    def __init__(self, config):
        self.config = config
        self.gate = FakeGate()
        self.commands = []
        self.halts = []
        FakeArm.last = self

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read_joints(self, retry_s):
        return SimpleNamespace(angles=[0, 0, 0, 0, 0, -45])

    def read_status(self):
        return SimpleNamespace(is_normal=True, describe=lambda: "normal")

    def command_joints(self, target, measured):
        assert self.gate.opened
        self.commands.append(list(target))

    def halt(self, reason):
        self.halts.append(reason)
        self.gate.close(reason)


class FakeRobot:
    def __init__(self):
        self.calls = []
        self.mode = 0
        self.angles = [0, 0, 0, 0, 0, -45]
        self.closed = False

    def set_fresh_mode(self, mode):
        self.mode = mode
        self.calls.append(("set_fresh_mode", mode))

    def get_fresh_mode(self):
        return self.mode

    def get_angles(self):
        return self.angles

    def get_error_information(self):
        return 0

    def is_power_on(self):
        return 1

    def send_angles(self, angles, speed, _async=False):
        self.calls.append(("send_angles", angles, speed, _async))

    def stop(self):
        self.calls.append(("stop",))

    def close(self):
        self.closed = True


class FakeRuntime:
    def __init__(self, detected=True):
        self.detected = detected

    def predict(self, rgb, angles, velocities, previous_action):
        assert rgb.shape == (4, 4, 3)
        assert angles[-1] == -45
        assert previous_action == [0.0] * 5
        return Prediction((0, 0, 0.2, 0.2, float(self.detected)), (0.1,) * 5, (5.0, 0.0, 0.0, 0.0, 0.0, -45.0))


def args(execute):
    return SimpleNamespace(
        camera="/dev/video0",
        execute=execute,
        port="/dev/ttyTHS1",
        baud=1000000,
        max_frames=1,
        rotate=0,
        flip_x=False,
        flip_y=False,
    )


def test_deployment_guards_and_feedback_step():
    check_start([0, 0, 0, 0, 0, -45])
    with pytest.raises(UnsafeTarget):
        check_start([0, 0, 0, 0, 0, 0])
    p = Prediction((0, 0, 0.2, 0.2, 1), (0,) * 5, (5, 0, 0, 0, 0, -45))
    assert plan_target(p, [0, 0, 0, 0, 0, -45]) == [1.0, 0.0, 0.0, 0.0, 0.0, -45.0]
    with pytest.raises(UnsafeTarget):
        plan_target(Prediction((0, 0, 0, 0, 0), p.action, p.target_deg), [0, 0, 0, 0, 0, -45])
    with pytest.raises(UnsafeTarget):
        plan_target(Prediction(p.box, p.action, (40, 0, 0, 0, 0, -45)), [0, 0, 0, 0, 0, -45])


def test_preview_never_sends_and_execute_halts_after_command():
    run_camera(args(False), FakeRuntime(), lambda **kw: SimpleNamespace(**kw), FakeArm, FakeCamera)
    arm = FakeArm.last
    assert arm is not None
    assert arm.config.read_only
    assert arm.commands == []
    run_camera(args(True), FakeRuntime(), lambda **kw: SimpleNamespace(**kw), FakeArm, FakeCamera)
    arm = FakeArm.last
    assert arm is not None
    assert arm.commands == [[1.0, 0.0, 0.0, 0.0, 0.0, -45.0]]
    assert arm.halts == ["deployment ended"]


def test_lost_hand_sends_no_command_and_stops_arm():
    with pytest.raises(UnsafeTarget, match="not detected"):
        run_camera(args(True), FakeRuntime(False), lambda **kw: SimpleNamespace(**kw), FakeArm, FakeCamera)
    arm = FakeArm.last
    assert arm is not None
    assert arm.commands == []
    assert arm.halts == ["deployment ended"]


def test_bad_start_pose_sends_no_command():
    class BadPoseArm(FakeArm):
        def read_joints(self, retry_s):
            return SimpleNamespace(angles=[20, 0, 0, 0, 0, -45])

    with pytest.raises(UnsafeTarget, match="start near"):
        run_camera(args(True), FakeRuntime(), lambda **kw: SimpleNamespace(**kw), BadPoseArm, FakeCamera)
    arm = FakeArm.last
    assert arm is not None
    assert arm.commands == []
    assert arm.halts == ["deployment ended"]


def test_missing_pymycobot_has_install_instruction(monkeypatch, capsys):
    original_import = builtins.__import__

    def missing_hardware_package(name, *args, **kwargs):
        if name == "pymycobot":
            raise ModuleNotFoundError("No module named 'pymycobot'", name="pymycobot")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_hardware_package)
    with pytest.raises(RuntimeError, match="pip install -r deploy/requirements.txt"):
        require_pymycobot()
    monkeypatch.setattr(sys, "argv", ["deploy.run", "--max-frames", "1"])
    with pytest.raises(SystemExit) as error:
        cli()
    assert error.value.code == 2
    assert "pip install -r deploy/requirements.txt" in capsys.readouterr().err


def test_direct_serial_preview_has_no_motion_commands(tmp_path):
    robot = FakeRobot()
    config = ArmConfig(read_only=True)
    with ArmSession(config, robot_factory=lambda _port, _baud: robot, lock_path=str(tmp_path / "arm.lock")) as arm:
        reading = arm.read_joints()
        assert reading.angles == (0, 0, 0, 0, 0, -45)
        assert arm.read_status().is_normal
        arm.gate.open("test")
        with pytest.raises(UnsafeTarget, match="preview"):
            arm.command_joints([0, 0, 0, 0, 0, -45], measured=reading)
    assert robot.calls == []
    assert robot.closed


def test_direct_serial_execute_checks_gate_feedback_and_step(tmp_path):
    robot = FakeRobot()
    config = ArmConfig(read_only=False)
    with ArmSession(config, robot_factory=lambda _port, _baud: robot, lock_path=str(tmp_path / "arm.lock")) as arm:
        reading = arm.read_joints()
        with pytest.raises(UnsafeTarget, match="gate"):
            arm.command_joints([1, 0, 0, 0, 0, -45], measured=reading)
        arm.gate.open("test")
        with pytest.raises(UnsafeTarget, match="step limit"):
            arm.command_joints([2, 0, 0, 0, 0, -45], measured=reading)
        arm.command_joints([1, 0, 0, 0, 0, -45], measured=reading)
        arm.gate.close("one command")
    assert robot.calls == [
        ("set_fresh_mode", 1),
        ("send_angles", [1.0, 0.0, 0.0, 0.0, 0.0, -45.0], 10, True),
        ("stop",),
    ]
    assert robot.closed


def test_camera_loop_with_direct_serial_session(tmp_path):
    robot = FakeRobot()

    class DirectSession(ArmSession):
        def __init__(self, config):
            super().__init__(config, robot_factory=lambda _port, _baud: robot, lock_path=str(tmp_path / "arm.lock"))

    run_camera(args(False), FakeRuntime(), ArmConfig, DirectSession, FakeCamera)
    assert robot.calls == []
    robot.closed = False
    run_camera(args(True), FakeRuntime(), ArmConfig, DirectSession, FakeCamera)
    assert robot.calls == [
        ("set_fresh_mode", 1),
        ("send_angles", [1.0, 0.0, 0.0, 0.0, 0.0, -45.0], 10, True),
        ("stop",),
    ]
    assert robot.closed


def test_direct_serial_rejects_invalid_or_stale_feedback(tmp_path):
    robot = FakeRobot()
    config = ArmConfig(read_only=False, max_observation_age_s=0.001)
    with ArmSession(config, robot_factory=lambda _port, _baud: robot, lock_path=str(tmp_path / "arm.lock")) as arm:
        robot.angles = -1
        with pytest.raises(UnsafeTarget, match="no valid six-joint feedback"):
            arm.read_joints(retry_s=0)
        robot.angles = [0, 0, 0, 0, 0, -45]
        reading = arm.read_joints()
        arm.gate.open("test")
        time.sleep(0.01)
        with pytest.raises(UnsafeTarget, match="expired"):
            arm.command_joints([0, 0, 0, 0, 0, -45], measured=reading)
    assert all(call[0] != "send_angles" for call in robot.calls)


def test_direct_serial_refuses_busy_lock(tmp_path):
    lock_path = str(tmp_path / "arm.lock")
    config = ArmConfig(read_only=True)
    with ArmSession(config, robot_factory=lambda _port, _baud: FakeRobot(), lock_path=lock_path):
        second = ArmSession(config, robot_factory=lambda _port, _baud: FakeRobot(), lock_path=lock_path)
        with pytest.raises(RuntimeError, match="lock is busy"):
            second.__enter__()


def test_direct_serial_requires_verified_fresh_mode(tmp_path):
    robot = FakeRobot()
    robot.get_fresh_mode = lambda: 0
    session = ArmSession(
        ArmConfig(read_only=False), robot_factory=lambda _port, _baud: robot, lock_path=str(tmp_path / "arm.lock")
    )
    with pytest.raises(UnsafeTarget, match="fresh command mode"), session:
        pass
    assert all(call[0] != "send_angles" for call in robot.calls)
    assert robot.closed


def test_checkpoint_hash_rejects_another_file(tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_checkpoint(path)


def test_rgb_square_crop_and_class_filter():
    bgr = np.zeros((4, 6, 3), dtype=np.uint8)
    bgr[0, 1] = [10, 20, 30]
    rgb = rgb_frame(bgr)
    assert rgb.shape == (4, 4, 3)
    assert rgb[0, 0].tolist() == [30, 20, 10]
    boxes = SimpleNamespace(
        xyxy=torch.tensor([[0.0, 0.0, 320.0, 320.0], [80.0, 80.0, 240.0, 240.0]]),
        cls=torch.tensor([1.0, 267.0]),
        conf=torch.tensor([0.99, 0.8]),
    )
    result = box_from_results([SimpleNamespace(boxes=boxes)])
    assert result == pytest.approx((0, 0, 0.5, 0.5, 1))


def test_deploy_source_parses_as_python_38():
    for name in ("runtime.py", "controller.py", "arm.py", "run.py"):
        source = (Path(__file__).resolve().parents[1] / "deploy" / name).read_text()
        ast.parse(source, filename=name, feature_version=(3, 8))
