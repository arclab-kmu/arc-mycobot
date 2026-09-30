"""Real-arm deployment guards using fake camera and serial robot."""

import ast
import builtins
import json
import sys
import time
import types
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from deploy.arm import ArmConfig, ArmSession, require_pymycobot
from deploy.controller import TargetSmoother, UnsafeTarget, check_start, plan_target
from deploy.preview import BrowserPreview
from deploy.run import cli, rgb_frame, run_camera
from deploy.runtime import HandPolicyRuntime, Prediction, box_from_results, verify_checkpoint
from deploy.train_hand_detector import prepare_detection_dataset


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
        self.halted = False
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
        self.halted = False

    def halt(self, reason):
        if not self.halted:
            self.halts.append(reason)
            self.halted = True
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
        self.warmups = 0

    def warmup(self):
        self.warmups += 1

    def predict(self, rgb, angles, velocities, previous_action):
        assert rgb.shape == (4, 4, 3)
        assert angles[-1] == -45
        return Prediction((0, 0, 0.2, 0.2, float(self.detected)), (0.1,) * 5, (5.0, 0.0, 0.0, 0.0, 0.0, -45.0))


def args(execute):
    return SimpleNamespace(
        camera="/dev/video0",
        execute=execute,
        port="/dev/ttyTHS1",
        baud=1000000,
        max_frames=1,
        web_preview_port=None,
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


def test_target_smoother_projects_overshoot_and_resets_after_loss():
    smoother = TargetSmoother()
    measured = [0, 0, 0, 0, 0, -45]
    filtered, clipped = smoother.update((40, -40, 0, 0, 0, -45), measured)
    assert clipped
    assert filtered[:2] == pytest.approx((10.5, -10.5))
    filtered, clipped = smoother.update((40, -40, 0, 0, 0, -45), measured)
    assert clipped
    assert filtered[:2] == pytest.approx((17.325, -17.325))
    smoother.reset()
    filtered, clipped = smoother.update((5, 0, 0, 0, 0, -45), measured)
    assert not clipped
    assert filtered[0] == pytest.approx(1.75)
    with pytest.raises(UnsafeTarget, match="non-finite"):
        smoother.update((float("nan"), 0, 0, 0, 0, -45), measured)


def test_smoothed_arm_step_keeps_j6_exactly_at_home():
    prediction = Prediction((0, 0, 0.2, 0.2, 1), (0,) * 5, (5, 0, 0, 0, 0, -45))
    assert plan_target(prediction, [0, 0, 0, 0, 0, -45.7]) == [1.0, 0.0, 0.0, 0.0, 0.0, -45.0]
    with pytest.raises(UnsafeTarget, match="too far"):
        plan_target(prediction, [0, 0, 0, 0, 0, -46.5])


def test_preview_never_sends_and_execute_halts_after_command():
    preview_runtime = FakeRuntime()
    run_camera(args(False), preview_runtime, lambda **kw: SimpleNamespace(**kw), FakeArm, FakeCamera)
    assert preview_runtime.warmups == 1
    arm = FakeArm.last
    assert arm is not None
    assert arm.config.read_only
    assert arm.commands == []
    execute_runtime = FakeRuntime()
    options = args(True)
    options.max_frames = 2
    run_camera(options, execute_runtime, lambda **kw: SimpleNamespace(**kw), FakeArm, FakeCamera)
    assert execute_runtime.warmups == 1
    arm = FakeArm.last
    assert arm is not None
    assert arm.commands == [[0.6, 0.0, 0.0, 0.0, 0.0, -45.0]]
    assert arm.halts == ["deployment ended"]


def test_policy_overshoot_is_projected_without_ending_camera_run(capsys):
    class OvershootRuntime(FakeRuntime):
        def predict(self, _rgb, _angles, _velocities, _previous_action):
            return Prediction((0, 0, 0.2, 0.2, 1), (3.0,) * 5, (45, 0, 0, 0, 0, -45))

    options = args(True)
    options.max_frames = 2
    run_camera(options, OvershootRuntime(), lambda **kw: SimpleNamespace(**kw), FakeArm, FakeCamera)
    assert FakeArm.last.commands == [[0.6, 0.0, 0.0, 0.0, 0.0, -45.0]]
    frames = [json.loads(line) for line in capsys.readouterr().out.splitlines() if '"frame"' in line]
    assert frames[-1]["target_clipped"] is True
    assert frames[-1]["policy_target_deg"][0] == 45
    assert frames[-1]["filtered_target_deg"][0] < 30
    assert frames[-1]["motion_state"] == "tracking"


def test_usb_camera_uses_v4l2_backend(monkeypatch):
    import deploy.run as deploy_run

    opened = []

    def capture(source, backend):
        opened.append((source, backend))
        return FakeCamera(source)

    monkeypatch.setattr(deploy_run.cv2, "VideoCapture", capture)
    run_camera(args(False), FakeRuntime(), lambda **kw: SimpleNamespace(**kw), FakeArm)
    assert opened == [("/dev/video0", cv2.CAP_V4L2)]


def test_browser_preview_serves_current_yolo_box():
    preview = BrowserPreview(0)
    url = preview.start()
    try:
        prediction = Prediction((0.0, 0.0, 0.5, 0.5, 1.0), (0.0,) * 5, (0.0,) * 5 + (-45.0,))
        preview.publish(torch.zeros((320, 320, 3), dtype=torch.uint8), prediction, "PREVIEW", 0.2)
        with urllib.request.urlopen(url) as response:
            assert b"myCobot hand preview" in response.read()
        with urllib.request.urlopen(url + "/frame.jpg") as response:
            assert response.headers["Content-Type"] == "image/jpeg"
            frame = cv2.imdecode(np.frombuffer(response.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
        assert frame.shape == (320, 320, 3)
        assert frame[80, 80, 1] > 150  # green corner of the Human hand box
    finally:
        preview.close()


def test_missing_hand_holds_but_keeps_camera_running(capsys):
    run_camera(args(True), FakeRuntime(False), lambda **kw: SimpleNamespace(**kw), FakeArm, FakeCamera)
    arm = FakeArm.last
    assert arm is not None
    assert arm.commands == []
    assert arm.halts == ["Human hand not detected"]
    frame = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert frame["motion_state"] == "waiting_for_hand"
    assert frame["sent_target_deg"] is None
    assert frame["hold_reason"] == "Human hand not detected"


def test_lost_hand_stops_and_resumes_after_two_detections(capsys):
    class IntermittentRuntime(FakeRuntime):
        def __init__(self, detections):
            super().__init__()
            self.detections = iter(detections)
            self.previous_actions = []

        def predict(self, _rgb, _angles, _velocities, previous_action):
            self.previous_actions.append(list(previous_action))
            detected = next(self.detections)
            return Prediction((0, 0, 0.2, 0.2, float(detected)), (0.1,) * 5, (5, 0, 0, 0, 0, -45))

    options = args(True)
    options.max_frames = 8
    runtime = IntermittentRuntime([True, True, False, True, False, True, True, False])
    run_camera(options, runtime, lambda **kw: SimpleNamespace(**kw), FakeArm, FakeCamera)
    arm = FakeArm.last
    assert arm.commands == [[0.6, 0.0, 0.0, 0.0, 0.0, -45.0]] * 2
    assert arm.halts == ["Human hand not detected"] * 2
    assert runtime.previous_actions[2] == [0.1] * 5
    assert runtime.previous_actions[3] == [0.0] * 5
    frames = [json.loads(line) for line in capsys.readouterr().out.splitlines() if '"frame"' in line]
    assert [frame["motion_state"] for frame in frames] == [
        "acquiring_hand",
        "tracking",
        "waiting_for_hand",
        "acquiring_hand",
        "waiting_for_hand",
        "acquiring_hand",
        "tracking",
        "waiting_for_hand",
    ]
    assert [frame["sent_target_deg"] is not None for frame in frames] == [
        False,
        True,
        False,
        False,
        False,
        False,
        True,
        False,
    ]


def test_slow_inference_sends_no_command_and_reports_timing(monkeypatch):
    import deploy.run as deploy_run

    class SlowRuntime(FakeRuntime):
        def predict(self, rgb, angles, velocities, previous_action):
            time.sleep(0.02)
            return super().predict(rgb, angles, velocities, previous_action)

    monkeypatch.setattr(deploy_run, "MAX_OBSERVATION_AGE_S", 0.001)
    with pytest.raises(UnsafeTarget, match="inference .*no command sent"):
        run_camera(args(True), SlowRuntime(), lambda **kw: SimpleNamespace(**kw), FakeArm, FakeCamera)
    arm = FakeArm.last
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


def test_direct_serial_accepts_small_j6_feedback_error_with_fixed_home_target(tmp_path):
    robot = FakeRobot()
    robot.angles[-1] = -45.7
    with ArmSession(
        ArmConfig(read_only=False), robot_factory=lambda _port, _baud: robot, lock_path=str(tmp_path / "arm.lock")
    ) as arm:
        reading = arm.read_joints()
        prediction = Prediction((0, 0, 0.2, 0.2, 1), (0,) * 5, (5, 0, 0, 0, 0, -45))
        target = plan_target(prediction, reading.angles)
        arm.gate.open("fresh hand")
        arm.command_joints(target, measured=reading)
    assert robot.calls[1] == ("send_angles", [1.0, 0.0, 0.0, 0.0, 0.0, -45.0], 10, True)


def test_direct_serial_stops_again_after_tracking_resumes(tmp_path):
    robot = FakeRobot()
    with ArmSession(
        ArmConfig(read_only=False), robot_factory=lambda _port, _baud: robot, lock_path=str(tmp_path / "arm.lock")
    ) as arm:
        for _ in range(2):
            reading = arm.read_joints()
            arm.gate.open("detected")
            arm.command_joints([1, 0, 0, 0, 0, -45], measured=reading)
            arm.halt("lost")
            arm.halt("still lost")
    assert [call[0] for call in robot.calls] == ["set_fresh_mode", "send_angles", "stop", "send_angles", "stop"]


def test_camera_loop_with_direct_serial_session(tmp_path):
    robot = FakeRobot()

    class DirectSession(ArmSession):
        def __init__(self, config):
            super().__init__(config, robot_factory=lambda _port, _baud: robot, lock_path=str(tmp_path / "arm.lock"))

    run_camera(args(False), FakeRuntime(), ArmConfig, DirectSession, FakeCamera)
    assert robot.calls == []
    robot.closed = False
    options = args(True)
    options.max_frames = 2
    run_camera(options, FakeRuntime(), ArmConfig, DirectSession, FakeCamera)
    assert robot.calls == [
        ("set_fresh_mode", 1),
        ("send_angles", [0.6, 0.0, 0.0, 0.0, 0.0, -45.0], 10, True),
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
    assert box_from_results([SimpleNamespace(boxes=boxes)], hand_class=0) == pytest.approx((0, 0, 0, 0, 0))


def test_two_hands_keep_the_nearby_box_when_confidence_order_changes():
    boxes = SimpleNamespace(
        xyxy=torch.tensor([[20.0, 20.0, 100.0, 100.0], [220.0, 20.0, 300.0, 100.0]]),
        cls=torch.tensor([0.0, 0.0]),
        conf=torch.tensor([0.3, 0.9]),
    )
    result = [SimpleNamespace(boxes=boxes)]
    assert box_from_results(result, hand_class=0, min_confidence=0.25)[0] > 0
    reference = (-0.625, -0.625, 0.25, 0.25, 1.0)
    assert box_from_results(result, hand_class=0, min_confidence=0.25, reference_box=reference) == pytest.approx(
        reference
    )


def test_runtime_filters_box_jumps_and_logs_subthreshold_confidence(monkeypatch):
    import deploy.runtime as deploy_runtime

    class FakeYOLO:
        names = {0: "Human hand"}

        def __init__(self, _weights):
            self.frames = iter(
                [
                    ([0, 0, 160, 160], 0.8),
                    ([160, 160, 320, 320], 0.8),
                    ([160, 160, 320, 320], 0.12),
                ]
            )

        def predict(self, **kwargs):
            assert kwargs["conf"] == 0.05
            assert kwargs["max_det"] == 5
            xyxy, confidence = next(self.frames)
            boxes = SimpleNamespace(
                xyxy=torch.tensor([xyxy], dtype=torch.float32),
                cls=torch.tensor([0.0]),
                conf=torch.tensor([confidence]),
            )
            return [SimpleNamespace(boxes=boxes)]

    class Actor:
        def __init__(self):
            self.observations = []

        def __call__(self, observation):
            self.observations.append(observation[0].tolist())
            return torch.zeros((1, 5))

    actor = Actor()
    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(YOLO=FakeYOLO))
    monkeypatch.setattr(deploy_runtime, "build_actor", lambda *_args: actor)
    runtime = HandPolicyRuntime("unused", yolo_weights="unused", yolo_confidence=0.25)
    image = torch.zeros((320, 320, 3), dtype=torch.uint8)
    predictions = [runtime.predict(image, [0, 0, 0, 0, 0, -45], [0] * 6, [0] * 5) for _ in range(3)]
    assert [item.confidence for item in predictions] == pytest.approx([0.8, 0.8, 0.12])
    assert predictions[1].box[:2] == pytest.approx((0.5, 0.5))
    assert predictions[1].policy_box[:2] == pytest.approx((-0.1, -0.1))
    assert actor.observations[1][:5] == pytest.approx(predictions[1].policy_box)
    assert predictions[2].box == (0, 0, 0, 0, 0)
    assert predictions[2].policy_box == (0, 0, 0, 0, 0)


def test_pose_labels_become_one_class_boxes_without_following_source_labels(tmp_path):
    pose = tmp_path / "pose"
    output = tmp_path / "detect"
    for split in ("train", "val"):
        images = pose / "images" / split
        labels = pose / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        (images / "hand.jpg").write_bytes(b"image bytes")
        (labels / "hand.txt").write_text("0 0.5 0.5 1.2 0.5 0.1 0.2 2\n")
    yaml = prepare_detection_dataset(pose, output)
    assert "0: Human hand" in yaml.read_text()
    assert (output / "labels/train/hand.txt").read_text() == "0 0.50000000 0.50000000 1.00000000 0.50000000\n"
    assert (output / "images/train/hand.jpg").samefile(pose / "images/train/hand.jpg")
    assert not (output / "images/train").is_symlink()


def test_deploy_source_parses_as_python_38():
    for name in ("runtime.py", "controller.py", "arm.py", "preview.py", "run.py", "train_hand_detector.py"):
        source = (Path(__file__).resolve().parents[1] / "deploy" / name).read_text()
        ast.parse(source, filename=name, feature_version=(3, 8))
