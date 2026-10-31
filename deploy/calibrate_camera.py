"""Calibrate a USB camera fixed next to the myCobot for the Isaac Lab external view.

Three read-only steps on the Jetson; none of them talks to the arm:

1. ``capture``: save checkerboard frames while the board is moved around the view.
2. ``intrinsics``: fit focal length, principal point, and lens distortion.
3. ``extrinsics``: with the board lying flat on the table at a measured spot,
   solve the camera pose in the robot base frame and write the JSON that
   ``arc_mycobot.vision.camera_calibration`` loads.

Robot base frame: origin on the table under the J1 axis, +x the arm's forward
direction at J1 = 0, +y to the robot's left, +z up.
"""

import argparse
import glob
import json
import math
import os
import time

import cv2
import numpy as np

from deploy.preview import BrowserPreview

MIN_INTRINSIC_IMAGES = 10
MIN_SAVE_INTERVAL_S = 1.0
MIN_BOARD_MOTION = 0.05
ROBOT_COLUMN_M = ((0.0, 0.0, 0.05), (0.0, 0.0, 0.25))
CORNER_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
PAGE = b"""<!doctype html>
<html lang="en"><meta charset="utf-8"><title>Camera calibration</title>
<style>body{background:#111;color:#eee;font:16px sans-serif;margin:24px}img{max-width:90vw}</style>
<h1>Camera calibration</h1><p>Colored corners: checkerboard found. Move the board slowly between saves.</p>
<img id="frame" alt="Waiting for camera frame">
<script>const frame=document.getElementById('frame');
function refresh(){frame.onload=()=>setTimeout(refresh,200);
frame.onerror=()=>setTimeout(refresh,500);
frame.src='/frame.jpg?t='+Date.now()}refresh()</script></html>"""


def parse_board(text):
    """``9x6`` inner corners (columns x rows); unequal sizes avoid a 90-degree ambiguity."""
    try:
        columns, rows = (int(value) for value in text.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("board must look like 9x6") from exc
    if columns < 3 or rows < 3 or columns == rows:
        raise argparse.ArgumentTypeError("board needs at least 3x3 inner corners and unequal sides")
    return columns, rows


def board_points(board, square_m):
    """Inner corners in the board frame, centred on the pattern, OpenCV order."""
    columns, rows = board
    points = np.zeros((rows * columns, 3), dtype=np.float64)
    for row in range(rows):
        for column in range(columns):
            points[row * columns + column, :2] = (
                (column - (columns - 1) / 2.0) * square_m,
                (row - (rows - 1) / 2.0) * square_m,
            )
    return points


def find_corners(gray, board):
    found, corners = cv2.findChessboardCorners(gray, board, flags=CORNER_FLAGS)
    if not found:
        return None
    return cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), SUBPIX_CRITERIA)


def calibrate_intrinsics(images, board, square_m):
    """Pinhole + 5 distortion coefficients from several views of the board."""
    object_points, image_points, size, used = [], [], None, []
    for path, image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if size is None:
            size = gray.shape[::-1]
        elif gray.shape[::-1] != size:
            raise ValueError(f"{path}: every image must have the size {size[0]}x{size[1]}")
        corners = find_corners(gray, board)
        if corners is not None:
            object_points.append(board_points(board, square_m).astype(np.float32))
            image_points.append(corners)
            used.append(str(path))
    if size is None or len(used) < MIN_INTRINSIC_IMAGES:
        raise ValueError(f"checkerboard found in {len(used)} images; need at least {MIN_INTRINSIC_IMAGES}")
    rms, matrix, distortion, _, _ = cv2.calibrateCamera(object_points, image_points, size, np.eye(3), np.zeros(5))
    return {
        "width": int(size[0]),
        "height": int(size[1]),
        "intrinsic_matrix": [float(value) for value in matrix.ravel()],
        "distortion": [float(value) for value in distortion.ravel()],
        "rms_px": float(rms),
        "images_used": used,
    }


def quaternion_wxyz(matrix):
    m = np.asarray(matrix, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = (0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = ((m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s)
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = ((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s)
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = ((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s)
    q = np.asarray(q) / np.linalg.norm(q)
    return tuple(float(value) for value in (q if q[0] >= 0 else -q))


def _project(points, rotation, translation, intrinsics):
    matrix = np.asarray(intrinsics["intrinsic_matrix"]).reshape(3, 3)
    camera = np.asarray(points) @ rotation.T + translation
    return camera[:, 2], camera[:, :2] / camera[:, 2:] @ matrix[:2, :2].T + matrix[:2, 2]


def camera_pose_from_board(corners, intrinsics, board, square_m, center_m, height_m):
    """Camera pose in the base frame from a flat board whose long side follows +x.

    The pattern is symmetric under a half turn, so OpenCV may number it from
    either end. Of the four axis assignments only one keeps the camera above
    the table and the robot column in front of the lens.
    """
    matrix = np.asarray(intrinsics["intrinsic_matrix"]).reshape(3, 3)
    distortion = np.asarray(intrinsics["distortion"])
    points = board_points(board, square_m)
    ok, rvec, tvec = cv2.solvePnP(points, corners, matrix, distortion, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        raise ValueError("solvePnP failed on the table checkerboard")
    rotation_cb = np.asarray(cv2.Rodrigues(rvec)[0], dtype=np.float64)
    translation_cb = tvec.ravel()
    projected, _ = cv2.projectPoints(points, rvec, tvec, matrix, distortion)
    residual = np.asarray(projected, dtype=np.float64) - np.asarray(corners, dtype=np.float64)
    reprojection = float(np.sqrt((residual**2).sum(axis=2)).mean())
    center = np.array([center_m[0], center_m[1], height_m])
    candidates = []
    for x_sign in (1.0, -1.0):
        for z_sign in (1.0, -1.0):
            x_axis = np.array([x_sign, 0.0, 0.0])
            z_axis = np.array([0.0, 0.0, z_sign])
            rotation_bb = np.stack((x_axis, np.cross(z_axis, x_axis), z_axis), axis=1)  # board -> base
            rotation = rotation_cb @ rotation_bb.T  # base -> camera
            translation = translation_cb - rotation @ center
            position = -rotation.T @ translation
            depth, pixels = _project(ROBOT_COLUMN_M, rotation, translation, intrinsics)
            visible = bool(
                (depth > 0).all()
                and (pixels[:, 0] >= 0).all()
                and (pixels[:, 0] < intrinsics["width"]).all()
                and (pixels[:, 1] >= 0).all()
                and (pixels[:, 1] < intrinsics["height"]).all()
            )
            if position[2] > height_m + 0.02 and visible:
                candidates.append((position, rotation.T))
    if len(candidates) != 1:
        raise ValueError(
            f"{len(candidates)} board orientations put the camera above the table with the robot in view; "
            "check --board-center-m and that the board's long side is parallel to the robot's +x"
        )
    position, camera_axes = candidates[0]
    return position, camera_axes, reprojection


def calibration_json(intrinsics, position, camera_axes, note):
    return {
        "width": intrinsics["width"],
        "height": intrinsics["height"],
        "intrinsic_matrix": intrinsics["intrinsic_matrix"],
        "distortion": intrinsics["distortion"],
        "position_m": [round(float(value), 5) for value in position],
        "quaternion_wxyz": [round(value, 6) for value in quaternion_wxyz(camera_axes)],
        "calibrated": True,
        "note": note,
    }


def draw_base_axes(image, intrinsics, position, camera_axes):
    """Base frame axes (x red, y green, z blue, 10 cm) and the J1 column for a visual check."""
    rotation = np.asarray(camera_axes).T
    translation = -rotation @ np.asarray(position)
    origin = np.zeros(3)
    points = [origin, (0.1, 0, 0), (0, 0.1, 0), (0, 0, 0.1), (0, 0, 0.44)]
    depth, pixels = _project(points, rotation, translation, intrinsics)
    out = image.copy()
    if (depth > 0).all():
        start = tuple(int(round(value)) for value in pixels[0])
        for pixel, color in zip(pixels[1:4], ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
            cv2.line(out, start, tuple(int(round(value)) for value in pixel), color, 3)
        cv2.line(out, start, tuple(int(round(value)) for value in pixels[4]), (0, 255, 255), 1)
    return out


def open_camera(device, width=None, height=None):
    camera = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"cannot open V4L2 camera {device}")
    if width and height:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return camera


def capture(args):
    os.makedirs(args.out, exist_ok=True)
    camera = open_camera(args.camera, args.width, args.height)
    preview = BrowserPreview(args.web_preview_port, PAGE) if args.web_preview_port is not None else None
    saved, last_saved_at, last_center = 0, 0.0, None
    try:
        if preview is not None:
            print(f"Calibration preview: {preview.start()}", flush=True)
        while saved < args.count:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("camera read failed")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, args.board, flags=CORNER_FLAGS | cv2.CALIB_CB_FAST_CHECK)
            annotated = frame.copy()
            if found:
                cv2.drawChessboardCorners(annotated, args.board, corners, found)
                center = corners.reshape(-1, 2).mean(axis=0) / np.array(gray.shape[::-1])
                moved = last_center is None or np.linalg.norm(center - last_center) > MIN_BOARD_MOTION
                if moved and time.monotonic() - last_saved_at > MIN_SAVE_INTERVAL_S:
                    path = os.path.join(args.out, f"frame_{saved:03d}.png")
                    cv2.imwrite(path, frame)
                    saved, last_saved_at, last_center = saved + 1, time.monotonic(), center
                    print(json.dumps({"saved": path, "count": saved, "size": list(gray.shape[::-1])}), flush=True)
            cv2.putText(
                annotated, f"saved {saved}/{args.count}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
            if preview is not None:
                preview.publish_bgr(annotated)
    except KeyboardInterrupt:
        print("Stopped by operator.", flush=True)
    finally:
        camera.release()
        if preview is not None:
            preview.close()


def intrinsics(args):
    paths = sorted(glob.glob(os.path.join(args.images, "*.png")) + glob.glob(os.path.join(args.images, "*.jpg")))
    images = [(path, cv2.imread(path, cv2.IMREAD_COLOR)) for path in paths]
    result = calibrate_intrinsics(
        [(path, image) for path, image in images if image is not None], args.board, args.square_mm / 1000.0
    )
    result["board"] = list(args.board)
    result["square_mm"] = args.square_mm
    with open(args.out, "w") as stream:
        json.dump(result, stream, indent=2)
    fx, _, cx, _, fy, cy = result["intrinsic_matrix"][:6]
    print(
        json.dumps(
            {
                "out": args.out,
                "images_used": len(result["images_used"]),
                "rms_px": round(result["rms_px"], 3),
                "horizontal_fov_deg": round(math.degrees(2 * math.atan(result["width"] / (2 * fx))), 2),
                "fx_fy_cx_cy": [round(fx, 1), round(fy, 1), round(cx, 1), round(cy, 1)],
            }
        ),
        flush=True,
    )
    if result["rms_px"] > 1.0:
        print("WARNING: RMS above 1 px; recapture with sharper, more varied board views.", flush=True)


def extrinsics(args):
    with open(args.intrinsics) as stream:
        lens = json.load(stream)
    if args.image is not None:
        frame = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if frame is None:
            raise FileNotFoundError(args.image)
    else:
        camera = open_camera(args.camera, lens["width"], lens["height"])
        try:
            for _ in range(30):
                ok, frame = camera.read()
            if not ok:
                raise RuntimeError("camera read failed")
        finally:
            camera.release()
    if frame.shape[1::-1] != (lens["width"], lens["height"]):
        raise ValueError(f"frame is {frame.shape[1]}x{frame.shape[0]}, intrinsics are {lens['width']}x{lens['height']}")
    corners = find_corners(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), args.board)
    if corners is None:
        raise ValueError("checkerboard not found; the whole board must be visible and flat on the table")
    height_m = args.board_thickness_mm / 1000.0
    position, axes, reprojection = camera_pose_from_board(
        corners, lens, args.board, args.square_mm / 1000.0, args.board_center_m, height_m
    )
    note = (
        f"{time.strftime('%Y-%m-%d')} deploy/calibrate_camera.py: {len(lens['images_used'])} intrinsic views, "
        f"RMS {lens['rms_px']:.3f} px; table board {args.board[0]}x{args.board[1]} @ {args.square_mm} mm centred at "
        f"({args.board_center_m[0]:.3f}, {args.board_center_m[1]:.3f}) m, reprojection {reprojection:.2f} px"
    )
    result = calibration_json(lens, position, axes, note)
    with open(args.out, "w") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    stem = os.path.splitext(args.out)[0]
    cv2.imwrite(stem + "_board.png", frame)
    cv2.imwrite(stem + "_check.png", draw_base_axes(frame, lens, position, axes))
    print(
        json.dumps(
            {
                "out": args.out,
                "position_m": result["position_m"],
                "quaternion_wxyz": result["quaternion_wxyz"],
                "reprojection_px": round(reprojection, 3),
                "check_image": stem + "_check.png",
            }
        ),
        flush=True,
    )


def cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    grab = commands.add_parser("capture", help="save checkerboard frames for intrinsics")
    grab.add_argument("--camera", default="/dev/video0")
    grab.add_argument("--out", required=True, help="directory for PNG frames")
    grab.add_argument("--board", type=parse_board, default=(9, 6), help="inner corners, e.g. 9x6")
    grab.add_argument("--count", type=int, default=25)
    grab.add_argument("--width", type=int, help="request a stream width; default keeps the camera default")
    grab.add_argument("--height", type=int)
    grab.add_argument("--web-preview-port", type=int, help="show detected corners in a local browser")

    lens = commands.add_parser("intrinsics", help="fit the lens from saved frames")
    lens.add_argument("--images", required=True)
    lens.add_argument("--board", type=parse_board, default=(9, 6))
    lens.add_argument("--square-mm", type=float, required=True, help="measured printed square size")
    lens.add_argument("--out", required=True)

    pose = commands.add_parser("extrinsics", help="camera pose from a board lying flat on the table")
    pose.add_argument("--intrinsics", required=True)
    source = pose.add_mutually_exclusive_group(required=True)
    source.add_argument("--camera", help="grab one frame from this device")
    source.add_argument("--image", help="use a saved frame instead")
    pose.add_argument("--board", type=parse_board, default=(9, 6))
    pose.add_argument("--square-mm", type=float, required=True)
    pose.add_argument("--board-center-m", type=float, nargs=2, required=True, metavar=("X", "Y"))
    pose.add_argument("--board-thickness-mm", type=float, default=0.0, help="board surface above the robot base")
    pose.add_argument("--out", required=True, help="camera JSON for Isaac Lab")

    args = parser.parse_args()
    if getattr(args, "count", 1) < 1:
        parser.error("--count must be positive")
    if getattr(args, "square_mm", 1.0) <= 0:
        parser.error("--square-mm must be positive")
    {"capture": capture, "intrinsics": intrinsics, "extrinsics": extrinsics}[args.command](args)


if __name__ == "__main__":
    cli()
