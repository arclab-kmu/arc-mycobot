"""Loopback browser view of the camera frame and the YOLO hand box."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

PAGE = b"""<!doctype html>
<html lang="en"><meta charset="utf-8"><title>myCobot hand preview</title>
<style>body{background:#111;color:#eee;font:16px sans-serif;margin:24px}
img{max-width:min(90vw,720px);image-rendering:auto}p{max-width:720px}</style>
<h1>myCobot hand preview</h1>
<p>Green: YOLO Human hand box. White: camera center. STOPPED means restart is required before any motion.</p>
<img id="frame" alt="Waiting for camera frame">
<script>const frame=document.getElementById('frame');
function refresh(){frame.onload=()=>setTimeout(refresh,200);
frame.onerror=()=>setTimeout(refresh,500);
frame.src='/frame.jpg?t='+Date.now()}refresh()</script></html>"""


def annotated_frame(rgb, prediction, mode, observation_age_s):
    """Draw the class-267 box on the exact square crop passed to YOLO."""
    frame = cv2.cvtColor(rgb.numpy(), cv2.COLOR_RGB2BGR)
    height, width = frame.shape[:2]
    cv2.drawMarker(frame, (width // 2, height // 2), (255, 255, 255), cv2.MARKER_CROSS, 16, 2)
    if prediction.box[4] == 1.0:
        cx, cy, box_width, box_height = prediction.box[:4]
        x1 = int(((cx + 1.0) - box_width) * width / 2.0)
        y1 = int(((cy + 1.0) - box_height) * height / 2.0)
        x2 = int(((cx + 1.0) + box_width) * width / 2.0)
        y2 = int(((cy + 1.0) + box_height) * height / 2.0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = "YOLO: Human hand"
    else:
        label = "YOLO: no Human hand"
    cv2.putText(frame, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(
        frame,
        f"{mode}  age {observation_age_s:.2f}s",
        (8, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
    )
    return frame


class BrowserPreview:
    """Serve only the latest annotated JPEG on 127.0.0.1."""

    def __init__(self, port):
        self.port = port
        self._jpeg = None
        self._lock = threading.Lock()
        self._server = None
        self._thread = None

    def start(self):
        preview = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path == "/":
                    body, content_type, status = PAGE, "text/html; charset=utf-8", 200
                elif path == "/frame.jpg":
                    with preview._lock:
                        body = preview._jpeg
                    content_type = "image/jpeg"
                    status = 200 if body is not None else 503
                    if body is None:
                        body = b"waiting for frame"
                else:
                    body, content_type, status = b"not found", "text/plain", 404
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format, *_args):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
        self._thread.start()
        return f"http://127.0.0.1:{self._server.server_port}"

    def publish(self, rgb, prediction, mode, observation_age_s):
        frame = annotated_frame(rgb, prediction, mode, observation_age_s)
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise RuntimeError("could not encode browser preview frame")
        with self._lock:
            self._jpeg = jpeg.tobytes()

    def close(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._thread.join(timeout=1.0)
            self._server = None
            self._thread = None
