"""Per-camera MJPEG server with two streams:

  /stream.mjpg   annotated (tag outlines, IDs, reject reasons, HUD)
  /raw.mjpg      untouched camera frames (focus tools, calibration, sanity checks)
  /              simple page showing both

Frames are encoded ONLY while someone is watching that stream -- an unwatched
stream costs nothing. Encoding happens on a separate thread (cv2.imencode releases
the GIL), and each frame is encoded once no matter how many viewers there are.
Full resolution; max_fps caps how often a new frame is encoded while watched.
"""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

_PAGE = """<!doctype html><html><head><title>{name}</title>
<style>body{{background:#111;color:#ddd;font-family:sans-serif;margin:12px}}
img{{max-width:100%;display:block;margin:6px 0 18px}}a{{color:#8cf}}</style></head>
<body><h3>{name}</h3>
<div>Annotated &middot; <a href="/stream.mjpg">/stream.mjpg</a></div><img src="/stream.mjpg">
<div>Raw &middot; <a href="/raw.mjpg">/raw.mjpg</a></div><img src="/raw.mjpg">
</body></html>"""


class _Channel:
    def __init__(self):
        self.cond = threading.Condition()
        self.jpeg: bytes | None = None
        self.seq = 0
        self.clients = 0
        self.pending = None  # newest un-encoded frame


class MjpegStreamServer:
    def __init__(self, name: str, port: int, jpeg_quality: int = 80):
        self.name = name
        self.port = port
        self.quality = jpeg_quality
        self.channels = {"stream": _Channel(), "raw": _Channel()}
        self._work = threading.Condition()
        self.encoded_count = 0

    # ---- camera side -------------------------------------------------------

    def wants(self, channel: str) -> bool:
        return self.channels[channel].clients > 0

    def submit(self, channel: str, frame) -> None:
        """Hand over a frame to encode. Drops the previous one if the encoder is behind."""
        ch = self.channels[channel]
        with self._work:
            ch.pending = frame
            self._work.notify()

    # ---- threads -----------------------------------------------------------

    def start(self) -> None:
        threading.Thread(target=self._encoder, daemon=True, name=f"{self.name}-encoder").start()
        server = ThreadingHTTPServer(("", self.port), self._make_handler())
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, daemon=True, name=f"{self.name}-http").start()

    def _encoder(self) -> None:
        params = [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        while True:
            with self._work:
                while not any(ch.pending is not None for ch in self.channels.values()):
                    self._work.wait()
                jobs = [(ch, ch.pending) for ch in self.channels.values() if ch.pending is not None]
                for ch, _ in jobs:
                    ch.pending = None
            for ch, frame in jobs:
                ok, buf = cv2.imencode(".jpg", frame, params)
                if not ok:
                    continue
                self.encoded_count += 1
                with ch.cond:
                    ch.jpeg = buf.tobytes()
                    ch.seq += 1
                    ch.cond.notify_all()

    def _make_handler(self):
        srv = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path = self.path.split("?")[0]
                if path in ("/", "/index.html"):
                    body = _PAGE.format(name=srv.name).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if path in ("/stream.mjpg", "/stream"):
                    self._serve(srv.channels["stream"])
                elif path in ("/raw.mjpg", "/raw"):
                    self._serve(srv.channels["raw"])
                else:
                    self.send_error(404)

            def _serve(self, ch: _Channel):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                with ch.cond:
                    ch.clients += 1
                last = -1
                try:
                    while True:
                        with ch.cond:
                            if ch.seq == last:
                                ch.cond.wait(timeout=2.0)
                            if ch.seq == last or ch.jpeg is None:
                                continue
                            data, last = ch.jpeg, ch.seq
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode())
                        self.wfile.write(data)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    with ch.cond:
                        ch.clients -= 1

            def log_message(self, *args):
                pass

        return Handler


def annotate(frame, detections, used_ids: set[int], result, reject: str | None, hud: list[str]):
    """Draws tag outlines + IDs and a status HUD onto a copy of the frame."""
    img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
    h, w = img.shape[:2]
    scale = max(0.5, w / 1280.0)
    thick = max(1, int(round(2 * scale)))
    font = cv2.FONT_HERSHEY_SIMPLEX

    for d in detections:
        pts = [(int(d.getCorner(k).x), int(d.getCorner(k).y)) for k in range(4)]
        color = (0, 220, 0) if d.getId() in used_ids else (0, 165, 255)  # green used, orange filtered
        for k in range(4):
            cv2.line(img, pts[k], pts[(k + 1) % 4], color, thick, cv2.LINE_AA)
        cv2.circle(img, pts[0], thick * 2, (0, 0, 255), -1)  # corner 0 (bottom-left)
        label = f"ID {d.getId()}"
        fs = 0.6 * scale
        (tw, th), base = cv2.getTextSize(label, font, fs, max(1, thick - 1))
        x = int(min(p[0] for p in pts))
        y = int(min(p[1] for p in pts)) - int(6 * scale)
        y = max(y, th + base + 2)
        cv2.rectangle(img, (x, y - th - base), (x + tw + 6, y + base), (0, 0, 0), -1)
        cv2.putText(img, label, (x + 3, y), font, fs, color, max(1, thick - 1), cv2.LINE_AA)

    lines = list(hud)
    if result is not None:
        p = result.robot_pose
        lines.append(f"pose x={p.X():.2f} y={p.Y():.2f} th={p.rotation().Z() * 57.2958:.1f}deg "
                     f"| tags {len(result.tag_ids)} | {result.avg_tag_distance:.1f} m")
    elif reject:
        lines.append(f"REJECTED: {reject}")

    line_h = int(26 * scale)
    box_h = line_h * len(lines) + int(10 * scale)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    for i, text in enumerate(lines):
        color = (80, 80, 255) if text.startswith("REJECTED") else (240, 240, 240)
        cv2.putText(img, text, (int(8 * scale), line_h * (i + 1)), font, 0.6 * scale, color,
                    max(1, thick - 1), cv2.LINE_AA)
    return img
