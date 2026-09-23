"""One process per camera: capture -> detect -> solve -> send result to the publisher.

A process (not a thread) per camera guarantees the cameras run in parallel on
separate CPU cores, independent of Python's GIL.

Inside each process:
  - a grabber thread keeps only the newest frame (no stale backlog) and applies
    exposure/gain changes,
  - an MJPEG server (stream_server.py) serves annotated + raw streams, encoding
    only while someone is watching,
  - live settings arrive from the main process over a control queue.
"""

from __future__ import annotations

import dataclasses
import queue
import sys
import threading
import time
import traceback

import cv2
import numpy as np

from camera_controls import apply_exposure_gain
from config import CameraConfig, load_field_layout, robot_to_camera_transform
from pose_solver import DetectorSettings, PoseSolver, SolverSettings, make_detector
from stream_server import MjpegStreamServer, annotate

STATS_PERIOD_S = 1.0
DETECTOR_KEYS = {f.name for f in dataclasses.fields(DetectorSettings)}
SOLVER_KEYS = {f.name for f in dataclasses.fields(SolverSettings)}


def is_video_file(device) -> bool:
    """Camera = an index (0, "1") or a /dev path. Anything else is treated as a video file."""
    if isinstance(device, int):
        return False
    return not device.startswith("/dev/") and not device.isdigit()


def open_camera(device) -> cv2.VideoCapture:
    """Open a camera with the right backend for this OS."""
    if isinstance(device, str) and device.isdigit():
        device = int(device)
    if sys.platform == "darwin":
        return cv2.VideoCapture(device, cv2.CAP_AVFOUNDATION)  # macOS: use indices (0, 1, ...)
    if sys.platform.startswith("linux"):
        return cv2.VideoCapture(device, cv2.CAP_V4L2)
    return cv2.VideoCapture(device)  # Windows: let OpenCV choose


class LatestFrameGrabber(threading.Thread):
    def __init__(self, cam: CameraConfig):
        super().__init__(daemon=True)
        self.cam = cam
        self.cond = threading.Condition()
        self.frame = None
        self.capture_time = 0.0
        self.seq = 0
        self.connected = False
        self.running = True
        self._controls = (cam.exposure, cam.gain)
        self._controls_dirty = True
        self._ctl_lock = threading.Lock()

    def set_controls(self, exposure, gain):
        with self._ctl_lock:
            self._controls = (exposure, gain)
            self._controls_dirty = True

    def _log(self, msg):
        print(f"[{self.cam.name}] {msg}", flush=True)

    def _open(self) -> cv2.VideoCapture | None:
        c = self.cam
        is_file = is_video_file(c.device)
        cap = cv2.VideoCapture(c.device) if is_file else open_camera(c.device)
        if not cap.isOpened():
            return None
        if not is_file:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*c.fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, c.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c.height)
            cap.set(cv2.CAP_PROP_FPS, c.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if (w, h) != (c.width, c.height):
                self._log(f"WARNING: asked for {c.width}x{c.height}, camera gave {w}x{h}. "
                          f"Calibration won't match; fix width/height in config.")
        with self._ctl_lock:
            self._controls_dirty = True  # re-apply controls after every (re)open
        return cap

    def run(self):
        c = self.cam
        is_file = is_video_file(c.device)
        frame_period = 1.0 / c.fps if is_file else 0.0
        cap = None
        while self.running:
            if cap is None:
                cap = self._open()
                if cap is None:
                    self.connected = False
                    time.sleep(1.0)
                    continue
                self.connected = True
            if not is_file:
                with self._ctl_lock:
                    dirty, (exposure, gain) = self._controls_dirty, self._controls
                    self._controls_dirty = False
                if dirty:
                    apply_exposure_gain(cap, c.device, exposure, gain, self._log)
            ok = cap.grab()
            t = time.monotonic()  # capture time; shared clock with the publisher process
            if ok:
                ok, frame = cap.retrieve()
            if not ok:
                if is_file and c.loop_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self._log("frame grab failed, reconnecting")
                cap.release()
                cap = None
                self.connected = False
                time.sleep(0.5)
                continue
            with self.cond:
                self.frame, self.capture_time, self.seq = frame, t, self.seq + 1
                self.cond.notify()
            if frame_period:
                time.sleep(frame_period)  # play test videos at real-time speed

    def wait_for_frame(self, last_seq: int, timeout: float = 0.5):
        with self.cond:
            if self.seq == last_seq:
                self.cond.wait(timeout)
            if self.seq == last_seq:
                return None
            return self.frame, self.capture_time, self.seq


def run_camera(cam: CameraConfig, layout_name: str, det_settings: DetectorSettings,
               solver_settings: SolverSettings, stream_cfg: dict, out_queue, ctrl_queue) -> None:
    try:
        _run(cam, layout_name, det_settings, solver_settings, stream_cfg, out_queue, ctrl_queue)
    except Exception:
        print(f"[{cam.name}] crashed:\n{traceback.format_exc()}", flush=True)
        raise


def _run(cam, layout_name, det_settings, solver_settings, stream_cfg, out_queue, ctrl_queue):
    layout = load_field_layout(layout_name)
    solver = PoseSolver(layout, solver_settings)
    detector = make_detector(det_settings)
    K = np.array(cam.camera_matrix, dtype=np.float64)
    D = np.array(cam.dist_coeffs, dtype=np.float64)
    r2c = robot_to_camera_transform(cam.robot_to_camera)
    std_dev_factor = cam.std_dev_factor
    exposure, gain = cam.exposure, cam.gain

    stream = None
    stream_max_fps = float(stream_cfg.get("max_fps", 30))
    if stream_cfg.get("enabled", True):
        stream = MjpegStreamServer(cam.name, cam.stream_port, int(stream_cfg.get("jpeg_quality", 80)))
        stream.start()

    grabber = LatestFrameGrabber(cam)
    grabber.start()
    print(f"[{cam.name}] started on {cam.device}"
          + (f", stream at http://<this-ip>:{cam.stream_port}/" if stream else ""), flush=True)

    last_seq = 0
    frames = tagged = 0
    proc_total = 0.0
    last_stats = time.monotonic()
    last_stream = 0.0
    last_reject = ""
    fps_shown = 0.0
    proc_shown = 0.0
    encoded_at_stats = 0

    while True:
        # ---- live settings from the main process ----
        try:
            while True:
                changes = ctrl_queue.get_nowait()
                det_changes = {k: v for k, v in changes.items() if k in DETECTOR_KEYS}
                if det_changes:
                    det_settings = dataclasses.replace(det_settings, **det_changes)
                    detector = make_detector(det_settings)
                for k, v in changes.items():
                    if k in SOLVER_KEYS and k != "tag_size_m":
                        setattr(solver.s, k, v)
                if "std_dev_factor" in changes:
                    std_dev_factor = changes["std_dev_factor"]
                if "stream_max_fps" in changes:
                    stream_max_fps = changes["stream_max_fps"]
                if "exposure" in changes or "gain" in changes:
                    exposure = changes.get("exposure", exposure)
                    gain = changes.get("gain", gain)
                    grabber.set_controls(exposure, gain)
        except queue.Empty:
            pass

        got = grabber.wait_for_frame(last_seq)
        now = time.monotonic()
        if now - last_stats >= STATS_PERIOD_S:
            dt = now - last_stats
            fps_shown = frames / dt
            proc_shown = (proc_total / frames * 1000) if frames else 0.0
            enc = stream.encoded_count if stream else 0
            viewers = sum(ch.clients for ch in stream.channels.values()) if stream else 0
            out_queue.put(("stats", cam.id, grabber.connected, fps_shown, proc_shown, tagged / dt,
                           (enc - encoded_at_stats) / dt, viewers, last_reject))
            encoded_at_stats = enc
            frames = tagged = 0
            proc_total = 0.0
            last_stats = now
        if got is None:
            continue
        frame, capture_time, last_seq = got

        t0 = time.monotonic()
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = detector.detect(gray)
        res = solver.solve(detections, K, D, r2c, std_dev_factor) if detections else None
        proc_total += time.monotonic() - t0
        frames += 1
        if res is None:
            if solver.last_reject:
                last_reject = solver.last_reject
        else:
            tagged += 1
            p = res.robot_pose
            q = p.rotation().getQuaternion()
            out_queue.put((
                "obs", cam.id, capture_time,
                p.X(), p.Y(), p.Z(), q.W(), q.X(), q.Y(), q.Z(),
                res.xy_std_dev, res.theta_std_dev, len(res.tag_ids),
                res.avg_tag_distance, res.ambiguity, res.tag_mask,
            ))

        # ---- stream: only does work while someone is watching ----
        if stream and (stream.wants("stream") or stream.wants("raw")):
            if stream_max_fps <= 0 or now - last_stream >= 1.0 / stream_max_fps:
                last_stream = now
                if stream.wants("raw"):
                    stream.submit("raw", frame)
                if stream.wants("stream"):
                    exp_txt = "auto" if exposure is None else f"{exposure:g}"
                    hud = [f"{cam.name}  {frame.shape[1]}x{frame.shape[0]}  {fps_shown:.0f} fps  "
                           f"detect {proc_shown:.1f} ms  exposure {exp_txt}  "
                           f"decimate {det_settings.quad_decimate:g}"]
                    used = set(res.tag_ids) if res else set()
                    reject = None if res else solver.last_reject
                    stream.submit("stream", annotate(frame, detections, used, res, reject, hud))
