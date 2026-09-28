"""One process per camera: capture -> detect -> solve -> send result to the publisher.

A process (not a thread) per camera guarantees the cameras run in parallel on
separate CPU cores, independent of Python's GIL.

Inside each process:
  - a grabber thread keeps only the newest frame (no stale backlog) and applies
    exposure/gain and resolution changes,
  - an MJPEG server (stream_server.py) serves annotated + raw streams, encoding
    only while someone is watching, plus the settings/calibration page,
  - live settings arrive from the main process over a control queue.

A camera with no calibration for its current resolution keeps running: it streams
and detects tags but publishes no poses until it's calibrated from its page.
"""

from __future__ import annotations

import dataclasses
import os
import queue
import sys
import threading
import time
import traceback

import cv2
import numpy as np

from calibration_session import CalibrationSession, board_png
from camera_controls import apply_exposure_gain, control_ranges, list_modes
from config import (CameraConfig, calibration_path, find_calibration, load_field_layout,
                    robot_to_camera_transform)
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
        self._reopen = False
        self._ctl_lock = threading.Lock()

    def set_controls(self, exposure, gain):
        with self._ctl_lock:
            self._controls = (exposure, gain)
            self._controls_dirty = True

    def set_resolution(self, width: int, height: int):
        with self._ctl_lock:
            self.cam = dataclasses.replace(self.cam, width=width, height=height)
            self._reopen = True

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
            # Two buffers: the camera fills one while we decode the other. With one, the
            # driver has nowhere to put the next frame during decode and every other frame
            # is lost (50 fps -> 25). The grabber reads continuously, so at most one frame
            # waits, and capture time comes from the driver's timestamp anyway (run()).
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if (w, h) != (c.width, c.height):
                self._log(f"WARNING: asked for {c.width}x{c.height}, camera gave {w}x{h}. "
                          f"Pick a resolution the camera lists on its settings page.")
        with self._ctl_lock:
            self._controls_dirty = True  # re-apply controls after every (re)open
        return cap

    def run(self):
        c = self.cam
        is_file = is_video_file(c.device)
        frame_period = 1.0 / c.fps if is_file else 0.0
        cap = None
        while self.running:
            with self._ctl_lock:
                reopen, self._reopen = self._reopen, False
            if reopen and cap is not None:
                cap.release()
                cap = None
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
            if ok and not is_file:
                # Prefer the driver's timestamp of when the frame was captured (on Linux
                # V4L2 it's CLOCK_MONOTONIC, the same clock as time.monotonic()), so a frame
                # that waited in the buffer still gets its true capture time. Only trusted
                # if it's plausible; otherwise keep the grab time.
                ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if 0.0 <= t - ts < 0.25:
                    t = ts
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
            if is_file and (frame.shape[1], frame.shape[0]) != (self.cam.width, self.cam.height):
                frame = cv2.resize(frame, (self.cam.width, self.cam.height))  # test videos follow resolution
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
               solver_settings: SolverSettings, stream_cfg: dict, out_queue, ctrl_queue,
               peers: list | None = None) -> None:
    try:
        _run(cam, layout_name, det_settings, solver_settings, stream_cfg, out_queue, ctrl_queue,
             peers or [])
    except Exception:
        print(f"[{cam.name}] crashed:\n{traceback.format_exc()}", flush=True)
        raise


def _board_from_values(values: dict) -> dict:
    return {
        "squares_x": values["board_squares_x"],
        "squares_y": values["board_squares_y"],
        "square_length_m": values["board_square_length_m"],
        "marker_length_m": values["board_marker_length_m"],
        "dict": values["board_dict"],
        "legacy": values["board_legacy"],
    }


def _run(cam, layout_name, det_settings, solver_settings, stream_cfg, out_queue, ctrl_queue, peers):
    layout = load_field_layout(layout_name)
    solver = PoseSolver(layout, solver_settings)
    detector = make_detector(det_settings)
    r2c = robot_to_camera_transform(cam.robot_to_camera)
    std_dev_factor = cam.std_dev_factor
    exposure, gain = cam.exposure, cam.gain

    # Calibration for the resolution frames actually arrive at (None = uncalibrated).
    K = D = None
    calib_info = None
    active_res = None

    def load_calibration_for(w, h):
        nonlocal K, D, calib_info, last_reject
        last_reject = ""  # drop a stale "not calibrated" (or other) reason
        found = find_calibration(cam, w, h)
        if found:
            K = np.array(found["K"], dtype=np.float64)
            D = np.array(found["D"], dtype=np.float64)
            calib_info = {"file": os.path.basename(found["path"]), "rms": found["rms"],
                          "calibrated_at": found["calibrated_at"]}
            print(f"[{cam.name}] {w}x{h}: using calibration {found['path']}", flush=True)
        else:
            K = D = None
            calib_info = None
            print(f"[{cam.name}] {w}x{h}: NOT CALIBRATED (no poses until you calibrate "
                  f"from http://<this-ip>:{cam.stream_port}/)", flush=True)

    calib: CalibrationSession | None = None
    calib_msg = ""
    calib_cmds: queue.Queue = queue.Queue()  # from the web page's HTTP threads

    # What the web settings page shows. Whole dicts are replaced (never mutated) so the
    # HTTP threads can read them without a lock.
    ui = {"values": {}, "stats": {}, "ranges": None, "modes": None}

    def web_state():
        if ui["ranges"] is None:
            ui["ranges"] = control_ranges(cam.device)
        if ui["modes"] is None:
            ui["modes"] = list_modes(cam.device, cam.fourcc)
        session = calib
        try:
            cal_state = session.status() if session else None
        except Exception:  # raced with the camera loop; next poll gets it
            cal_state = None
        return {"name": cam.name, "cameras": peers, "values": ui["values"], "stats": ui["stats"],
                "ranges": ui["ranges"], "modes": ui["modes"], "calibration": cal_state,
                "calib_info": calib_info, "calib_msg": calib_msg}

    def web_action(kind, data=None):
        if kind == "calib":
            calib_cmds.put(data or {})
            return
        # The main process owns settings (NT + save); it sends the result back via ctrl_queue.
        out_queue.put(("set", cam.id, data) if kind == "set" else ("save",))

    def board_image():
        return "image/png", board_png(_board_from_values(ui["values"]))

    stream = None
    stream_max_fps = float(stream_cfg.get("max_fps", 30))
    if stream_cfg.get("enabled", True):
        stream = MjpegStreamServer(cam.name, cam.stream_port, int(stream_cfg.get("jpeg_quality", 80)),
                                   web_state, web_action, {"/api/board.png": board_image})
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
    seq_at_stats = 0
    last_res = None

    while True:
        # ---- live settings from the main process ----
        try:
            while True:
                changes = ctrl_queue.get_nowait()
                ui["values"] = {**ui["values"], **changes}
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
                if "resolution" in changes:
                    w, h = (int(x) for x in changes["resolution"].split("x"))
                    if (w, h) != (cam.width, cam.height):
                        cam = dataclasses.replace(cam, width=w, height=h)
                        grabber.set_resolution(w, h)
                        if calib is not None:
                            calib = None
                            calib_msg = "calibration cancelled: resolution changed"
                        print(f"[{cam.name}] switching to {w}x{h}", flush=True)
        except queue.Empty:
            pass

        # ---- calibration commands from the web page ----
        try:
            while True:
                cmd = calib_cmds.get_nowait()
                op = cmd.get("op")
                try:
                    if op == "start":
                        size = active_res or (cam.width, cam.height)
                        calib = CalibrationSession(_board_from_values(ui["values"]), size,
                                                   bool(cmd.get("wide_lens")))
                        calib_msg = ""
                    elif op == "stop":
                        calib = None
                        calib_msg = "calibration cancelled"
                    elif calib is None:
                        pass
                    elif op == "capture":
                        calib.request_capture()
                    elif op == "undo":
                        calib.undo()
                    elif op == "clear":
                        calib.clear()
                    elif op == "auto":
                        calib.auto = bool(cmd.get("on"))
                    elif op == "calibrate":
                        calib.start_calibration()
                    elif op == "discard":
                        calib.discard_result()
                    elif op == "preview":
                        calib.preview = bool(cmd.get("on"))
                    elif op == "save":
                        path = calibration_path(cam, calib.w, calib.h)
                        saved = calib.save(path)
                        calib = None
                        calib_msg = f"saved {os.path.basename(path)} (RMS {saved['rms']:.3f} px)"
                        print(f"[{cam.name}] {calib_msg}", flush=True)
                        active_res = None  # reload on the next frame
                except Exception as e:
                    calib_msg = f"{op} failed: {e}"
                    print(f"[{cam.name}] calibration {calib_msg}", flush=True)
        except queue.Empty:
            pass

        got = grabber.wait_for_frame(last_seq)
        now = time.monotonic()
        if now - last_stats >= STATS_PERIOD_S:
            dt = now - last_stats
            fps_shown = frames / dt
            cam_seq = grabber.seq  # frames the camera delivered (processed or not)
            cam_fps = (cam_seq - seq_at_stats) / dt
            seq_at_stats = cam_seq
            proc_shown = (proc_total / frames * 1000) if frames else 0.0
            enc = stream.encoded_count if stream else 0
            viewers = sum(ch.clients for ch in stream.channels.values()) if stream else 0
            out_queue.put(("stats", cam.id, grabber.connected, fps_shown, proc_shown, tagged / dt,
                           (enc - encoded_at_stats) / dt, viewers, last_reject))
            encoded_at_stats = enc
            pose_txt = ""
            if last_res is not None:
                p = last_res.robot_pose
                pose_txt = (f"x={p.X():.2f} y={p.Y():.2f} th={p.rotation().Z() * 57.2958:.1f}deg, "
                            f"tags {sorted(last_res.tag_ids)}")
            ui["stats"] = {"connected": grabber.connected, "fps": fps_shown, "processMs": proc_shown,
                           "tagFps": tagged / dt, "lastReject": last_reject, "pose": pose_txt,
                           "cameraFps": cam_fps,
                           "resolution": f"{active_res[0]}x{active_res[1]}" if active_res else None,
                           "calibrated": K is not None}
            last_res = None  # only show a pose accepted during this stats period
            frames = tagged = 0
            proc_total = 0.0
            last_stats = now
        if got is None:
            continue
        frame, capture_time, last_seq = got

        res_now = (frame.shape[1], frame.shape[0])
        if res_now != active_res:
            active_res = res_now
            load_calibration_for(*res_now)

        t0 = time.monotonic()
        gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        want_stream = stream and (stream.wants("stream") or stream.wants("raw")) and (
            stream_max_fps <= 0 or now - last_stream >= 1.0 / stream_max_fps)

        # ---- calibration mode: no pose pipeline, just the calibration session ----
        if calib is not None and (calib.w, calib.h) == res_now:
            det = calib.process(gray, now)
            proc_total += time.monotonic() - t0
            frames += 1
            if want_stream:
                last_stream = now
                if stream.wants("raw"):
                    stream.submit("raw", frame)
                if stream.wants("stream"):
                    stream.submit("stream", calib.draw(frame, det, now))
            continue

        detections = detector.detect(gray)
        res = None
        if K is None:
            reject = f"NOT CALIBRATED at {res_now[0]}x{res_now[1]} - calibrate on this page"
            last_reject = reject
        elif detections:
            res = solver.solve(detections, K, D, r2c, std_dev_factor)
            reject = None if res else solver.last_reject
        else:
            reject = None
        proc_total += time.monotonic() - t0
        frames += 1
        if res is None:
            if K is not None and solver.last_reject:
                last_reject = solver.last_reject
        else:
            tagged += 1
            last_res = res
            p = res.robot_pose
            q = p.rotation().getQuaternion()
            out_queue.put((
                "obs", cam.id, capture_time,
                p.X(), p.Y(), p.Z(), q.W(), q.X(), q.Y(), q.Z(),
                res.std_dev_factor, len(res.tag_ids),
                res.avg_tag_distance, res.ambiguity, res.tag_mask,
            ))

        # ---- stream: only does work while someone is watching ----
        if want_stream:
            last_stream = now
            if stream.wants("raw"):
                stream.submit("raw", frame)
            if stream.wants("stream"):
                exp_txt = "auto" if exposure is None else f"{exposure:g}"
                hud = [f"{cam.name}  {res_now[0]}x{res_now[1]}  {fps_shown:.0f} fps  "
                       f"detect {proc_shown:.1f} ms  exposure {exp_txt}  "
                       f"decimate {det_settings.quad_decimate:g}"]
                used = set(res.tag_ids) if res else set()
                stream.submit("stream", annotate(frame, detections, used, res, reject, hud))
