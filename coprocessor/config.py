"""Loads config.json and camera calibrations into plain, picklable objects.

Calibrations are per camera AND per resolution: calibrations/<name>_<W>x<H>.json
(what the settings page's calibration saves). A camera without a calibration for
its current resolution still runs -- it streams and detects tags, but publishes no
poses until it's calibrated.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

from pose_solver import DetectorSettings, SolverSettings

# OV2311 full sensor (16:13). Used when a camera in config.json has no width/height.
DEFAULT_WIDTH, DEFAULT_HEIGHT = 1600, 1300

# 6328's Northstar calibration board (12x9 squares, 5x5 markers, 30mm/23mm).
DEFAULT_BOARD = {
    "squares_x": 12,
    "squares_y": 9,
    "square_length_m": 0.030,
    "marker_length_m": 0.023,
    "dict": "DICT_5X5_1000",
    "legacy": False,
}


@dataclass
class CameraConfig:
    id: int
    name: str
    device: str | int
    width: int
    height: int
    fps: int
    fourcc: str
    robot_to_camera: dict  # x, y, z (m), roll_deg, pitch_deg, yaw_deg
    calib_dir: str  # where <name>_<W>x<H>.json calibrations live
    calibration: str | None = None  # optional explicit file (older configs), used if its resolution matches
    std_dev_factor: float = 1.0
    exposure: float | None = None  # raw V4L2 value; None = leave camera on auto
    gain: float | None = None
    loop_video: bool = False  # for testing with a video file instead of a camera
    stream_port: int = 1181


@dataclass
class AppConfig:
    team: int | None
    server: str | None
    standalone: bool
    client_name: str
    topic: str
    field_layout: str
    detector: DetectorSettings
    solver: SolverSettings
    stream: dict = field(default_factory=dict)  # enabled, base_port, max_fps, jpeg_quality
    path: str = ""
    cameras: list[CameraConfig] = field(default_factory=list)
    calibration_board: dict = field(default_factory=lambda: dict(DEFAULT_BOARD))


def _find(d: dict, *keys):
    for k in keys:
        if k in d:
            return d[k]
    return None


def load_calibration(path: str, width: int, height: int) -> tuple[list, list]:
    """Accepts either a simple {"camera_matrix": ..., "dist_coeffs": ...} JSON or a
    calibration JSON exported from PhotonVision's camera settings page."""
    with open(path) as f:
        c = json.load(f)

    km = _find(c, "camera_matrix", "cameraMatrix", "cameraIntrinsics")
    dc = _find(c, "dist_coeffs", "distCoeffs", "distortion")
    if isinstance(km, dict):  # PhotonVision stores {"rows", "cols", "type", "data"}
        km = km.get("data")
    if isinstance(dc, dict):
        dc = dc.get("data")
    if km is None or dc is None:
        raise ValueError(f"{path}: couldn't find camera matrix / distortion coefficients")

    K = np.array(km, dtype=np.float64).reshape(3, 3)
    D = np.array(dc, dtype=np.float64).flatten()

    res = c.get("resolution")
    if isinstance(res, dict) and "width" in res and "height" in res:
        if (int(res["width"]), int(res["height"])) != (width, height):
            raise ValueError(
                f"{path} was calibrated at {res['width']}x{res['height']} but the camera is "
                f"configured for {width}x{height}. Calibrate at the resolution you run at."
            )
    return K.tolist(), D.tolist()


def calibration_path(cam: CameraConfig, width: int, height: int) -> str:
    return os.path.join(cam.calib_dir, f"{cam.name}_{width}x{height}.json")


def find_calibration(cam: CameraConfig, width: int, height: int) -> dict | None:
    """The calibration for this camera at this resolution, or None if there isn't one.
    Looks for <name>_<W>x<H>.json, then the config's explicit file, then <name>.json
    (calibrate.py's output); files recorded at another resolution are skipped."""
    candidates = [calibration_path(cam, width, height)]
    if cam.calibration:
        candidates.append(cam.calibration)
    candidates.append(os.path.join(cam.calib_dir, f"{cam.name}.json"))
    for p in dict.fromkeys(candidates):
        if not os.path.exists(p):
            continue
        try:
            K, D = load_calibration(p, width, height)
        except ValueError:
            continue  # other resolution (or unreadable): keep looking
        with open(p) as f:
            meta = json.load(f)
        return {"K": K, "D": D, "path": p,
                "rms": meta.get("rms"), "calibrated_at": meta.get("calibrated_at")}
    return None


def load_config(path: str) -> AppConfig:
    with open(path) as f:
        raw = json.load(f)
    base = os.path.dirname(os.path.abspath(path))

    nt = raw.get("nt", {})
    det = DetectorSettings(**raw.get("detector", {}))
    # Std dev coefficients live on the robot now (Preferences); an old "stddev" section is ignored.
    solver = SolverSettings(**raw.get("filters", {}))
    if "tag_size_m" in raw.get("field", {}):
        solver.tag_size_m = raw["field"]["tag_size_m"]

    stream = {"enabled": True, "base_port": 1181, "max_fps": 30, "jpeg_quality": 80, **raw.get("stream", {})}

    calib_dir = os.path.join(base, raw.get("calibration_dir", "calibrations"))
    board = {**DEFAULT_BOARD, **raw.get("calibration_board", {})}

    cams = []
    for i, c in enumerate(raw["cameras"]):
        explicit = c.get("calibration")
        if explicit and not os.path.isabs(explicit):
            explicit = os.path.join(base, explicit)
        cams.append(
            CameraConfig(
                id=i,
                name=c.get("name", f"camera{i}"),
                device=c["device"],
                width=int(c.get("width") or DEFAULT_WIDTH),
                height=int(c.get("height") or DEFAULT_HEIGHT),
                fps=c.get("fps", 30),
                fourcc=c.get("fourcc", "MJPG"),
                robot_to_camera=c["robot_to_camera"],
                calib_dir=calib_dir,
                calibration=explicit,
                std_dev_factor=c.get("std_dev_factor", 1.0),
                exposure=c.get("exposure"),
                gain=c.get("gain"),
                loop_video=c.get("loop_video", False),
                stream_port=int(stream["base_port"]) + i,
            )
        )
    if len({c.name for c in cams}) != len(cams):
        raise ValueError("camera names must be unique")

    return AppConfig(
        team=nt.get("team"),
        server=nt.get("server"),
        standalone=nt.get("standalone", False),
        client_name=nt.get("client_name", "vision-coprocessor"),
        topic=raw.get("topic", "/Vision/observations"),
        field_layout=raw.get("field", {}).get("layout", "k2026RebuiltWelded"),
        detector=det,
        solver=solver,
        stream=stream,
        path=os.path.abspath(path),
        cameras=cams,
        calibration_board=board,
    )


def robot_to_camera_transform(rc: dict):
    from wpimath.geometry import Rotation3d, Transform3d, Translation3d

    return Transform3d(
        Translation3d(rc["x"], rc["y"], rc["z"]),
        Rotation3d(
            math.radians(rc.get("roll_deg", 0.0)),
            math.radians(rc.get("pitch_deg", 0.0)),
            math.radians(rc.get("yaw_deg", 0.0)),
        ),
    )


def load_field_layout(name_or_path: str):
    import robotpy_apriltag as ra

    if name_or_path.endswith(".json"):
        return ra.AprilTagFieldLayout(name_or_path)
    return ra.AprilTagFieldLayout.loadField(getattr(ra.AprilTagField, name_or_path))
