"""Exposure / gain control.

On Linux with a /dev camera path and v4l2-ctl installed (sudo apt install v4l-utils),
controls are set through v4l2-ctl. Many USB cameras ignore or mis-scale OpenCV's
CAP_PROP_EXPOSURE, while v4l2-ctl talks to the driver directly and uses the
camera's native units (check the range with: v4l2-ctl -d <device> --list-ctrls).

Otherwise falls back to OpenCV properties. macOS (AVFoundation) doesn't support
manual exposure through OpenCV at all, so it's ignored there.

exposure: None = auto exposure.   gain: None = leave the camera's gain alone.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

import cv2

_ctrl_cache: dict[str, set[str]] = {}


def _list_ctrls(device: str) -> set[str]:
    if device not in _ctrl_cache:
        try:
            out = subprocess.run(["v4l2-ctl", "-d", device, "--list-ctrls"],
                                 capture_output=True, text=True, timeout=3).stdout
        except (OSError, subprocess.TimeoutExpired):
            out = ""
        _ctrl_cache[device] = {line.split()[0] for line in out.splitlines() if " 0x" in line}
    return _ctrl_cache[device]


def _v4l2_set(device: str, **ctrls) -> bool:
    args = ["v4l2-ctl", "-d", device]
    for k, v in ctrls.items():
        args += ["-c", f"{k}={v}"]
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=3)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0


def control_ranges(device) -> dict[str, tuple[float, float]]:
    """{"exposure": (min, max), "gain": (min, max)} as reported by the driver, for the
    web UI's sliders. Empty when v4l2-ctl isn't available (the UI then uses defaults)."""
    if not uses_v4l2ctl(device):
        return {}
    try:
        out = subprocess.run(["v4l2-ctl", "-d", device, "--list-ctrls"],
                             capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}
    found = {}
    for line in out.splitlines():
        parts = line.split()
        if not parts or " 0x" not in line:
            continue
        kv = dict(p.split("=", 1) for p in parts if "=" in p)
        if "min" in kv and "max" in kv:
            found[parts[0]] = (float(kv["min"]), float(kv["max"]))
    ranges = {}
    for key, names in (("exposure", ("exposure_time_absolute", "exposure_absolute")), ("gain", ("gain",))):
        name = next((n for n in names if n in found), None)
        if name:
            ranges[key] = found[name]
    return ranges


def list_modes(device, fourcc: str = "MJPG") -> list[dict]:
    """Resolutions the camera offers in this pixel format, largest first:
    [{"width", "height", "fps"}] (fps = the fastest rate for that size).
    Empty when v4l2-ctl isn't available (the web UI then offers common sizes)."""
    if not uses_v4l2ctl(device):
        return []
    try:
        out = subprocess.run(["v4l2-ctl", "-d", device, "--list-formats-ext"],
                             capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    modes: dict[tuple[int, int], float] = {}
    in_format = False
    size = None
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("[") and "'" in s:  # e.g. [0]: 'MJPG' (Motion-JPEG, compressed)
            in_format = s.split("'")[1] == fourcc
            size = None
        elif in_format and s.startswith("Size:"):
            try:
                w, h = s.split()[-1].split("x")
                size = (int(w), int(h))
                modes.setdefault(size, 0.0)
            except ValueError:
                size = None
        elif in_format and size and s.startswith("Interval:") and "fps)" in s:
            try:
                fps = float(s.rsplit("(", 1)[1].split()[0])
                modes[size] = max(modes[size], fps)
            except (IndexError, ValueError):
                pass
    return [{"width": w, "height": h, "fps": fps}
            for (w, h), fps in sorted(modes.items(), key=lambda m: -m[0][0] * m[0][1])]


def uses_v4l2ctl(device) -> bool:
    return (sys.platform.startswith("linux") and isinstance(device, str)
            and device.startswith("/dev/") and shutil.which("v4l2-ctl") is not None)


def apply_exposure_gain(cap: cv2.VideoCapture, device, exposure, gain, log) -> None:
    if uses_v4l2ctl(device):
        ctrls = _list_ctrls(device)
        ae = next((c for c in ("auto_exposure", "exposure_auto") if c in ctrls), None)
        ex = next((c for c in ("exposure_time_absolute", "exposure_absolute") if c in ctrls), None)
        ok = True
        if exposure is None:
            if ae:
                ok &= _v4l2_set(device, **{ae: 3})  # 3 = aperture priority (auto)
        else:
            if ae:
                ok &= _v4l2_set(device, **{ae: 1})  # 1 = manual; must be set before exposure
            if ex:
                ok &= _v4l2_set(device, **{ex: int(round(exposure))})
            else:
                log("camera has no absolute exposure control")
        if gain is not None and "gain" in ctrls:
            ok &= _v4l2_set(device, gain=int(round(gain)))
        if not ok:
            log("v4l2-ctl rejected a setting; check ranges with --list-ctrls")
        return

    if sys.platform.startswith("linux"):
        if exposure is None:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
        else:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        if gain is not None:
            cap.set(cv2.CAP_PROP_GAIN, gain)
    elif exposure is not None or gain is not None:
        log("manual exposure/gain isn't supported on this OS; ignoring")
