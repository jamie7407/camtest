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
