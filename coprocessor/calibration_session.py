"""ChArUco camera calibration driven from a camera's settings page.

The flow is 6328's Northstar one (hold up a ChArUco board, capture views, calibrate),
with extras so the result is good the first time:

  - auto-capture: a view is taken when the board has been held still for a moment
    in a pose that's different enough (position, size or tilt) from every view so far,
  - a live coverage map on the stream and hints for where/how to move next
    (uncovered areas, missing tilts),
  - per-view reprojection error, with bad views (blur, misdetections) dropped and
    the calibration re-run automatically,
  - uncertainty (std dev) of the focal length / principal point, and an undistorted
    preview before saving.

Runs inside the camera process. The camera loop calls process() on each frame and
draw() for the stream; the web page calls the command methods through a queue.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import struct
import tempfile
import threading
import time
import zlib

import cv2
import numpy as np

GRID_COLS, GRID_ROWS = 12, 10      # coverage map cells
MIN_CORNER_FRAC = 0.2              # a view needs at least this fraction of the board's corners
MIN_CORNERS = 8
STILL_PX_FRAC = 0.002              # mean corner motion (fraction of image diagonal) that counts as still
STILL_TIME_S = 0.4                 # ...for this long before an auto-capture
AUTO_COOLDOWN_S = 0.6
NOVEL_CENTER = 0.10                # a new view must differ from every captured one by one of these:
NOVEL_SCALE = 0.25                 #   log size ratio
NOVEL_TILT_DEG = 12.0
TILT_BUCKET_DEG = 20.0             # "tilted" for the tilt-coverage hints
MIN_VIEWS = 15
GOOD_COVERAGE = 0.65


def make_board(b: dict):
    size = (int(b["squares_x"]), int(b["squares_y"]))
    full = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, b["dict"]))
    # The board only uses the first N markers. Matching candidates against just those
    # instead of the whole dictionary (e.g. 54 vs 1000) makes detection ~7x faster,
    # and the board is identical.
    n = (size[0] * size[1]) // 2
    d = cv2.aruco.Dictionary(full.bytesList[:n], full.markerSize, full.maxCorrectionBits)
    board = cv2.aruco.CharucoBoard(size, float(b["square_length_m"]), float(b["marker_length_m"]), d)
    board.setLegacyPattern(bool(b.get("legacy", False)))
    return board


def board_png(b: dict, dpi: int = 300) -> bytes:
    """Printable board image with its DPI recorded, so it prints at true size at 100%."""
    board = make_board(b)
    cols, rows = board.getChessboardSize()
    px = max(1, round(board.getSquareLength() / 0.0254 * dpi))
    img = board.generateImage((cols * px, rows * px), marginSize=px // 2, borderBits=1)
    ok, buf = cv2.imencode(".png", img)
    data = buf.tobytes()
    # Insert a pHYs chunk (pixels per meter) right after IHDR so viewers know the DPI.
    ppm = round(dpi / 0.0254)
    body = b"pHYs" + struct.pack(">IIB", ppm, ppm, 1)
    chunk = struct.pack(">I", 9) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    ihdr_end = 8 + 8 + 13 + 4
    return data[:ihdr_end] + chunk + data[ihdr_end:]


def _region_name(col: float, row: float) -> str:
    v = ("top", "middle", "bottom")[min(2, int(row * 3))]
    h = ("left", "center", "right")[min(2, int(col * 3))]
    return "center" if (v, h) == ("middle", "center") else f"{v} {h}".replace("middle ", "")


class CalibrationSession:
    def __init__(self, board_cfg: dict, size: tuple[int, int], wide_lens: bool = False):
        self.board_cfg = dict(board_cfg)
        self.board = make_board(board_cfg)
        self.detector = cv2.aruco.CharucoDetector(self.board)
        self.w, self.h = size
        self.diag = math.hypot(self.w, self.h)
        self.wide_lens = wide_lens
        sx, sy = self.board.getChessboardSize()
        self.total_corners = (sx - 1) * (sy - 1)
        # A rough camera matrix, only for judging board tilt between views.
        f = float(self.w)
        self._k_guess = np.array([[f, 0, self.w / 2], [0, f, self.h / 2], [0, 0, 1]])

        self.views: list[dict] = []
        self.covered = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.int32)
        self.auto = True
        self.preview = False
        self.state = "capturing"  # capturing | calibrating | result
        self.result: dict | None = None
        self.message = ""
        self._capture_requested = False
        self._prev: dict[int, np.ndarray] = {}
        self._still_since: float | None = None
        self._last_capture_t = 0.0
        self._flash_until = 0.0
        self._coverage_overlay = None
        self._undistort_maps = None
        self.live = {"corners": 0, "still": False, "novel": False, "tilt": None}

    # ---- commands (from the web page) -------------------------------------

    def request_capture(self):
        self._capture_requested = True

    def undo(self):
        if self.views and self.state == "capturing":
            self.views.pop()
            self._rebuild_coverage()
            self.message = f"removed last view ({len(self.views)} left)"

    def clear(self):
        if self.state == "capturing":
            self.views.clear()
            self._rebuild_coverage()
            self.message = "cleared all views"

    def start_calibration(self):
        if self.state != "capturing":
            return
        if len(self.views) < MIN_VIEWS:
            self.message = f"need at least {MIN_VIEWS} views (have {len(self.views)})"
            return
        self.state = "calibrating"
        self.message = "calibrating..."
        threading.Thread(target=self._calibrate, daemon=True, name="calibrate").start()

    def discard_result(self):
        if self.state == "result":
            self.state, self.result, self.preview = "capturing", None, False
            self._undistort_maps = None
            self.message = "result discarded; keep capturing or calibrate again"

    def save(self, path: str) -> dict:
        """Write the result to path (atomically). Returns the saved JSON."""
        if self.state != "result" or not self.result:
            raise RuntimeError("nothing to save; run Calibrate first")
        r = self.result
        out = {
            "camera_matrix": r["K"],
            "dist_coeffs": r["D"],
            "resolution": {"width": self.w, "height": self.h},
            "rms": r["rms"],
            "calibrated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "model": r["model"],
            "views_used": r["views_used"],
            "board": self.board_cfg,
            "std_dev": r["std"],
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".calib-", suffix=".json", dir=os.path.dirname(path))
        with os.fdopen(fd, "w") as f:
            json.dump(out, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return out

    # ---- per frame ---------------------------------------------------------

    def process(self, gray: np.ndarray, now: float) -> dict:
        """Detect the board; auto/manual capture. Returns the detection for draw()."""
        det = {"corners": None, "ids": None, "ok": False}
        if self.state != "capturing":
            return det
        corners, ids, _, _ = self.detector.detectBoard(gray)
        n = 0 if corners is None or ids is None or len(corners) != len(ids) else len(corners)
        need = max(MIN_CORNERS, int(MIN_CORNER_FRAC * self.total_corners))
        if n < 4:
            self._prev, self._still_since = {}, None
            self.live = {"corners": 0, "still": False, "novel": False, "tilt": None}
            return det
        pts = corners.reshape(-1, 2)
        idl = ids.flatten()
        det.update(corners=pts, ids=idl, ok=n >= need)

        # Stillness: mean motion of corners seen in both this and the last frame.
        common = [(self._prev[i], p) for i, p in zip(idl, pts) if i in self._prev]
        motion = (np.mean([np.linalg.norm(a - b) for a, b in common]) if len(common) >= 4 else 1e9)
        self._prev = dict(zip(idl, pts))
        still = bool(motion < STILL_PX_FRAC * self.diag)
        if not still:
            self._still_since = None
        elif self._still_since is None:
            self._still_since = now

        desc = self._describe(corners, ids) if det["ok"] else None
        novel = desc is not None and self._is_novel(desc)
        self.live = {"corners": int(n), "still": still, "novel": bool(novel),
                     "tilt": None if desc is None else [int(round(desc[3])), int(round(desc[4]))]}
        det["novel"] = novel
        det["still"] = still

        if desc is None:
            return det
        manual = self._capture_requested
        auto_ready = (self.auto and novel and self._still_since is not None
                      and now - self._still_since >= STILL_TIME_S
                      and now - self._last_capture_t >= AUTO_COOLDOWN_S)
        if manual or auto_ready:
            self._capture_requested = False
            self._capture(corners, ids, desc, pts, now, manual)
        return det

    def _describe(self, corners, ids):
        """(center x, center y, log size, tilt left/right deg, tilt up/down deg)."""
        obj, img = self.board.matchImagePoints(corners, ids)
        if obj is None or len(obj) < 4:
            return None
        pts = img.reshape(-1, 2)
        cx, cy = pts.mean(axis=0)
        hull = cv2.convexHull(pts.astype(np.float32))
        size = math.sqrt(max(cv2.contourArea(hull), 1.0) / (self.w * self.h))
        ok, rvec, _ = cv2.solvePnP(obj, img, self._k_guess, None, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            return None
        n = cv2.Rodrigues(rvec)[0][:, 2]
        if n[2] > 0:
            n = -n  # normal pointing back at the camera
        tilt_h = math.degrees(math.atan2(n[0], -n[2]))
        tilt_v = math.degrees(math.atan2(n[1], -n[2]))
        return (cx / self.w, cy / self.h, math.log(size), tilt_h, tilt_v)

    def _is_novel(self, d) -> bool:
        for v in self.views:
            e = v["desc"]
            if (abs(d[0] - e[0]) < NOVEL_CENTER and abs(d[1] - e[1]) < NOVEL_CENTER
                    and abs(d[2] - e[2]) < NOVEL_SCALE
                    and abs(d[3] - e[3]) < NOVEL_TILT_DEG and abs(d[4] - e[4]) < NOVEL_TILT_DEG):
                return False
        return True

    def _capture(self, corners, ids, desc, pts, now, manual):
        obj, img = self.board.matchImagePoints(corners, ids)
        self.views.append({"obj": obj, "img": img, "desc": desc, "pts": pts.copy()})
        self._mark_coverage(pts)
        self._last_capture_t = now
        self._flash_until = now + 0.25
        self.message = f"captured view {len(self.views)}" + (" (manual)" if manual else "")

    def _mark_coverage(self, pts):
        cols = np.clip((pts[:, 0] / self.w * GRID_COLS).astype(int), 0, GRID_COLS - 1)
        rows = np.clip((pts[:, 1] / self.h * GRID_ROWS).astype(int), 0, GRID_ROWS - 1)
        np.add.at(self.covered, (rows, cols), 1)
        self._coverage_overlay = None

    def _rebuild_coverage(self):
        self.covered[:] = 0
        for v in self.views:
            self._mark_coverage(v["pts"])

    # ---- guidance ----------------------------------------------------------

    def coverage(self) -> float:
        return float(np.count_nonzero(self.covered)) / self.covered.size

    def tilt_counts(self) -> dict:
        c = {"left": 0, "right": 0, "up": 0, "down": 0}
        for v in self.views:
            th, tv = v["desc"][3], v["desc"][4]
            if th <= -TILT_BUCKET_DEG: c["left"] += 1
            if th >= TILT_BUCKET_DEG: c["right"] += 1
            if tv <= -TILT_BUCKET_DEG: c["up"] += 1
            if tv >= TILT_BUCKET_DEG: c["down"] += 1
        return c

    def hint(self) -> str:
        if self.state == "calibrating":
            return "Calibrating..."
        if self.state == "result":
            return "Review the result, then Save (or Discard and keep capturing)."
        lv = self.live
        need = max(MIN_CORNERS, int(MIN_CORNER_FRAC * self.total_corners))
        if lv["corners"] == 0:
            return "Hold the board up to the camera."
        if lv["corners"] < need:
            return "Show more of the board (move back or bring it fully into view)."
        if not lv["still"]:
            return "Hold still..."
        if not lv["novel"]:
            return "Already have this view: " + self._next_goal()
        return "Capturing..." if self.auto else "New view. Press Capture."

    def _next_goal(self) -> str:
        if self.coverage() < GOOD_COVERAGE:
            # Emptiest third-of-the-frame region.
            best, best_n = None, None
            for r in range(3):
                for c in range(3):
                    blk = self.covered[r * GRID_ROWS // 3:(r + 1) * GRID_ROWS // 3,
                                       c * GRID_COLS // 3:(c + 1) * GRID_COLS // 3]
                    n = np.count_nonzero(blk)
                    if best_n is None or n < best_n:
                        best, best_n = (c / 3 + 0.1, r / 3 + 0.1), n
            return f"move the board to the {_region_name(*best)} of the frame."
        tc = self.tilt_counts()
        missing = [k for k, n in tc.items() if n < 2]
        if missing:
            return f"tilt the board ~30 deg ({', '.join(missing)})."
        if len(self.views) < MIN_VIEWS:
            return "move closer/farther or to a new spot."
        return "you have enough views. Press Calibrate (more views are fine too)."

    def status(self) -> dict:
        tc = self.tilt_counts()
        return {
            "state": self.state,
            "views": len(self.views),
            "min_views": MIN_VIEWS,
            "coverage": round(self.coverage(), 3),
            "good_coverage": GOOD_COVERAGE,
            "tilts": tc,
            "auto": self.auto,
            "preview": self.preview,
            "live": self.live,
            "hint": self.hint(),
            "message": self.message,
            "resolution": f"{self.w}x{self.h}",
            "board": self.board_cfg,
            "wide_lens": self.wide_lens,
            "result": self.result,
            "ready": len(self.views) >= MIN_VIEWS,
        }

    # ---- calibration -------------------------------------------------------

    def _run_opencv(self, views):
        flags = cv2.CALIB_RATIONAL_MODEL if self.wide_lens else 0
        obj = [v["obj"] for v in views]
        img = [v["img"] for v in views]
        rms, K, D, _, _, std_int, _, per_view = cv2.calibrateCameraExtended(
            obj, img, (self.w, self.h), None, None, flags=flags)
        return rms, K, D, std_int.flatten(), per_view.flatten()

    def _calibrate(self):
        try:
            views = list(self.views)
            rms, K, D, std, per_view = self._run_opencv(views)
            removed: dict[int, float] = {}
            # Drop views far worse than the rest (blur, bad detections), then re-run.
            med = float(np.median(per_view))
            limit = max(3.0 * med, 0.75)
            bad = [i for i in np.argsort(-per_view) if per_view[i] > limit][: len(views) // 5]
            if bad and len(views) - len(bad) >= MIN_VIEWS - 3:
                removed = {int(i): float(per_view[i]) for i in bad}
                kept = [v for i, v in enumerate(views) if i not in removed]
                rms, K, D, std, per_view_kept = self._run_opencv(kept)
                it = iter(per_view_kept)
                per_view = [removed[i] if i in removed else float(next(it)) for i in range(len(views))]
            per_view = [float(e) for e in per_view]

            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            rms = float(rms)
            verdict = ("excellent" if rms < 0.3 else "good" if rms < 0.6
                       else "ok" if rms < 1.0 else "poor. Recalibrate (steadier views, flatter board, better light)")
            self.result = {
                "rms": rms,
                "verdict": verdict,
                "K": K.tolist(),
                "D": D.flatten().tolist(),
                "model": "rational (8 coeffs)" if self.wide_lens else "standard (5 coeffs)",
                "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
                "std": {"fx": float(std[0]), "fy": float(std[1]), "cx": float(std[2]), "cy": float(std[3])},
                "fov_h_deg": math.degrees(2 * math.atan(self.w / (2 * fx))),
                "fov_v_deg": math.degrees(2 * math.atan(self.h / (2 * fy))),
                "per_view": per_view,
                "removed": sorted(removed),
                "views_used": len(views) - len(removed),
            }
            self._undistort_maps = cv2.initUndistortRectifyMap(K, D, None, K, (self.w, self.h), cv2.CV_16SC2)
            self.state = "result"
            self.message = (f"RMS {rms:.3f} px from {len(views) - len(removed)} views"
                            + (f" ({len(removed)} outlier views dropped)" if removed else ""))
        except Exception as e:  # keep the session usable
            self.state = "capturing"
            self.message = f"calibration failed: {e}"

    # ---- drawing -----------------------------------------------------------

    def draw(self, frame: np.ndarray, det: dict, now: float) -> np.ndarray:
        img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame.copy()
        if self.state == "result" and self.preview and self._undistort_maps is not None:
            img = cv2.remap(img, *self._undistort_maps, cv2.INTER_LINEAR)
            self._banner(img, [f"UNDISTORTED PREVIEW  |  RMS {self.result['rms']:.3f} px  |  "
                               "straight lines should look straight"])
            return img

        # Coverage map: green where captured corners have landed.
        if self._coverage_overlay is None or self._coverage_overlay.shape[:2] != img.shape[:2]:
            mask = (self.covered > 0).astype(np.uint8) * 255
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            self._coverage_overlay = mask
        green = np.zeros_like(img)
        green[:, :, 1] = 255
        m = self._coverage_overlay > 0
        img[m] = (img[m] * 0.7 + green[m] * 0.3).astype(np.uint8)
        for c in range(1, GRID_COLS):
            x = c * img.shape[1] // GRID_COLS
            cv2.line(img, (x, 0), (x, img.shape[0]), (70, 70, 70), 1)
        for r in range(1, GRID_ROWS):
            y = r * img.shape[0] // GRID_ROWS
            cv2.line(img, (0, y), (img.shape[1], y), (70, 70, 70), 1)

        scale = max(0.6, img.shape[1] / 1280.0)
        if det.get("corners") is not None:
            if not det["ok"]:
                color = (0, 0, 255)          # red: too little of the board
            elif not det.get("novel"):
                color = (160, 160, 160)      # gray: already have this view
            elif not det.get("still"):
                color = (0, 200, 255)        # yellow: new, hold still
            else:
                color = (0, 230, 0)          # green: capturing
            hull = cv2.convexHull(det["corners"].astype(np.float32)).astype(np.int32)
            cv2.polylines(img, [hull], True, color, max(1, int(2 * scale)), cv2.LINE_AA)
            for x, y in det["corners"]:
                cv2.circle(img, (int(x), int(y)), max(2, int(3 * scale)), color, -1, cv2.LINE_AA)

        if now < self._flash_until:
            cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (255, 255, 255),
                          max(6, int(12 * scale)))

        tc = self.tilt_counts()
        self._banner(img, [
            f"CALIBRATING {self.w}x{self.h}  |  views {len(self.views)}/{MIN_VIEWS}  |  "
            f"coverage {self.coverage() * 100:.0f}%  |  tilts L{tc['left']} R{tc['right']} "
            f"U{tc['up']} D{tc['down']}",
            self.hint(),
        ])
        return img

    @staticmethod
    def _banner(img, lines):
        scale = max(0.6, img.shape[1] / 1280.0)
        lh = int(28 * scale)
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (img.shape[1], lh * len(lines) + int(10 * scale)), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        for i, t in enumerate(lines):
            cv2.putText(img, t, (int(8 * scale), lh * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65 * scale, (240, 240, 240), max(1, int(1.5 * scale)), cv2.LINE_AA)
