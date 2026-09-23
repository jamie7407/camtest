"""Live-tunable settings over NetworkTables, with a save button.

  /Vision/settings/<name>                 shared settings (detector, filters, std devs, stream)
  /Vision/cameras/<camera>/settings/<name> per-camera settings (exposure, gain, std_dev_factor)
  /Vision/settings/save                   set true -> current values written to config.json
  /Vision/settings/status                 "saved 14:02:11" / "unsaved changes" / errors

config.json is the source of truth: on startup every entry is set FROM the file.
Changes apply immediately but only persist when you press save, so experimenting
never overwrites a good config, and the SD card isn't rewritten on every nudge.

NT has no null, so exposure/gain use -1 to mean "auto" / "leave alone"
(saved to config.json as null).
"""

from __future__ import annotations

import json
import os
import tempfile
import time

import ntcore

from config import AppConfig

# name -> (config.json section, NT type)
GLOBAL_SETTINGS = {
    "quad_decimate": ("detector", float),
    "min_cluster_pixels": ("detector", int),
    "min_decision_margin": ("filters", float),
    "max_single_tag_ambiguity": ("filters", float),
    "max_tag_distance_m": ("filters", float),
    "xy_coefficient": ("stddev", float),
    "theta_coefficient": ("stddev", float),
    "trust_single_tag_theta": ("stddev", bool),
    "stream_max_fps": ("stream", float),
}
CAMERA_SETTINGS = {"exposure": float, "gain": float, "std_dev_factor": float}


def _entry(inst, path, typ, value):
    if typ is bool:
        e = inst.getBooleanTopic(path).getEntry(bool(value))
        e.set(bool(value))
    elif typ is int:
        e = inst.getIntegerTopic(path).getEntry(int(value))
        e.set(int(value))
    else:
        e = inst.getDoubleTopic(path).getEntry(float(value))
        e.set(float(value))
    return e


class LiveSettings:
    def __init__(self, inst: ntcore.NetworkTableInstance, cfg: AppConfig, base: str = "/Vision"):
        self.cfg = cfg
        self.values: dict[str, object] = {}
        self.cam_values: dict[int, dict[str, object]] = {}
        self.entries: dict[str, object] = {}
        self.cam_entries: dict[int, dict[str, object]] = {}

        for name, (section, typ) in GLOBAL_SETTINGS.items():
            v = self._initial_global(name)
            self.values[name] = typ(v)
            self.entries[name] = _entry(inst, f"{base}/settings/{name}", typ, v)

        for cam in cfg.cameras:
            vals = {
                "exposure": -1.0 if cam.exposure is None else float(cam.exposure),
                "gain": -1.0 if cam.gain is None else float(cam.gain),
                "std_dev_factor": float(cam.std_dev_factor),
            }
            self.cam_values[cam.id] = vals
            self.cam_entries[cam.id] = {
                k: _entry(inst, f"{base}/cameras/{cam.name}/settings/{k}", CAMERA_SETTINGS[k], v)
                for k, v in vals.items()
            }

        self.save_entry = inst.getBooleanTopic(f"{base}/settings/save").getEntry(False)
        self.save_entry.set(False)
        self.status = inst.getStringTopic(f"{base}/settings/status").publish()
        self.status.set("loaded from config.json")
        self.dirty = False

    def _initial_global(self, name):
        if name == "stream_max_fps":
            return self.cfg.stream.get("max_fps", 30)
        if hasattr(self.cfg.detector, name):
            return getattr(self.cfg.detector, name)
        return getattr(self.cfg.solver, name)

    @staticmethod
    def camera_message(vals: dict) -> dict:
        """Convert NT-side camera values into what the camera process expects."""
        out = {}
        for k, v in vals.items():
            out[k] = None if k in ("exposure", "gain") and v < 0 else v
        return out

    def full_camera_settings(self, cam_id: int) -> dict:
        """Everything a (re)started camera process needs to match current live values."""
        return {**self.values, **self.camera_message(self.cam_values[cam_id])}

    def poll(self) -> tuple[dict, dict[int, dict]]:
        """Returns (global changes, per-camera changes) since the last poll."""
        g = {}
        for name, e in self.entries.items():
            typ = GLOBAL_SETTINGS[name][1]
            v = typ(e.get())
            if v != self.values[name]:
                self.values[name] = v
                g[name] = v
        per_cam = {}
        for cam_id, ents in self.cam_entries.items():
            changed = {}
            for k, e in ents.items():
                v = float(e.get())
                if v != self.cam_values[cam_id][k]:
                    self.cam_values[cam_id][k] = v
                    changed[k] = v
            if changed:
                per_cam[cam_id] = self.camera_message(changed)
        if g or per_cam:
            self.dirty = True
            self.status.set("unsaved changes")

        if self.save_entry.get():
            self.save_entry.set(False)
            self._save()
        return g, per_cam

    def _save(self):
        try:
            with open(self.cfg.path) as f:
                raw = json.load(f)
            for name, (section, _) in GLOBAL_SETTINGS.items():
                key = "max_fps" if name == "stream_max_fps" else name
                raw.setdefault(section, {})[key] = self.values[name]
            for cam in self.cfg.cameras:
                entry = raw["cameras"][cam.id]
                vals = self.camera_message(self.cam_values[cam.id])
                entry["exposure"] = vals["exposure"]
                entry["gain"] = vals["gain"]
                entry["std_dev_factor"] = vals["std_dev_factor"]

            # Atomic write: a power cut mid-save leaves either the old or the new file, never half.
            d = os.path.dirname(self.cfg.path)
            fd, tmp = tempfile.mkstemp(prefix=".config-", suffix=".json", dir=d)
            with os.fdopen(fd, "w") as f:
                json.dump(raw, f, indent=2)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.cfg.path)
            self.dirty = False
            msg = f"saved {time.strftime('%H:%M:%S')}"
            print(f"settings: {msg} -> {self.cfg.path}", flush=True)
            self.status.set(msg)
        except Exception as e:  # keep running; report on NT
            print(f"settings: save failed: {e}", flush=True)
            self.status.set(f"save failed: {e}")
