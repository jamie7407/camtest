# Batched multi-camera AprilTag vision (Orange Pi)

One process per camera does capture → AprilTag detection → pose solve → std devs on the coprocessor. All cameras publish into **one** NT4 struct-array topic, so the robot does a single NT read per loop and just loops `addVisionMeasurement()`, with no per-camera photonlibpy deserialization in RobotPy.

Everything is run from one web page, like PhotonVision: live stream, settings, resolution and camera calibration for every camera, with a camera dropdown at the top. Settings are also live-tunable over NetworkTables.

Adding a camera is just adding an entry to `config.json`. Nothing else changes on either side.

```
coprocessor/              -> goes on the Orange Pi
  main.py                 spawns camera processes, publishes the batched topic
  camera_process.py       per-camera capture/detect/solve (own process = own CPU core)
  pose_solver.py          multi-tag SQPNP / single-tag IPPE, filters, std dev factor
  config.py               config + calibration loading (accepts PhotonVision exports)
  settings.py             live settings (NT + web page) and save-to-config.json
  stream_server.py        per-camera web server: MJPEG streams + settings/calibration API
  web_ui.html             the settings page (stream, settings, resolution, calibration)
  calibration_session.py  in-browser ChArUco calibration
  camera_controls.py      exposure/gain/resolution modes (v4l2-ctl on Linux, OpenCV fallback)
  vision_types.py         the NT4 struct (must be identical on both sides)
  test_synthetic.py       renders tags from known poses and checks the math
  list_cameras.py         finds cameras, saves a snapshot from each
  calibrate.py            ChArUco calibration from a terminal (alternative to the page)
  config.example.json, vision.service
  calibrations/           <camera>_<W>x<H>.json, one per camera per resolution (not in git)
robot/                    -> goes in your RobotPy project
  batched_vision.py       reads the topic, feeds your pose estimator
  vision_types.py         identical copy
```

## Quick reference

Our setup: Pi at **192.168.50.2**, user **choate7407**, code in `~/vision/coprocessor`, run by the `vision` systemd service. Commands below run on your Mac from the repo root.

| What | How |
|---|---|
| Settings page (all cameras) | `http://192.168.50.2:1181/` (camera dropdown at the top) |
| AdvantageScope / Elastic | connect to `192.168.50.2` (needs `"standalone": true` when there's no roboRIO) |
| Is it running? | `ssh choate7407@192.168.50.2 "systemctl status vision --no-pager"` |
| Logs | `ssh choate7407@192.168.50.2 "journalctl -u vision -n 50 --no-pager"` (`-f` to follow) |
| Restart | `ssh -t choate7407@192.168.50.2 "sudo systemctl restart vision"` |
| Deploy code | `scp coprocessor/*.py coprocessor/web_ui.html choate7407@192.168.50.2:~/vision/coprocessor/` then restart |
| Pull tuned config back | `scp choate7407@192.168.50.2:~/vision/coprocessor/config.json coprocessor/config.json` |
| Back up calibrations | `scp -r choate7407@192.168.50.2:~/vision/coprocessor/calibrations coprocessor/` |

Don't `scp` your `config.json` *to* the Pi without pulling it first: settings saved on the Pi (exposure, resolution, filters) live only in the Pi's copy. Tip: `ssh-copy-id choate7407@192.168.50.2` once, and you stop typing the SSH password.

## What's been verified

- **Pose math** (`test_synthetic.py`): renders real tag36h11 images on the 2026 field from known robot poses through a pitched/yawed/offset camera. Median error 0.3 cm multi-tag, 0.7 cm single-tag.
- **End to end** in a sandbox: two synthetic camera feeds → `main.py` → NT4 → `batched_vision.py` → `SwerveDrive4PoseEstimator` started 1.8 m off converged to 1.8 cm of truth.
- **Calibration**: a simulated OV2311 at 1600×1300 with strong barrel distortion, calibrated through the web flow (auto-capture from a video), recovered fx/fy/cx/cy within ~0.5 px of truth at 0.28 px RMS.
- **Live settings / page**: changes from the page, Elastic and AdvantageScope stay in sync; save writes config atomically; resolution switching loads the matching calibration.
- **On the Orange Pi 5**: runs as a service with two USB cameras, streams and settings page work, exposure/gain via `v4l2-ctl`.

Two things found along the way that are worth knowing even if you never use this code. First, WPILib's `AprilTagDetector` defaults `minClusterPixels` to 300, which silently drops small or far tags once `quadDecimate` is 2; 20 found more tags than no decimation at about a third of the CPU. Second, OpenCV's generated tag36h11 images are rotated 180° from the official FRC tags, so don't print test tags from `cv2.aruco`; use the official AprilRobotics images.

## Developing on a Mac

Everything runs on macOS, and it's the easiest way to iterate: robot code in RobotPy simulation, the vision code, and AdvantageScope all on one machine.

```bash
cd coprocessor
pip3 install -r requirements.txt
python3 test_synthetic.py          # should print PASS
python3 list_cameras.py            # which index is which camera (saves snapshots)
```

The first time you open a camera, macOS asks for camera permission for whatever app launched Python (Terminal, iTerm, VS Code). If `list_cameras.py` finds nothing, check System Settings > Privacy & Security > Camera.

On a Mac, use camera **indices** in `config.json` (`"device": 0`), not `/dev/...` paths. The order can change when you replug cameras, and the built-in camera or a nearby iPhone (Continuity Camera) may take index 0. Manual exposure and resolution lists are Linux-only; on macOS cameras run on auto exposure. A video file also works as a `device` (add `"loop_video": true`), which is handy for testing the page and calibration without a camera.

```bash
# Vision only: this process is the NT server. Point AdvantageScope at localhost.
python3 main.py --config config.json --standalone

# Full loop: start your robot code in sim first (it becomes the NT server)...
python -m robotpy sim
# ...then in another terminal, connect vision to it:
python3 main.py --config config.json --server 127.0.0.1
```

Judge frame rate and `processMs` on the Pi, not the Mac; Apple Silicon is much faster.

## 1. Set up the Orange Pi

Use a **second** SD card so the competition PhotonVision card stays untouched. We run Ubuntu 22.04 (the Orange Pi image) with Python 3.14 in a venv at `~/vision/coprocessor/venv`.

The Pi usually has **no internet**, so install from wheels downloaded on a Mac. `coprocessor/wheels/` (not in git) holds the aarch64 / cp314 wheels for everything in `requirements.txt`:

```bash
# On the Pi, in ~/vision/coprocessor:
<your python3.14> -m venv venv
venv/bin/pip install --no-index --find-links wheels -r requirements.txt
venv/bin/python test_synthetic.py          # should print PASS
```

**v4l2-ctl** is needed for manual exposure/gain and for the resolution list. With internet it's `sudo apt install v4l-utils`, but on the Orange Pi image that package conflicts with the Rockchip-patched `libv4l-0` (`...+rkmpp`), and downgrading that library can break hardware video decoding. Instead, extract just the binary from Ubuntu 22.04's arm64 `v4l-utils` .deb on the Mac and install it:

```bash
scp v4l2-ctl choate7407@192.168.50.2:~/
ssh -t choate7407@192.168.50.2 "sudo install -m 755 ~/v4l2-ctl /usr/local/bin/ && v4l2-ctl --version"
```

It only needs `libv4l2.so.0`, which the patched library provides.

**Network.** Give the Pi a static IP (ours is 192.168.50.2). To work on it over a direct Ethernet cable, set your Mac's Ethernet adapter to a manual address on the same subnet (e.g. 192.168.50.10, mask 255.255.255.0, no router) and check with `ping 192.168.50.2`.

## 2. Cameras

```bash
ls -l /dev/v4l/by-path/
v4l2-ctl -d /dev/video0 --list-formats-ext    # resolutions/fps per format
```

Use the `/dev/v4l/by-path/...-video-index0` paths in `config.json` (each camera also shows an `index1`; that's metadata, ignore it). By-path names are tied to the physical USB port, so identical cameras can't swap between reboots. Moving a cable to another port does change them.

**Resolution.** Pick it on the settings page; the dropdown lists what the camera offers in MJPG, with 16:13 (the OV2311's full 1600×1300 sensor) first when it's available. If it isn't, use **1600×1200** (4:3, nearly the full sensor). Avoid 16:9 modes: they crop the vertical field of view. Higher resolution gives more range and less noise at distance; 25–50 fps per camera is plenty for localization, since the pose estimator compensates for latency.

**Which camera is which.** Cover one lens and see which camera's stream goes dark. Don't go by what's in the picture: with cameras angled toward each other (toed in), the physical *left* camera sees the *right* side of the scene. Each config entry's `device` must be the physical camera its `robot_to_camera` describes, and calibration files are named after the entry. If two cameras on the same robot report very different robot poses (tens of cm, tens of degrees), check this first.

## 3. Configure

`cp config.example.json config.json` and edit. `robot_to_camera` uses WPILib conventions: x forward, y left, z up (meters from robot center at floor level); **pitch negative = camera tilted up**; yaw is CCW-positive (a camera facing the robot's right side is ~−90°, a rear camera ~180°).

`nt.standalone: true` makes the Pi its own NT server, for bench testing without a roboRIO. **Set it back to `false` for the robot**, or the roboRIO never sees the data. Frame rate (`fps`) is per camera in the config; set it to what the mode supports (the resolution dropdown shows the max).

Low exposure matters more than anything else for tag detection on a moving robot, since motion blur kills detections. Tune it live on the settings page: turn off Auto, drop exposure until a waved tag stays crisp, raise gain if the image gets too dark, then save.

## 4. Calibrate

Calibrate from the settings page. A camera boots and streams without a calibration; it just doesn't publish poses (the header says **NOT CALIBRATED**) until it has one for its current resolution. Pick the resolution first: each resolution needs its own calibration.

1. **Board.** The default is 6328's Northstar board (12×9 squares, 5×5 markers, 30/23 mm); a PhotonVision 8×8 preset is there too, or type in your own. **Print board** gives an image that prints at true size at 100% scale. The 6328 board is 360×270 mm, bigger than letter, so print it on 11×17 or show it full-screen on a flat monitor. Mount it flat; measure a square.
2. **Start calibration.** Pose output pauses for that camera. Hold the board in view and keep it still for a moment: a view is captured automatically whenever the board is held still in a pose that's new (different spot, distance or tilt). The stream shows a green coverage map and a hint for what to do next (e.g. "move the board to the top left", "tilt the board ~30 deg (left, up)"). **Capture now** / **Undo** / **Clear** are there for manual control.
3. Aim for 15+ views, 65%+ coverage including the corners, and at least two views tilted each way. Then **Calibrate**.
4. **Review.** RMS error (under 0.5 px is good), focal length and principal point with their uncertainty, field of view, and error per view. Views much worse than the rest (blur, bad detections) are dropped automatically and the calibration re-run. **Preview undistorted** should make straight edges look straight.
5. **Save calibration.** Written to `calibrations/<camera>_<W>x<H>.json` and used immediately; no restart.

Tick **Wide-angle lens** before starting if your lens is wider than ~100° (OpenCV's 8-coefficient model). Older files (`calibrations/<camera>.json` from `calibrate.py`, or PhotonVision exports) still load when their recorded resolution matches. Calibrations aren't in git; back them up with the `scp` in the quick reference.

## 5. Settings page, live tuning, and saving

**The page.** `http://<pi-ip>:1181/` (any camera's port works; `#cam=<name>` in the URL opens that camera). The header shows **camera fps** (what the camera delivers), **processed** fps, detect time and calibration status. The stream has two views: *Annotated* (green outlines for tags used in the pose, orange for filtered, red dot on corner 0, and the pose or **the reason a frame was rejected**) and *Raw*.

Streams are full resolution and only encode while someone is watching, so an unwatched stream costs nothing; `stream_max_fps` caps encoding while watched. Both streams are registered under `/CameraPublisher`, so Elastic's camera widget finds them. The field network caps robot bandwidth, so keep streams closed during matches.

**Live tuning.** Every setting applies within about 0.1 s, from the page, Elastic or AdvantageScope; they all edit the same NT entries:

```
/Vision/settings/quad_decimate, min_cluster_pixels            detector
/Vision/settings/min_decision_margin, max_single_tag_ambiguity,
                 max_tag_distance_m                           filters
/Vision/settings/stream_max_fps
/Vision/settings/board_*                                      calibration board
/Vision/cameras/<name>/settings/exposure, gain, std_dev_factor, resolution
```

Camera settings apply to that camera only; the rest apply to all cameras. For exposure and gain, **-1 means auto** (NT has no null). Exposure uses the camera's native units (often 100 µs steps; the page reads the range from the camera).

**Saving.** Changes apply immediately but are **not** saved until you press **Save to config** (or set `/Vision/settings/save` true). The Pi writes `config.json` atomically, so a power cut mid-save can't corrupt it. On startup `config.json` always wins, so unsaved experiments vanish on restart. After a tuning session, pull the Pi's `config.json` into the repo.

NT status per camera is under `/Vision/cameras/<name>/`: `fps`, `processMs`, `tagFps`, `connected`, `lastReject`, and `robotPose` (a plain Pose3d you can drag onto AdvantageScope's field view). The robot reads `/Vision/observations`.

## 6. Run on boot

`vision.service` is set up for our Pi (user `choate7407`, the venv's Python). For another setup, change `User`, `WorkingDirectory` and `ExecStart`. `ExecStart` must be the full path to the venv's `python`, because systemd doesn't use your shell's PATH or an activated venv. Then:

```bash
sudo cp vision.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now vision
```

After editing the installed service, run `daemon-reload` and restart.

## 7. Robot side

Copy `robot/batched_vision.py` and `robot/vision_types.py` into your RobotPy project:

```python
from batched_vision import BatchedVision

class Drivetrain(commands2.Subsystem):
    def __init__(self):
        ...
        self.vision = BatchedVision()

    def periodic(self):
        self.pose_estimator.update(self.gyro.getRotation2d(), self.module_positions())
        self.vision.update(self.pose_estimator)
```

**Std devs are tuned on the robot.** Each observation carries `stdDevFactor` = avgTagDistance² / tagCount² × the camera's `std_dev_factor` (set on the Pi). `BatchedVision` multiplies it by coefficients stored as WPILib Preferences, so you tune them in Elastic under `/Preferences/Vision/` and they persist on the roboRIO:

| Preference | Default | Meaning |
|---|---|---|
| `XYStdDevCoefficient` | 0.01 | xy std dev (m) = coefficient × stdDevFactor |
| `ThetaStdDevCoefficient` | 0.03 | heading std dev (rad) = coefficient × stdDevFactor |
| `TrustSingleTagTheta` | false | false = single-tag heading ignored (gyro owns heading) |

Lower coefficient = trust vision more: the pose snaps to vision faster but jitters more.

Capture timestamps are handled for you: each observation carries its age at publish time (measured from the camera driver's frame timestamp when it's available, otherwise from when the frame was read), and NT4 converts the publish timestamp to the robot's clock. Single-tag observations more than 1 m from the current estimate are ignored while enabled (tune `max_single_tag_jump_m`); while disabled everything is accepted so vision can seed your starting pose. `self.vision.last_observations` has the latest batch if you want to log it.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `status=203/EXEC` in `systemctl status` | `ExecStart` path is wrong. Check it with `ls -l <path>`; it's usually a missing username in `/home/<user>/...`. |
| `status=1/FAILURE`, restarting | `main.py` itself failed; the reason is in `journalctl -u vision -n 50 --no-pager` (config typo, bad device path...). |
| Page header says **NOT CALIBRATED** | No calibration for this camera at this resolution. Calibrate on the page, or switch to a resolution you've calibrated. |
| Camera fps low | Auto exposure lengthens frames in dim light: set exposure manually. Also check the mode's max fps in the dropdown and `fps` in the config. |
| Camera fps fine, processed fps low | CPU-bound: detect ms is above the frame period. Raise quad decimate or lower resolution. Check with the page closed, since stream encoding costs CPU too. |
| Two cameras disagree on robot pose | `device` and `robot_to_camera` don't match the same physical camera (see "Which camera is which"), or a wrong yaw/pitch sign. |
| Rejected: robot height wrong | `robot_to_camera` is off (height or pitch sign). Normal at a desk too, since the camera isn't where the config says it is. |
| AdvantageScope can't connect | Nothing is an NT server: set `"standalone": true` for bench tests. Also check your laptop is on the Pi's subnet. |
| Exposure slider does nothing | `v4l2-ctl` isn't installed; see section 1. |

## Tuning notes

**Std devs.** The robot-side coefficients (xy 0.01, theta 0.03) are a starting point, not tuned for our robot. Tune them in Elastic (`/Preferences/Vision/`) while driving: if the pose lags or drifts from vision, lower them; if it jitters or jumps, raise them. Use `std_dev_factor` per camera (on the Pi) to trust a worse camera less.

**Detector.** `quad_decimate` finds tag outlines on an image shrunk by that factor (corners are still refined at full resolution): higher is much faster but loses small, far tags, so it trades range for speed. `min_cluster_pixels` is the smallest candidate outline considered (on the decimated image); it's a noise filter, and WPILib's default of 300 loses far tags, so keep it low (~20). `min_decision_margin` is how confidently a tag's bits were read; raise it if you see phantom tags or jumpy far-tag poses, lower it (~15) if good tags in dim light or at distance get dropped.

If detect time is near or above the frame period (20 ms at 50 fps), raise `quad_decimate` or lower the resolution. The grabber always processes the newest frame, so falling behind costs frame rate, not latency.

## Not included (yet)

A fused single-pose output (by design, the robot fuses per-camera observations itself) and object detection.
