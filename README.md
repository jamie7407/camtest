# Batched multi-camera AprilTag vision (Orange Pi proof of concept)

One process per camera does capture → AprilTag detection → pose solve → std devs on the coprocessor. Each camera also serves a full-resolution annotated stream, and the important settings are live-tunable over NetworkTables with a save button. All cameras publish into **one** NT4 struct-array topic, so the robot does a single NT read per loop and just loops `addVisionMeasurement()` — no per-camera photonlibpy deserialization in RobotPy.

Adding a camera is just adding an entry to `config.json`. Nothing else changes on either side.

```
coprocessor/          -> goes on the Orange Pi
  main.py             spawns camera processes, publishes the batched topic
  camera_process.py   per-camera capture/detect/solve (own process = own CPU core)
  pose_solver.py      multi-tag SQPNP / single-tag IPPE, filters, 6328 std devs
  config.py           config + calibration loading (accepts PhotonVision exports)
  settings.py         live NT settings + save-to-config.json button
  stream_server.py    per-camera MJPEG streams (annotated + raw), encode only when watched
  camera_controls.py  exposure/gain (v4l2-ctl on Linux, OpenCV fallback)
  vision_types.py     the NT4 struct (must be identical on both sides)
  test_synthetic.py   renders tags from known poses and checks the math
  list_cameras.py     finds cameras, saves a snapshot from each
  calibrate.py        ChArUco camera calibration, run on your Mac
  config.example.json, vision.service, calibrations/
robot/                -> goes in your RobotPy project
  batched_vision.py   reads the topic, feeds your pose estimator
  vision_types.py     identical copy
```

## What's been verified

Tested in a sandbox (not yet on real hardware): the synthetic test renders real tag36h11 images on the 2026 field from known robot poses through a pitched/yawed/offset camera. Median error was 0.3 cm multi-tag and 0.7 cm single-tag. A full end-to-end run (two synthetic camera feeds → `main.py` → NT4 → `batched_vision.py` → `SwerveDrive4PoseEstimator` started 1.8 m off) converged to 1.8 cm of truth with both cameras arriving on the one topic. The streams were checked at full resolution with zero encoding when unwatched. Live settings were checked end to end: raising `xy_coefficient` 5× scaled every observation's std dev exactly 5×, tightening max distance rejected frames with the right reason on the stream, and the save button wrote the changed values atomically. Exposure control via `v4l2-ctl` was checked against real `--list-ctrls` output but not on a physical camera.

Two things found along the way that are worth knowing even if you never use this code. First, WPILib's `AprilTagDetector` defaults `minClusterPixels` to 300, which silently drops small or far tags once `quadDecimate` is 2; 20 found more tags than no decimation at about a third of the CPU. Second, OpenCV's generated tag36h11 images are rotated 180° from the official FRC tags, so don't print test tags from `cv2.aruco`; use the official AprilRobotics images.

## Developing on a Mac (recommended dev loop)

Everything runs on macOS, and it's the easiest way to iterate: robot code in RobotPy simulation, the vision code, and AdvantageScope all on one machine.

```bash
cd coprocessor
pip3 install -r requirements.txt
python3 test_synthetic.py          # should print PASS
python3 list_cameras.py            # which index is which camera (saves snapshots)
```

The first time you open a camera, macOS asks for camera permission for whatever app launched Python (Terminal, iTerm, VS Code). If `list_cameras.py` finds nothing, check System Settings > Privacy & Security > Camera.

In `config.json`, use camera **indices** on a Mac (`"device": 0`), not `/dev/...` paths. The order can change when you replug cameras, and the built-in FaceTime camera or a nearby iPhone (Continuity Camera) may take index 0, so rerun `list_cameras.py` if things move. Manual `exposure` is ignored on macOS (it's a Linux/V4L2 feature); cameras run on auto exposure, which is fine for a desk test but not for a moving robot.

Two ways to run it:

```bash
# Vision only: this process is the NT server. Point AdvantageScope at localhost.
python3 main.py --config config.json --standalone

# Full loop: start your robot code in sim first (it becomes the NT server)...
python -m robotpy sim
# ...then in another terminal, connect vision to it:
python3 main.py --config config.json --server 127.0.0.1
```

Everything you build this way carries over to the Pi unchanged except `config.json` (device paths, and exposure once you're on Linux). The Mac's timing won't represent the Pi, though: an Apple Silicon Mac detects tags far faster, so judge `processMs` and frame rate on the real coprocessor.

## 1. Set up the Orange Pi (use a spare SD card)

Flash a fresh Ubuntu/Armbian image on a **second** SD card so your competition PhotonVision card stays untouched. Then:

```bash
sudo apt install -y python3-pip v4l-utils
cd ~/vision/coprocessor
pip install -r requirements.txt        # add --break-system-packages on newer Ubuntu/Debian, or use a venv
python3 test_synthetic.py              # should print PASS
```

If pip can't find `robotpy-*` / `pyntcore` wheels, your Python or OS version is probably older than RobotPy's aarch64 wheels support; check RobotPy's install docs for supported platforms before trying to build from source.

## 2. Find your cameras

```bash
v4l2-ctl --list-devices
ls -l /dev/v4l/by-path/
v4l2-ctl -d /dev/video0 --list-formats-ext    # which resolutions/fps each format supports
```

On the Pi, use the `/dev/v4l/by-path/...-video-index0` paths in `config.json`. They're tied to the physical USB port, so identical cameras can't swap places between reboots the way `/dev/video0` and `/dev/video2` can. Pick a width/height/fps that's actually listed under MJPG.

## 3. Calibrate

Each camera needs its own calibration **at the exact resolution you run at** (the loader refuses mismatches). Calibration belongs to the camera + lens + resolution, not the computer, so you can calibrate anywhere and carry the JSON over:

- **`calibrate.py` on your Mac (easiest):** plug the camera into your Mac, print a ChArUco board (`python3 calibrate.py --generate-board board.png`), then `python3 calibrate.py --camera 0 --name front_left --width 1280 --height 800` -- hold the board in view, press SPACE to capture views covering the whole frame at different angles, then `c` to calibrate and save straight into `calibrations/front_left.json`. See the script's docstring for details.
- **Swap SD cards:** boot your PhotonVision card on the Pi, calibrate each camera, download the JSON from the Cameras tab, swap back.
- **PhotonVision on a laptop:** it has desktop builds; plug the cameras in and calibrate there.

Don't change the lens focus after calibrating. Drop the files in `calibrations/`. Both `calibrate.py`'s output and the PhotonVision JSON format load directly; if a file doesn't, the loader's error says which keys it couldn't find.

## 4. Configure

`cp config.example.json config.json` and edit. `robot_to_camera` uses WPILib conventions: x forward, y left, z up (meters from robot center at floor level); **pitch negative = camera tilted up**; yaw is CCW-positive (a rear camera is ~180°).

Low exposure matters more than anything else for tag detection on a moving robot, since motion blur kills detections. Tune it live while watching the stream (see section 7): drop exposure until tags are crisp while you wave one around, raise gain if the image gets too dark, then save.

## 5. Bench test without a robot

```bash
python3 main.py --config config.json --standalone
```

This runs its own NT server. Open `http://<pi-ip>:1181/` in a browser to see the first camera, and connect AdvantageScope or Elastic to the Pi's IP: `/Vision/observations` decodes natively (it's a struct), and `/Vision/cameras/<name>/` shows `fps`, `processMs`, `tagFps`, `connected`, and `lastReject` per camera. Hold a printed tag in front of each camera and check it gets a green outline.

To test against robot code running in **simulation** on a laptop: `python3 main.py --config config.json --server <laptop IP>`.

## 6. Robot side

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

That's it. Capture timestamps are handled for you: each observation carries its age at publish time, and NT4 converts the publish timestamp to the robot's clock (on the roboRIO that's the same FPGA timebase the pose estimator uses). Single-tag observations more than 1 m from the current estimate are ignored while enabled (tune `max_single_tag_jump_m`); while disabled everything is accepted so vision can seed your starting pose. `self.vision.last_observations` has the latest batch if you want to log it.

## 7. Streams, live tuning, and saving

**Streams.** Each camera serves `http://<pi-ip>:<port>/`, where the port is `stream.base_port` + the camera's position in the list (1181, 1182, ...). The page shows two feeds. `/stream.mjpg` is annotated: green outlines for tags used in the pose, orange for tags filtered out, a red dot on each tag's corner 0, and a header with fps, detection time, exposure, the current pose, or **the reason a frame was rejected** (tags too far, ambiguity too high, robot height wrong, which usually means a bad `robot_to_camera`). `/raw.mjpg` is untouched frames.

Streams are full resolution and only encode while someone is watching, so an unwatched stream costs nothing. `stream.max_fps` (default 30) caps how often a new frame is encoded *while watched*, because full-res JPEG encoding shares the Pi's cores with detection. `streamFps` and `streamViewers` under `/Vision/cameras/<name>/` show what it's doing. Both streams are registered under `/CameraPublisher`, so Elastic's camera widget lists them automatically. The field network caps robot bandwidth, so during matches keep streams closed or view them in the pits.

**Live tuning.** Every setting below appears in NT and applies within about 0.1 s. Edit them from Elastic (number fields or sliders, toggle for the boolean) or AdvantageScope:

```
/Vision/settings/quad_decimate, min_cluster_pixels            detector
/Vision/settings/min_decision_margin, max_single_tag_ambiguity,
                 max_tag_distance_m                           filters
/Vision/settings/xy_coefficient, theta_coefficient,
                 trust_single_tag_theta                       std devs
/Vision/settings/stream_max_fps
/Vision/cameras/<name>/settings/exposure, gain, std_dev_factor
```

For exposure and gain, **-1 means auto / leave alone** (NT has no null). Exposure uses the camera's native units, so check its range with `v4l2-ctl -d <device> --list-ctrls`. On Linux, exposure and gain are set with `v4l2-ctl` when it's installed (`sudo apt install v4l-utils`), because many USB cameras ignore OpenCV's exposure property.

**Saving.** Changes apply immediately but are **not** saved until you set `/Vision/settings/save` to true (it resets itself). The Pi then writes the current values into `config.json` atomically, so a power cut mid-save can't corrupt it. `/Vision/settings/status` shows `unsaved changes`, `saved HH:MM:SS`, or an error. On startup, `config.json` always wins: every NT entry is reset from the file, so unsaved experiments vanish on restart. After a tuning session, copy the Pi's `config.json` into your repo so it's version-controlled.

Resolution and calibration aren't live settings on purpose: changing resolution invalidates the calibration, so those stay file edits.

## 8. Run on boot

Edit the user/paths in `vision.service`, then follow the commands at the top of that file.

## Tuning notes

The std-dev coefficients (`xy_coefficient` 0.01, `theta_coefficient` 0.03) and the `distance² / tagCount²` shape are copied from 6328's 2026 code. They're a starting point, not tuned for your cameras. Use `std_dev_factor` per camera to trust a worse camera less. Single-tag headings are sent with infinite theta std dev (heading comes from the gyro) unless you set `trust_single_tag_theta`.

If `processMs` is near or above your frame period (20 ms at 50 fps), raise `quad_decimate` or lower the resolution. The grabber always processes the newest frame, so falling behind costs frame rate, not latency.

## Not included (yet)

A web settings UI (settings are edited through NT dashboards instead), fused single-pose output (deliberately; see our earlier discussion), and object detection.
