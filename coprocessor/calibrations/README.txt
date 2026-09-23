Put one calibration JSON per camera here, named to match config.json.

Easiest: run ../calibrate.py on your Mac with a ChArUco board (see its
docstring, or `python3 calibrate.py --help`). It saves directly into this
folder at the right resolution.

Alternative: export each camera's calibration from PhotonVision (Cameras tab ->
calibration at the SAME resolution you'll run at -> download JSON). That format
loads directly too.

Or write a simple file yourself:
{
  "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "dist_coeffs": [k1, k2, p1, p2, k3],
  "resolution": {"width": 1280, "height": 800}
}
