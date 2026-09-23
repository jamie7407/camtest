"""Camera calibration using a ChArUco board, run interactively on your Mac.

For each camera: hold the board in front of it, cover the frame (corners and
edges, not just the center), and vary distance and tilt. Press SPACE to
capture a view once the board outline turns green, 'u' to undo the last
capture, 'c' to calibrate and save once you have enough views, 'q'/ESC to
quit without saving.

    python3 list_cameras.py                     # find out which index is which camera
    python3 calibrate.py --camera 0 --name front_left --width 1280 --height 800

Calibrate at the SAME resolution the camera will run at in config.json --
config.py refuses to load a calibration whose recorded resolution doesn't
match. The saved file goes straight into calibrations/<name>.json, matching
what config.json expects.

Don't have a board yet? Generate one:

    python3 calibrate.py --generate-board board.png

Print it at 100% scale ("fit to page" OFF) and measure a square with a
ruler -- if it's off from the default 35mm, pass --square-length with the
real value in meters next time. (This only matters if some other tool
consumes real-world units from this board; the camera_matrix/dist_coeffs
saved by this script are intrinsics only and don't depend on it.) Mount the
printed board on something flat -- a warped board will hurt accuracy.

This needs OpenCV built with GUI support (the `opencv-python` package, not
`opencv-python-headless`, which is what coprocessor/requirements.txt installs
for the headless Pi). If both are installed and imshow fails, run this in a
separate venv with only `opencv-python`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2

WINDOW = "calibrate"
GRID_COLS, GRID_ROWS = 4, 3


def open_camera(device):
    """Same backend choice as camera_process.open_camera, copied here so this
    script only depends on opencv + numpy, not the rest of the coprocessor stack."""
    if sys.platform == "darwin":
        return cv2.VideoCapture(device, cv2.CAP_AVFOUNDATION)
    if sys.platform.startswith("linux"):
        return cv2.VideoCapture(device, cv2.CAP_V4L2)
    return cv2.VideoCapture(device)


def check_gui_support():
    try:
        cv2.namedWindow(WINDOW)
        cv2.destroyWindow(WINDOW)
    except cv2.error as e:
        raise SystemExit(
            "OpenCV has no GUI support in this environment (imshow failed).\n"
            "Install the non-headless build: pip install opencv-python\n"
            "If opencv-python-headless is also installed it can shadow the GUI build; "
            "use a separate venv with only opencv-python.\n"
            f"Underlying error: {e}"
        )


def generate_board(board, path, dpi):
    cols, rows = board.getChessboardSize()
    px_per_square = max(1, round(board.getSquareLength() / 0.0254 * dpi))
    margin = px_per_square // 2
    img = board.generateImage(
        (cols * px_per_square, rows * px_per_square), marginSize=margin, borderBits=1
    )
    cv2.imwrite(path, img)
    print(f"saved {path} ({img.shape[1]}x{img.shape[0]}px, {dpi} dpi)")
    print(f"print at 100% scale (no 'fit to page'); square = "
          f"{board.getSquareLength() * 1000:.1f}mm, marker = {board.getMarkerLength() * 1000:.1f}mm -- "
          f"measure a printed square to check your printer didn't rescale it")


def cell_for_point(x, y, w, h):
    col = min(GRID_COLS - 1, max(0, int(x / w * GRID_COLS)))
    row = min(GRID_ROWS - 1, max(0, int(y / h * GRID_ROWS)))
    return col, row


def draw_overlay(frame, covered, detected_ok, n_corners, n_captures, min_captures):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    for col in range(GRID_COLS):
        for row in range(GRID_ROWS):
            if (col, row) in covered:
                x0, y0 = col * w // GRID_COLS, row * h // GRID_ROWS
                x1, y1 = (col + 1) * w // GRID_COLS, (row + 1) * h // GRID_ROWS
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 120, 0), -1)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, dst=frame)
    for col in range(1, GRID_COLS):
        x = col * w // GRID_COLS
        cv2.line(frame, (x, 0), (x, h), (80, 80, 80), 1)
    for row in range(1, GRID_ROWS):
        y = row * h // GRID_ROWS
        cv2.line(frame, (0, y), (w, y), (80, 80, 80), 1)

    status_color = (0, 200, 0) if detected_ok else (0, 0, 220)
    status = f"board detected ({n_corners} corners)" if detected_ok else "no board"
    lines = [
        f"captures: {n_captures}/{min_captures} min   coverage: {len(covered)}/{GRID_COLS * GRID_ROWS} cells",
        status,
        "SPACE capture   U undo   C calibrate+save   Q/ESC quit",
    ]
    for i, text in enumerate(lines):
        y = 26 + i * 24
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color if i == 1 else (255, 255, 255), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default="0", help="camera index (see list_cameras.py)")
    ap.add_argument("--name", help="output filename stem -> calibrations/<name>.json")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--fourcc", default="MJPG")
    ap.add_argument("--squares-x", type=int, default=5)
    ap.add_argument("--squares-y", type=int, default=7)
    ap.add_argument("--square-length", type=float, default=0.035, help="meters")
    ap.add_argument("--marker-length", type=float, default=0.026, help="meters")
    ap.add_argument("--dict", default="DICT_4X4_50", help="cv2.aruco dictionary name")
    ap.add_argument("--min-captures", type=int, default=15)
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibrations"))
    ap.add_argument("--generate-board", metavar="PATH", help="write a printable board image to PATH and exit")
    ap.add_argument("--dpi", type=int, default=300, help="for --generate-board")
    args = ap.parse_args()

    dict_id = getattr(cv2.aruco, args.dict, None)
    if dict_id is None:
        raise SystemExit(f"unknown --dict {args.dict!r} (expected something like DICT_4X4_50, DICT_5X5_100, ...)")
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y), args.square_length, args.marker_length, dictionary
    )

    if args.generate_board:
        generate_board(board, args.generate_board, args.dpi)
        return

    if not args.name:
        raise SystemExit("--name is required (e.g. --name front_left) so the calibration lands in the right file")

    check_gui_support()

    device = int(args.camera) if args.camera.isdigit() else args.camera
    cap = open_camera(device)
    if not cap.isOpened():
        raise SystemExit(
            f"couldn't open camera {device}. On macOS, check System Settings > Privacy & Security > "
            "Camera for your terminal app, and run list_cameras.py to confirm the index."
        )
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (actual_w, actual_h) != (args.width, args.height):
        print(f"WARNING: asked for {args.width}x{args.height}, camera gave {actual_w}x{actual_h}. "
              f"The saved calibration will record {actual_w}x{actual_h} -- make sure config.json matches.")
    width, height = actual_w, actual_h

    detector = cv2.aruco.CharucoDetector(board)

    all_obj_points, all_img_points, capture_cells = [], [], []
    covered = set()
    calibrate_now = False

    print(f"[{args.name}] camera ready at {width}x{height}. Move the board around and press SPACE to capture.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("frame grab failed, retrying...")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
        # detectBoard can return mismatched corner/id counts (e.g. board partially out of
        # frame or at a steep angle) -- treat that as "not detected" rather than crashing.
        if charuco_corners is None or charuco_ids is None or len(charuco_corners) != len(charuco_ids):
            charuco_corners = charuco_ids = None
            n_corners = 0
        else:
            n_corners = len(charuco_corners)
        detected_ok = n_corners >= 6

        display = frame.copy()
        if marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(display, marker_corners, marker_ids)
        if detected_ok:
            # cv2.aruco.drawDetectedCornersCharuco asserts on this OpenCV build because
            # detectBoard returns charucoCorners as an (N, 2) array instead of the (N, 1, 2)
            # Point2f Mat it expects (its .total() then comes out as 2N, not N). Draw
            # manually instead -- also sidesteps needing charuco_ids to be well-formed.
            for x, y in charuco_corners.reshape(-1, 2):
                cv2.circle(display, (int(x), int(y)), 5, (0, 255, 0), -1)
        draw_overlay(display, covered, detected_ok, n_corners, len(all_obj_points), args.min_captures)
        cv2.imshow(WINDOW, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            print("quit without saving")
            break
        elif key == ord("u"):
            if all_obj_points:
                all_obj_points.pop()
                all_img_points.pop()
                capture_cells.pop()
                covered = set(capture_cells)
                print(f"undid last capture ({len(all_obj_points)} remaining)")
        elif key == ord(" "):
            if not detected_ok:
                print("board not detected clearly enough, reposition and try again")
                continue
            obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
            if obj_pts is None or len(obj_pts) < 4:
                print("not enough matched points in this view, try again")
                continue
            all_obj_points.append(obj_pts)
            all_img_points.append(img_pts)
            cx, cy = charuco_corners.reshape(-1, 2).mean(axis=0)
            cell = cell_for_point(cx, cy, width, height)
            capture_cells.append(cell)
            covered.add(cell)
            print(f"captured {len(all_obj_points)}/{args.min_captures}")
        elif key == ord("c"):
            if len(all_obj_points) < args.min_captures:
                print(f"need at least {args.min_captures} captures first ({len(all_obj_points)} so far)")
            else:
                calibrate_now = True
                break

    cap.release()
    cv2.destroyAllWindows()

    if not calibrate_now:
        return

    print(f"calibrating from {len(all_obj_points)} views...")
    ret, K, D, _rvecs, _tvecs = cv2.calibrateCamera(
        all_obj_points, all_img_points, (width, height), None, None
    )
    print(f"reprojection RMS error: {ret:.3f} px")
    if ret > 1.0:
        print("WARNING: that's higher than you'd like (aim well under 1.0 px). Consider recalibrating with "
              "more/steadier views, better lighting, or a flatter board.")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.name}.json")
    with open(out_path, "w") as f:
        json.dump({
            "camera_matrix": K.tolist(),
            "dist_coeffs": D.flatten().tolist(),
            "resolution": {"width": width, "height": height},
        }, f, indent=2)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
