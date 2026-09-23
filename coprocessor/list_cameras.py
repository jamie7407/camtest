"""Find your cameras: probes indices 0-7, prints what each one reports, and saves a
snapshot (camera_<index>.jpg) so you can see which physical camera is which.

    python3 list_cameras.py
    python3 list_cameras.py --width 1280 --height 800   # check a specific mode

On macOS the order can change when cameras are replugged, and the built-in FaceTime
camera / an iPhone (Continuity Camera) may take index 0. Re-run this if things move.
On Linux, prefer the /dev/v4l/by-path/ paths in config.json instead of indices.
"""

import argparse

import cv2

from camera_process import open_camera


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-index", type=int, default=8)
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    args = ap.parse_args()

    found = 0
    for i in range(args.max_index):
        cap = open_camera(i)
        if not cap.isOpened():
            continue
        if args.width and args.height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        for _ in range(5):  # let auto-exposure settle a little
            ok, frame = cap.read()
        if ok:
            h, w = frame.shape[:2]
            cv2.imwrite(f"camera_{i}.jpg", frame)
            print(f"index {i}: {w}x{h} @ {cap.get(cv2.CAP_PROP_FPS):.0f} fps reported -> saved camera_{i}.jpg")
            found += 1
        else:
            print(f"index {i}: opened but returned no frames")
        cap.release()
    if not found:
        print("No cameras found. On macOS, make sure your terminal app has camera permission "
              "(System Settings > Privacy & Security > Camera).")


if __name__ == "__main__":
    main()
