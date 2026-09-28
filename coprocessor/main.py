"""Multi-camera AprilTag coprocessor.

Spawns one process per camera, collects their pose observations, and publishes
them as ONE NT4 StructArray topic. Publishing happens as soon as any camera has a
result (anything that arrived meanwhile is batched in) -- no fixed publish rate,
so no added latency and no waiting on the slowest camera.

Also: live-tunable settings over NT with a save button (settings.py), and an MJPEG
stream per camera registered under /CameraPublisher so dashboards find it.

    python3 main.py --config config.json
    python3 main.py --config config.json --standalone   # bench test with no robot
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import signal
import socket
import time

from config import load_config


def local_ips() -> list[str]:
    """Best-effort list of this machine's IPv4 addresses (for dashboard stream URLs)."""
    ips = set()
    for target in ("10.255.255.255", "8.8.8.8"):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect((target, 1))  # UDP connect sends nothing; just picks the route
            ips.add(s.getsockname()[0])
            s.close()
        except OSError:
            pass
    try:
        ips.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    real = sorted(ip for ip in ips if not ip.startswith("127."))
    return real or ["127.0.0.1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--standalone", action="store_true",
                    help="run our own NT server (bench testing without a roboRIO)")
    ap.add_argument("--server", help="NT server IP override (e.g. 127.0.0.1 for sim)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.standalone:
        cfg.standalone = True
    if args.server:
        cfg.server = args.server

    # Start camera processes BEFORE initializing NetworkTables ("spawn" also keeps
    # them from inheriting any NT/thread state from this process).
    from camera_process import run_camera

    ctx = mp.get_context("spawn")
    q = ctx.Queue(maxsize=256)
    procs, ctrls = {}, {}

    # Every camera's page lists all cameras so one page can switch between them.
    peers = [{"name": c.name, "port": c.stream_port} for c in cfg.cameras]

    def start(cam):
        ctrls[cam.id] = ctx.Queue()
        p = ctx.Process(target=run_camera, name=cam.name, daemon=True,
                        args=(cam, cfg.field_layout, cfg.detector, cfg.solver, cfg.stream,
                              q, ctrls[cam.id], peers))
        p.start()
        procs[cam.id] = p

    for cam in cfg.cameras:
        start(cam)

    import ntcore
    from wpimath.geometry import Pose3d, Quaternion, Rotation3d, Translation3d

    from settings import LiveSettings
    from vision_types import VisionObservation

    inst = ntcore.NetworkTableInstance.getDefault()
    if cfg.standalone:
        inst.startServer()
        print("NT: running standalone server", flush=True)
    else:
        inst.startClient4(cfg.client_name)
        if cfg.server:
            inst.setServer(cfg.server)
        elif cfg.team is not None:
            inst.setServerTeam(cfg.team)
        else:
            raise SystemExit("config needs nt.team or nt.server (or use --standalone)")
        print(f"NT: client -> {cfg.server or f'team {cfg.team}'}", flush=True)

    opts = ntcore.PubSubOptions(sendAll=True, keepDuplicates=True)
    obs_pub = inst.getStructArrayTopic(cfg.topic, VisionObservation).publish(opts)
    base = cfg.topic.rsplit("/", 1)[0] or "/Vision"
    heartbeat_pub = inst.getIntegerTopic(f"{base}/heartbeat").publish()
    settings = LiveSettings(inst, cfg, base)

    stat_pubs = {}
    for cam in cfg.cameras:
        t = inst.getTable(f"{base}/cameras/{cam.name}")
        t.getIntegerTopic("id").publish().set(cam.id)
        stat_pubs[cam.id] = {
            "connected": t.getBooleanTopic("connected").publish(),
            "fps": t.getDoubleTopic("fps").publish(),
            "processMs": t.getDoubleTopic("processMs").publish(),
            "tagFps": t.getDoubleTopic("tagFps").publish(),
            "streamFps": t.getDoubleTopic("streamFps").publish(),
            "streamViewers": t.getIntegerTopic("streamViewers").publish(),
            "lastReject": t.getStringTopic("lastReject").publish(),
            # Same pose as in the batched topic, as a plain Pose3d so AdvantageScope can
            # drag it straight onto a field view. Debug only; the robot reads the batch.
            "robotPose": t.getStructTopic("robotPose", Pose3d).publish(),
        }

    # Register streams where Elastic / Shuffleboard look for cameras.
    stream_pubs = {}
    if cfg.stream.get("enabled", True):
        for cam in cfg.cameras:
            for suffix, path in (("", "stream.mjpg"), ("_raw", "raw.mjpg")):
                t = inst.getTable(f"/CameraPublisher/{cam.name}{suffix}")
                stream_pubs[(cam.id, path)] = t.getStringArrayTopic("streams").publish()
                t.getStringTopic("source").publish().set(f"vision:{cam.name}")
    last_ips = None

    def shutdown(*_):
        for p in procs.values():
            p.terminate()
        os._exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Give every camera the full current settings so its web page can show them
    # (after that it only receives changes).
    for cam in cfg.cameras:
        ctrls[cam.id].put({**settings.full_camera_settings(cam.id), "_status": settings.status_text})
    last_status = settings.status_text

    heartbeat = 0
    last_health = last_settings = 0.0

    def handle(msg, batch):
        if msg[0] == "obs":
            (_, cam_id, cap_t, x, y, z, qw, qx, qy, qz, factor, n, dist, amb, mask) = msg
            batch.append((cam_id, cap_t, Pose3d(Translation3d(x, y, z),
                          Rotation3d(Quaternion(qw, qx, qy, qz))), factor, n, dist, amb, mask))
        elif msg[0] == "stats":
            _, cam_id, connected, fps, proc_ms, tag_fps, stream_fps, viewers, reject = msg
            p = stat_pubs[cam_id]
            p["connected"].set(bool(connected)); p["fps"].set(fps); p["processMs"].set(proc_ms)
            p["tagFps"].set(tag_fps); p["streamFps"].set(stream_fps)
            p["streamViewers"].set(viewers); p["lastReject"].set(reject)
        elif msg[0] == "set":  # from a camera's web settings page
            settings.set_from_web(msg[1], msg[2])
        elif msg[0] == "save":
            settings.request_save()

    while True:
        batch = []
        try:
            handle(q.get(timeout=0.1), batch)
            while True:  # batch in anything else that's already waiting
                handle(q.get_nowait(), batch)
        except queue.Empty:
            pass

        if batch:
            now_mono = time.monotonic()
            obs_pub.set(
                [VisionObservation(cam_id, pose, now_mono - cap_t, factor, n, dist, amb, mask)
                 for cam_id, cap_t, pose, factor, n, dist, amb, mask in batch],
                ntcore._now(),
            )
            for cam_id, _cap_t, pose, *_ in batch:
                stat_pubs[cam_id]["robotPose"].set(pose)

        now = time.monotonic()
        if now - last_settings > 0.1:  # live settings -> camera processes
            last_settings = now
            g, per_cam = settings.poll()
            for cam in cfg.cameras:
                msg = {**g, **per_cam.get(cam.id, {})}
                if msg:
                    ctrls[cam.id].put(msg)
            if settings.status_text != last_status:  # "unsaved changes" / "saved ..." on the web pages
                last_status = settings.status_text
                for cam in cfg.cameras:
                    ctrls[cam.id].put({"_status": last_status})

        if now - last_health > 1.0:
            last_health = now
            heartbeat += 1
            heartbeat_pub.set(heartbeat)
            for cam in cfg.cameras:  # restart crashed camera processes with current settings
                if not procs[cam.id].is_alive():
                    print(f"[{cam.name}] process died, restarting", flush=True)
                    start(cam)
                    ctrls[cam.id].put({**settings.full_camera_settings(cam.id), "_status": settings.status_text})
            ips = local_ips()  # IP can change after boot (robot radio DHCP), so refresh
            if ips != last_ips:
                last_ips = ips
                for (cam_id, path), pub in stream_pubs.items():
                    port = cfg.cameras[cam_id].stream_port
                    pub.set([f"mjpg:http://{ip}:{port}/{path}" for ip in ips])


if __name__ == "__main__":
    main()
