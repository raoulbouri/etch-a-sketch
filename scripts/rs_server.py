"""RealSense camera server: opens the D435 colour stream ONCE and serves frame bursts on request.

Why a server: a calibration session needs a capture after every knob turn. Opening the camera once and
keeping it is faster and gentler than restarting librealsense per capture (on macOS each restart had to
kill the system camera services, which froze the Mac; macOS is now blocked, see rs_capture.sh).

Start it in its own terminal (Linux) and leave it running for the whole session:

    sh scripts/rs_capture.sh serve

Clients (scripts/hand_calibrate.py) send JSON lines to 127.0.0.1:8766:
    {"cmd": "grab", "out": "/abs/dir", "n": 10}  -> saves rgb_00.png ... in out, replies {"ok": true, "saved": n}
    {"cmd": "ping"} / {"cmd": "quit"}
It exits on "quit", Ctrl-C, or after IDLE_S without requests. Only numpy, OpenCV and pyrealsense2 are used.
"""
import json
import os
import signal
import socket
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

HOST, PORT = "127.0.0.1", 8766
IDLE_S = 20 * 60
READY = os.environ.get("RS_READY_FILE")


def owner_ids():
    """If run as root under sudo, hand written files back to the user who ran sudo (not needed on Linux)."""
    try:
        return int(os.environ["SUDO_UID"]), int(os.environ["SUDO_GID"])
    except (KeyError, ValueError):
        return None


def grab(pipe, out: Path, n: int, ids) -> int:
    for _ in range(6):                               # drop queued frames (~1 s at 6 fps): images are current
        pipe.wait_for_frames(3000)
    frames = [np.asanyarray(pipe.wait_for_frames(3000).get_color_frame().get_data()).copy() for _ in range(n)]
    out.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        p = out / f"rgb_{i:02d}.png"
        cv2.imwrite(str(p), f)
        if ids:
            os.chown(p, *ids)
    if ids:
        for d in [out, *out.parents]:                # folders created here belong to the user too
            if d.exists() and d.stat().st_uid == 0 and "test_images" in d.parts:
                os.chown(d, *ids)
    return len(frames)


def main() -> None:
    # the wrapper stops this process with TERM: turn that into SystemExit so the camera is released properly
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)
    pipe.start(cfg)
    try:
        for _ in range(12):                          # ~2 s for auto exposure to settle
            pipe.wait_for_frames(3000)
        if READY:
            Path(READY).touch()                      # tells the wrapper to stop pushing camera services aside
        ids = owner_ids()
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((HOST, PORT))
        srv.listen(1)
        srv.settimeout(5.0)
        print(f"camera streaming; serving on {HOST}:{PORT}. Leave this running; Ctrl-C to stop.", flush=True)
        last = time.time()
        while time.time() - last < IDLE_S:
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                pipe.poll_for_frames()               # keep the stream drained while idle
                continue
            with conn, conn.makefile("rw") as f:
                for line in f:
                    last = time.time()
                    try:
                        req = json.loads(line)
                        if req.get("cmd") == "grab":
                            n = grab(pipe, Path(req["out"]), int(req.get("n", 10)), ids)
                            reply = {"ok": True, "saved": n}
                            print(f"saved {n} frames to {req['out']}", flush=True)
                        elif req.get("cmd") == "ping":
                            reply = {"ok": True}
                        elif req.get("cmd") == "quit":
                            f.write(json.dumps({"ok": True}) + "\n")
                            f.flush()
                            return
                        else:
                            reply = {"ok": False, "error": f"unknown command {req.get('cmd')}"}
                    except Exception as e:          # a failed grab is reported, the server keeps the camera
                        reply = {"ok": False, "error": str(e)}
                    f.write(json.dumps(reply) + "\n")
                    f.flush()
        print(f"no requests for {IDLE_S // 60} min: stopping", flush=True)
    finally:
        pipe.stop()


if __name__ == "__main__":
    sys.exit(main())
