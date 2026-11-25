"""Visit wrist-camera views through the aims-kinova motion server and save a frame plus the arm status at each.

The motion server runs a bounded routine that a person approved when starting it; any refusal or failure ends the
routine and this run (nothing retries). Used by scripts/teach_knobs.py and scripts/relocate.py.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from . import board_model as B
from . import wrist_camera as W


class ViewRunner:
    """One motion-server connection plus the wrist camera stream, for step-by-step move / capture."""

    def __init__(self, port: int = 8765, camera_ip: str | None = None):
        from .backend import ArmClient
        self.c = ArmClient(port=port)
        st = self.c.call("status")["state"]
        if st.get("arm_state") != "ARMSTATE_SERVOING_READY":
            self.c.close()
            raise SystemExit(f"arm is {st.get('arm_state')}, not SERVOING_READY")
        self.grab = W.RtspGrabber(camera_ip or st["ip"])

    def status(self) -> dict:
        return self.c.call("status")["state"]

    def move(self, p, label: str = "") -> None:
        print(f"move{(' ' + label) if label else ''}: {[round(v, 4) for v in p]}", flush=True)
        self.c.call("move_pose", x=p[0], y=p[1], z=p[2], theta_x=p[3], theta_y=p[4], theta_z=p[5])

    def capture(self, stem: Path, extra: dict | None = None) -> dict:
        """Settle, grab a frame, save it with the arm status; returns the detected markers {id: corners}."""
        time.sleep(0.8)
        before = self.status()
        img = self.grab.frame_after(time.time() + 0.2)
        after = self.status()
        still = np.max(np.abs(np.subtract(before["joint_angles_deg"], after["joint_angles_deg"]))) <= 0.05
        m = B.detect_markers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        cv2.imwrite(f"{stem}.png", img)
        stem.with_suffix(".json").write_text(json.dumps({"status": after, "t": time.time(), **(extra or {})}, indent=1))
        print(f"  {stem.name}: markers {sorted(m)}" + ("" if still else "  (arm still moving: frame unreliable)"), flush=True)
        return m if still else {}

    def close(self) -> None:
        self.grab.close()
        self.c.close()


def run_views(moves: list[dict], out_dir: Path, prefix: str = "view", port: int = 8765) -> int:
    """moves: [{"pose": [x, y, z, tx, ty, tz], "capture": bool}, ...]. Returns the number of frames saved."""
    from .backend import ArmRefused
    r = ViewRunner(port)
    n0, k = len(list(out_dir.glob(f"{prefix}_*.png"))), 0
    try:
        for j, mv in enumerate(moves, 1):
            r.move(mv["pose"], f"{j}/{len(moves)}{'' if mv['capture'] else ' (via)'}")
            if mv["capture"]:
                r.capture(out_dir / f"{prefix}_{n0 + k:02d}", {"planned": mv["pose"]})
                k += 1
    except ArmRefused as e:
        print(f"REFUSED: {e}\nThe routine has ended; nothing retries. Frames saved so far are kept.")
        raise SystemExit(1)
    finally:
        r.close()
    return k


def load_views(out_dir: Path, prefix: str = "view", min_markers: int = 2) -> list[tuple[dict, dict]]:
    views = []
    for p in sorted(out_dir.glob(f"{prefix}_*.png")):
        m = B.detect_markers(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))
        if len(m) >= min_markers:
            views.append((m, json.loads(p.with_suffix(".json").read_text())["status"]))
    return views
