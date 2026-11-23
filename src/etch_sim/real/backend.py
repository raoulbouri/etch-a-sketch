"""Skills backend for the real Kinova Gen3, as a client of the aims-kinova motion server.

This process never talks to the arm. It sends skill-level requests to `kinova.server`, which holds
the Kortex session and checks every motion against the envelope a person approved when starting
it. If the server refuses or a motion fails, ArmRefused is raised and the script stops: nothing is
retried here.

Knob poses are taught once (scripts/teach_pose.py) with the toy held in a fixture; re-localizing the
board with the wrist camera comes later. The pen is measured from a scene camera with the same
perception code as in simulation.
"""
from __future__ import annotations

import json
import math
import socket
import time
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from .. import perception as P
from .. import run_tracker as R
from .. import sketch_extract as S
from ..interface import Axis, EraseResult, RotateResult
from ..mj.toy_model import ToyGeometry

POSES_FILE = Path(__file__).resolve().parents[3] / "hardware" / "taught_poses.json"


def down_yaw_deg(tool_pose: dict) -> float:
    """Wrist yaw of a tool pointing down, as theta_z for the (180, 0, theta_z) Euler form the taught poses use.

    Kortex can report the same downward orientation as (~0, ~180, theta_z + 180) (blue arm, 2026-11-22, after a 180 deg
    knob turn). Feeding that theta_z back with theta_x=180, theta_y=0 asks for a 180 deg wrist spin. The tool x axis
    heading, cos(ty) * (cos tz, sin tz), is the same in both forms.
    """
    ty, tz = math.radians(tool_pose["theta_y"]), math.radians(tool_pose["theta_z"])
    return math.degrees(math.atan2(math.sin(tz) * math.cos(ty), math.cos(tz) * math.cos(ty)))


class ArmRefused(RuntimeError):
    """The motion server rejected a command or a motion did not complete. Needs a human."""


class ArmClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8765, timeout_s: float = 180.0):
        self.sock = socket.create_connection((host, port), timeout=timeout_s)
        self.f = self.sock.makefile("rw")

    def call(self, cmd: str, **kw) -> dict:
        self.f.write(json.dumps({"cmd": cmd, **kw}) + "\n")
        self.f.flush()
        line = self.f.readline()
        if not line:
            raise ArmRefused("motion server closed the connection")
        reply = json.loads(line)
        if not reply.get("ok"):
            why = reply.get("error") or reply.get("event")
            extra = {k: reply[k] for k in ("abort_details", "rotated_deg", "chunks", "routine_stopped") if k in reply}
            raise ArmRefused(f"{cmd}: {why}" + (f" {extra}" if extra else "")
                             + ("  (abort_details 18 = ROBOT_MOVEMENT_IN_PROGRESS: another controller, e.g. the Web App"
                                " or the Xbox pad, is driving the arm)" if str(reply.get("abort_details")) == "18" else ""))
        return reply

    def close(self) -> None:
        self.f.close()
        self.sock.close()


def webcam(index: int = 0, width: int = 1920, height: int = 1080) -> Callable[[], np.ndarray]:
    """Returns a function that grabs one RGB frame from a USB camera."""
    import cv2
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def grab() -> np.ndarray:
        for _ in range(3):                               # drop buffered frames so the image is current
            ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f"camera {index} returned no frame")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return grab


class RealBackend:
    def __init__(self, client: ArmClient, poses: dict | None = None,
                 camera: Callable[[], np.ndarray] | None = None, geometry: ToyGeometry | None = None):
        self.c = client
        self.poses = poses if poses is not None else json.loads(POSES_FILE.read_text())
        missing = [f"{a}.{k}" for a in ("x", "y") for k in ("above", "grasp") if not self.poses["knobs"][a].get(k)]
        if missing:
            raise ValueError(f"taught poses missing: {missing}. Teach them with scripts/teach_pose.py")
        self.camera = camera
        self.g = geometry or ToyGeometry()
        self.n_switches = 0
        self.held: Axis | None = None
        self._last_axis: Axis | None = None
        # one run = from the first board sighting until the next erase (run_tracker rules 1-3)
        self._track_p = R.TrackParams()
        self._tracker = R.RunTracker(R.knobs_from_toy(), self._track_p)
        self._run_started = False
        self._moves: list[tuple[str, float]] = []           # (axis, knob degrees) since the last observation
        self._theta_z = down_yaw_deg(self.c.call("status")["state"]["tool_pose"])
        self._t0 = time.time()
        self.log: list[dict] = []

    @property
    def clock_s(self) -> float:
        return time.time() - self._t0

    # ---- arm ---------------------------------------------------------------------
    def _move(self, pose: list[float]) -> None:
        """Go to a taught position, keeping the wrist's current rotation about the vertical.

        After turning a knob the wrist is at an arbitrary angle; the knob is round, so there is no
        reason to unwind it.
        """
        x, y, z, tx, ty, _ = pose
        r = self.c.call("move_pose", x=x, y=y, z=z, theta_x=tx, theta_y=ty, theta_z=self._theta_z)
        self._theta_z = down_yaw_deg(r["tool_pose"])

    def grasp_knob(self, axis: Axis) -> bool:
        if self.held == axis:
            return True
        if self.held is not None:
            self.release()
        knob, grip = self.poses["knobs"][axis], self.poses["gripper"]
        switching = self._last_axis is not None and self._last_axis != axis
        self.c.call("gripper", position=grip["pre"])
        self._move(knob["above"])
        self._move(knob["grasp"])
        r = self.c.call("gripper", position=grip["closed"])
        held = "STALLED" in r.get("event", "") and grip["held_min"] <= r["position"] <= grip["held_max"]
        self.log.append({"t": self.clock_s, "skill": "grasp_knob", "axis": axis, "held": held, "gripper": r.get("position")})
        self._last_axis = axis
        self.n_switches += int(switching)
        if not held:                                     # fingers closed on nothing, or on the wrong thing
            self.c.call("gripper", position=grip["pre"])
            self._move(knob["above"])
            return False
        # A stall leaves the gripper command running, and the arm then fails the next action (rotate: ACTION_ABORT,
        # abort_details 1, blue arm 2026-11-22). The motion server's gripper ends a stalled command with Base.Stop()
        # (aims-kinova patch gripper-stall-stop); retargeting the gripper from here did not help.
        self.held = axis
        return True

    def rotate(self, deg: float) -> RotateResult:
        if self.held is None:
            return RotateResult(deg, 0.0, ok=False, note="no knob held")
        sign = float(self.poses["j6_deg_per_knob_deg"][self.held])
        r = self.c.call("rotate", deg=deg * sign, speed=self.poses.get("knob_speed_dps"))
        self._theta_z = down_yaw_deg(r["tool_pose"])
        wrist = float(r["rotated_deg"]) / sign
        self._moves.append((self.held, wrist))               # what the wrist turned; the knob may have slipped
        self.log.append({"t": self.clock_s, "skill": "rotate", "axis": self.held, "deg": deg, "wrist_deg": wrist})
        return RotateResult(deg, wrist)

    def release(self) -> None:
        if self.held is None:
            return
        self.c.call("gripper", position=self.poses["gripper"]["pre"])
        self._move(self.poses["knobs"][self.held]["above"])
        self.log.append({"t": self.clock_s, "skill": "release", "axis": self.held})
        self.held = None

    # ---- camera ------------------------------------------------------------------
    def _frame(self) -> np.ndarray:
        if self.camera is None:
            raise RuntimeError("no scene camera configured")
        return self.camera()

    def photo(self, camera: str = "scene") -> Image.Image:
        if camera != "scene":
            raise NotImplementedError("wrist camera frames are not wired up yet")
        return Image.fromarray(self._frame())

    def rectified(self) -> np.ndarray | None:
        img = self._frame()
        H = P.board_homography(P.detect_markers(img), self.g)
        return None if H is None else P.rectify(img, H, self.g)

    def _lines(self, rect: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Line mask (real-image extraction) and the pixels the camera can judge."""
        ex = S.extract(rect, None, S.ExtractParams())
        return ex.mask, R.visible_from(ex.flat, ex.unknown, self._track_p)

    def locate_board(self) -> bool:
        """True if the scene camera sees enough markers to measure in the screen frame. The first sighting
        starts the run: what is on the screen then is never taken for the pen's line."""
        rect = self.rectified()
        if rect is not None and not self._run_started:
            self._tracker.start(self._lines(rect)[0])
            self._run_started, self._moves = True, []
        return rect is not None

    def observe_pen(self) -> tuple[float, float]:
        rect = self.rectified()
        if rect is not None:
            mask, vis = self._lines(rect)
            if not self._run_started:
                self._tracker.start(mask)
                self._run_started = True
            else:
                self._tracker.update(mask, self._moves, vis)
            self._moves = []
        pen = self._tracker.pen_mm
        self.log.append({"t": self.clock_s, "skill": "observe_pen", "pen": pen})
        return pen if pen is not None else (float("nan"), float("nan"))

    def erase(self, peak_accel_ms2: float = 9.0, duration_s: float = 30.0) -> EraseResult:
        # When built: an erase ends the run; set self._run_started = False so the next board sighting starts a new one.
        raise NotImplementedError("erase on the arm is a separate routine with its own approval; not built yet")
