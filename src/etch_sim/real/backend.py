"""Skills backend for the real Kinova Gen3, as a client of the aims-kinova motion server.

This process never talks to the arm. It sends skill-level requests to `kinova.server`, which holds
the Kortex session and checks every motion against the envelope a person approved when starting
it. If the server refuses or a motion fails, ArmRefused is raised and the script stops: nothing is
retried here.

Knob poses are taught once (scripts/teach_pose.py) with the toy held in a fixture; re-localizing the
board with the wrist camera comes later.

The pen is measured from the scene camera: a burst of frames (camera.RealSenseServer), rectified to the
screen frame, line extracted with sketch_extract, pen followed by run_tracker. The screen is mapped either
from the four hand-clicked corners in hardware/scene_camera_screen.json (board="corners", the default:
the real toy carries no markers yet; camera and toy must not move after picking) or from the ArUco
markers as in simulation (board="markers").
"""
from __future__ import annotations

import json
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
            raise ArmRefused(f"{cmd}: {reply.get('error') or reply.get('event')}")
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


Frames = Callable[[], "np.ndarray | list[np.ndarray]"]      # one RGB frame or a burst


class RealBackend:
    def __init__(self, client: ArmClient, poses: dict | None = None, camera: Frames | None = None,
                 geometry: ToyGeometry | None = None, board: str = "corners", corners: np.ndarray | None = None):
        """camera: returns one RGB frame or a burst (camera.RealSenseServer, webcam). board: "corners" maps the
        screen from `corners` (toy order BL, BR, TR, TL, image px), else from hardware/scene_camera_screen.json;
        "markers" uses the ArUco markers."""
        self.c = client
        self.poses = poses if poses is not None else json.loads(POSES_FILE.read_text())
        missing = [f"{a}.{k}" for a in ("x", "y") for k in ("above", "grasp") if not self.poses["knobs"][a].get(k)]
        if missing:
            raise ValueError(f"taught poses missing: {missing}. Teach them with scripts/teach_pose.py")
        self.camera = camera
        self.g = geometry or ToyGeometry()
        if board not in ("corners", "markers"):
            raise ValueError(f"board must be 'corners' or 'markers', not {board!r}")
        self.board = board
        if board == "markers":
            self.corners, self._xp = None, S.ExtractParams()
        elif corners is not None:
            self.corners, self._xp = np.asarray(corners, np.float32), S.ExtractParams(toy_order=True)
        else:                                                # None if never picked: reported at the first capture
            self.corners, self._xp = S.load_screen_corners(), S.screen_params()
        self.n_switches = 0
        self.held: Axis | None = None
        self._last_axis: Axis | None = None
        # one run = from the first board sighting until the next erase (run_tracker rules 1-3)
        self._track_p = R.TrackParams()
        self._tracker = R.RunTracker(R.knobs_from_toy(), self._track_p, screen_mm=self.g.screen_mm)
        self._run_started = False
        self._moves: list[tuple[str, float]] = []           # (axis, knob degrees) since the last observation
        self._theta_z = float(self.c.call("status")["state"]["tool_pose"]["theta_z"])
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
        self._theta_z = float(r["tool_pose"]["theta_z"])

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
        self.held = axis
        return True

    def rotate(self, deg: float) -> RotateResult:
        if self.held is None:
            return RotateResult(deg, 0.0, ok=False, note="no knob held")
        sign = float(self.poses["j6_deg_per_knob_deg"][self.held])
        r = self.c.call("rotate", deg=deg * sign, speed=self.poses.get("knob_speed_dps"))
        self._theta_z = float(r["tool_pose"]["theta_z"])
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
    def _frames(self) -> list[np.ndarray]:
        if self.camera is None:
            raise RuntimeError("no scene camera configured")
        f = self.camera()
        return [f] if isinstance(f, np.ndarray) else list(f)

    def photo(self, camera: str = "scene") -> Image.Image:
        if camera != "scene":
            raise NotImplementedError("wrist camera frames are not wired up yet")
        return Image.fromarray(self._frames()[0])

    def _extract(self) -> S.Extraction | None:
        """One capture: rectify the burst to the screen frame and extract the line. None if the board is not found
        (markers mode only; with corners the board is wherever it was when they were picked)."""
        frames = self._frames()
        if self.board == "corners":
            if self.corners is None:
                raise RuntimeError("no screen corners: capture once and run scripts/analyze_sketch.py --pick")
            return S.extract(frames, self.corners, self._xp, self.g)
        H = next((h for h in (P.board_homography(P.detect_markers(f), self.g) for f in frames) if h is not None), None)
        if H is None:
            return None
        rects = [P.rectify(f, H, self.g, self._xp.px_per_mm) for f in frames]
        return S.extract(rects, None, self._xp, self.g)

    def rectified(self) -> np.ndarray | None:
        ex = self._extract()
        return None if ex is None else ex.rectified

    def locate_board(self) -> bool:
        """True if the screen can be measured from the scene camera. The first sighting starts the run: what is
        on the screen then is never taken for the pen's line."""
        ex = self._extract()
        if ex is not None and not self._run_started:
            self._tracker.start(ex.mask)
            self._run_started, self._moves = True, []
        return ex is not None

    def observe_pen(self) -> tuple[float, float]:
        ex = self._extract()
        if ex is not None:
            if not self._run_started:
                self._tracker.start(ex.mask)
                self._run_started = True
            else:
                self._tracker.update(ex.mask, self._moves, R.visible_from(ex.flat, ex.unknown, self._track_p))
            self._moves = []
        pen = self._tracker.pen_mm
        self.log.append({"t": self.clock_s, "skill": "observe_pen", "pen": pen})
        return pen if pen is not None else (float("nan"), float("nan"))

    def erase(self, peak_accel_ms2: float = 9.0, duration_s: float = 30.0) -> EraseResult:
        # When built: an erase ends the run; set self._run_started = False so the next board sighting starts a new one.
        raise NotImplementedError("erase on the arm is a separate routine with its own approval; not built yet")
