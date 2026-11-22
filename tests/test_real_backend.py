"""RealBackend against a stand-in motion server that speaks the aims-kinova protocol (no arm)."""
import json
import socketserver
import threading

import pytest

from etch_sim.real import ArmClient, ArmRefused, RealBackend

DOWN = [180.0, 0.0, 90.0]
POSES = {"knobs": {"x": {"above": [0.38, 0.09, 0.15, *DOWN], "grasp": [0.38, 0.09, 0.10, *DOWN]},
                   "y": {"above": [0.38, -0.09, 0.15, *DOWN], "grasp": [0.38, -0.09, 0.10, *DOWN]}},
         "gripper": {"pre": 0.45, "closed": 1.0, "held_min": 0.30, "held_max": 0.90},
         "j6_deg_per_knob_deg": {"x": -1.0, "y": -1.0}, "knob_speed_dps": 30.0}


class Stub(socketserver.ThreadingTCPServer):
    allow_reuse_address, daemon_threads = True, True

    def __init__(self):
        self.pose = dict(x=0.38, y=0.0, z=0.20, theta_x=180.0, theta_y=0.0, theta_z=90.0)
        self.calls, self.knob_present, self.dead = [], True, False
        super().__init__(("127.0.0.1", 0), Handler)

    def reply(self, req):
        self.calls.append(req)
        cmd = req["cmd"]
        if cmd == "status":
            return {"ok": True, "state": {"tool_pose": self.pose}}
        if self.dead:
            return {"ok": False, "error": "Routine has ended", "routine_stopped": True}
        if cmd == "move_pose":
            if req["z"] < 0.07:
                self.dead = True
                return {"ok": False, "error": "SAFETY: outside envelope", "routine_stopped": True}
            self.pose = {k: req[k] for k in self.pose}
            return {"ok": True, "event": "ACTION_END", "tool_pose": self.pose}
        if cmd == "rotate":
            self.pose["theta_z"] = (self.pose["theta_z"] + req["deg"] + 180) % 360 - 180
            return {"ok": True, "event": "ACTION_END", "rotated_deg": req["deg"], "tool_pose": self.pose}
        if cmd == "gripper":
            if req["position"] > 0.62 and self.knob_present:
                return {"ok": True, "event": "STALLED (object?)", "position": 0.62, "tool_pose": self.pose}
            return {"ok": True, "event": "REACHED", "position": req["position"], "tool_pose": self.pose}


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        for raw in self.rfile:
            self.wfile.write((json.dumps(self.server.reply(json.loads(raw))) + "\n").encode())
            self.wfile.flush()


@pytest.fixture
def rig():
    srv = Stub()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    client = ArmClient(*srv.server_address)
    yield srv, client
    client.close()
    srv.shutdown()
    srv.server_close()


def test_grasp_rotate_switch(rig):
    srv, client = rig
    b = RealBackend(client, POSES)
    assert b.grasp_knob("x")
    r = b.rotate(470.0)
    assert r.wrist_deg == pytest.approx(470.0)
    assert [c for c in srv.calls if c["cmd"] == "rotate"][0]["deg"] == pytest.approx(-470.0)   # sign from the teach file
    assert b.grasp_knob("y") and b.n_switches == 1
    moves = [c for c in srv.calls if c["cmd"] == "move_pose"]
    assert moves[-1]["y"] == -0.09 and moves[-1]["z"] == 0.10
    assert all(m["theta_z"] == pytest.approx(-20.0) for m in moves[2:])       # wrist is not unwound between knobs


def test_missed_grasp_is_reported(rig):
    srv, client = rig
    srv.knob_present = False
    b = RealBackend(client, POSES)
    assert b.grasp_knob("x") is False and b.held is None
    assert b.rotate(90.0).ok is False


def test_refusal_stops_the_script(rig):
    srv, client = rig
    bad = json.loads(json.dumps(POSES))
    bad["knobs"]["x"]["grasp"][2] = 0.01
    b = RealBackend(client, bad)
    with pytest.raises(ArmRefused):
        b.grasp_knob("x")
    with pytest.raises(ArmRefused):
        b.grasp_knob("y")


def test_untaught_poses_are_rejected(rig):
    _, client = rig
    with pytest.raises(ValueError):
        RealBackend(client)                      # hardware/taught_poses.json still has nulls


# ---- scene camera: camera-server client and clicked-corners board mode ------------------------------
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from etch_sim.kernel import ToyKernel, ToyParams  # noqa: E402
from etch_sim.real import CameraServerError, RealSenseServer  # noqa: E402


class CamStub(socketserver.ThreadingTCPServer):
    """Speaks the scripts/rs_server.py protocol; a grab saves n pure-red frames."""
    allow_reuse_address, daemon_threads = True, True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), CamHandler)


class CamHandler(socketserver.StreamRequestHandler):
    def handle(self):
        for raw in self.rfile:
            req = json.loads(raw)
            if req["cmd"] == "grab":
                out = __import__("pathlib").Path(req["out"])
                out.mkdir(parents=True, exist_ok=True)
                for i in range(req["n"]):
                    cv2.imwrite(str(out / f"rgb_{i:02d}.png"), np.full((8, 8, 3), (0, 0, 255), np.uint8))
                reply = {"ok": True, "saved": req["n"]}
            else:
                reply = {"ok": True}
            self.wfile.write((json.dumps(reply) + "\n").encode())
            self.wfile.flush()


def test_camera_server_client(tmp_path):
    srv = CamStub()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        cam = RealSenseServer(*srv.server_address, n=4)
        assert cam.ping()
        frames = cam()
        assert len(frames) == 4 and frames[0][0, 0].tolist() == [255, 0, 0]       # BGR on disk -> RGB
        kept = RealSenseServer(*srv.server_address, n=2, save_dir=tmp_path)
        kept()
        assert len(list(tmp_path.glob("*/rgb_*.png"))) == 2                      # bursts kept for later checks
    finally:
        srv.shutdown()
        srv.server_close()
    dead = RealSenseServer(*srv.server_address, timeout_s=1.0)
    assert not dead.ping()
    with pytest.raises(CameraServerError):
        dead()


# corners as picked on 2026-11-22 (toy upside down in the view), scaled up 1.4x about their centre
_C = np.array([[869, 337], [615, 375], [632, 540], [924, 497]], np.float32)
CORNERS = (_C - _C.mean(0)) * 1.4 + _C.mean(0)


def camera_view(toy: ToyKernel, corners: np.ndarray, rng, n: int = 3) -> list[np.ndarray]:
    """A burst of 1280x720 frames of the toy's screen seen through the given corners (toy order BL, BR, TR, TL):
    faint blurred line (30 grey levels) on a light screen, dark surroundings, sensor noise."""
    ppm = 10.0
    w, h = round(153 * ppm), round(107 * ppm)
    ink = np.zeros((h, w), np.float32)
    for st in toy.strokes:
        if len(st.points) > 1:
            pts = np.array([(x * ppm, (107 - y) * ppm) for x, y in st.points]).round().astype(np.int32)
            cv2.polylines(ink, [pts], False, 1.0, 6)
    screen = 165 - 30 * cv2.GaussianBlur(ink, (0, 0), 2.0)
    M = cv2.getPerspectiveTransform(np.array([[0, h - 1], [w - 1, h - 1], [w - 1, 0], [0, 0]], np.float32),
                                    np.asarray(corners, np.float32))
    on = cv2.warpPerspective(np.ones((h, w), np.float32), M, (1280, 720)) > 0.5
    view = np.where(on, cv2.warpPerspective(screen, M, (1280, 720)), 60.0)
    out = []
    for _ in range(n):
        g = np.clip(view + rng.normal(0, 2.0, view.shape), 0, 255).astype(np.uint8)
        out.append(np.dstack([g, g, g]))
    return out


def test_corners_mode_tracks_the_pen(rig):
    _, client = rig
    toy = ToyKernel(ToyParams(), start_mm=(30.0, 25.0))           # the measured knob k and backlash
    rng = np.random.default_rng(0)
    b = RealBackend(client, POSES, camera=lambda: camera_view(toy, CORNERS, rng), corners=CORNERS)
    assert b.locate_board()                                         # starts the run on the blank screen
    for ax, deg in [("x", 360.0), ("y", 360.0), ("x", -360.0), ("y", -180.0)]:
        assert b.grasp_knob(ax)
        r = b.rotate(deg)
        toy.turn(ax, r.wrist_deg)                                   # the knob follows the wrist (no slip)
        pen = b.observe_pen()
        # 2 mm, not 1: run_tracker reports the line end, which sits past the stylus by part of the line's round
        # end cap in the direction of travel (1.0-1.6 mm here, every step measured, none predicted). Only the
        # calibration fit removes that bias today (turn_measure.fit_session).
        assert np.hypot(*np.subtract(pen, toy.stylus)) < 2.0, (ax, deg, pen, toy.stylus)


def test_corners_missing_is_reported(rig):
    _, client = rig
    b = RealBackend(client, POSES, camera=lambda: np.zeros((720, 1280, 3), np.uint8))
    b.corners = None                                                # as if never picked
    with pytest.raises(RuntimeError, match="analyze_sketch.py --pick"):
        b.observe_pen()
