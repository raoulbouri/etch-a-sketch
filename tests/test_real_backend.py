"""RealBackend against a stand-in motion server that speaks the aims-kinova protocol (no arm)."""
import json
import socketserver
import threading

import pytest

from etch_sim.real import ArmClient, ArmRefused, RealBackend
from etch_sim.real.backend import down_yaw_deg

DOWN = [180.0, 0.0, 90.0]
POSES = {"knobs": {"x": {"above": [0.38, 0.09, 0.15, *DOWN], "grasp": [0.38, 0.09, 0.10, *DOWN]},
                   "y": {"above": [0.38, -0.09, 0.15, *DOWN], "grasp": [0.38, -0.09, 0.10, *DOWN]}},
         "gripper": {"pre": 0.45, "closed": 1.0, "held_min": 0.30, "held_max": 0.90},
         "j6_deg_per_knob_deg": {"x": -1.0, "y": -1.0}, "knob_speed_dps": 30.0}


class Stub(socketserver.ThreadingTCPServer):
    allow_reuse_address, daemon_threads = True, True

    def __init__(self):
        self.pose = dict(x=0.38, y=0.0, z=0.20, theta_x=180.0, theta_y=0.0, theta_z=90.0)
        self.calls, self.knob_present, self.dead, self.flip_euler = [], True, False, False
        self.floor = 0.07                             # stand-in envelope floor (m)
        super().__init__(("127.0.0.1", 0), Handler)

    def reply(self, req):
        self.calls.append(req)
        cmd = req["cmd"]
        if cmd == "status":
            return {"ok": True, "state": {"tool_pose": self.pose}}
        if self.dead:
            return {"ok": False, "error": "Routine has ended", "routine_stopped": True}
        if cmd == "move_pose":
            if req["z"] < self.floor:
                self.dead = True
                return {"ok": False, "error": "SAFETY: outside envelope", "routine_stopped": True}
            self.pose = {k: req[k] for k in ("x", "y", "z", "theta_x", "theta_y", "theta_z")}
            return {"ok": True, "event": "ACTION_END", "tool_pose": self.pose}
        if cmd == "rotate":
            self.pose["theta_z"] = (self.pose["theta_z"] + req["deg"] + 180) % 360 - 180
            if self.flip_euler:                       # same orientation, the other Euler form (as the blue arm reports)
                self.pose = dict(self.pose, theta_x=0.4, theta_y=-179.8, theta_z=(self.pose["theta_z"] + 360) % 360 - 180)
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


def test_down_yaw_is_the_same_in_both_euler_forms():
    assert down_yaw_deg(dict(theta_x=180.0, theta_y=0.0, theta_z=-119.1)) == pytest.approx(-119.1)
    assert down_yaw_deg(dict(theta_x=0.4, theta_y=-179.8, theta_z=60.9)) == pytest.approx(-119.1, abs=0.01)


def test_release_after_a_flipped_readback_does_not_spin_the_wrist(rig):
    srv, client = rig
    srv.pose["theta_z"] = 61.1
    srv.flip_euler = True
    b = RealBackend(client, POSES)
    assert b.grasp_knob("x")
    b.rotate(180.0)
    b.release()
    last = [c for c in srv.calls if c["cmd"] == "move_pose"][-1]
    assert last["theta_x"] == 180.0 and last["theta_z"] == pytest.approx(61.1 - 180.0, abs=0.01)


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
    untaught = json.loads(json.dumps(POSES))
    untaught["knobs"]["y"]["grasp"] = None       # as in a fresh hardware/taught_poses.json
    with pytest.raises(ValueError, match="y.grasp"):
        RealBackend(client, untaught)


def test_refusal_reports_the_arms_abort_details(rig):
    srv, client = rig
    srv.reply = lambda req: ({"ok": True, "state": {"tool_pose": srv.pose}} if req["cmd"] == "status" else
                             {"ok": False, "event": "ACTION_ABORT", "abort_details": "18", "rotated_deg": 0.0})
    b = RealBackend(client, POSES)
    with pytest.raises(ArmRefused, match=r"abort_details.*18.*ROBOT_MOVEMENT_IN_PROGRESS"):
        b.grasp_knob("x")



def test_relocated_live_poses_drive_the_drawer(rig, tmp_path, monkeypatch):
    """relocate.py solve -> hardware/knob_poses_live.json -> RealBackend -> OpenLoopDrawer: the arm goes to the
    re-located knob poses and turns the knobs with the taught signs."""
    import importlib.util
    from pathlib import Path
    from etch_sim.draw import Calib, OpenLoopDrawer, square
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("relocate", root / "scripts" / "relocate.py")
    rl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rl)
    monkeypatch.setattr(rl, "LIVE_POSES", tmp_path / "knob_poses_live.json")
    monkeypatch.setattr(rl, "ROOT", tmp_path)
    (tmp_path / "s").mkdir()
    grasp_x, grasp_y = [0.2082, 0.139, 0.0538, 180.0, 0.0, 135.2], [0.3095, -0.0174, 0.0539, 180.0, 0.0, -143.8]
    res = {"board": {"x": 0.32, "y": 0.10, "yaw_deg": -57.8}, "date": "test",
           "knobs": {"x": {"grasp": grasp_x, "approach": grasp_x[:2] + [0.1038] + grasp_x[3:]},
                     "y": {"grasp": grasp_y, "approach": grasp_y[:2] + [0.1039] + grasp_y[3:]}}}
    rl.write_live_poses(res, tmp_path / "s")
    live = json.loads((tmp_path / "knob_poses_live.json").read_text())
    assert live["knobs"]["x"]["grasp"] == grasp_x and live["j6_deg_per_knob_deg"] == {"x": -1.0, "y": -1.0}
    srv, client = rig
    srv.floor = 0.03                                         # the drawing envelope's floor is below the grasps
    b = RealBackend(client, live)
    OpenLoopDrawer(b, Calib({"x": 0.095, "y": 0.100}, {"x": 10.6, "y": 8.4})).draw(square(10.0))
    moves = [c for c in srv.calls if c["cmd"] == "move_pose"]
    grasps = [(m["x"], m["y"], m["z"]) for m in moves if m["z"] < 0.06]
    assert (0.2082, 0.139, 0.0538) in grasps and (0.3095, -0.0174, 0.0539) in grasps
    turns = [c["deg"] for c in srv.calls if c["cmd"] == "rotate"]
    assert len(turns) == 4 and turns[0] < 0                  # +x (knob clockwise) = joint 6 negative


def test_light_grip_closes_to_its_target_and_counts_as_held(rig):
    srv, client = rig
    poses = json.loads(json.dumps(POSES))
    poses["gripper"]["light"] = 0.60                         # below the stub's 0.62 stall point: reaches its target
    b = RealBackend(client, poses)
    assert b.grasp_knob("x")
    closes = [c["position"] for c in srv.calls if c["cmd"] == "gripper"]
    assert closes[-1] == 0.60                                 # light target, not the full 1.0 squeeze
    b2 = RealBackend(client, POSES)                          # without "light": full squeeze, stall required
    assert b2.grip_target is None


def test_light_grip_that_closes_on_nothing_is_not_held(rig):
    srv, client = rig
    srv.knob_present = False
    poses = json.loads(json.dumps(POSES))
    poses["gripper"]["light"] = 0.95                         # outside held_max 0.90: fingers met, nothing between
    assert not RealBackend(client, poses).grasp_knob("x")
