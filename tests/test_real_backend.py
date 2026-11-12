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
