"""One motion-server command per run, printing the server's full reply and the arm's state afterwards.
For diagnosing the rotate abort. Run from the etch-a-sketch folder with the motion server running.

    uv run python arm_diag.py status         # read-only
    uv run python arm_diag.py rotate 10      # spin joint 6 by 10 deg (+ = the knob's clockwise direction x -1, as RealBackend)
    uv run python arm_diag.py grasp-x        # RealBackend grasp of the x knob: open to 'pre', above, descend, close (stalls on the knob)
    uv run python arm_diag.py hold           # gripper to where it stopped: still gripping, nothing left to 'run'
    uv run python arm_diag.py open           # gripper to 'pre' (0.45)
    uv run python arm_diag.py up             # tool straight up 5 cm (server move_pose)

A refused or failed command ends the server's routine: restart the server (and approve) before the next motion.
"""
import json
import socket
import sys

from etch_sim.real import ArmClient, ArmRefused, RealBackend


def raw(cmd: str, **kw) -> dict:
    """Send one command and return the server's reply as is (no exception on refusal)."""
    with socket.create_connection(("127.0.0.1", 8765), timeout=180) as s, s.makefile("rw") as f:
        f.write(json.dumps({"cmd": cmd, **kw}) + "\n")
        f.flush()
        return json.loads(f.readline())


def state() -> dict:
    st = raw("status").get("state", {})
    return {k: st.get(k) for k in ("arm_state", "joint_angles_deg", "tool_pose", "gripper")}


def show(title: str, reply: dict) -> None:
    print(f"{title}: {json.dumps(reply)}")
    print(f"arm now: {json.dumps(state())}")


what = sys.argv[1] if len(sys.argv) > 1 else "status"
if what == "status":
    print(json.dumps(state(), indent=1))
elif what == "rotate":
    deg = float(sys.argv[2])
    show(f"rotate {deg:+.1f}", raw("rotate", deg=deg, speed=30.0))
elif what == "grasp-x":
    try:
        held = RealBackend(ArmClient()).grasp_knob("x")
        print(f"grasp-x: held={held}")
    except ArmRefused as e:
        print(f"grasp-x REFUSED: {e}")
    print(f"arm now: {json.dumps(state())}")
elif what == "hold":
    pos = float(state()["gripper"])
    show(f"hold at {pos:.4f}", raw("gripper", position=pos))
elif what == "open":
    show("open to 0.45", raw("gripper", position=0.45))
elif what == "up":
    p = state()["tool_pose"]
    show("up 5 cm", raw("move_pose", x=p["x"], y=p["y"], z=p["z"] + 0.05, theta_x=p["theta_x"],
                        theta_y=p["theta_y"], theta_z=p["theta_z"]))
else:
    raise SystemExit(__doc__)
