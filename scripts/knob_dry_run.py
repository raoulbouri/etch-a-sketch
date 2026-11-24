"""Knob dry run through the aims-kinova motion server, using the poses in hardware/taught_poses.json.

Run from the etch-a-sketch folder with the motion server already running in another window.
    uv run python knob_dry_run.py                     # default: x:90 y:-90 x:360 (fake-mode test)
    uv run python knob_dry_run.py x:30 --pause        # real-arm test: one knob, 30 deg, Enter before each step
Each step "axis:deg" = grasp that knob (gripper to 'pre', move above, descend, close), turn it deg
(+ = clockwise seen from above); release (open, back up to 'above') at the end.

Refuses to start unless the arm reports SERVOING_READY: the arm rejects our commands while the Web App
or the Xbox pad holds manual control (abort_details 18, ROBOT_MOVEMENT_IN_PROGRESS). Any refusal or
failed motion stops the run and prints the arm's state; nothing retries.
"""
import argparse
import json

from etch_sim.real import ArmClient, ArmRefused, RealBackend

ap = argparse.ArgumentParser()
ap.add_argument("steps", nargs="*", default=["x:90", "y:-90", "x:360"])
ap.add_argument("--pause", action="store_true", help="wait for Enter before every arm step")
a = ap.parse_args()


def wait(what: str) -> None:
    if a.pause:
        input(f"  next: {what}. Enter to go, Ctrl-C to stop ")


def state(c: ArmClient) -> dict:
    return c.call("status")["state"]


c = ArmClient()
st = state(c)
print("start:", st.get("arm_state"), st["tool_pose"], "gripper", st.get("gripper"))
if st.get("arm_state", "ARMSTATE_SERVOING_READY") != "ARMSTATE_SERVOING_READY":
    raise SystemExit(f"arm is {st.get('arm_state')}, not SERVOING_READY. Close the Web App's pose/admittance panel, "
                     "leave the Xbox pad alone (RB then LB resets it), then check with `kinova --json status`.")
b = RealBackend(c)
try:
    for step in a.steps:
        axis, deg = step.split(":")
        deg = float(deg)
        wait(f"grasp the {axis} knob (open to 'pre', move above it, descend 5 cm, close)")
        held = b.grasp_knob(axis)
        print(f"{axis}: held={held}  gripper={b.log[-1].get('gripper')}")
        if not held:
            print("  not held: the gripper did not stall in the expected range; stopping here")
            break
        wait(f"turn the {axis} knob {deg:+.0f} deg")
        r = b.rotate(deg)
        print(f"{axis}: rotate {deg:+.0f} -> wrist {r.wrist_deg:+.1f} deg  switches={b.n_switches}")
    wait("release (open, back up 5 cm)")
    b.release()
except ArmRefused as e:
    print(f"REFUSED: {e}")
    try:
        st = state(c)
        print("arm now:", st.get("arm_state"), st["tool_pose"], "gripper", st.get("gripper"))
    except Exception as e2:
        print("status unavailable:", e2)
    print("The routine has ended. If the gripper is closed on a knob: open it first (kinova gripper 0.45), then lift.")
    raise SystemExit(1)
st = state(c)
print("end:", st.get("arm_state"), st["tool_pose"])
print("log:", json.dumps(b.log, indent=1))
