"""Record the arm's current tool pose as a taught knob pose. Read-only on the arm.

Jog the arm by hand or from the Kinova web app until the open fingers are around the knob
(grasp) or 5 cm straight above that (above), then run, with the motion server up:
    uv run python scripts/teach_pose.py x grasp
    uv run python scripts/teach_pose.py x above
Axis x = left knob (horizontal), y = right knob (vertical).
"""
import json
import sys

from etch_sim.real.backend import POSES_FILE, ArmClient

axis, name = sys.argv[1], sys.argv[2]
assert axis in ("x", "y") and name in ("above", "grasp"), __doc__
pose = ArmClient().call("status")["state"]["tool_pose"]
poses = json.loads(POSES_FILE.read_text())
poses["knobs"][axis][name] = [round(pose[k], 4) for k in ("x", "y", "z", "theta_x", "theta_y", "theta_z")]
POSES_FILE.write_text(json.dumps(poses, indent=2) + "\n")
print(f"saved {axis}.{name} = {poses['knobs'][axis][name]}  ->  {POSES_FILE}")
