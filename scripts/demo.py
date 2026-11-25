"""Demo: find the Etch-a-Sketch with the wrist camera, then draw on it. One command, two approvals.

    uv run python scripts/demo.py spiral --size 60 --steps 6 --start 86,52              # the demo
    uv run python scripts/demo.py spiral --size 60 --steps 6 --start 86,52 --dry-run    # checks only, no motion

Run it from an interactive terminal on the Mac (the motion server asks YOU to type `yes` for each envelope).

  1. Checks: the drawing fits the end-stop guard from --start; the arm is SERVOING_READY and inside the re-locate
     envelope (park it there first; demo.py ends there too); the handover pose fits both envelopes.
  2. Re-locate (envelope relocate_workspace_10, approve with `yes`): look, survey poses if the tags are not all seen,
     5 views, board pose -> hardware/knob_poses_live.json (scripts/relocate.py auto).
  3. Handover: the arm goes above the left knob at z 0.20 m, a pose inside both envelopes, and that routine ends.
  4. Draw (envelope etch_draw_workspace_10, approve with `yes`; knobs up to --knob-dps): scripts/draw_shapes.py with
     the live knob poses: two-step light grip, end-stop guard, slow 15 deg steps near edges, 0.7 Nm torque backstop,
     stop on any failed grasp.
  5. Park: back up to the handover pose (z 0.20, J2 does not heat up there), routine ends.

Every arm motion happens inside an envelope you approved for this run; any refusal or failure ends the routine and
the demo (nothing retries). Shake (erase) the toy before the run if you like: the re-locate finds it wherever it is,
and shaking does not move the pen, so --start stays valid. Unplug the Xbox pad's USB: its input aborts commands.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from etch_sim.real import view_plan as V
from etch_sim.real import wrist_camera as W

ROOT = Path(__file__).resolve().parents[1]
RELOCATE_ENV = ROOT / "hardware" / "envelopes" / "relocate_workspace_10.json"
DRAW_ENV = ROOT / "hardware" / "envelopes" / "etch_draw_workspace_10.json"
LIVE = ROOT / "hardware" / "knob_poses_live.json"
HANDOVER_Z_M = 0.20


def inside(p, env: dict) -> bool:
    return all(lo <= v <= hi for v, lo, hi in zip(p[:3], env["box_min"], env["box_max"]))


def handover_pose(grasp: list[float], env_r: dict, env_d: dict, margin_m: float = 0.01) -> list[float]:
    """Pose where one routine hands over to the next: straight down at HANDOVER_Z_M, as close above the knob as the
    overlap of both envelopes' boxes allows (the re-locate box only covers camera positions, so for some board
    placements the point straight above the knob is outside it). The drawing routine then descends to the knob."""
    lo = np.maximum(env_r["box_min"][:2], env_d["box_min"][:2]) + margin_m
    hi = np.minimum(env_r["box_max"][:2], env_d["box_max"][:2]) - margin_m
    x, y = np.clip(grasp[:2], lo, hi)
    return [float(x), float(y), HANDOVER_Z_M, 180.0, 0.0, float(grasp[5])]


def say(msg: str) -> None:
    print(f"\n=== {msg}", flush=True)


class Server:
    """The aims-kinova motion server for one envelope, started in the container. Its approval prompt is shown here and
    read from this terminal: the person running the demo types `yes`. Stopping it ends the routine (Base.Stop)."""

    def __init__(self, envelope: Path, rotate_dps: float | None = None):
        name = f"/tmp/demo_{envelope.name}"
        subprocess.run(["docker", "compose", "cp", str(envelope), f"kinova:{name}"], cwd=W.KINOVA_DIR, check=True,
                       capture_output=True)
        cmd = ["docker", "compose", "exec", "-i"]
        if rotate_dps:
            cmd += ["-e", f"KINOVA_ROTATE_MAX_DPS={rotate_dps:g}"]
        cmd += ["kinova", "with-ros.sh", "python3", "-u", "-m", "kinova.server", "--envelope", name, "--host", "0.0.0.0"]
        self.p = subprocess.Popen(cmd, cwd=W.KINOVA_DIR, stdin=None, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.out, self.ready = "", threading.Event()
        threading.Thread(target=self._echo, daemon=True).start()
        while not self.ready.wait(0.2):
            if self.p.poll() is not None:
                raise SystemExit(f"motion server for {envelope.name} did not start (not approved?). Demo stopped.")

    def _echo(self):
        while True:
            ch = self.p.stdout.read(1)
            if not ch:
                return
            c = ch.decode(errors="replace")
            sys.stdout.write(c)
            sys.stdout.flush()
            self.out += c
            if "Serving on" in self.out:
                self.ready.set()

    def stop(self):
        subprocess.run(["docker", "compose", "exec", "-T", "kinova", "pkill", "-INT", "-f", "kinova.server"],
                       cwd=W.KINOVA_DIR, capture_output=True)
        try:
            self.p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.p.terminate()


def move_path(goal_pose: list[float], label: str) -> None:
    """Move the tool to goal_pose through the running server, in steps of <= 0.22 m and <= 35 deg."""
    from etch_sim.real import ArmClient
    c = ArmClient()
    try:
        cur = W.tool_pose(c.call("status")["state"])
        goal = W.Pose(Rotation.from_euler("xyz", goal_pose[3:], degrees=True), np.array(goal_pose[:3]))
        for w in V.path_to(cur, goal):
            p = V.kortex_pose(w)
            print(f"  {label}: {p}", flush=True)
            c.call("move_pose", x=p[0], y=p[1], z=p[2], theta_x=p[3], theta_y=p[4], theta_z=p[5])
    finally:
        c.close()


def run(cmd: list[str]) -> int:
    print("  $ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT).returncode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shape", choices=["square", "rectangle", "triangle", "circle", "spiral"])
    ap.add_argument("--size", type=float, default=60.0)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--start", required=True, help="pen 'x,y' mm from the screen's left / bottom edge (by eye)")
    ap.add_argument("--start-unc", type=float, default=5.0)
    ap.add_argument("--knob-dps", type=float, default=30.0, help="knob speed (green arm: up to 30)")
    ap.add_argument("--slow-dps", type=float, default=10.0, help="knob speed near the edges, 15 deg steps")
    ap.add_argument("--dry-run", action="store_true", help="checks only: no server, no motion")
    a = ap.parse_args()
    shape = [a.shape, "--size", f"{a.size:g}", "--steps", str(a.steps), "--start", a.start, "--start-unc", f"{a.start_unc:g}",
             "--knob-dps", f"{a.knob_dps:g}", "--slow-dps", f"{a.slow_dps:g}"]
    t0 = time.time()

    say("1/5 checks")
    if run(["uv", "run", "python", "scripts/draw_shapes.py", *shape, "--plan"]) != 0:
        raise SystemExit("the drawing does not pass the end-stop guard from --start: nothing moved")
    st = W.arm_status()
    tp = st["tool_pose"]
    here = (tp["x"], tp["y"], tp["z"])
    env_r, env_d = json.loads(RELOCATE_ENV.read_text()), json.loads(DRAW_ENV.read_text())
    print(f"  arm {st['arm_state']} at {tuple(round(v, 3) for v in here)}, gripper {st.get('gripper', 0):.2f}")
    if not inside(here, env_r):
        raise SystemExit(f"the arm is outside the re-locate envelope {env_r['box_min']}..{env_r['box_max']}: park it "
                         "inside first (an approved move, e.g. straight up), then rerun. Nothing moved.")
    if not (env_r["box_min"][2] <= HANDOVER_Z_M <= env_r["box_max"][2] and env_d["box_min"][2] <= HANDOVER_Z_M <= env_d["box_max"][2]):
        raise SystemExit(f"handover height {HANDOVER_Z_M} m is not inside both envelopes: regenerate the drawing envelope "
                         "(scripts/relocate.py draw-envelope)")
    print(f"  envelopes: re-locate z {env_r['box_min'][2]}-{env_r['box_max'][2]}, drawing z {env_d['box_min'][2]}-"
          f"{env_d['box_max'][2]}, handover z {HANDOVER_Z_M}: OK")
    if a.dry_run:
        print("\ndry run: all checks passed; nothing moved")
        return

    say("2/5 re-locate the board (type yes to approve the re-locate envelope)")
    srv = Server(RELOCATE_ENV)
    try:
        if run(["uv", "run", "python", "scripts/relocate.py", "auto"]) != 0:
            raise SystemExit("re-locate failed: demo stopped")
        live = json.loads(LIVE.read_text())
        g = live["knobs"]["x"]["grasp"]
        handover = handover_pose(g, env_r, env_d)
        if not (inside(handover, env_r) and inside(handover, env_d)):
            raise SystemExit(f"handover {handover[:3]} is not inside both envelopes (board outside the workspace?)")
        say("3/5 handover: above the left knob")
        move_path(handover, "handover")
    finally:
        srv.stop()

    say(f"4/5 draw (type yes to approve the drawing envelope; knobs up to {a.knob_dps:g} deg/s)")
    srv = Server(DRAW_ENV, rotate_dps=a.knob_dps)
    code = 1
    try:
        code = run(["uv", "run", "python", "scripts/draw_shapes.py", *shape, "--poses", str(LIVE.relative_to(ROOT))])
        say("5/5 park at the handover height")
        from etch_sim.real import ArmClient, ArmRefused
        c = ArmClient()
        try:
            p = W.tool_pose(c.call("status")["state"])
            heading = float(np.degrees(np.arctan2(*p.R.apply([1, 0, 0])[1::-1])))       # tool x-axis heading
            c.call("move_pose", x=float(p.t[0]), y=float(p.t[1]), z=HANDOVER_Z_M, theta_x=180.0, theta_y=0.0, theta_z=heading)
        except ArmRefused as e:                    # the routine already ended (the drawing stopped): park by hand
            print(f"  park refused ({e}); lift the arm with an approved move")
        finally:
            c.close()
    finally:
        srv.stop()
    say(f"demo {'finished' if code == 0 else 'STOPPED early'} in {(time.time() - t0) / 60:.1f} min")
    if code != 0:
        sys.exit(code)


if __name__ == "__main__":
    main()
