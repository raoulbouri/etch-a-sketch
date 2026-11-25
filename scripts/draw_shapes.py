"""Draw a basic shape on the real Etch A Sketch: the base test (arm reaches both knobs and draws).

Open loop through the Skills interface: OpenLoopDrawer turns each knob by mm / k, adding the backlash
on direction changes, with k and backlash from the hand calibration (etch_sim.kernel.ToyParams).
Needs the aims-kinova motion server running with the toy's envelope, and the arm SERVOING_READY.

    uv run python scripts/draw_shapes.py square --size 30 --start 36,16 --plan    # print the plan only
    uv run python scripts/draw_shapes.py square --size 30 --start 36,16 --pause   # draw, Enter before each line
    uv run python scripts/draw_shapes.py picture --image house --start 36,16 --budget 600 --plan

picture: Andrew's S3 planner (etch_sim.planner) turns an image into one continuous knob route from the pen,
fitted to the whole screen inside an 8 mm margin, at the coarsest-needed staircase tolerance for --budget.
--image is a file, or one of s3_demo's test pictures: house, circle, heart, pieces.

--start is where the pen is now, in mm from the screen's left and bottom edges (estimate by eye or with
a card). It is used to refuse shapes that would drive the pen into an end stop (that can snap the string).
Shapes start at the pen and go right/up first. Each knob switch costs a release and a grasp (~10 s).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from etch_sim.draw import Calib, OpenLoopDrawer, polyline, rectangle, spiral_out, square, staircase, stepped_circle
from etch_sim.draw import GraspFailed
from etch_sim.endstop import EndStopRisk, GuardedDrawer, GuardParams, TravelGuard
from etch_sim.kernel import ToyParams
from etch_sim.planner import PlanParams, plan_for_budget

MARGIN_MM = 5.0
SWITCH_S = 4.0           # release + grasp: 3.7 s median on the green arm 2026-11-24 (blue arm 2026-11-22: ~10 s)
ROUTINE_MAX_S = 1800.0   # envelope max_duration_s: the server ends the routine after this


def shape_moves(name: str, size: float, steps: int):
    if name == "square":
        return square(size)
    if name == "rectangle":
        return rectangle(size, round(size * 0.6, 1))
    if name == "triangle":
        return staircase(size / steps, steps)
    if name == "circle":
        return stepped_circle(size / 2, steps)
    if name == "spiral":                                 # centre-out rectangular spiral, `steps` loops, 3:4 high
        return spiral_out(size, steps)
    raise SystemExit(f"unknown shape {name}")


def picture(name: str):
    if Path(name).exists():
        return name
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from s3_demo import test_pictures
    pics = test_pictures()
    if name not in pics:
        raise SystemExit(f"--image {name}: no such file, and not one of {sorted(pics)}")
    return pics[name]


def plan(moves, calib: Calib, speed_dps: float):
    """Knob degrees per move as OpenLoopDrawer will turn them, and a time estimate."""
    last, rows, t = {"x": 1.0, "y": 1.0}, [], 0.0          # preload leaves both knobs loaded in +
    prev_axis = None
    for axis, mm in moves:
        s = 1.0 if mm > 0 else -1.0
        deg = mm / calib.k_mm_per_deg[axis] + (s * 2 * calib.delta_deg[axis] if s != last[axis] else 0.0)
        last[axis] = s
        t += abs(deg) / speed_dps + (SWITCH_S if axis != prev_axis else 0.0)
        prev_axis = axis
        rows.append((axis, mm, deg))
    return rows, t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shape", choices=["square", "rectangle", "triangle", "circle", "spiral", "picture"])
    ap.add_argument("--image", help="picture: image file, or house / circle / heart / pieces")
    ap.add_argument("--budget", type=float, default=600.0, help="picture: time budget, s (sets the staircase tolerance)")
    ap.add_argument("--size", type=float, default=30.0, help="side / width / triangle leg / circle diameter / spiral outer width, mm")
    ap.add_argument("--steps", type=int, default=4, help="triangle stairs, circle steps per quadrant, or spiral loops")
    ap.add_argument("--start", required=True, help="pen position now, 'x,y' mm from the left and bottom edges")
    ap.add_argument("--no-preload", action="store_true", help="skip taking up the slack in + on both knobs first")
    ap.add_argument("--plan", action="store_true", help="print the plan and exit; no arm")
    ap.add_argument("--pause", action="store_true", help="wait for Enter before every line")
    ap.add_argument("--start-unc", type=float, default=5.0,
                    help="how sure --start is, mm (by eye ~5, wrist camera ~1.5); the end-stop guard grows it per move")
    ap.add_argument("--knob-dps", type=float, help="knob speed, deg/s (default: knob_speed_dps from --poses); the motion "
                    "server caps it (KINOVA_ROTATE_MAX_DPS, 15 unless raised for the arm)")
    ap.add_argument("--slow-dps", type=float, default=5.0, help="knob speed within 15 mm of an edge, deg/s (15 deg steps)")
    ap.add_argument("--poses", default="hardware/taught_poses.json",
                    help="knob poses for RealBackend; hardware/knob_poses_live.json after relocate.py solve")
    a = ap.parse_args()

    toy = ToyParams()
    calib = Calib({ax: toy.axis(ax).k_mm_per_deg for ax in "xy"}, {ax: toy.axis(ax).delta_deg for ax in "xy"})
    poses = json.loads(Path(a.poses).read_text())
    if a.knob_dps:
        poses["knob_speed_dps"] = a.knob_dps
    speed = float(poses.get("knob_speed_dps") or 15.0)
    start = np.array([float(v) for v in a.start.split(",")])
    pre_mm = {ax: 0.0 if a.no_preload else (2 * calib.delta_deg[ax] + 10.0) * calib.k_mm_per_deg[ax] for ax in "xy"}  # at most
    origin = start + np.array([pre_mm["x"], pre_mm["y"]])
    if a.shape == "picture":
        if not a.image:
            raise SystemExit("picture needs --image")
        pp = PlanParams(screen_mm=(toy.x.travel_mm, toy.y.travel_mm), switch_s=SWITCH_S, knob_speed_dps=speed, calib=calib)
        route = plan_for_budget(picture(a.image), a.budget, pp, pen_mm=tuple(origin))
        moves, origin = route.moves, np.array(route.start_mm)
        st = route.stats
        print(f"planner: tol {st['tol_mm']:g} mm, {st['switches']} switches, picture->route {st['target_to_route_mm']:.1f} mm, "
              f"retrace {st['retrace_mm']:.0f} mm, est {st['est_time_s'] / 60:.1f} min")
    else:
        moves = shape_moves(a.shape, a.size, a.steps)
    pts = polyline(moves, origin)
    lo, hi = pts.min(0), pts.max(0)
    travel = np.array([toy.x.travel_mm, toy.y.travel_mm])
    rows, t = plan(moves, calib, speed)
    t += 0.0 if a.no_preload else sum((2 * calib.delta_deg[ax] + 10) / speed + SWITCH_S for ax in "xy")

    print(f"{a.shape} {a.image or f'{a.size:g} mm'}: {len(moves)} lines, {sum(r[0] != p[0] for r, p in zip(rows[1:], rows)) + 1} knob grasps, "
          f"~{t / 60:.1f} min at {speed:g} deg/s")
    print(f"calib: k x {calib.k_mm_per_deg['x']} y {calib.k_mm_per_deg['y']} mm/deg, "
          f"backlash 2x{calib.delta_deg['x']:.1f} / 2x{calib.delta_deg['y']:.1f} deg on reversal")
    print(f"pen box: x {lo[0]:.0f}..{hi[0]:.0f}  y {lo[1]:.0f}..{hi[1]:.0f} mm  (screen 0..{travel[0]:.0f} x 0..{travel[1]:.0f})")
    for i, (axis, mm, deg) in enumerate(rows, 1):
        if i <= 12 or i > len(rows) - 3:
            print(f"  {i:2d}. {axis} {mm:+6.1f} mm -> knob {deg:+7.1f} deg")
        elif i == 13:
            print(f"  ... {len(rows) - 15} more")
    if (lo < MARGIN_MM).any() or (hi > travel - MARGIN_MM).any():
        raise SystemExit(f"refused: the shape comes within {MARGIN_MM:g} mm of an end stop from --start {a.start}. "
                         "Smaller --size, or move the pen first.")
    # the whole drawing through the end-stop guard before anything moves: the guard's uncertainty grows with every
    # line drawn open loop, so a drawing can be safe at the start and refused later; refuse it now, not half-drawn
    sim = TravelGuard(GuardParams(travel_mm={"x": toy.x.travel_mm, "y": toy.y.travel_mm}, margin_mm=MARGIN_MM))
    for i, ax in enumerate("xy"):
        sim.anchor(ax, float(start[i]), a.start_unc)
    plan_moves = ([] if a.no_preload else [(ax, pre_mm[ax]) for ax in "xy"]) + list(moves)
    for i, (ax, mm) in enumerate(plan_moves, 1):
        try:
            sim.check(ax, mm)
        except EndStopRisk as e:
            raise SystemExit(f"refused: line {i}/{len(plan_moves)} could reach an end stop ({e}). Draw smaller, start "
                             "nearer the middle, or measure the pen (D435) so the uncertainty stays small.")
        sim.commit(ax, mm)
    print(f"end-stop guard: whole drawing OK; pen uncertainty at the end +-{sim.unc['x']:.1f} / +-{sim.unc['y']:.1f} mm")
    if t > 0.8 * ROUTINE_MAX_S:
        raise SystemExit(f"refused: ~{t / 60:.0f} min is too close to the routine's {ROUTINE_MAX_S / 60:.0f} min limit.")
    if a.plan:
        return

    from etch_sim.real import ArmClient, ArmRefused, RealBackend
    c = ArmClient()
    st = c.call("status")["state"]
    if st.get("arm_state", "ARMSTATE_SERVOING_READY") != "ARMSTATE_SERVOING_READY":
        raise SystemExit(f"arm is {st.get('arm_state')}, not SERVOING_READY. Close the Web App, leave the pad alone.")
    b = RealBackend(c, poses)
    guard = TravelGuard(GuardParams(travel_mm={"x": toy.x.travel_mm, "y": toy.y.travel_mm}, margin_mm=MARGIN_MM))
    for i, ax in enumerate("xy"):
        guard.anchor(ax, float(start[i]), a.start_unc)
    d = GuardedDrawer(b, calib, guard, slow_dps=a.slow_dps)   # refuses any move that could reach an end stop, before turning
    t0 = time.time()
    try:
        if not a.no_preload:
            if a.pause:
                input("  next: preload (both knobs + to take up slack). Enter to go, Ctrl-C to stop ")
            d.preload()
        for i, (axis, mm) in enumerate(moves, 1):
            if a.pause:
                input(f"  next: line {i}/{len(moves)}: {axis} {mm:+.1f} mm. Enter to go, Ctrl-C to stop ")
            d.move(axis, mm)
            r = b.log[-1]
            print(f"  {i:2d}. {axis} {mm:+6.1f} mm: wrist {r.get('wrist_deg', float('nan')):+7.1f} deg   t={time.time() - t0:5.0f} s")
        b.release()
    except GraspFailed as e:
        print(f"STOPPED: {e}")
        if b.held:                                       # a turn failed with the knob in hand: let go and lift
            b.release()                                  # (a failed grasp has already opened and lifted)
        print("Nothing more was drawn. Re-locate the board (relocate.py auto) if it was moved or shaken, then rerun.")
        raise SystemExit(1)
    except EndStopRisk as e:
        print(f"STOPPED before turning: {e}")
        b.release()
        print("The knob was released and the arm is above it. Re-measure the pen position, or move the pen off the edge.")
        raise SystemExit(1)
    except ArmRefused as e:
        print(f"REFUSED: {e}")
        try:
            st = c.call("status")["state"]
            print("arm now:", st.get("arm_state"), st["tool_pose"], "gripper", st.get("gripper"))
        except Exception as e2:
            print("status unavailable:", e2)
        print("The routine has ended. If the gripper is closed on a knob: open it first (kinova gripper 0.45), then lift.")
        raise SystemExit(1)
    out = Path("hardware/draw_runs") / time.strftime(f"%Y%m%d-%H%M%S_{a.shape}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"shape": a.shape, "image": a.image, "budget_s": a.budget, "size_mm": a.size, "steps": a.steps, "start_mm": list(start),
                               "moves": moves, "calib": {"k": calib.k_mm_per_deg, "delta": calib.delta_deg},
                               "speed_dps": speed, "log": b.log}, indent=1))
    print(f"done in {time.time() - t0:.0f} s; log {out}. Photograph the screen square-on with a card in view.")


if __name__ == "__main__":
    main()
