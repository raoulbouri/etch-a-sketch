"""Teach the knob grasps relative to the toy's markers. Read-only: this script never moves the arm.

You drive the arm with the Xbox controller; the script only grabs wrist frames and reads the arm's status.
The toy must not move during a session.

    uv run python scripts/teach_knobs.py view            # arm placed so the wrist camera sees the toy; grabs 3 frames
    uv run python scripts/teach_knobs.py view            # repeat from 2-4 more viewpoints (different sides / tilts)
    uv run python scripts/teach_knobs.py knob x          # gripper CLOSED on the left (x, horizontal) knob
    uv run python scripts/teach_knobs.py knob y          # gripper CLOSED on the right (y, vertical) knob
    uv run python scripts/teach_knobs.py solve [--save]  # board pose, marker layout, knob grasps relative to the board

All files go to the newest session in test_images/teach/ (--new starts one). solve --save writes hardware/board_model.json.
Checks printed by solve: marker layout vs the ruler (hardware/phase0_measurements.yaml), marker height vs the table
(+10 mm in the base frame), knob spacing, knob-to-marker distances vs the ruler.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from etch_sim.real import board_model as B
from etch_sim.real import wrist_camera as W

ROOT = Path(__file__).resolve().parents[1]
TEACH = ROOT / "test_images" / "teach"
MARKER_SURFACE_Z = 0.010             # toy marker surface in the base frame: 28 mm above a table at -18 mm (ruler)


def session(new: bool) -> Path:
    olds = sorted(TEACH.glob("2*")) if TEACH.exists() else []
    if new or not olds:
        s = TEACH / time.strftime("%Y%m%d-%H%M%S")
        s.mkdir(parents=True)
        return s
    return olds[-1]


def view(a) -> None:
    s = session(a.new)
    st = W.arm_status()
    grab = W.RtspGrabber(st["ip"])
    try:
        n = len(list(s.glob("view_*.png")))
        for k in range(a.frames):
            before = W.arm_status()
            img = grab.frame_after(time.time() + 0.2)
            after = W.arm_status()
            if np.max(np.abs(np.subtract(before["joint_angles_deg"], after["joint_angles_deg"]))) > 0.05:
                raise SystemExit("the arm moved during the capture: hold still and rerun")
            m = B.detect_markers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            stem = s / f"view_{n + k:02d}"
            cv2.imwrite(f"{stem}.png", img)
            stem.with_suffix(".json").write_text(json.dumps({"status": after, "t": time.time()}, indent=1))
            print(f"{stem.name}: markers {sorted(m)}")
    finally:
        grab.close()
    if len(m) < 2:
        print("  fewer than 2 markers visible: aim the camera at the toy")


def knob(a) -> None:
    s = session(False)
    st = W.arm_status()
    g = st.get("gripper", 0.0)
    (s / f"knob_{a.axis}.json").write_text(json.dumps({"status": st, "t": time.time()}, indent=1))
    t = st["tool_pose"]
    print(f"knob {a.axis}: tool ({t['x']:.4f}, {t['y']:.4f}, {t['z']:.4f}) m, angles ({t['theta_x']:.1f}, {t['theta_y']:.1f}, "
          f"{t['theta_z']:.1f}), gripper {g:.3f}")
    if g < 0.3:
        print("  WARNING: the gripper looks open; close it on the knob (it should stall around 0.6-0.7) and rerun")


def solve(a) -> None:
    s = session(False)
    K, D, mount = W.load_calibration()
    views = []
    for p in sorted(s.glob("view_*.png")):
        m = B.detect_markers(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))
        if len(m) >= 2:
            views.append((m, json.loads(p.with_suffix(".json").read_text())["status"]))
    if not views:
        raise SystemExit(f"no usable views in {s}")
    fit_fixed = B.fit_markers(views, K, D, mount, z_fixed=MARKER_SURFACE_Z)
    fit = B.fit_markers(views, K, D, mount)          # height fitted by the camera: teach and re-locate share this chain,
    print(f"{len(views)} views; markers seen {sorted(fit.markers)}")   # so its height bias cancels in the stored grasps
    print(f"  marker height, fitted: {fit.z * 1000:+.1f} mm (table-based {MARKER_SURFACE_Z * 1000:+.1f}); "
          f"rms fitted z {fit.rms_px:.2f} px, table-based z {fit_fixed.rms_px:.2f} px")
    print("  per view rms (px): " + ", ".join(f"{r:.1f}" for r in fit.per_view_rms))
    if set(fit.markers) != set(B.MARKER_IDS):
        raise SystemExit("all 4 markers must be seen in at least one view")
    board = B.board_frame(fit.markers, fit.z)
    layout = B.layout_in_board(fit.markers, board)
    print(f"  board in base: centre ({board.t[0]:.4f}, {board.t[1]:.4f}) m, yaw {np.degrees(board.R.as_euler('zyx')[0]):.1f} deg")
    ruler = yaml.safe_load(open(ROOT / "hardware" / "phase0_measurements.yaml"))["markers"]
    print("  marker centre distances, camera vs ruler (mm):")
    for pair, rv in ruler["centre_distances_mm"].items():
        i, j = map(int, pair.split("-"))
        d = np.hypot(layout[i][0] - layout[j][0], layout[i][1] - layout[j][1])
        print(f"    {pair}: {d:6.1f} vs {rv:6.1f}  ({d - rv:+.1f})")
    grasps = {}
    for ax in ("x", "y"):
        f = s / f"knob_{ax}.json"
        if not f.exists():
            print(f"  knob {ax}: not taught yet")
            continue
        st = json.loads(f.read_text())["status"]
        rel = B.relative_grasp(board, st)
        grasps[ax] = rel
        pinch = board.inv().apply(W.tool_pose(st).apply([0, 0, 0.0358]))      # model pad-centre point, for reporting
        near = 3 if ax == "x" else 2
        mx, my = layout[near][:2]
        dist = np.hypot(pinch[0] * 1000 - mx, pinch[1] * 1000 - my)
        edge = dist - B.MARKER_M * 500
        key = f"{ax}_knob-{near}"
        rv = ruler.get("knob_to_marker_mm", {}).get(key)
        print(f"  knob {ax}: grasp point in board ({pinch[0] * 1000:.1f}, {pinch[1] * 1000:.1f}, {pinch[2] * 1000:+.1f}) mm; "
              f"to marker {near} centre {dist:.1f} mm (edge ~{edge:.1f} mm)" + (f"; ruler {rv:.1f} mm" if rv else ""))
    if len(grasps) == 2:
        px = board.inv().apply(W.tool_pose(json.loads((s / "knob_x.json").read_text())["status"]).apply([0, 0, 0.0358]))
        py = board.inv().apply(W.tool_pose(json.loads((s / "knob_y.json").read_text())["status"]).apply([0, 0, 0.0358]))
        print(f"  knob spacing {np.hypot(*(px - py)[:2]) * 1000:.1f} mm (blue-arm toy 196.6 mm); height difference "
              f"{abs(px[2] - py[2]) * 1000:.1f} mm")
    if a.save:
        if len(grasps) < 2:
            raise SystemExit("not saved: teach both knobs first")
        B.save_model(B.MODEL_FILE, layout, fit.z, grasps,
                     {"session": str(s.relative_to(ROOT)), "views": len(views), "rms_px": fit.rms_px,
                      "arm": views[0][1]["ip"], "date": time.strftime("%Y-%m-%d %H:%M")})
        print(f"wrote {B.MODEL_FILE}")


def board_guess(s: Path) -> tuple[np.ndarray, W.Pose, dict]:
    """Rough marker corners in the base frame (for view planning only, +-2 cm): from camera views of the toy if the
    session has any, else from the two taught knob grasps and the ruler layout."""
    K, D, mount = W.load_calibration()
    views = []
    for p in sorted(s.glob("view_*.png")):
        m = B.detect_markers(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))
        if len(m) >= 2:
            views.append((m, json.loads(p.with_suffix(".json").read_text())["status"]))
    if views:
        fit = B.fit_markers(views, K, D, mount, z_fixed=MARKER_SURFACE_Z)
        if set(fit.markers) == set(B.MARKER_IDS):
            board = B.board_frame(fit.markers, fit.z)
            pts = np.vstack([B.flat_pose(x, y, fit.z, a).apply(B.LOCAL) for x, y, a in fit.markers.values()])
            return pts, board, json.loads(sorted(s.glob("view_*.json"))[-1].read_text())["status"]
    kx = W.tool_pose(json.loads((s / "knob_x.json").read_text())["status"]).t
    ky = W.tool_pose(json.loads((s / "knob_y.json").read_text())["status"]).t
    ex = (ky - kx)[:2] / np.linalg.norm((ky - kx)[:2])          # left knob -> right knob = board x
    ey = np.array([-ex[1], ex[0]])                               # up the screen, away from the knobs
    centre = (kx + ky)[:2] / 2 + 0.105 * ey                      # marker centroid ~105 mm from the knob line (ruler)
    board = B.flat_pose(*centre, MARKER_SURFACE_Z, float(np.arctan2(ex[1], ex[0])))
    layout = {0: (-0.094, 0.0725), 1: (0.094, 0.0725), 2: (0.064, -0.0725), 3: (-0.064, -0.0725)}   # ruler, mm/1000
    pts = np.vstack([(board * B.flat_pose(x, y, 0.0, 0.0)).apply(B.LOCAL) for x, y in layout.values()])
    return pts, board, None


def plan(a) -> None:
    from etch_sim.mj.scene import EtchScene
    from etch_sim.real import view_plan as V
    s = session(False)
    pts, board, _ = board_guess(s)
    st = W.arm_status()
    start = W.tool_pose(st)
    lim = V.ViewLimits(rho_min_m=0.17, rho_max_m=a.rho_max, dist_m=(0.35, 0.50), n_dist=3, tilts_deg=(0, 10, 20, 30),
                       min_visible=1.0, margin_px=60.0, tool_z_min_m=0.20)
    ref_x = W.camera_pose(st).R.apply([1.0, 0.0, 0.0])
    # predict the views with the CALIBRATED camera (the K_GUESS principal point is 117 px off: a marker fell out of view)
    cands = V.candidates(pts, lim, K=W.load_calibration()[0], ref_x=ref_x)
    model = EtchScene().model
    views = V.order(V.select(cands, a.n, start, lambda tool: V.reachable(tool, model)), start)
    moves = V.with_waypoints(views, start, lim.max_step_m, lim.max_turn_deg)
    print(f"board guess: centre ({board.t[0]:.3f}, {board.t[1]:.3f}) m; {len(cands)} candidate views see all 4 markers; "
          f"{len(views)} picked")
    prev = start
    for tool, cap in moves:
        step, turn = np.linalg.norm(tool.t - prev.t), np.degrees((prev.R.inv() * tool.R).magnitude())
        print(f"  {'view' if cap else 'via '} ({tool.t[0]:.3f}, {tool.t[1]:+.3f}, {tool.t[2]:.3f})  tilt {V.tilt_deg(tool):3.0f}  "
              f"step {step * 100:4.1f} cm  turn {turn:3.0f} deg  radius {np.hypot(*tool.t[:2]):.3f}")
        prev = tool
    # the box also holds the start lifted to the routine floor: the arm must already be inside when the server starts
    lifted = W.Pose(start.R, np.r_[start.t[:2], max(start.t[2], lim.tool_z_min_m)])
    env = V.envelope_for([m[0] for m in moves] + [lifted], f"board_views_{st['ip'].split('.')[-1]}_{time.strftime('%Y%m%d')}",
                         f"Wrist-camera views of the Etch-a-Sketch markers (etch-sim scripts/teach_knobs.py plan, session "
                         f"{s.name}). Camera only: no gripper or knob commands. Valid for this toy placement.")
    env_path = ROOT / "hardware" / "envelopes" / f"{env['name']}.json"
    env_path.parent.mkdir(exist_ok=True)
    env_path.write_text(json.dumps(env, indent=2) + "\n")
    (s / "plan.json").write_text(json.dumps({"start_status": st, "envelope": str(env_path.relative_to(ROOT)),
                                             "moves": [{"pose": V.kortex_pose(t), "capture": c} for t, c in moves]}, indent=1))
    print(f"envelope proposal {env_path.name}: box_min {env['box_min']} box_max {env['box_max']} tilt <= {env['max_tilt_deg']}")


def run(a) -> None:
    """Visit the planned views through the motion server (bounded routine approved when it was started) and capture.
    Any refusal or failure stops the run; nothing retries."""
    from etch_sim.real import ArmClient, ArmRefused
    s = session(False)
    moves = json.loads((s / "plan.json").read_text())["moves"]
    c = ArmClient()
    st = c.call("status")["state"]
    if st.get("arm_state") != "ARMSTATE_SERVOING_READY":
        raise SystemExit(f"arm is {st.get('arm_state')}, not SERVOING_READY")
    grab = W.RtspGrabber(st["ip"])
    n0 = len(list(s.glob("view_*.png")))
    k = 0
    try:
        for j, mv in enumerate(moves, 1):
            p = mv["pose"]
            print(f"move {j}/{len(moves)}{'' if mv['capture'] else ' (via)'}: {p}", flush=True)
            c.call("move_pose", x=p[0], y=p[1], z=p[2], theta_x=p[3], theta_y=p[4], theta_z=p[5])
            if not mv["capture"]:
                continue
            time.sleep(0.8)
            before = c.call("status")["state"]
            img = grab.frame_after(time.time() + 0.2)
            after = c.call("status")["state"]
            still = np.max(np.abs(np.subtract(before["joint_angles_deg"], after["joint_angles_deg"]))) <= 0.05
            m = B.detect_markers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            stem = s / f"view_{n0 + k:02d}"
            cv2.imwrite(f"{stem}.png", img)
            stem.with_suffix(".json").write_text(json.dumps({"status": after, "t": time.time(), "planned": p}, indent=1))
            print(f"  {stem.name}: markers {sorted(m)}" + ("" if still else "  (arm still moving: frame unreliable)"), flush=True)
            k += 1
    except ArmRefused as e:
        print(f"REFUSED: {e}\nThe routine has ended; nothing retries. Frames saved so far are kept.")
        raise SystemExit(1)
    finally:
        grab.close()
        c.close()
    print(f"done: {k} views. Next: uv run python scripts/teach_knobs.py solve --save")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("view")
    v.add_argument("--frames", type=int, default=3)
    v.add_argument("--new", action="store_true", help="start a new session")
    k = sub.add_parser("knob")
    k.add_argument("axis", choices=["x", "y"])
    s = sub.add_parser("solve")
    s.add_argument("--save", action="store_true")
    p = sub.add_parser("plan", help="plan wrist views of the toy's markers (no motion)")
    p.add_argument("-n", type=int, default=5)
    p.add_argument("--rho-max", type=float, default=0.25, help="max tool distance from the base axis (m)")
    sub.add_parser("run", help="visit the planned views through the motion server and capture")
    a = ap.parse_args()
    {"view": view, "knob": knob, "solve": solve, "plan": plan, "run": run}[a.cmd](a)


if __name__ == "__main__":
    main()
