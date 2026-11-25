"""Re-locate the Etch-a-Sketch after it was moved: board pose from wrist views, then both knob grasp poses.

Uses hardware/wrist_camera_calib.json (camera) and hardware/board_model.json (marker layout + knob grasps relative to
the board, from scripts/teach_knobs.py). Only `run` moves the arm, through the motion server (a bounded routine a person
approves when starting it); every other step is read-only or offline.

    uv run python scripts/relocate.py workspace       # once: survey poses + ONE envelope for the whole board workspace
    uv run python scripts/relocate.py draw-envelope   # once: drawing envelope (knob poses + turns) for that workspace
    uv run python scripts/relocate.py auto            # server running with that envelope: look -> survey if needed ->
                                                      #   5 views -> solve, in one routine (~30 s)
  or step by step:
    uv run python scripts/relocate.py sight --new     # read-only: one frame from where the arm is now (rough board pose)
    uv run python scripts/relocate.py plan            # 5 views around the rough pose + an envelope proposal (no motion)
    uv run python scripts/relocate.py run             # motion server running with that envelope: visit views, capture
    uv run python scripts/relocate.py solve           # board x, y, yaw -> knob grasp + approach (5 cm above) poses
    uv run python scripts/relocate.py taught x        # read-only: gripper closed on the knob by hand -> ground truth
    uv run python scripts/relocate.py compare         # predicted vs hand-taught grasps
    uv run python scripts/relocate.py learn           # add the hand corrections to the model (averaged per knob)

Files go to the newest session in test_images/relocate/ (sight --new starts one).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from etch_sim.real import board_model as B
from etch_sim.real import knob_poses as KP
from etch_sim.real import view_run as R
from etch_sim.real import wrist_camera as W

ROOT = Path(__file__).resolve().parents[1]
RELOC = ROOT / "test_images" / "relocate"
APPROACH_M = 0.05                    # approach pose: straight above the grasp
HANDOVER_Z_M = 0.20                  # above a knob, inside BOTH the re-locate envelope (floor 0.188 m) and the
                                     # drawing envelope, so demo.py can switch routines without a move outside one
ROUTINE_FLOOR_M = 0.20               # views keep the tool frame at least this high
SIDEWAYS_TOL_MM = 3.0               # knob grasp, predicted vs hand-corrected
HEIGHT_TOL_MM = 5.0                 # hand grasp heights of one knob spread ~4 mm (2026-11-24): height is grip choice


def session(new: bool = False) -> Path:
    olds = sorted(RELOC.glob("2*")) if RELOC.exists() else []
    if new or not olds:
        s = RELOC / time.strftime("%Y%m%d-%H%M%S")
        s.mkdir(parents=True)
        return s
    return olds[-1]


def model():
    d = B.load_model()
    return d, d["markers"], float(d["marker_surface_z_m_when_taught"])


def last_board(s: Path):
    """Best available board pose: this session's frames, else the previous re-locate, else the teach session."""
    d, layout, z = model()
    K, D, mount = W.load_calibration()
    views = R.load_views(s, "sight") + R.load_views(s, "view")
    if views:
        return B.fit_board(views, layout, z, K, D, mount).board, "this session's frames"
    for prev in sorted(RELOC.glob("2*"), reverse=True):
        if (prev / "result.json").exists():
            b = json.loads((prev / "result.json").read_text())["board"]
            return B.flat_pose(b["x"], b["y"], z, math.radians(b["yaw_deg"])), f"previous re-locate {prev.name}"
    teach = ROOT / d["session"]
    return B.fit_board(R.load_views(teach, "view"), layout, z, K, D, mount).board, f"teach session {teach.name}"


def sight(a) -> None:
    s = session(a.new)
    st = W.arm_status()
    grab = W.RtspGrabber(st["ip"])
    try:
        img = grab.frame_after(time.time() + 0.2)
    finally:
        grab.close()
    after = W.arm_status()
    if np.max(np.abs(np.subtract(st["joint_angles_deg"], after["joint_angles_deg"]))) > 0.05:
        raise SystemExit("the arm moved during the capture: hold still and rerun")
    n = len(list(s.glob("sight_*.png")))
    stem = s / f"sight_{n:02d}"
    cv2.imwrite(f"{stem}.png", img)
    stem.with_suffix(".json").write_text(json.dumps({"status": after, "t": time.time()}, indent=1))
    m = B.detect_markers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    print(f"{stem}: markers {sorted(m)}")
    if len(m) >= 2:
        b, src = last_board(s)
        print(f"  rough board pose: ({b.t[0]:.3f}, {b.t[1]:.3f}) m, yaw {math.degrees(b.R.as_euler('zyx')[0]):.1f} deg")
    else:
        print("  fewer than 2 markers: planning will use the previous board pose")


def plan(a) -> None:
    from etch_sim.mj.scene import EtchScene
    from etch_sim.real import view_plan as V
    s = session()
    d, layout, z = model()
    board, src = last_board(s)
    pts = np.vstack([(board * B.flat_pose(v[0] / 1000, v[1] / 1000, 0.0, math.radians(v[2]))).apply(B.LOCAL)
                     for v in layout.values()])
    st = W.arm_status()
    start = W.tool_pose(st)
    lim = V.ViewLimits(rho_min_m=0.17, rho_max_m=a.rho_max, dist_m=(0.35, 0.50), n_dist=3, tilts_deg=(0, 10, 20, 30),
                       min_visible=1.0, margin_px=60.0, tool_z_min_m=ROUTINE_FLOOR_M)
    # predict the views with the CALIBRATED camera (the K_GUESS principal point is 117 px off: a marker fell out of view)
    cands = V.candidates(pts, lim, K=W.load_calibration()[0], ref_x=W.camera_pose(st).R.apply([1.0, 0.0, 0.0]))
    mdl = EtchScene().model
    views = V.order(V.select(cands, a.n, start, lambda tool: V.reachable(tool, mdl)), start)
    moves = V.with_waypoints(views, start, lim.max_step_m, lim.max_turn_deg)
    print(f"board guess from {src}: ({board.t[0]:.3f}, {board.t[1]:.3f}) m, yaw {math.degrees(board.R.as_euler('zyx')[0]):.1f} deg; "
          f"{len(cands)} candidate views see all 4 markers; {len(views)} picked")
    prev = start
    for tool, cap in moves:
        print(f"  {'view' if cap else 'via '} ({tool.t[0]:.3f}, {tool.t[1]:+.3f}, {tool.t[2]:.3f})  tilt {V.tilt_deg(tool):3.0f}  "
              f"step {np.linalg.norm(tool.t - prev.t) * 100:4.1f} cm  turn {np.degrees((prev.R.inv() * tool.R).magnitude()):3.0f} deg  "
              f"radius {np.hypot(*tool.t[:2]):.3f}")
        prev = tool
    if len(views) < a.n:
        print(f"WARNING: only {len(views)} views found")
    lifted = W.Pose(start.R, np.r_[start.t[:2], max(start.t[2], ROUTINE_FLOOR_M)])
    env = V.envelope_for([m[0] for m in moves] + [lifted], f"relocate_{st['ip'].split('.')[-1]}_{time.strftime('%Y%m%d_%H%M')}",
                         f"Wrist-camera views of the Etch-a-Sketch markers for a re-locate (etch-sim scripts/relocate.py, "
                         f"session {s.name}). Camera only: no gripper or knob commands.")
    env_path = ROOT / "hardware" / "envelopes" / f"{env['name']}.json"
    env_path.parent.mkdir(exist_ok=True)
    env_path.write_text(json.dumps(env, indent=2) + "\n")
    (s / "plan.json").write_text(json.dumps({"start_status": st, "envelope": str(env_path.relative_to(ROOT)),
                                             "moves": [{"pose": V.kortex_pose(t), "capture": c} for t, c in moves]}, indent=1))
    print(f"envelope proposal {env_path.name}: box_min {env['box_min']} box_max {env['box_max']} tilt <= {env['max_tilt_deg']}")


def run(a) -> None:
    s = session()
    moves = json.loads((s / "plan.json").read_text())["moves"]
    n = R.run_views(moves, s, "view", port=a.port)
    print(f"done: {n} views. Next: uv run python scripts/relocate.py solve")


def solve(a) -> None:
    s = session()
    d, layout, z = model()
    K, D, mount = W.load_calibration()
    views = R.load_views(s, "view") + R.load_views(s, "sight")
    if len(views) < 3:
        raise SystemExit(f"only {len(views)} usable frames (need >= 3 with >= 2 markers each)")
    fit = B.fit_board(views, layout, z, K, D, mount)
    b = fit.board
    yaw = math.degrees(b.R.as_euler("zyx")[0])
    print(f"{fit.n_views} frames, {fit.n_markers} marker sightings; fit rms {fit.rms_px:.2f} px "
          f"(per frame: {', '.join(f'{r:.1f}' for r in fit.per_view_rms)})")
    print(f"board: centre ({b.t[0]:.4f}, {b.t[1]:.4f}) m, yaw {yaw:.1f} deg, height {z * 1000:+.1f} mm (taught)")
    out = {"board": {"x": float(b.t[0]), "y": float(b.t[1]), "yaw_deg": yaw}, "rms_px": fit.rms_px, "frames": fit.n_views,
           "knobs": {}, "date": time.strftime("%Y-%m-%d %H:%M")}
    ok = fit.rms_px <= 6.0
    for ax, rel in d["grasps"].items():
        g = B.grasp_in_base(b, rel)
        ap = g[:2] + [round(g[2] + APPROACH_M, 4)] + g[3:]
        errs = KP.check_pose(ap) + KP.check_pose(g)
        print(f"  knob {ax}: grasp {g}\n          approach {ap}" + (f"\n          REJECT: {errs}" if errs else ""))
        out["knobs"][ax] = {"grasp": g, "approach": ap, "checks": errs}
        ok &= not errs
    out["ok"] = bool(ok)
    (s / "result.json").write_text(json.dumps(out, indent=1))
    if ok:
        write_live_poses(out, s)
    print(("OK" if ok else "NOT OK (see above)") + f"; saved {s / 'result.json'}")


LIVE_POSES = ROOT / "hardware" / "knob_poses_live.json"
TAUGHT_POSES = ROOT / "hardware" / "taught_poses.json"


def write_live_poses(res: dict, s: Path) -> None:
    """Knob poses of this re-locate in the format RealBackend reads (hardware/taught_poses.json), so draw_shapes.py and
    knob_dry_run.py can use them with --poses hardware/knob_poses_live.json. Gripper settings and the joint-6 signs come
    from taught_poses.json: knob clockwise (seen from above) = joint 6 negative on both arms (aims-kinova docs/05)."""
    base = json.loads(TAUGHT_POSES.read_text())
    live = {"_comment": "Written by scripts/relocate.py solve: knob poses for the board where it was last re-located. "
                        "Valid only until the toy moves.",
            "_source": {"session": str(s.relative_to(ROOT)), "board": res["board"], "date": res["date"]},
            "knobs": {ax: {"above": k["approach"], "grasp": k["grasp"]} for ax, k in res["knobs"].items()},
            "gripper": base["gripper"], "j6_deg_per_knob_deg": base["j6_deg_per_knob_deg"],
            "knob_speed_dps": base.get("knob_speed_dps", 15.0)}
    LIVE_POSES.write_text(json.dumps(live, indent=2) + "\n")
    print(f"wrote {LIVE_POSES.relative_to(ROOT)} (use: draw_shapes.py ... --poses {LIVE_POSES.relative_to(ROOT)})")


def taught(a) -> None:
    s = session()
    st = W.arm_status()
    (s / f"taught_{a.axis}.json").write_text(json.dumps({"status": st, "t": time.time()}, indent=1))
    t = st["tool_pose"]
    print(f"taught {a.axis}: tool ({t['x']:.4f}, {t['y']:.4f}, {t['z']:.4f}), gripper {st.get('gripper', 0):.3f}")
    if st.get("gripper", 0) < 0.3:
        print("  WARNING: the gripper looks open; close it on the knob and rerun")


def compare(a) -> None:
    s = session()
    res = json.loads((s / "result.json").read_text())
    for ax in ("x", "y"):
        f = s / f"taught_{ax}.json"
        if not f.exists():
            print(f"knob {ax}: no hand-taught grasp yet")
            continue
        t = W.tool_pose(json.loads(f.read_text())["status"]).t
        p = np.array(res["knobs"][ax]["grasp"][:3])
        d = (t - p) * 1000
        ok = math.hypot(d[0], d[1]) <= SIDEWAYS_TOL_MM and abs(d[2]) <= HEIGHT_TOL_MM
        print(f"knob {ax}: taught - predicted = ({d[0]:+.1f}, {d[1]:+.1f}, {d[2]:+.1f}) mm; sideways {math.hypot(d[0], d[1]):.1f} mm, "
              f"height {d[2]:+.1f} mm  ->  {'PASS' if ok else 'FAIL'} (sideways <= {SIDEWAYS_TOL_MM:g} mm, height <= {HEIGHT_TOL_MM:g} mm)")


WORKSPACE_FILE = ROOT / "hardware" / "relocate_workspace.json"


def placements(region: dict) -> list:
    """Board poses spanning the workspace: centre radius x azimuth x board yaw (relative to its azimuth)."""
    d, layout, z = model()
    out = []
    for rho in np.linspace(*region["rho_m"], 3):
        for az in np.radians(np.linspace(*region["azimuth_deg"], 3)):
            for dy in np.radians(np.linspace(-region["yaw_range_deg"], region["yaw_range_deg"], 3)):
                out.append(B.flat_pose(rho * math.cos(az), rho * math.sin(az), z, az + math.radians(region["yaw_offset_deg"]) + dy))
    return out


def marker_points(board) -> np.ndarray:
    d, layout, z = model()
    return np.vstack([(board * B.flat_pose(v[0] / 1000, v[1] / 1000, 0.0, math.radians(v[2]))).apply(B.LOCAL)
                      for v in layout.values()])


def visible(cam, pts, K, margin_px=40.0) -> bool:
    pc = cam.inv().apply(pts)
    if (pc[:, 2] < 0.05).any():
        return False
    uv = (K @ (pc / pc[:, 2:]).T).T[:, :2]
    return bool(((uv[:, 0] > margin_px) & (uv[:, 0] < 1280 - margin_px) & (uv[:, 1] > margin_px) & (uv[:, 1] < 720 - margin_px)).all())


def view_limits(rho_max):
    from etch_sim.real import view_plan as V
    return V.ViewLimits(rho_min_m=0.17, rho_max_m=rho_max, dist_m=(0.35, 0.50), n_dist=3, tilts_deg=(0, 10, 20, 30),
                        min_visible=1.0, margin_px=60.0, tool_z_min_m=ROUTINE_FLOOR_M)


def workspace(a) -> None:
    """Offline: survey poses that together see every board placement in the workspace, and ONE envelope that holds the
    surveys and the close views for any placement. A person approves that envelope once per session."""
    from etch_sim.mj.scene import EtchScene
    from etch_sim.real import view_plan as V
    K = W.load_calibration()[0]
    region = {"rho_m": a.rho, "azimuth_deg": a.azimuth, "yaw_offset_deg": a.yaw_offset, "yaw_range_deg": a.yaw_range}
    boards = placements(region)
    mdl = EtchScene().model
    st = W.arm_status()
    ref_x = W.camera_pose(st).R.apply([1.0, 0.0, 0.0])
    # survey candidates: high views at the middle placement
    mid = boards[len(boards) // 2]
    lim_s = V.ViewLimits(rho_min_m=0.17, rho_max_m=a.rho_max, dist_m=(0.50, 0.70), n_dist=3, tilts_deg=(0, 10, 20, 30),
                         min_visible=0.5, margin_px=40.0, tool_z_min_m=0.30)
    cands = V.candidates(marker_points(mid), lim_s, K=K, ref_x=ref_x)
    cover = [{i for i, b in enumerate(boards) if visible(c.cam, marker_points(b), K)} for c in cands]
    surveys, todo = [], set(range(len(boards)))
    while todo and len(surveys) < 3:
        order = sorted(range(len(cands)), key=lambda i: -len(cover[i] & todo))
        pick = next((i for i in order if cover[i] & todo and V.reachable(cands[i].tool, mdl)), None)
        if pick is None:
            break
        surveys.append(cands[pick])
        todo -= cover[pick]
    print(f"{len(boards)} placements; {len(surveys)} survey poses see all 4 markers of {len(boards) - len(todo)} of them")
    if todo:
        print(f"  WARNING: {len(todo)} placements are seen fully by no survey pose (the routine will then use the best partial view)")
    # close views for every placement -> envelope box
    tools = [s.tool for s in surveys]
    lim = view_limits(a.rho_max)
    for b in boards:
        cs = V.candidates(marker_points(b), lim, K=K, ref_x=ref_x)
        tools += [v.tool for v in V.select(cs, 5, surveys[0].tool, lambda t: V.reachable(t, mdl))]
    start = W.tool_pose(st)
    tools.append(W.Pose(start.R, np.r_[start.t[:2], max(start.t[2], ROUTINE_FLOOR_M)]))
    env = V.envelope_for(tools, f"relocate_workspace_{st['ip'].split('.')[-1]}",
                         "Re-locate workspace (etch-sim scripts/relocate.py workspace): high survey poses and wrist views of the "
                         f"Etch-a-Sketch markers for any board placement with centre radius {a.rho} m, azimuth {a.azimuth} deg. "
                         "Camera only: no gripper or knob commands.")
    env["max_commands"], env["max_duration_s"] = 60, 900.0
    env_path = ROOT / "hardware" / "envelopes" / f"{env['name']}.json"
    env_path.write_text(json.dumps(env, indent=2) + "\n")
    WORKSPACE_FILE.write_text(json.dumps({"region": region, "rho_max_m": a.rho_max, "envelope": str(env_path.relative_to(ROOT)),
                                          "survey_poses": [V.kortex_pose(s.tool) for s in surveys],
                                          "date": time.strftime("%Y-%m-%d %H:%M")}, indent=1))
    for i, s in enumerate(surveys):
        print(f"  survey {i}: tool ({s.tool.t[0]:.3f}, {s.tool.t[1]:+.3f}, {s.tool.t[2]:.3f}), tilt {V.tilt_deg(s.tool):.0f} deg, "
              f"radius {np.hypot(*s.tool.t[:2]):.3f} m, sees {len(cover[cands.index(s)])} placements")
    print(f"envelope {env_path.name}: box_min {env['box_min']} box_max {env['box_max']} tilt <= {env['max_tilt_deg']}")


def drawing_envelope(boards: list, st: dict) -> Path:
    """ONE drawing envelope for the whole workspace: a box around the knob approach and grasp poses of every board
    placement (3 cm sideways and above, 1.5 cm below the lowest grasp), tool within 5 deg of straight down, joint 6
    may turn the knobs. Approved once per session; the drawer only ever goes to knob poses inside it."""
    d, layout, z = model()
    pts = []
    for b in boards:
        for rel in d["grasps"].values():
            g = B.grasp_in_base(b, rel)
            pts += [g[:3], g[:2] + [g[2] + APPROACH_M], g[:2] + [HANDOVER_Z_M]]
    P = np.array(pts)
    lo, hi = P.min(0) - 0.03, P.max(0) + 0.03
    lo[2] = P[:, 2].min() - 0.015
    hi[2] = HANDOVER_Z_M + 0.02          # the handover pose (above a knob, inside the re-locate box too) fits
    env = {"name": f"etch_draw_workspace_{st['ip'].split('.')[-1]}",
           "box_min": [round(float(v), 3) for v in lo], "box_max": [round(float(v), 3) for v in hi],
           "max_lin_speed": 0.05, "max_ang_speed": 30.0, "max_tilt_deg": 5.0, "rotate_joint": 6,
           "max_rotation_deg": 1800.0, "max_joint_torque_nm": 0.7, "max_step_m": 0.25, "max_commands": 2000,
           "max_duration_s": 1800.0,
           "notes": "Etch-a-Sketch drawing (etch-sim RealBackend with hardware/knob_poses_live.json from scripts/relocate.py): "
                    "knob approach/grasp poses for any board placement in the re-locate workspace, gripper, joint-6 knob turns. "
                    "Lift the arm out of it between sessions (J2 heats up when parked low)."}
    path = ROOT / "hardware" / "envelopes" / f"{env['name']}.json"
    path.write_text(json.dumps(env, indent=2) + "\n")
    rho = np.hypot(P[:, 0], P[:, 1])
    print(f"drawing envelope {path.name}: box_min {env['box_min']} box_max {env['box_max']}; knob poses "
          f"{rho.min():.3f}-{rho.max():.3f} m from the base axis")
    return path


def draw_envelope(a) -> None:
    """Offline: the drawing envelope for the saved workspace region (survey poses and the re-locate envelope stay as
    they are)."""
    ws = json.loads(WORKSPACE_FILE.read_text())
    drawing_envelope(placements(ws["region"]), W.arm_status())


def auto(a) -> None:
    """The whole re-locate in one approved routine: look; if not all 4 markers are seen, go to the survey poses until
    one sees them; plan 5 close views inside the approved envelope; capture; solve."""
    from etch_sim.mj.scene import EtchScene
    from etch_sim.real import view_plan as V
    from etch_sim.real.backend import ArmRefused
    ws = json.loads(WORKSPACE_FILE.read_text())
    env = json.loads((ROOT / ws["envelope"]).read_text())
    K = W.load_calibration()[0]
    d, layout, z = model()
    s = session(True)
    r = R.ViewRunner(a.port, a.camera_ip)
    t0 = time.time()
    try:
        cur = W.tool_pose(r.status())
        m = r.capture(s / "sight_00")
        best = len(m)
        if best < 4:
            for i, p in enumerate(ws["survey_poses"]):
                goal = W.Pose(Rotation.from_euler("xyz", p[3:], degrees=True), np.array(p[:3]))
                for w in V.path_to(cur, goal):
                    r.move(V.kortex_pose(w), f"survey {i}")
                cur = goal
                m = r.capture(s / f"sight_{i + 1:02d}")
                best = max(best, len(m))
                if len(m) == 4:
                    break
        sights = R.load_views(s, "sight")
        if not sights:
            raise SystemExit("board not found: no survey pose saw 2 markers. Check the board is inside the workspace.")
        K_, D_, mount = W.load_calibration()
        board = B.fit_board(sights, layout, z, K_, D_, mount).board
        print(f"board found at ({board.t[0]:.3f}, {board.t[1]:.3f}) m, yaw {math.degrees(board.R.as_euler('zyx')[0]):.1f} deg "
              f"(best sighting: {best} markers)")
        st = r.status()
        cur = W.tool_pose(st)
        cands = [c for c in V.candidates(marker_points(board), view_limits(ws["rho_max_m"]), K=K,
                                         ref_x=W.camera_pose(st).R.apply([1.0, 0.0, 0.0])) if V.inside_envelope(c.tool, env)]
        mdl = EtchScene().model
        views = V.order(V.select(cands, a.n, cur, lambda t: V.reachable(t, mdl)), cur)
        print(f"{len(cands)} candidate views inside the envelope; {len(views)} picked")
        for j, v in enumerate(views):
            for w in V.path_to(cur, v.tool):
                r.move(V.kortex_pose(w), f"view {j}")
            cur = v.tool
            r.capture(s / f"view_{j:02d}", {"planned": V.kortex_pose(v.tool)})
    except ArmRefused as e:
        print(f"REFUSED: {e}\nThe routine has ended; nothing retries. Frames saved so far are kept.")
        raise SystemExit(1)
    finally:
        r.close()
    print(f"captured in {time.time() - t0:.0f} s")
    solve(a)


def learn(a) -> None:
    """Add this session's hand-corrected grasps to the board model (stored relative to the board, averaged per knob),
    so future re-locates follow how you grip."""
    s = session()
    res = json.loads((s / "result.json").read_text())
    d, layout, z = model()
    board = B.flat_pose(res["board"]["x"], res["board"]["y"], z, math.radians(res["board"]["yaw_deg"]))
    before = {ax: g.t.copy() for ax, g in d["grasps"].items()}
    for ax in ("x", "y"):
        f = s / f"taught_{ax}.json"
        if not f.exists():
            continue
        rel = B.relative_grasp(board, json.loads(f.read_text())["status"])
        B.add_sample(B.MODEL_FILE, ax, rel, f"relocate {s.name}")
    d2 = B.load_model()
    for ax, g in d2["grasps"].items():
        dt = (g.t - before[ax]) * 1000
        print(f"knob {ax}: {len(d2['grasp_samples'].get(ax, [])) or 1} samples; grasp moved ({dt[0]:+.1f}, {dt[1]:+.1f}, {dt[2]:+.1f}) mm "
              f"in the board frame")
    print(f"grip_raise_mm = {d2.get('grip_raise_mm', 0.0):g} (edit hardware/board_model.json to grip higher or lower)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sight")
    p.add_argument("--new", action="store_true")
    p = sub.add_parser("plan")
    p.add_argument("-n", type=int, default=5)
    p.add_argument("--rho-max", type=float, default=0.25)
    p = sub.add_parser("run")
    p.add_argument("--port", type=int, default=8765)
    sub.add_parser("solve")
    p = sub.add_parser("taught")
    p.add_argument("axis", choices=["x", "y"])
    sub.add_parser("compare")
    sub.add_parser("learn", help="add this session's hand-corrected grasps to the board model")
    p = sub.add_parser("workspace", help="offline: survey poses + one envelope covering the board workspace")
    p.add_argument("--rho", type=float, nargs=2, default=[0.30, 0.40], help="board centre distance from the base axis (m)")
    p.add_argument("--azimuth", type=float, nargs=2, default=[0.0, 30.0], help="board centre direction range (deg)")
    p.add_argument("--yaw-offset", type=float, default=-104.0, help="board yaw minus its azimuth, as placed today (deg)")
    p.add_argument("--yaw-range", type=float, default=20.0, help="+- board rotation allowed (deg)")
    p.add_argument("--rho-max", type=float, default=0.25, help="max tool distance from the base axis (m)")
    sub.add_parser("draw-envelope", help="offline: drawing envelope for the saved workspace (knob poses, knob turns)")
    p = sub.add_parser("auto", help="look, survey if needed, plan, capture, solve: one approved routine")
    p.add_argument("-n", type=int, default=5)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--camera-ip", help="wrist camera IP (default: the arm's; set it for a dry run on the fake server)")
    a = ap.parse_args()
    {"sight": sight, "plan": plan, "run": run, "solve": solve, "taught": taught, "compare": compare, "learn": learn,
     "workspace": workspace, "draw-envelope": draw_envelope, "auto": auto}[a.cmd](a)


if __name__ == "__main__":
    main()
