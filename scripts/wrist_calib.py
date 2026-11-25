"""Wrist camera calibration from the ChArUco chart taped flat on the table. Read-only: this script never moves the arm.

You drive the arm with the Xbox controller; at each pose the script grabs a wrist frame (RTSP) and the arm's status.

    uv run python scripts/wrist_calib.py capture              # Enter = capture this pose, q = finish
    uv run python scripts/wrist_calib.py capture --once       # one capture, then exit (for an agent)
    uv run python scripts/wrist_calib.py capture --session test_images/wrist_calib/<session>   # add to a session
    uv run python scripts/wrist_calib.py solve test_images/wrist_calib/<session> [--save]

Poses that calibrate well (aim for 15-20): the camera 25-45 cm from the chart; tilted 15-35 deg from straight down,
from all four sides; the wrist rolled differently; the chart in the centre, the corners and the edges of the image.
Hold still at each pose (lock with RB) before pressing Enter: a capture is refused if the arm moved during it.

solve: intrinsics (focal lengths, principal point, distortion), then the camera mount on the flange and the chart's
pose in the base frame from all views together. Checks: reprojection error, mount vs Kinova's model, and the chart
height in the base frame (table = -1.8 mm: the arm base stands on a 1.8 mm pad). --save writes
hardware/wrist_camera_calib.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from etch_sim.real import wrist_camera as W

ROOT = Path(__file__).resolve().parents[1]
TABLE_Z_M = -0.018                     # table in the base frame: base on an 18 mm rubber pad (Raoul, 2026-11-24)
GRID = (4, 3)                          # coverage cells across the image


def coverage(points_list) -> np.ndarray:
    g = np.zeros(GRID[::-1], int)
    for pts in points_list:
        cells = {(min(int(y / W.IMAGE_SIZE[1] * GRID[1]), GRID[1] - 1), min(int(x / W.IMAGE_SIZE[0] * GRID[0]), GRID[0] - 1))
                 for x, y in pts}
        for c in cells:
            g[c] += 1
    return g


def view_geometry(status: dict, det, K) -> tuple[float, float]:
    """Camera distance to the chart (m) and tilt of the camera axis from straight down (deg)."""
    cam = W.camera_pose(status)
    z = cam.R.apply([0, 0, 1])
    tilt = float(np.degrees(np.arccos(np.clip(-z[2], -1, 1))))
    _, _, t = cv2.solvePnP(det[0], det[1], K, None)
    return float(np.linalg.norm(t)), tilt


def capture(a) -> None:
    sess = Path(a.session) if a.session else ROOT / "test_images" / "wrist_calib" / time.strftime("%Y%m%d-%H%M%S")
    sess.mkdir(parents=True, exist_ok=True)
    st = W.arm_status()
    print(f"arm {st['ip']} {st['arm_state']}; session {sess}")
    grab = W.RtspGrabber(st["ip"])
    try:
        while True:
            n = len(sorted(sess.glob("pose_*.png")))
            if not a.once:
                cmd = input(f"[{n} captured] pose the arm, lock it, Enter to capture (q to finish): ").strip().lower()
                if cmd == "q":
                    break
            before = W.arm_status()
            img = grab.frame_after(time.time() + 0.3)
            after = W.arm_status()
            moved = np.max(np.abs(np.subtract(before["joint_angles_deg"], after["joint_angles_deg"])))
            if moved > 0.05:
                print(f"  refused: the arm moved {moved:.2f} deg during the capture; hold still and retry")
                if a.once:
                    sys.exit(1)
                continue
            det = W.detect_chart(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            if det is None:
                print("  refused: fewer than 8 chart corners visible; aim the camera at the chart")
                if a.once:
                    sys.exit(1)
                continue
            stem = sess / f"pose_{n:02d}"
            cv2.imwrite(f"{stem}.png", img)
            stem.with_suffix(".json").write_text(json.dumps({"status": after, "t": time.time()}, indent=1))
            dist, tilt = view_geometry(after, det, W.K_GUESS)
            print(f"  saved {stem.name}: {len(det[2])}/28 corners, ~{dist * 100:.0f} cm away, camera tilt {tilt:.0f} deg")
            dets = [W.detect_chart(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)) for p in sorted(sess.glob("pose_*.png"))]
            print("  image coverage (views per cell):\n" + "\n".join("    " + " ".join(f"{v:2d}" for v in row)
                                                                    for row in coverage([d[1] for d in dets if d])))
            if a.once:
                break
    finally:
        grab.close()


def load_session(sess: Path):
    views, statuses, names = [], [], []
    for p in sorted(sess.glob("pose_*.png")):
        det = W.detect_chart(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))
        if det is None:
            print(f"  {p.name}: chart not found, skipped")
            continue
        views.append(det)
        statuses.append(json.loads(p.with_suffix(".json").read_text())["status"])
        names.append(p.stem)
    return views, statuses, names


def solve(a) -> None:
    sess = Path(a.session)
    views, statuses, names = load_session(sess)
    if len(views) < 6:
        raise SystemExit(f"only {len(views)} usable views; capture at least 10-15")
    drop = set(a.drop or [])
    keep = [i for i, n in enumerate(names) if n not in drop]
    views, statuses, names = [views[i] for i in keep], [statuses[i] for i in keep], [names[i] for i in keep]
    print(f"{len(views)} views, {sum(len(v[0]) for v in views)} corners")

    # chart alone (no arm poses): shows how flat the chart is; its focal length is NOT trusted (paper bends)
    intr = W.calibrate_intrinsics(views)
    print(f"chart-only lens fit: fx {intr.K[0, 0]:.1f} fy {intr.K[1, 1]:.1f} cx {intr.K[0, 2]:.1f} cy {intr.K[1, 2]:.1f}, "
          f"rms {intr.rms_px:.2f} px (above ~0.5 px: chart not flat or blurred frames)")

    fit = W.solve_joint(views, statuses, fix_mount_rotation=a.fix_mount_rotation)
    K, d = fit.K, fit.dist
    dm = np.linalg.norm(fit.mount.t - W.MOUNT_MODEL.t) * 1000
    dang = np.degrees((W.MOUNT_MODEL.R.inv() * fit.mount.R).magnitude())
    allc = fit.chart_in_base.apply(np.asarray(W.Chart().board().getChessboardCorners()) * fit.chart_scale)
    print(f"joint fit (arm poses + images): rms {fit.rms_px:.2f} px")
    print(f"  intrinsics fx {K[0, 0]:.1f} fy {K[1, 1]:.1f} cx {K[0, 2]:.1f} cy {K[1, 2]:.1f}  "
          f"dist k1 {d[0]:+.4f} k2 {d[1]:+.4f} p1 {d[2]:+.5f} p2 {d[3]:+.5f}   (Kortex: 1297.7 1298.6 620.9 238.3, none)")
    print(f"  mount t {np.round(fit.mount.t * 1000, 1)} mm: {dm:.1f} mm and {dang:.2f} deg from Kinova's model")
    print(f"  chart printed at {fit.chart_scale * 100:.2f} % (square {W.Chart().square_m * fit.chart_scale * 1000:.2f} mm); "
          f"height in base {allc[:, 2].mean() * 1000:+.1f} mm (table expected {TABLE_Z_M * 1000:+.1f})")
    med = np.median(fit.per_view_rms)
    print("  per view rms (px): " + ", ".join(f"{nm[-2:]} {r:.1f}{'!' if r > 2.5 * med else ''}" for nm, r in zip(names, fit.per_view_rms)))

    # the number that matters for re-locating a target: where does EACH view alone put the chart?
    cen = []
    full = np.asarray(W.Chart().board().getChessboardCorners()) * fit.chart_scale
    for (o, i, _), s in zip(views, statuses):
        _, rv, tv = cv2.solvePnP(o * fit.chart_scale, i, K, d)
        cb = W.flange_pose(s) * fit.mount * W.Pose.from_vec(np.r_[rv.ravel(), tv.ravel()])
        cen.append(cb.apply(full).mean(0) * 1000)
    cen = np.array(cen)
    dev = np.linalg.norm(cen - np.median(cen, 0), axis=1)
    print(f"single-view repeatability of the chart centre: std x {cen[:, 0].std():.1f} y {cen[:, 1].std():.1f} "
          f"z {cen[:, 2].std():.1f} mm, worst {dev.max():.1f} mm ({names[int(dev.argmax())]})")

    if a.save:
        W.save_calibration(W.Intrinsics(K, d, fit.rms_px), W.HandEye(fit.mount, fit.chart_in_base, fit.rms_px, fit.per_view_rms),
                           {"session": str(sess), "views": names, "arm_ip": statuses[0]["ip"], "method": "joint",
                            "chart_scale": fit.chart_scale, "chart_height_mm": float(allc[:, 2].mean() * 1000),
                            "single_view_std_mm": cen.std(0).round(2).tolist(), "date": time.strftime("%Y-%m-%d %H:%M")})
        print(f"wrote {W.CALIB_FILE}")


def chart_points_in_base(sess: Path) -> tuple[np.ndarray, dict]:
    """Chart corners (inner corners + the board's outer corners) in the base frame from the session's first frame,
    using the Kortex focal length, a centred principal point and Kinova's camera mount (good to ~1-2 cm)."""
    first = sorted(sess.glob("pose_*.png"))[0]
    det = W.detect_chart(cv2.imread(str(first), cv2.IMREAD_GRAYSCALE))
    st = json.loads(first.with_suffix(".json").read_text())["status"]
    _, r, t = cv2.solvePnP(det[0], det[1], W.K_GUESS, None)
    chart = W.camera_pose(st) * W.Pose.from_vec(np.r_[r.ravel(), t.ravel()])
    ch = W.Chart()
    w, h = ch.squares[0] * ch.square_m, ch.squares[1] * ch.square_m
    pts = np.vstack([W.Chart().board().getChessboardCorners(), [[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]]])
    return chart.apply(pts), st


def plan(a) -> None:
    from etch_sim.mj.scene import EtchScene
    from etch_sim.real import view_plan as V
    sess = Path(a.session)
    pts, st = chart_points_in_base(sess)
    start = W.tool_pose(st)
    lim = V.ViewLimits(rho_min_m=a.rho_min, rho_max_m=a.rho_max, dist_m=tuple(a.dist), tilts_deg=(0, 10, 20, 30, 40))
    ref_x = W.camera_pose(st).R.apply([1.0, 0.0, 0.0])          # keep the wrist near its current yaw
    cands = V.candidates(pts, lim, ref_x=ref_x)
    model = EtchScene().model
    views = V.order(V.select(cands, a.n, start, lambda tool: V.reachable(tool, model)), start)
    moves = V.with_waypoints(views, start, lim.max_step_m, lim.max_turn_deg)
    c = pts.mean(0)
    print(f"chart centre in base ({c[0]:.3f}, {c[1]:.3f}, {c[2] * 1000:+.0f} mm), {np.hypot(c[0], c[1]) * 100:.0f} cm from the base axis")
    print(f"{len(cands)} candidate views pass the image/safety checks; {len(views)} picked (reachable in the MuJoCo Gen3)")
    print(" #    tool x,y,z (m)           tilt  cam dist  step  turn   chart corners in view")
    prev, k, bad = start, 0, []
    for tool, cap in moves:
        step = np.linalg.norm(tool.t - prev.t)
        turn = np.degrees((prev.R.inv() * tool.R).magnitude())
        rho, z = np.hypot(*tool.t[:2]), tool.t[2]
        if rho < lim.rho_min_m - 0.03 or rho > lim.rho_max_m + 0.01 or z < lim.tool_z_min_m or step > lim.max_step_m + 1e-6 \
                or turn > lim.max_turn_deg + 1e-6 or not V.reachable(tool, model):
            bad.append(tool)
        if cap:
            v = views[k]
            k += 1
            vis = ((v.img_pts[:28, 0] > 0) & (v.img_pts[:28, 0] < 1280) & (v.img_pts[:28, 1] > 0) & (v.img_pts[:28, 1] < 720)).sum()
            print(f"{k:2d}   ({tool.t[0]:.3f}, {tool.t[1]:+.3f}, {tool.t[2]:.3f})  {V.tilt_deg(tool):4.0f}  "
                  f"{np.linalg.norm(v.cam.t - c) * 100:5.0f} cm  {step * 100:4.0f} cm  {turn:3.0f}°   {vis}/28")
        else:
            print(f" via ({tool.t[0]:.3f}, {tool.t[1]:+.3f}, {tool.t[2]:.3f})  {V.tilt_deg(tool):4.0f}            {step * 100:4.0f} cm  {turn:3.0f}°   (no capture)")
        prev = tool
    if bad:
        raise SystemExit(f"plan rejected: {len(bad)} moves break a limit (step, height, distance from base, reach)")
    env = V.envelope_for([m[0] for m in moves], f"wrist_calib_{st['ip'].split('.')[-1]}_{time.strftime('%Y%m%d')}",
                         "Wrist-camera calibration views of the ChArUco chart on the table (etch-sim scripts/wrist_calib.py "
                         f"plan, session {sess.name}). Camera only: no gripper or knob commands. Valid for this chart placement.")
    env_path = ROOT / "hardware" / "envelopes" / f"{env['name']}.json"
    env_path.parent.mkdir(exist_ok=True)
    env_path.write_text(json.dumps(env, indent=2) + "\n")
    (sess / "plan.json").write_text(json.dumps({"start_status": st, "envelope": str(env_path.relative_to(ROOT)),
                                                "moves": [{"pose": V.kortex_pose(t), "capture": cap} for t, cap in moves]},
                                               indent=1))
    # contact sheet of where the chart should appear in each view
    tiles = []
    for i, v in enumerate(views, 1):
        img = np.full((180, 320, 3), 255, np.uint8)
        q = (v.img_pts * 0.25).astype(np.int32)
        cv2.polylines(img, [q[-4:]], True, (0, 0, 0), 2)
        for p in q[:-4]:
            cv2.circle(img, tuple(int(x) for x in p), 2, (0, 0, 200), -1)
        cv2.rectangle(img, (0, 0), (319, 179), (160, 160, 160), 1)
        cv2.putText(img, f"{i}  tilt {V.tilt_deg(v.tool):.0f}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        tiles.append(img)
    while len(tiles) % 5:
        tiles.append(np.full_like(tiles[0], 255))
    sheet = np.vstack([np.hstack(tiles[r:r + 5]) for r in range(0, len(tiles), 5)])
    cv2.imwrite(str(sess / "plan_preview.png"), sheet)
    print(f"envelope proposal: {env_path}\n  box_min {env['box_min']} box_max {env['box_max']} tilt <= {env['max_tilt_deg']} deg, "
          f"{env['max_lin_speed']} m/s, {env['max_ang_speed']} deg/s, <= {env['max_commands']} commands, 15 min")
    print(f"plan: {sess / 'plan.json'}; preview: {sess / 'plan_preview.png'}")


def run(a) -> None:
    """Visit the planned poses through the aims-kinova motion server (bounded routine approved by a person when the
    server was started) and capture a frame at each. Any refusal or failure stops the run; nothing retries."""
    from etch_sim.real import ArmClient, ArmRefused
    sess = Path(a.session)
    moves = json.loads((sess / "plan.json").read_text())["moves"]
    c = ArmClient(port=a.port)
    st = c.call("status")["state"]
    if st.get("arm_state") != "ARMSTATE_SERVOING_READY":
        raise SystemExit(f"arm is {st.get('arm_state')}, not SERVOING_READY: release the Xbox pad / Web App first")
    grab = W.RtspGrabber(a.camera_ip or st["ip"])
    n0 = len(sorted(sess.glob("pose_*.png")))
    i = 0
    try:
        for j, mv in enumerate(moves, 1):
            p = mv["pose"]
            print(f"move {j}/{len(moves)}{'' if mv['capture'] else ' (via)'}: {p}", flush=True)
            c.call("move_pose", x=p[0], y=p[1], z=p[2], theta_x=p[3], theta_y=p[4], theta_z=p[5])
            if not mv["capture"]:
                continue
            time.sleep(0.8)                                    # let the arm settle
            before = c.call("status")["state"]
            img = grab.frame_after(time.time() + 0.2)
            after = c.call("status")["state"]
            moved = np.max(np.abs(np.subtract(before["joint_angles_deg"], after["joint_angles_deg"])))
            det = W.detect_chart(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            stem = sess / f"pose_{n0 + i:02d}"
            cv2.imwrite(f"{stem}.png", img)
            stem.with_suffix(".json").write_text(json.dumps({"status": after, "t": time.time(), "planned": p}, indent=1))
            print(f"  saved {stem.name}: {0 if det is None else len(det[2])}/28 corners"
                  + (f"  (arm still moving {moved:.2f} deg: view will be dropped)" if moved > 0.05 else ""), flush=True)
            i += 1
    except ArmRefused as e:
        print(f"REFUSED: {e}\nThe routine has ended. Nothing retries; the frames saved so far are kept.")
        raise SystemExit(1)
    finally:
        grab.close()
        c.close()
    print(f"done: {i} views. Solve with: uv run python scripts/wrist_calib.py solve {sess}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--session")
    c.add_argument("--once", action="store_true")
    p = sub.add_parser("plan", help="plan views of the chart from the session's first frame (no motion)")
    p.add_argument("session")
    p.add_argument("-n", type=int, default=20)
    p.add_argument("--rho-min", type=float, default=0.17, help="min tool distance from the base axis (m)")
    p.add_argument("--rho-max", type=float, default=0.25, help="max tool distance from the base axis (m)")
    p.add_argument("--dist", type=float, nargs=2, default=[0.40, 0.60], help="camera-to-chart distance range (m)")
    r = sub.add_parser("run", help="visit the planned views through the motion server and capture")
    r.add_argument("session")
    r.add_argument("--port", type=int, default=8765, help="motion server port")
    r.add_argument("--camera-ip", help="wrist camera IP (default: the arm's)")
    s = sub.add_parser("solve")
    s.add_argument("session")
    s.add_argument("--drop", nargs="*", help="view names to leave out, e.g. pose_03")
    s.add_argument("--save", action="store_true")
    s.add_argument("--fix-mount-rotation", action="store_true", help="hold the mount rotation at Kinova's model")
    a = ap.parse_args()
    {"capture": capture, "plan": plan, "run": run, "solve": solve}[a.cmd](a)


if __name__ == "__main__":
    main()
