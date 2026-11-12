"""Stage S2 demo and exit check: the arm grasps and turns the knobs in MuJoCo, the scene camera
measures the pen, and calibration recovers toy parameters it was never told.

  1. board localization from the markers vs the true toy pose
  2. zigzag calibration sweep on both knobs through the camera -> k and backlash vs hidden truth
  3. what the sweep reveals: knob slip in the gripper, pen-tracking error, switch time, toy drift
  4. open-loop square drawn by the arm with the calibrated values
Ground truth is used only to grade the results. Writes images and results.json to out/s2/.
Run:  uv run python scripts/s2_demo.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from etch_sim import AxisParams, Calib, OpenLoopDrawer, ToyKernel, ToyParams, chamfer, fit_axis, polyline, square, sweep_axis
from etch_sim.mj import EtchScene
from etch_sim.mj.backend import MujocoBackend

OUT = Path(__file__).resolve().parent.parent / "out" / "s2"


def hidden_toy() -> ToyKernel:
    """A toy whose parameters differ from every default in the code."""
    return ToyKernel(ToyParams(x=AxisParams(k_mm_per_deg=0.091, half_dead_deg=7.0, transition_deg=3.0, travel_mm=153.0),
                               y=AxisParams(k_mm_per_deg=0.079, half_dead_deg=11.0, transition_deg=4.0, travel_mm=107.0)),
                     start_mm=(60.0, 30.0))


def toy_pose(b: MujocoBackend):
    body = b.d.body("toy")
    return body.xpos.copy(), body.xmat.reshape(3, 3).copy()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {}
    b = MujocoBackend(EtchScene(kernel=hidden_toy()))
    toy = b.s.kernel
    pos0, R0 = toy_pose(b)

    # 1. Board localization.
    assert b.locate_board()
    R_est, t_est = b._board
    res["board_pos_err_mm"] = float(np.linalg.norm(t_est - pos0) * 1000)
    res["board_yaw_err_deg"] = float(np.rad2deg(np.arccos(np.clip((np.trace(R_est @ R0.T) - 1) / 2, -1, 1))))
    print(f"1. Board located from the camera: position error {res['board_pos_err_mm']:.2f} mm, "
          f"rotation error {res['board_yaw_err_deg']:.2f} deg")

    # 2. Calibration sweeps, recording truth alongside for grading.
    pen_err, slip = [], []
    observe, rotate = b.observe_pen, b.rotate

    def observe_graded():
        pen = observe()
        if not np.isnan(pen[0]):
            pen_err.append(float(np.hypot(pen[0] - toy.stylus[0], pen[1] - toy.stylus[1])))
        return pen

    def rotate_graded(deg):
        k0 = b.s.knob_deg(b.held)
        r = rotate(deg)
        slip.append((r.wrist_deg, b.s.knob_deg(b.held) - k0))
        return r

    b.observe_pen, b.rotate = observe_graded, rotate_graded
    fits = {}
    print("2. Calibration through the camera (fit vs hidden truth)")
    for axis in ("x", "y"):
        theta, pen = sweep_axis(b, axis)
        f = fits[axis] = fit_axis(theta, pen)
        ax = toy.p.axis(axis)
        res[f"fit_{axis}"] = {"k": f.k_mm_per_deg, "k_true": ax.k_mm_per_deg, "delta": f.delta_deg, "delta_true": ax.delta_deg,
                              "rms_mm": f.rms_mm, "n_used": f.n_used, "n_dropped": f.n_dropped}
        print(f"   {axis}: k={f.k_mm_per_deg:.4f} (true {ax.k_mm_per_deg:.4f}, {100 * (f.k_mm_per_deg / ax.k_mm_per_deg - 1):+.1f}%)   "
              f"delta={f.delta_deg:.1f} (true {ax.delta_deg:.1f}) deg   rms={f.rms_mm:.2f} mm   used {f.n_used}, dropped {f.n_dropped}")
    b.release()
    Image.fromarray(b.s.render(b.camera)).save(OUT / "after_calibration_scene.png")
    Image.fromarray(b.rectified()).save(OUT / "after_calibration_rectified.png")

    # 3. What the sweep revealed.
    wrist, knob = np.array(slip).T
    res["slip_pct"] = float(100 * (1 - np.abs(knob).sum() / np.abs(wrist).sum()))
    res["pen_err_mm"] = {"median": float(np.median(pen_err)), "p95": float(np.percentile(pen_err, 95)), "max": float(max(pen_err)), "n": len(pen_err)}
    grasps = [e for e in b.log if e.get("skill") in ("grasp_knob", "release", "rotate", "observe_pen")]
    switch_t = []
    for i, e in enumerate(grasps):
        if e["skill"] == "release":
            nxt = next((g for g in grasps[i + 1:] if g["skill"] == "grasp_knob"), None)
            prev = grasps[i - 1]
            if nxt is not None:
                switch_t.append(nxt["t"] - prev["t"])
    res["switch_time_s"] = float(np.mean(switch_t))
    pos1, R1 = toy_pose(b)
    res["toy_drift_mm"] = float(np.linalg.norm(pos1 - pos0) * 1000)
    res["toy_yaw_drift_deg"] = float(np.rad2deg(np.arctan2((R1 @ R0.T)[1, 0], (R1 @ R0.T)[0, 0])))
    res["calibration_time_s"] = b.clock_s
    res["calibration_switches"] = b.n_switches
    print(f"3. Knob slip in the gripper {res['slip_pct']:.2f}% of rotation;  pen tracking error median "
          f"{res['pen_err_mm']['median']:.2f} mm, 95th pct {res['pen_err_mm']['p95']:.2f} mm, max {res['pen_err_mm']['max']:.2f} mm ({len(pen_err)} photos)")
    print(f"   knob switch {res['switch_time_s']:.1f} s on average ({b.n_switches} switches);  whole calibration {b.clock_s:.0f} s of arm time;  "
          f"toy moved {res['toy_drift_mm']:.2f} mm, {res['toy_yaw_drift_deg']:.3f} deg")

    # 4. Open-loop square with the calibrated values, on a fresh toy with the same hidden parameters.
    calib = Calib({a: f.k_mm_per_deg for a, f in fits.items()}, {a: f.delta_deg for a, f in fits.items()})
    b2 = MujocoBackend(EtchScene(kernel=hidden_toy()))
    toy2 = b2.s.kernel
    drawer = OpenLoopDrawer(b2, calib)
    drawer.preload()
    mark, start, t0, n0 = toy2.mark(), toy2.stylus, b2.clock_s, b2.n_switches
    moves = square(40.0)
    drawer.draw(moves)
    drawn, target = toy2.path_since(mark), polyline(moves, start)
    score = chamfer(drawn, target)
    sides = drawn.max(axis=0) - drawn.min(axis=0)          # width and height of what was drawn
    res["square"] = {**score, "side_err_mm": float(np.abs(sides - 40.0).max()), "closure_mm": float(np.linalg.norm(drawn[-1] - drawn[0])),
                     "time_s": b2.clock_s - t0, "switches": b2.n_switches - n0}
    print(f"4. 40 mm square drawn by the arm, open loop: chamfer {score['chamfer_mm']:.2f} mm, worst side error "
          f"{res['square']['side_err_mm']:.2f} mm, closure {res['square']['closure_mm']:.2f} mm, "
          f"{res['square']['time_s']:.0f} s, {res['square']['switches']} switches")
    Image.fromarray(b2.s.render(b2.camera)).save(OUT / "square_scene.png")
    rect = b2.rectified()
    Image.fromarray(np.hstack([rect, np.asarray(toy2.render(overlay=target))])).save(OUT / "square_rectified_vs_target.png")

    (OUT / "results.json").write_text(json.dumps(res, indent=2))
    ok = (all(abs(res[f"fit_{a}"]["k"] / res[f"fit_{a}"]["k_true"] - 1) < 0.02 for a in "xy")
          and all(abs(res[f"fit_{a}"]["delta"] - res[f"fit_{a}"]["delta_true"]) < 2.5 for a in "xy")
          and res["pen_err_mm"]["median"] < 0.4 and res["square"]["side_err_mm"] < 3.0 and res["toy_drift_mm"] < 1.0)
    print(f"\nS2 exit check: {'PASS' if ok else 'FAIL'}   (images and results.json in {OUT})")


if __name__ == "__main__":
    main()
