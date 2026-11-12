"""Stage S1 demo and exit check: MuJoCo scene with arm, gripper, cameras, toy and live screen.

Checks, with numbers:
  1. model vs real arm: forward kinematics against tool positions read back from the green arm
  2. dynamics sanity: toy rests on the table, arm holds Home
  3. reach: IK puts the gripper over each knob, tool pointing down
  4. live screen: turning the knob joints draws; the scene camera image, rectified through the
     ArUco markers, matches the kernel's own picture
  5. cameras: markers found and screen visibility from each scene-camera mount and the wrist camera
Writes images and results.json to out/s1/. Run:  uv run python scripts/s1_demo.py
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from etch_sim import perception as P
from etch_sim.mj import EtchScene, ik
from etch_sim.mj.scene import SCREEN_PX_PER_MM

OUT = Path(__file__).resolve().parent.parent / "out" / "s1"
# (joint angles deg, tool xyz m) read back from the green arm, aims-kinova logs/selftest_A.log, 2026-11-09.
REAL_READBACKS = [([-0.022, 15.089, -129.961, -0.014, 55.126, 89.992], (0.574, -0.004, 0.426)),
                  ([0.041, 8.339, -125.367, 0.333, 43.725, 89.656], (0.574, -0.005, 0.476)),
                  ([5.823, -14.823, -141.257, 5.213, -61.188, 93.139], (0.109, -0.043, 0.212))]
REAL_TOOL_OFFSET_M = 0.12            # the arm's configured tool frame sits 0.12 m beyond the wrist flange
PARK_DEG = REAL_READBACKS[2][0]      # the arm's rest pose, clear of the board
SCENE_CAMS = ("scene_overhead", "scene_opposite")


def save(img: np.ndarray, name: str) -> None:
    Image.fromarray(img).save(OUT / name)


def turn_knob(s: EtchScene, axis: str, kernel_deg: float, n: int = 40) -> None:
    """Rotate a knob joint directly (no gripper yet; grasping is S2)."""
    sign = -1.0 if s.cfg.toy.knob_cw_positive else 1.0
    adr = s._knob_qadr[axis]
    for _ in range(n):
        s.data.qpos[adr] += sign * np.deg2rad(kernel_deg / n)
        mujoco.mj_forward(s.model, s.data)
        s.sync_toy()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    s = EtchScene()
    m, d, g = s.model, s.data, s.cfg.toy
    res: dict = {}

    # 1. Forward kinematics vs the real arm.
    ee = m.site("ee").id
    errs = []
    for q_deg, tool in REAL_READBACKS:
        s.set_arm(np.deg2rad(q_deg))
        z = d.site_xmat[ee].reshape(3, 3)[:, 2]
        errs.append(float(np.linalg.norm(d.site_xpos[ee] + REAL_TOOL_OFFSET_M * z - tool) * 1000))
    res["fk_error_mm"] = [round(e, 1) for e in errs]
    print(f"1. FK vs real arm read-backs: error {', '.join(f'{e:.1f}' for e in errs)} mm")

    # 2. Dynamics sanity.
    s.reset()
    toy0 = d.body("toy").xpos.copy()
    s.step(2.0)
    res["toy_drift_mm"] = float(np.linalg.norm(d.body("toy").xpos - toy0) * 1000)
    res["arm_hold_error_deg"] = float(np.abs(np.rad2deg(d.qpos[s.arm_qadr] - d.ctrl[s.arm_act])).max())
    print(f"2. After 2 s: toy moved {res['toy_drift_mm']:.2f} mm, arm sags {res['arm_hold_error_deg']:.2f} deg from Home")

    # 3. Reach: gripper over each knob, pads at knob mid-height, tool down.
    knob_q = {}
    for side in ("left", "right"):
        top = d.site_xpos[m.site(f"knob_{side}_top").id]
        target = top - [0, 0, g.knob_height_mm / 2000]
        q, pe, ae = ik.solve(m, target)
        knob_q[side] = q
        res[f"ik_{side}_knob"] = {"reached": q is not None, "pos_err_mm": round(pe, 3), "axis_err_deg": round(ae, 3),
                                  "q_deg": None if q is None else [round(float(v), 1) for v in np.rad2deg(q)]}
        print(f"3. IK {side} knob at {np.round(target, 3)}: {'reached' if q is not None else 'NOT REACHED'} "
              f"(pos {pe:.2f} mm, axis {ae:.2f} deg)")

    # 4. Live screen: draw by turning the knob joints, then compare camera vs kernel.
    s.set_arm(np.deg2rad(PARK_DEG))
    before = s.render("scene_overhead")
    for axis, deg in (("x", 400), ("y", 350), ("x", -700), ("y", -250), ("x", 300)):
        turn_knob(s, axis, deg)
    after = s.render("scene_overhead")
    res["screen_pixels_changed"] = int((np.abs(after.astype(int) - before.astype(int)).sum(-1) > 30).sum())
    kernel_img = np.asarray(s.kernel.render(SCREEN_PX_PER_MM))
    kernel_mask = P.line_mask(kernel_img)
    print(f"4. Knobs turned: stylus at ({s.kernel.stylus[0]:.1f}, {s.kernel.stylus[1]:.1f}) mm, "
          f"{res['screen_pixels_changed']} overhead-camera pixels changed")
    for cam in SCENE_CAMS:
        img = s.render(cam)
        markers = P.detect_markers(img)
        H = P.board_homography(markers, g)
        entry = {"markers_found": sorted(markers)}
        if H is not None:
            rect = P.rectify(img, H, g, SCREEN_PX_PER_MM)
            dist = P.mask_distance_mm(P.line_mask(rect), kernel_mask, SCREEN_PX_PER_MM)
            entry.update(camera_to_kernel_mm=round(dist["a_to_b_mm"], 3), kernel_to_camera_mm=round(dist["b_to_a_mm"], 3),
                         max_mm=round(dist["max_mm"], 2))
            save(np.hstack([rect, kernel_img]), f"rectified_vs_kernel_{cam}.png")
        res[f"rectify_{cam}"] = entry
        save(img, f"{cam}_parked.png")
        print(f"   {cam}: markers {entry['markers_found']}, line offset camera->kernel "
              f"{entry.get('camera_to_kernel_mm', float('nan'))} mm, kernel->camera {entry.get('kernel_to_camera_mm', float('nan'))} mm")

    # 5a. Scene-camera visibility while the gripper is at a knob, over a full turn of joint 6.
    print("5. Screen visibility and markers with the gripper at a knob (min over one turn of joint 6):")
    for side, q in knob_q.items():
        if q is None:
            continue
        for cam in SCENE_CAMS:
            vis, nmark = [], []
            for yaw in np.arange(0, 360, 30):
                qq = q.copy()
                qq[5] += np.deg2rad(yaw)
                s.set_arm(qq)
                vis.append(s.visible_fraction(cam))
                nmark.append(len(P.detect_markers(s.render(cam))))
            res[f"visibility_{cam}_{side}"] = {"screen_min": round(min(vis), 3), "screen_mean": round(float(np.mean(vis)), 3),
                                               "markers_min": min(nmark)}
            print(f"   {cam:<15} gripper at {side:<5} knob: screen visible min {min(vis):.0%} mean {np.mean(vis):.0%}, "
                  f"markers min {min(nmark)}/4")
        s.set_arm(q)
        for cam in SCENE_CAMS:
            save(s.render(cam), f"{cam}_at_{side}_knob.png")
    if knob_q["left"] is not None:
        s.set_arm(knob_q["left"])
        save(s.render("wrist"), "wrist_at_left_knob.png")

    # 5b. Wrist camera looking down at the board.
    top_z = g.body_mm[2] / 1000
    centre = d.site_xpos[m.site("screen_origin").id] * 0 + [s.cfg.toy_pos[0] + g.screen_centre_mm[1] / 1000, s.cfg.toy_pos[1], top_z]
    best = (0, None, None)
    for height in (0.25, 0.30, 0.35):
        q, *_ = ik.solve(m, centre + [-0.05, 0, height])
        if q is None:
            continue
        for yaw in (0, 90, 180, 270):
            qq = q.copy()
            qq[5] += np.deg2rad(yaw)
            s.set_arm(qq)
            img = s.render("wrist")
            n = len(P.detect_markers(img))
            if n > best[0]:
                best = (n, img, (height, yaw))
    res["wrist_look"] = {"markers_found": best[0], "height_m_yaw_deg": best[2]}
    if best[1] is not None:
        save(best[1], "wrist_look.png")
    print(f"   wrist camera looking down: {best[0]}/4 markers at (height m, joint-6 offset deg) = {best[2]}")

    # Overview image for humans.
    if knob_q["left"] is not None:
        s.set_arm(knob_q["left"])
    r = mujoco.Renderer(m, 720, 1280)
    cam = mujoco.MjvCamera()
    cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = (0.3, 0, 0.15), 1.3, 140, -28
    r.update_scene(d, cam)
    save(r.render(), "overview.png")
    r.close()

    (OUT / "results.json").write_text(json.dumps(res, indent=2))
    ok = (max(errs) < 15 and all(res[f"ik_{k}_knob"]["reached"] for k in ("left", "right"))
          and all(len(res[f"rectify_{c}"]["markers_found"]) == 4 for c in SCENE_CAMS)
          and all(res[f"rectify_{c}"].get("camera_to_kernel_mm", 9) < 0.5 for c in SCENE_CAMS)
          and res["wrist_look"]["markers_found"] >= 3)
    print(f"\nS1 exit check: {'PASS' if ok else 'FAIL'}   (images and results.json in {OUT})")
    s.close()


if __name__ == "__main__":
    main()
