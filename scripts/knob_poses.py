"""Knob approach and grasp poses for the real arm, from two hand-guided readings. No arm motion.

At the arm: in Cartesian admittance, guide the open gripper around the left (x, horizontal) knob,
close it so the fingers centre the arm on the knob, and save the arm's state (Web App data export or
`kinova --json status > x.json`). Same for the right (y, vertical) knob. Then:

    uv run python scripts/knob_poses.py x.json y.json
    uv run python scripts/knob_poses.py x.json y.json --envelope ../aims-kinova/envelopes/etch_draw.json
    uv run python scripts/knob_poses.py x.json y.json --write          # -> hardware/taught_poses.json

Prints the knob centres, the toy's placement, the poses (tool straight down), every safety and
envelope check, the joint angles our Gen3 model needs, and a proposed envelope box. --write stores
the poses in hardware/taught_poses.json, which RealBackend reads. The envelope itself belongs to the
aims-kinova repo and is approved by a person; this script only proposes it.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from etch_sim.real import knob_poses as K
from etch_sim.real.backend import POSES_FILE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("x_reading", help="reading at the left (x, horizontal) knob, gripper closed in admittance")
    ap.add_argument("y_reading", help="reading at the right (y, vertical) knob")
    ap.add_argument("--lift", type=float, default=0.05, help="approach height above the grasp (m)")
    ap.add_argument("--envelope", help="aims-kinova envelope JSON to check against")
    ap.add_argument("--write", action="store_true", help="store the poses in hardware/taught_poses.json")
    ap.add_argument("--no-ik", action="store_true", help="skip the MuJoCo joint-angle check")
    a = ap.parse_args()

    rx, ry = K.parse_reading(a.x_reading), K.parse_reading(a.y_reading)
    p = K.placement(rx, ry)
    print("Knob centres at grasp height (pinch point, arm base frame, m):")
    for ax, r in (("x", rx), ("y", ry)):
        c = p.knob[ax]
        print(f"  {ax}: ({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f})   reading tilt {p.tilt_deg[ax]:.1f} deg   {Path(r.source).name}")
    print(f"Knob spacing {p.spacing_mm:.1f} mm; toy rotated {p.toy_yaw_offset_deg:+.1f} deg from square to the arm; "
          f"finger yaw {p.tool_yaw_deg:.1f} deg")
    if abs(p.knob['x'][2] - p.knob['y'][2]) > 0.005:
        print(f"  WARNING: knob heights differ by {abs(p.knob['x'][2] - p.knob['y'][2]) * 1000:.0f} mm (toy not flat?)")

    poses = K.knob_poses(p, a.lift)
    env = json.loads(Path(a.envelope).read_text()) if a.envelope else None
    print(f"\nPoses [x, y, z, theta_x, theta_y, theta_z] (tool straight down; above = grasp + {a.lift:.3f} m):")
    bad = 0
    for ax in ("x", "y"):
        for kind in ("above", "grasp"):
            pose = poses[ax][kind]
            errs = K.check_pose(pose)
            env_errs = [e for e in K.check_pose(pose, env) if e.startswith("envelope")] if env else []
            joints, ik_errs = (None, []) if a.no_ik else K.ik_check(pose)
            bad += len(errs) + len(ik_errs)
            status = "OK" if not (errs or ik_errs) else "REJECT: " + "; ".join(errs + ik_errs)
            print(f"  {ax} {kind:5s} {pose}  safety {status}")
            if joints is not None:
                print(f"        model joints (deg) {joints}  (joint 6 free: the arm sets it from theta_z)")
            if env_errs:
                print(f"        current envelope: {'; '.join(env_errs)}")
    prop = K.propose_envelope(poses, base=env)
    print(f"\nProposed envelope box (for the aims-kinova envelope file, approved by a person):\n"
          f"  box_min {prop['box_min']}  box_max {prop['box_max']}")

    if a.write:
        if bad:
            raise SystemExit("not written: some poses failed the checks above")
        data = json.loads(POSES_FILE.read_text())
        data["knobs"] = poses
        data["_knobs_source"] = (f"computed by scripts/knob_poses.py on {time.strftime('%Y-%m-%d %H:%M')} from "
                                 f"{Path(rx.source).name}, {Path(ry.source).name}: spacing {p.spacing_mm:.1f} mm, "
                                 f"toy yaw {p.toy_yaw_offset_deg:+.1f} deg, lift {a.lift} m")
        POSES_FILE.write_text(json.dumps(data, indent=2) + "\n")
        print(f"\nwrote {POSES_FILE}")


if __name__ == "__main__":
    main()
