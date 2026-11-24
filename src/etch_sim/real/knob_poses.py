"""Knob poses for the real arm, from two hand-guided readings of the arm itself.

At the arm, a person guides the open gripper around each knob in Cartesian admittance, closes it so
the fingers centre the arm on the knob, and copies the arm's state (Kortex Web App data export, or
`kinova --json status`). From the two tool poses this module finds the knob centres in the arm's
base frame, the toy's placement (knob spacing, rotation), and the commanded approach and grasp poses
for each knob: tool pointing straight down, fingers at the measured grasp height.

Commands go to the arm as Cartesian tool poses, so Kortex's own kinematics place the gripper; our
MuJoCo model (8-9 mm from the blue and green arms' read-backs) is only used to check that each pose
is reachable within the joint limits and to report the joint angles.

Conventions (checked on the blue arm 2026-11-22, hardware/arm_readings/20261122_blue/home_like.json):
- Kortex theta_x/y/z are fixed-axis XYZ Euler angles in degrees (0.7 deg vs the model; the
  intrinsic reading is 120 deg off).
- The tool frame is 0.12 m beyond the wrist flange (8 mm fit; 0 m and 0.156 m give 126 and 31 mm).
- The gripper's pinch point (between the finger pads) is TOOL_TO_PINCH_M beyond the tool frame,
  from the MuJoCo Robotiq 2F-85 model.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

TOOL_TO_PINCH_M = 0.0358          # model: pinch site minus (flange + 0.12 m), along the tool axis

# Mirrored from aims-kinova python/kinova/safety.py (branch etch-bounded-routine, 7d0dab2). The motion
# server enforces its own copy; these only reject poses here before anything is sent.
R_MAX, Z_MIN, RHO_MIN, MAX_STEP = 0.80, 0.02, 0.15, 0.30
JOINT_LIMITS_DEG = [None, (-128.9, 128.9), (-147.8, 147.8), None, (-120.3, 120.3), None]


@dataclass
class Reading:
    joints_deg: list[float]       # wrapped to (-180, 180]
    tool: list[float]             # x, y, z (m), theta_x, theta_y, theta_z (deg), arm base frame
    source: str = ""


def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def parse_reading(path: str | Path) -> Reading:
    """A Kortex Web App data export (list of sections) or the output of `kinova --json status`."""
    d = json.loads(Path(path).read_text())
    if isinstance(d, dict) and "tool_pose" in d:
        tp = d["tool_pose"]
        tool = [tp[k] for k in ("x", "y", "z", "theta_x", "theta_y", "theta_z")]
        return Reading([_wrap(a) for a in d["joint_angles_deg"]], [float(v) for v in tool], str(path))
    joints, ee = None, None
    for sec in d:
        if sec.get("title") == "actuators":
            vals = [c["value"] for c in sec["data"] if "value" in c]
            joints = [_wrap(float(v.split()[0])) for v in vals[:6]]
        if sec.get("title") == "end effector":
            ee = {c["title"]: float(c["value"].split()[0]) for c in sec["data"]}
    if joints is None or ee is None:
        raise ValueError(f"{path}: neither a Web App export nor `kinova --json status` output")
    return Reading(joints, [ee[k] for k in ("x", "y", "z", "thetax", "thetay", "thetaz")], str(path))


def tool_rotation(theta_deg) -> np.ndarray:
    """Tool frame axes (columns) in the base frame, from Kortex fixed-axis XYZ Euler angles."""
    return Rotation.from_euler("xyz", theta_deg, degrees=True).as_matrix()


def tilt_from_down_deg(theta_x: float, theta_y: float) -> float:
    """Same formula as aims-kinova envelope.tilt_from_down_deg."""
    c = math.cos(math.radians(theta_x)) * math.cos(math.radians(theta_y))
    return math.degrees(math.acos(max(-1.0, min(1.0, -c))))


def pinch_point(tool) -> np.ndarray:
    """Base-frame position of the point between the finger pads, for a Kortex tool pose."""
    return np.asarray(tool[:3], float) + TOOL_TO_PINCH_M * tool_rotation(tool[3:])[:, 2]


@dataclass
class Placement:
    knob: dict                    # axis -> knob centre at grasp height (pinch point), base frame (m)
    spacing_mm: float
    knob_line_deg: float          # direction from the x knob to the y knob, about base z
    tool_yaw_deg: float           # mean measured theta_z: the finger orientation that fitted both knobs
    tilt_deg: dict                # axis -> tilt of the reading (not used for commands, which are vertical)

    @property
    def toy_yaw_offset_deg(self) -> float:
        """0 when the knob line runs along base -y (the simulation's layout, knob edge facing the arm)."""
        return _wrap(self.knob_line_deg + 90.0)


def placement(x: Reading, y: Reading) -> Placement:
    kx, ky = pinch_point(x.tool), pinch_point(y.tool)
    v = ky - kx
    return Placement({"x": kx, "y": ky}, float(np.hypot(v[0], v[1]) * 1000),
                     float(math.degrees(math.atan2(v[1], v[0]))),
                     float(np.degrees(np.angle(np.mean(np.exp(1j * np.radians([x.tool[5], y.tool[5]])))))),
                     {"x": tilt_from_down_deg(*x.tool[3:5]), "y": tilt_from_down_deg(*y.tool[3:5])})


def knob_poses(p: Placement, lift_m: float = 0.05, tool_yaw_deg: float | None = None) -> dict:
    """Approach ('above') and grasp poses per knob as [x, y, z, theta_x, theta_y, theta_z]: tool pointing
    straight down, pinch point on the knob centre at the measured grasp height."""
    yaw = p.tool_yaw_deg if tool_yaw_deg is None else tool_yaw_deg
    out = {}
    for ax, c in p.knob.items():
        grasp = [float(c[0]), float(c[1]), float(c[2] + TOOL_TO_PINCH_M), 180.0, 0.0, float(yaw)]
        above = grasp[:2] + [grasp[2] + lift_m] + grasp[3:]
        out[ax] = {"above": [round(v, 4) for v in above], "grasp": [round(v, 4) for v in grasp]}
    return out


def check_pose(pose, envelope: dict | None = None, current=None) -> list[str]:
    """Errors (empty = accepted) for the arm repo's safety limits and, if given, an envelope dict."""
    x, y, z, tx, ty = pose[:5]
    errs = []
    r, rho = math.sqrt(x * x + y * y + z * z), math.hypot(x, y)
    if r > R_MAX:
        errs.append(f"|p|={r:.3f} m > R_MAX={R_MAX}")
    if z < Z_MIN:
        errs.append(f"z={z:.3f} m < Z_MIN={Z_MIN}")
    if rho < RHO_MIN:
        errs.append(f"rho={rho:.3f} m < RHO_MIN={RHO_MIN}")
    if current is not None and math.dist(pose[:3], current[:3]) > MAX_STEP:
        errs.append(f"step {math.dist(pose[:3], current[:3]):.3f} m > MAX_STEP={MAX_STEP}")
    if envelope:
        for axis, v, lo, hi in zip("xyz", (x, y, z), envelope["box_min"], envelope["box_max"]):
            if not lo <= v <= hi:
                errs.append(f"envelope: {axis}={v:.3f} outside [{lo:.3f}, {hi:.3f}]")
        tilt = tilt_from_down_deg(tx, ty)
        if tilt > envelope.get("max_tilt_deg", 5.0):
            errs.append(f"envelope: tilt {tilt:.1f} deg > {envelope.get('max_tilt_deg', 5.0)}")
        if current is not None and math.dist(pose[:3], current[:3]) > envelope.get("max_step_m", MAX_STEP):
            errs.append(f"envelope: step {math.dist(pose[:3], current[:3]):.3f} m > {envelope.get('max_step_m')}")
    return errs


def propose_envelope(poses: dict, margin_m: float = 0.03, floor_margin_m: float = 0.015,
                     base: dict | None = None) -> dict:
    """Smallest box around all approach and grasp poses, plus margins (sideways and up: margin_m; below
    the lowest grasp: floor_margin_m). Other fields are kept from `base` (the current envelope file)."""
    pts = np.array([p[k][:3] for p in poses.values() for k in ("above", "grasp")])
    lo, hi = pts.min(0) - margin_m, pts.max(0) + margin_m
    lo[2] = pts[:, 2].min() - floor_margin_m
    env = dict(base or {})
    env.update(box_min=[round(max(float(v), Z_MIN) if i == 2 else float(v), 3) for i, v in enumerate(lo)],
               box_max=[round(float(v), 3) for v in hi])
    return env


def ik_check(pose) -> tuple[list[float] | None, list[str]]:
    """Joint angles (deg) our Gen3 model needs for a vertical tool pose, and any joint-limit errors.
    Joint 6 sets the finger orientation and is continuous, so only joints 2, 3 and 5 can fail."""
    from ..mj import ik
    from ..mj.scene import EtchScene
    m = EtchScene().model
    target = pinch_point(pose)
    q, pe, ae = ik.solve(m, target, z_axis=(0.0, 0.0, -1.0))
    if q is None:
        return None, [f"IK did not converge (best {pe:.1f} mm, {ae:.1f} deg)"]
    deg = [math.degrees(a) for a in q]
    errs = [f"joint {i + 1} = {a:.1f} deg outside {lim}" for i, (a, lim) in enumerate(zip(deg, JOINT_LIMITS_DEG))
            if lim is not None and not lim[0] <= a <= lim[1]]
    return [round(_wrap(a), 1) for a in deg], errs
