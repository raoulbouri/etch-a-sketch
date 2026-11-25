"""Plan wrist-camera views of a flat target (ChArUco chart or the toy's markers) from one rough sighting.

Given the target's corner points in the base frame (from one frame: PnP + the arm's pose), generate camera poses on
a dome above it, keep those that are safe, reachable and see the whole target, pick a diverse subset, and order it
into short moves. Output: Kortex tool poses [x, y, z, theta_x, theta_y, theta_z] for the motion server's move_pose.
No arm motion here.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from . import knob_poses as KP
from .wrist_camera import IMAGE_SIZE, K_GUESS, MOUNT_MODEL, TOOL_Z, Pose


@dataclass
class ViewLimits:
    dist_m: tuple = (0.30, 0.55)          # camera to target centre
    n_dist: int = 4
    min_visible: float = 0.6              # fraction of the target points that must be in the image
    tilts_deg: tuple = (0, 10, 20, 30)    # camera axis from straight down
    azimuth_step_deg: float = 30.0
    rolls_deg: tuple = (-30.0, 0.0, 30.0)  # about the camera axis, relative to the reference x direction
    aim_offsets_m: tuple = ((0, 0), (0.06, 0), (-0.06, 0), (0, 0.045), (0, -0.045))   # target off-centre in the image
    margin_px: float = 30.0               # whole target inside the image by this much
    tool_z_min_m: float = 0.15            # tool frame height: keeps the fingers >= ~10 cm above the table
    rho_min_m: float = 0.25               # aims-kinova RHO_MIN is 0.15; near the base axis Kortex needs big joint swings
                                          # for small tool moves (a 2 cm move with a wrist turn timed out at 0.19 m, 2026-11-24)
    r_max_m: float = 0.75                 # aims-kinova R_MAX 0.80 - margin
    rho_max_m: float = 0.75               # tool distance from the base axis; Raoul 2026-11-24: 0.44 m is too far on the table
    max_step_m: float = 0.22              # per move (envelope max_step_m 0.25)
    max_turn_deg: float = 35.0            # per move: larger reorientations are split (each move must finish well within 30 s)


def look_at(cam_pos, aim, roll_deg, ref_x=None) -> Rotation:
    """OpenCV camera axes (z forward, x right, y down) looking at aim, rolled about z.

    ref_x: the camera x axis is the reference direction projected into the image plane (then rolled), so views all
    around the target keep the wrist near one yaw. Without it, x is horizontal, which turns the wrist with azimuth.
    """
    z = np.asarray(aim, float) - cam_pos
    z /= np.linalg.norm(z)
    x = None
    if ref_x is not None:
        x = np.asarray(ref_x, float) - np.dot(ref_x, z) * z
        x = x if np.linalg.norm(x) > 1e-3 else None
    if x is None:
        x = np.cross([0.0, 0.0, 1.0], z) if abs(z[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
    x /= np.linalg.norm(x)
    return Rotation.from_matrix(np.c_[x, np.cross(z, x), z]) * Rotation.from_euler("z", roll_deg, degrees=True)


def tool_from_camera(cam: Pose, mount: Pose = MOUNT_MODEL) -> Pose:
    return cam * mount.inv() * Pose(Rotation.identity(), np.array([0.0, 0.0, TOOL_Z]))


def kortex_pose(tool: Pose) -> list[float]:
    tx, ty, tz = tool.R.as_euler("xyz", degrees=True)
    return [round(float(v), 4) for v in (*tool.t, tx, ty, tz)]


def tilt_deg(tool: Pose) -> float:
    return float(np.degrees(np.arccos(np.clip(-tool.R.apply([0, 0, 1])[2], -1, 1))))


@dataclass
class View:
    tool: Pose
    cam: Pose
    feat: np.ndarray                      # for diversity: tilt, azimuth, roll, image centroid, distance
    img_pts: np.ndarray


def candidates(target_pts: np.ndarray, lim: ViewLimits = ViewLimits(), K=K_GUESS, ref_x=None) -> list[View]:
    c = target_pts.mean(0)
    span = target_pts - c
    e1 = span[np.argmax(np.linalg.norm(span, axis=1))]
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross([0, 0, 1.0], e1)
    out = []
    for d in np.linspace(*lim.dist_m, lim.n_dist):
        for t in lim.tilts_deg:
            for a in (np.arange(0, 360, lim.azimuth_step_deg) if t else [0.0]):
                dirn = np.array([np.sin(np.radians(t)) * np.cos(np.radians(a)),
                                 np.sin(np.radians(t)) * np.sin(np.radians(a)), np.cos(np.radians(t))])
                p = c + d * dirn
                for ox, oy in lim.aim_offsets_m:
                    aim = c + ox * e1 + oy * e2
                    for roll in lim.rolls_deg:
                        cam = Pose(look_at(p, aim, roll, ref_x), p)
                        tool = tool_from_camera(cam)
                        x, y, z = tool.t
                        rho = np.hypot(x, y)
                        if z < lim.tool_z_min_m or rho < lim.rho_min_m or rho > lim.rho_max_m or np.linalg.norm(tool.t) > lim.r_max_m:
                            continue
                        cf = cam.inv()
                        pc = cf.apply(target_pts)
                        if (pc[:, 2] < 0.05).any():
                            continue
                        uv = (K @ (pc / pc[:, 2:]).T).T[:, :2]
                        m = lim.margin_px
                        inside = (uv[:, 0] > m) & (uv[:, 0] < IMAGE_SIZE[0] - m) & (uv[:, 1] > m) & (uv[:, 1] < IMAGE_SIZE[1] - m)
                        if inside.mean() < lim.min_visible:
                            continue
                        cen = uv[inside].mean(0) / IMAGE_SIZE
                        f = np.r_[t / 30, np.sin(np.radians(t)) * np.cos(np.radians(a)), np.sin(np.radians(t)) * np.sin(np.radians(a)),
                                  np.cos(np.radians(roll)), np.sin(np.radians(roll)), 2 * cen, (d - lim.dist_m[0]) / 0.17]
                        out.append(View(tool, cam, f, uv))
    return out


def reachable(tool: Pose, model=None) -> bool:
    """MuJoCo Gen3 model reaches the gripper position and pointing direction within the joint limits (joint 6, which
    sets the roll, is continuous)."""
    pose = kortex_pose(tool)
    pinch = KP.pinch_point(pose)
    from ..mj import ik
    q, _, _ = ik.solve(model, pinch, z_axis=tuple(tool.R.apply([0, 0, 1])))
    if q is None:
        return False
    deg = np.degrees(q)
    return all(lim is None or lim[0] <= a <= lim[1] for a, lim in zip(deg, KP.JOINT_LIMITS_DEG))


def select(cands: list[View], n: int, start_tool: Pose, is_reachable=None) -> list[View]:
    """Farthest-point sampling in feature space, starting near the arm's current pose; unreachable picks are skipped."""
    if not cands:
        return []
    F = np.array([v.feat for v in cands])
    first = int(np.argmin([np.linalg.norm(v.tool.t - start_tool.t) for v in cands]))
    chosen, dmin, banned = [], np.full(len(cands), np.inf), set()
    nxt = first
    while len(chosen) < n:
        if is_reachable is None or is_reachable(cands[nxt].tool):
            chosen.append(nxt)
            dmin = np.minimum(dmin, np.linalg.norm(F - F[nxt], axis=1))
        else:
            banned.add(nxt)
        dmin_masked = dmin.copy()
        dmin_masked[list(banned) + chosen] = -1
        if dmin_masked.max() <= 0:
            break
        nxt = int(np.argmax(dmin_masked))
    return [cands[i] for i in chosen]


def order(views: list[View], start_tool: Pose) -> list[View]:
    """Nearest-neighbour tour from the current tool position (position plus a small orientation term)."""
    left, out, cur = list(views), [], start_tool
    while left:
        cost = [np.linalg.norm(v.tool.t - cur.t) + 0.2 * (cur.R.inv() * v.tool.R).magnitude() for v in left]
        cur = left.pop(int(np.argmin(cost)))
        out.append(cur)
        cur = cur.tool
    return out


def inside_envelope(tool: Pose, env: dict, margin_m: float = 0.005) -> bool:
    """True if the tool pose satisfies an aims-kinova envelope's box (with a small margin) and tilt limit."""
    lo, hi = np.array(env["box_min"]) + margin_m, np.array(env["box_max"]) - margin_m
    return bool(np.all(tool.t >= lo) and np.all(tool.t <= hi) and tilt_deg(tool) <= env.get("max_tilt_deg", 5.0) - 0.5)


def path_to(start: Pose, goal: Pose, max_step_m: float = 0.22, max_turn_deg: float = 35.0) -> list[Pose]:
    """Intermediate poses (then the goal) so no single move exceeds max_step_m or max_turn_deg."""
    from scipy.spatial.transform import Slerp
    d = np.linalg.norm(goal.t - start.t)
    turn = np.degrees((start.R.inv() * goal.R).magnitude())
    k = max(1, int(np.ceil(d / max_step_m)), int(np.ceil(turn / max_turn_deg)))
    sl = Slerp([0, 1], Rotation.concatenate([start.R, goal.R]))
    return [Pose(sl(s), start.t + s * (goal.t - start.t)) for s in np.arange(1, k + 1) / k]


def with_waypoints(views: list[View], start_tool: Pose, max_step_m: float, max_turn_deg: float = 35.0) -> list[tuple[Pose, bool]]:
    """(tool pose, capture?) list: moves longer than max_step_m or turning more than max_turn_deg get evenly spaced
    via poses (no capture)."""
    from scipy.spatial.transform import Slerp
    out, cur = [], start_tool
    for v in views:
        d = np.linalg.norm(v.tool.t - cur.t)
        turn = np.degrees((cur.R.inv() * v.tool.R).magnitude())
        k = max(int(np.ceil(d / max_step_m)), int(np.ceil(turn / max_turn_deg)))
        if k > 1:
            sl = Slerp([0, 1], Rotation.concatenate([cur.R, v.tool.R]))
            for s in np.arange(1, k) / k:
                out.append((Pose(sl(s), cur.t + s * (v.tool.t - cur.t)), False))
        out.append((v.tool, True))
        cur = v.tool
    return out


def envelope_for(tools: list[Pose], name: str, notes: str, margin_m: float = 0.04) -> dict:
    P = np.array([t.t for t in tools])
    lo, hi = P.min(0) - margin_m, P.max(0) + margin_m
    lo[2] = max(lo[2], 0.10)
    return {"name": name, "box_min": [round(float(v), 3) for v in lo], "box_max": [round(float(v), 3) for v in hi],
            "max_lin_speed": 0.05, "max_ang_speed": 30.0,
            "max_tilt_deg": float(np.ceil(max(tilt_deg(t) for t in tools)) + 3),
            "rotate_joint": 6, "max_rotation_deg": 1.0, "max_step_m": 0.25,
            "max_commands": len(tools) * 2 + 10, "max_duration_s": 900.0, "notes": notes}
