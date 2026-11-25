"""The toy's board model: where its 4 ArUco markers sit, and the taught knob grasps, in a frame fixed to the toy.

Board frame (metres): origin at the centroid of the 4 marker centres, x from marker 3 (bottom-left, next to the left
knob) towards marker 2 (bottom-right, next to the right knob), z up out of the screen. The markers are stuck to the
toy, so this model never changes when the toy is moved or shaken; only the board's pose in the arm's base frame does.

Knob grasps are stored as tool poses RELATIVE to the board frame, taken from hand-guided grasps (gripper closed on the
knob). Replaying one is base_from_board * board_from_tool: no gripper pad offset is needed, because the taught grasp
already contains it.

All fitting uses the wrist camera calibration (wrist_camera.load_calibration) and the arm's reported poses. Read-only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .wrist_camera import Pose, flange_pose, tool_pose

MARKER_IDS = (0, 1, 2, 3)            # seen from the knob side: 0 top-left, 1 top-right, 2 bottom-right, 3 bottom-left
MARKER_M = 0.0216                    # printed black square (ruler, 2026-11-24)
DICTIONARY = cv2.aruco.DICT_4X4_50
MODEL_FILE = Path(__file__).resolve().parents[3] / "hardware" / "board_model.json"
_H = MARKER_M / 2
LOCAL = np.array([[-_H, _H, 0], [_H, _H, 0], [_H, -_H, 0], [-_H, -_H, 0]])    # ArUco corner order, marker y up


def detect_markers(gray: np.ndarray) -> dict[int, np.ndarray]:
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    c, ids, _ = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICTIONARY), p).detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): q.reshape(4, 2).astype(float) for i, q in zip(ids.ravel(), c) if int(i) in MARKER_IDS}


def flat_pose(x: float, y: float, z: float, yaw: float) -> Pose:
    return Pose(Rotation.from_euler("z", yaw), np.array([x, y, z]))


@dataclass
class MarkerFit:
    markers: dict                     # id -> (x, y, yaw) in the base frame (m, rad), all at height z, lying flat
    z: float
    rms_px: float
    per_view_rms: list
    n_views: int


def fit_markers(views, K, D, mount: Pose, z_fixed: float | None = None) -> MarkerFit:
    """Each marker's x, y, yaw in the base frame from several views [(markers {id: 4x2 px}, arm status), ...].

    The markers lie flat on one plane (the toy on the table): one common height z, fitted unless z_fixed is given."""
    seen = sorted({i for m, _ in views for i in m})
    flanges = [flange_pose(s) for _, s in views]
    x0 = []
    for i in seen:                                    # start: PnP of the marker in the first view that has it
        m, _ = next((m, s) for (m, s) in views if i in m)
        f = flanges[next(k for k, (mm, _) in enumerate(views) if i in mm)]
        _, rv, tv = cv2.solvePnP(LOCAL, m[i], K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        p = f * mount * Pose(Rotation.from_rotvec(rv.ravel()), tv.ravel())
        x0 += [p.t[0], p.t[1], p.R.as_euler("zyx")[0]]
    z0 = z_fixed if z_fixed is not None else 0.0
    if z_fixed is None:                               # starting height: median PnP height
        zs = []
        for (m, _), f in zip(views, flanges):
            for i, q in m.items():
                _, rv, tv = cv2.solvePnP(LOCAL, q, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                zs.append((f * mount * Pose(Rotation.from_rotvec(rv.ravel()), tv.ravel())).t[2])
        z0 = float(np.median(zs))

    def unpack(x):
        mk = {i: tuple(x[3 * k:3 * k + 3]) for k, i in enumerate(seen)}
        return mk, (z_fixed if z_fixed is not None else x[-1])

    def residuals(x):
        mk, z = unpack(x)
        out = []
        for (m, _), f in zip(views, flanges):
            cam_inv = (f * mount).inv()
            for i, q in m.items():
                P = cam_inv * flat_pose(*mk[i][:2], z, mk[i][2])
                pr, _ = cv2.projectPoints(LOCAL, P.R.as_rotvec(), P.t, K, D)
                out.append((pr.reshape(-1, 2) - q).ravel())
        return np.concatenate(out)

    x_init = np.r_[x0, [] if z_fixed is not None else [z0]]
    sol = least_squares(residuals, x_init, x_scale="jac", loss="soft_l1", f_scale=2.0)
    mk, z = unpack(sol.x)
    r = sol.fun.reshape(-1, 2)
    per, k = [], 0
    for m, _ in views:
        n = 4 * len(m)
        per.append(float(np.sqrt(np.mean(np.sum(r[k:k + n] ** 2, 1)))) if n else float("nan"))
        k += n
    return MarkerFit(mk, float(z), float(np.sqrt(np.mean(np.sum(r ** 2, 1)))), per, len(views))


@dataclass
class BoardFit:
    board: Pose                       # board frame in the base frame (flat, at the taught height)
    rms_px: float
    per_view_rms: list
    n_views: int
    n_markers: int                    # marker sightings used (sum over views)


def fit_board(views, layout_mm_deg: dict, z: float, K, D, mount: Pose, guess: Pose | None = None) -> BoardFit:
    """Board x, y, yaw from several views, with the marker layout known (board_model.json) and the height fixed
    (the toy lies on the table, so it is the taught height). views: [(markers {id: 4x2 px}, arm status), ...]."""
    flanges = [flange_pose(s) for _, s in views]
    local = {i: flat_pose(v[0] / 1000, v[1] / 1000, 0.0, np.radians(v[2])) for i, v in layout_mm_deg.items()}
    if guess is None:                                 # PnP of the first visible marker gives the start
        m, _ = views[0]
        i = next(iter(m))
        _, rv, tv = cv2.solvePnP(LOCAL, m[i], K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        mp = flanges[0] * mount * Pose(Rotation.from_rotvec(rv.ravel()), tv.ravel())
        yaw = mp.R.as_euler("zyx")[0] - local[i].R.as_euler("zyx")[0]
        b = flat_pose(mp.t[0], mp.t[1], z, yaw)
        b = flat_pose(*(mp.t[:2] - b.R.apply(local[i].t)[:2]), z, yaw)
        guess = b
    x0 = np.r_[guess.t[:2], guess.R.as_euler("zyx")[0]]

    def residuals(x):
        b = flat_pose(x[0], x[1], z, x[2])
        out = []
        for (m, _), f in zip(views, flanges):
            cam_inv = (f * mount).inv()
            for i, q in m.items():
                if i not in local:
                    continue
                P = cam_inv * b * local[i]
                pr, _ = cv2.projectPoints(LOCAL, P.R.as_rotvec(), P.t, K, D)
                out.append((pr.reshape(-1, 2) - q).ravel())
        return np.concatenate(out)

    sol = least_squares(residuals, x0, x_scale="jac", loss="soft_l1", f_scale=2.0)
    r = sol.fun.reshape(-1, 2)
    per, k = [], 0
    for m, _ in views:
        n = 4 * len([i for i in m if i in local])
        per.append(float(np.sqrt(np.mean(np.sum(r[k:k + n] ** 2, 1)))) if n else float("nan"))
        k += n
    return BoardFit(flat_pose(sol.x[0], sol.x[1], z, sol.x[2]), float(np.sqrt(np.mean(np.sum(r ** 2, 1)))), per,
                    len(views), sum(len([i for i in m if i in local]) for m, _ in views))


def board_frame(markers: dict, z: float) -> Pose:
    """Board frame in the base frame from the 4 marker centres (needs markers 2 and 3 for the x direction)."""
    c = np.array([[markers[i][0], markers[i][1]] for i in MARKER_IDS])
    ex = c[MARKER_IDS.index(2)] - c[MARKER_IDS.index(3)]
    return flat_pose(*c.mean(0), z, float(np.arctan2(ex[1], ex[0])))


def layout_in_board(markers: dict, board: Pose) -> dict:
    """Marker centres and yaws in the board frame: {id: [x_mm, y_mm, yaw_deg]}."""
    yaw_b = board.R.as_euler("zyx")[0]
    out = {}
    for i, (x, y, yaw) in markers.items():
        p = board.inv().apply([x, y, board.t[2]])
        out[i] = [round(float(p[0]) * 1000, 2), round(float(p[1]) * 1000, 2),
                  round(float(np.degrees((yaw - yaw_b + np.pi) % (2 * np.pi) - np.pi)), 2)]
    return out


def relative_grasp(board: Pose, status: dict) -> Pose:
    """The taught tool pose expressed in the board frame."""
    return board.inv() * tool_pose(status)


def grasp_in_base(board: Pose, rel: Pose) -> list[float]:
    """Kortex tool pose [x, y, z, theta_x, theta_y, theta_z] for a stored relative grasp on a re-located board."""
    t = board * rel
    tx, ty, tz = t.R.as_euler("xyz", degrees=True)
    return [round(float(v), 4) for v in (*t.t, tx, ty, tz)]


def save_model(path: Path, layout: dict, z_board: float, grasps: dict, meta: dict) -> None:
    data = {"frame": "origin at the centroid of the 4 marker centres; x from marker 3 to marker 2; z up",
            "marker_m": MARKER_M, "dictionary": "DICT_4X4_50", "markers_mm_deg": {str(k): v for k, v in layout.items()},
            "marker_surface_z_m_when_taught": z_board,
            "knob_grasps_in_board": {ax: {"t_m": g.t.round(5).tolist(), "rotvec": g.R.as_rotvec().round(6).tolist()}
                                     for ax, g in grasps.items()}, **meta}
    path.write_text(json.dumps(data, indent=1) + "\n")


def mean_grasp(samples: list[dict]) -> Pose:
    """Average of grasp samples ({"t_m", "rotvec"} in the board frame): mean position, rotation mean."""
    t = np.mean([s["t_m"] for s in samples], 0)
    R = Rotation.from_rotvec([s["rotvec"] for s in samples]).mean()
    return Pose(R, t)


def _pose_json(p: Pose) -> dict:
    return {"t_m": np.round(p.t, 5).tolist(), "rotvec": np.round(p.R.as_rotvec(), 6).tolist()}


def load_model(path: Path = MODEL_FILE) -> dict:
    """The model plus d["markers"] {id: [x_mm, y_mm, yaw_deg]} and d["grasps"] {axis: Pose in the board frame}.

    A knob's grasp is the mean of its samples (the taught grasp plus every hand correction added with add_sample),
    raised by d["grip_raise_mm"] along the board's z."""
    d = json.loads(path.read_text())
    d["markers"] = {int(k): v for k, v in d["markers_mm_deg"].items()}
    raise_m = float(d.get("grip_raise_mm", 0.0)) / 1000
    d["grasps"] = {}
    for ax, g in d["knob_grasps_in_board"].items():
        samples = d.get("grasp_samples", {}).get(ax) or [g]
        m = mean_grasp(samples)
        d["grasps"][ax] = Pose(m.R, m.t + [0.0, 0.0, raise_m])
    return d


def add_sample(path: Path, axis: str, rel: Pose, source: str) -> dict:
    """Append a hand-corrected grasp (in the board frame) to the model's samples for this knob and save."""
    d = json.loads(path.read_text())
    samples = d.setdefault("grasp_samples", {})
    if axis not in samples:                            # start from the taught grasp
        samples[axis] = [dict(d["knob_grasps_in_board"][axis], source="teach")]
    samples[axis].append(dict(_pose_json(rel), source=source))
    d.setdefault("grip_raise_mm", 0.0)
    path.write_text(json.dumps(d, indent=1) + "\n")
    return d
