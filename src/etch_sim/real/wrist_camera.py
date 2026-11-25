"""Kinova Gen3 wrist colour camera: frame chain, live frames, ChArUco chart, and calibration solvers.

Frame chain (metres; Kortex angles are fixed-axis XYZ Euler, degrees):

    base --(Kortex tool pose, live)--> tool --(-TOOL_Z along tool z)--> flange --(MOUNT)--> camera --(PnP)--> target

- Kortex reports the TOOL frame, TOOL_Z = 0.12 m beyond the flange (tool transform read from both arms, 2026-11-24).
- MOUNT: camera in the flange frame, from Kinova's gen3_macro.xacro (color_module: xyz 0 0.05639 -0.00305, rpy pi pi 0).
  Camera axes are OpenCV's (x right, y down, z forward); z points along the tool axis.
- Calibration replaces the Kortex intrinsics (cy is wrong on the green arm, see MEMORY.md) and refines MOUNT.

No arm motion here: frames come from the arm's RTSP stream and poses from `kinova --json status`.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

TOOL_Z = 0.12
MOUNT_T = np.array([0.0, 0.05639, -0.00305])
MOUNT_R = Rotation.from_euler("xyz", [np.pi, np.pi, 0.0])
IMAGE_SIZE = (1280, 720)
# Kortex focal length (checked against a ruler, 2026-11-24) with a centred principal point: a starting guess only.
K_GUESS = np.array([[1298.2, 0.0, 640.0], [0.0, 1298.2, 360.0], [0.0, 0.0, 1.0]])
KINOVA_DIR = Path(os.environ.get("KINOVA_DOCKER_DIR", Path.home() / "Developer" / "kinova-gen3-docker"))
CALIB_FILE = Path(__file__).resolve().parents[3] / "hardware" / "wrist_camera_calib.json"


# ---- frame chain -----------------------------------------------------------------------------------------------
@dataclass
class Pose:
    """Rigid transform: maps points in the child frame to the parent frame, p_parent = R p_child + t."""
    R: Rotation
    t: np.ndarray

    def __mul__(self, o: "Pose") -> "Pose":
        return Pose(self.R * o.R, self.R.apply(o.t) + self.t)

    def apply(self, p) -> np.ndarray:
        return self.R.apply(np.asarray(p, float)) + self.t

    def inv(self) -> "Pose":
        Ri = self.R.inv()
        return Pose(Ri, -Ri.apply(self.t))

    def vec(self) -> np.ndarray:
        return np.r_[self.R.as_rotvec(), self.t]

    @staticmethod
    def from_vec(v) -> "Pose":
        return Pose(Rotation.from_rotvec(v[:3]), np.asarray(v[3:6], float))


def tool_pose(status: dict) -> Pose:
    tp = status["tool_pose"]
    return Pose(Rotation.from_euler("xyz", [tp["theta_x"], tp["theta_y"], tp["theta_z"]], degrees=True),
                np.array([tp["x"], tp["y"], tp["z"]], float))


def flange_pose(status: dict) -> Pose:
    return tool_pose(status) * Pose(Rotation.identity(), np.array([0.0, 0.0, -TOOL_Z]))


MOUNT_MODEL = Pose(MOUNT_R, MOUNT_T)


def camera_pose(status: dict, mount: Pose = MOUNT_MODEL) -> Pose:
    """Camera in the base frame."""
    return flange_pose(status) * mount


# ---- arm status and live frames --------------------------------------------------------------------------------
def arm_status(kinova_dir: Path = KINOVA_DIR) -> dict:
    """`kinova --json status` from the aims-kinova container (read-only)."""
    out = subprocess.run(["docker", "compose", "exec", "-T", "kinova", "with-ros.sh", "kinova", "--json", "status"],
                         cwd=kinova_dir, capture_output=True, text=True, timeout=30)
    if out.returncode != 0:
        raise RuntimeError(f"kinova status failed: {out.stderr.strip()[-300:]}")
    return json.loads(out.stdout[out.stdout.index("{"):])


class RtspGrabber:
    """Keeps the newest frame of the arm's colour stream, so a capture is current and instant."""

    def __init__(self, ip: str):
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;10000000")
        self.cap = cv2.VideoCapture(f"rtsp://{ip}/color", cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open rtsp://{ip}/color")
        self._frame, self._t, self._lock, self._run = None, 0.0, threading.Lock(), True
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def _loop(self):
        while self._run:
            ok, f = self.cap.read()
            if ok:
                with self._lock:
                    self._frame, self._t = f, time.time()

    def frame_after(self, t0: float, timeout: float = 5.0) -> np.ndarray:
        """First frame received after time t0."""
        end = time.time() + timeout
        while time.time() < end:
            with self._lock:
                if self._frame is not None and self._t > t0:
                    return self._frame.copy()
            time.sleep(0.01)
        raise RuntimeError("no new frame from the wrist camera")

    def close(self):
        self._run = False
        self._th.join(timeout=2)
        self.cap.release()


# ---- ChArUco chart --------------------------------------------------------------------------------------------
@dataclass
class Chart:
    squares: tuple = (8, 5)
    square_m: float = 0.030
    marker_m: float = 0.0225          # fitted from the image (ratio 0.75); square size from Raoul, 2026-11-24
    dictionary: int = cv2.aruco.DICT_5X5_100

    def board(self):
        return cv2.aruco.CharucoBoard(self.squares, self.square_m, self.marker_m,
                                      cv2.aruco.getPredefinedDictionary(self.dictionary))


def detect_chart(gray: np.ndarray, chart: Chart = Chart(), min_corners: int = 8):
    """(object points N x 3 in the chart frame, image points N x 2, corner ids) or None."""
    b = chart.board()
    cc, cid, _, _ = cv2.aruco.CharucoDetector(b).detectBoard(gray)
    if cid is None or len(cid) < min_corners:
        return None
    ids = cid.ravel()
    return np.asarray(b.getChessboardCorners(), np.float64)[ids], cc.reshape(-1, 2).astype(np.float64), ids


# ---- solvers ---------------------------------------------------------------------------------------------------
@dataclass
class Intrinsics:
    K: np.ndarray
    dist: np.ndarray                  # k1 k2 p1 p2 k3
    rms_px: float = float("nan")
    per_view_rms: list = field(default_factory=list)
    rvecs: list = field(default_factory=list)
    tvecs: list = field(default_factory=list)


def calibrate_intrinsics(views, image_size=IMAGE_SIZE, K0=K_GUESS, fix_k3: bool = True) -> Intrinsics:
    """OpenCV calibration from ChArUco detections [(obj, img, ids), ...] of a planar chart."""
    obj = [v[0].astype(np.float32) for v in views]
    img = [v[1].astype(np.float32) for v in views]
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | (cv2.CALIB_FIX_K3 if fix_k3 else 0)
    rms, K, dist, rv, tv = cv2.calibrateCamera(obj, img, image_size, K0.copy(), np.zeros(5), flags=flags)
    per = []
    for o, i, r, t in zip(obj, img, rv, tv):
        p, _ = cv2.projectPoints(o, r, t, K, dist)
        per.append(float(np.sqrt(np.mean(np.sum((p.reshape(-1, 2) - i) ** 2, 1)))))
    return Intrinsics(K, dist.ravel(), float(rms), per, [r.ravel() for r in rv], [t.ravel() for t in tv])


@dataclass
class HandEye:
    mount: Pose                       # camera in the flange frame
    chart_in_base: Pose
    rms_px: float
    per_view_rms: list


def solve_hand_eye(views, statuses, intr: Intrinsics, mount0: Pose = MOUNT_MODEL) -> HandEye:
    """Camera mount on the flange and the fixed chart's pose in the base, by minimising reprojection error over
    all views (the chart does not move; the arm reports the flange pose of every view)."""
    flanges = [flange_pose(s) for s in statuses]
    # starting chart pose: first view through the model mount
    _, r0, t0 = cv2.solvePnP(views[0][0], views[0][1], intr.K, intr.dist)
    chart0 = flanges[0] * mount0 * Pose(Rotation.from_rotvec(r0.ravel()), t0.ravel())

    def residuals(x):
        m, c = Pose.from_vec(x[:6]), Pose.from_vec(x[6:])
        out = []
        for (o, i, _), f in zip(views, flanges):
            cam_from_chart = (f * m).inv() * c
            p, _ = cv2.projectPoints(o, cam_from_chart.R.as_rotvec(), cam_from_chart.t, intr.K, intr.dist)
            out.append((p.reshape(-1, 2) - i).ravel())
        return np.concatenate(out)

    sol = least_squares(residuals, np.r_[mount0.vec(), chart0.vec()], x_scale="jac")
    r = sol.fun.reshape(-1, 2)
    per, k = [], 0
    for o, _, _ in views:
        per.append(float(np.sqrt(np.mean(np.sum(r[k:k + len(o)] ** 2, 1)))))
        k += len(o)
    return HandEye(Pose.from_vec(sol.x[:6]), Pose.from_vec(sol.x[6:]), float(np.sqrt(np.mean(np.sum(r ** 2, 1)))), per)


@dataclass
class JointFit:
    K: np.ndarray
    dist: np.ndarray                  # k1 k2 p1 p2 k3 (k3 = 0)
    mount: Pose                       # camera in the flange frame
    chart_in_base: Pose
    chart_scale: float                # printed size / nominal size
    rms_px: float
    per_view_rms: list


def solve_joint(views, statuses, chart: Chart = Chart(), fit_scale: bool = True, fix_mount_rotation: bool = False) -> JointFit:
    """Intrinsics, mount translation, the fixed chart's pose and its printed scale in one least-squares fit over all
    views, using the arm's reported flange pose for every view.

    Why joint: a chart alone fixes the focal length only through its own flatness; a paper chart that is not flat
    biased the focal length by 5 % (1366 vs ~1290, 2026-11-24). The arm's poses give the metric scale instead.
    Caveat: a principal-point shift and a small camera pitch/yaw look almost the same in the images, so with weak view
    diversity the fit can trade cy for a few degrees of mount tilt (rays stay right; check both against the model).
    fix_mount_rotation holds the rotation at Kinova's model. A soft-L1 loss limits the pull of badly detected corners."""
    flanges = [flange_pose(s) for s in statuses]
    _, r0, t0 = cv2.solvePnP(views[0][0], views[0][1], K_GUESS, None)
    chart0 = flanges[0] * MOUNT_MODEL * Pose(Rotation.from_rotvec(r0.ravel()), t0.ravel())

    nm = 3 if fix_mount_rotation else 6

    def unpack(x):
        K = np.array([[x[0], 0, x[2]], [0, x[1], x[3]], [0, 0, 1.0]])
        m = Pose(MOUNT_MODEL.R, x[8:11]) if fix_mount_rotation else Pose.from_vec(x[8:14])
        return K, np.r_[x[4:8], 0.0], m, Pose.from_vec(x[8 + nm:14 + nm]), (x[14 + nm] if fit_scale else 1.0)

    def residuals(x):
        K, D, m, c, s = unpack(x)
        out = []
        for (o, i, _), f in zip(views, flanges):
            cc = (f * m).inv() * c
            p, _ = cv2.projectPoints(o * s, cc.R.as_rotvec(), cc.t, K, D)
            out.append((p.reshape(-1, 2) - i).ravel())
        return np.concatenate(out)

    x0 = np.r_[K_GUESS[0, 0], K_GUESS[1, 1], K_GUESS[0, 2], K_GUESS[1, 2], np.zeros(4),
               MOUNT_MODEL.t if fix_mount_rotation else MOUNT_MODEL.vec(), chart0.vec(),
               [1.0] if fit_scale else []]
    sol = least_squares(residuals, x0, x_scale="jac", loss="soft_l1", f_scale=2.0)
    K, D, m, c, s = unpack(sol.x)
    r = sol.fun.reshape(-1, 2)
    per, k = [], 0
    for o, _, _ in views:
        per.append(float(np.sqrt(np.mean(np.sum(r[k:k + len(o)] ** 2, 1)))))
        k += len(o)
    return JointFit(K, D, m, c, float(s), float(np.sqrt(np.mean(np.sum(r ** 2, 1)))), per)


def save_calibration(intr: Intrinsics, he: HandEye, meta: dict, path: Path = CALIB_FILE) -> None:
    data = {"image_size": list(IMAGE_SIZE), "K": intr.K.tolist(), "dist_k1k2p1p2k3": intr.dist.tolist(),
            "intrinsics_rms_px": intr.rms_px,
            "mount_in_flange": {"t_m": he.mount.t.tolist(), "rotvec": he.mount.R.as_rotvec().tolist()},
            "hand_eye_rms_px": he.rms_px, **meta}
    path.write_text(json.dumps(data, indent=1) + "\n")


def load_calibration(path: Path = CALIB_FILE) -> tuple[np.ndarray, np.ndarray, Pose]:
    d = json.loads(path.read_text())
    m = d["mount_in_flange"]
    return np.array(d["K"]), np.array(d["dist_k1k2p1p2k3"]), Pose(Rotation.from_rotvec(m["rotvec"]), np.array(m["t_m"]))
