"""Wrist camera chain and calibration solvers on synthetic views (no arm, no camera)."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from etch_sim.real import wrist_camera as W

K_TRUE = np.array([[1290.0, 0, 645.0], [0, 1291.0, 356.0], [0, 0, 1]])
D_TRUE = np.array([-0.06, 0.03, 0.0005, -0.0004, 0.0])
MOUNT_TRUE = W.MOUNT_MODEL * W.Pose(Rotation.from_euler("xyz", [0.6, -0.8, 1.0], degrees=True), np.array([0.002, -0.0015, 0.001]))
CHART_TRUE = W.Pose(Rotation.from_euler("z", 25, degrees=True), np.array([0.28, 0.12, -0.0018]))   # flat on the table


def status_of(tool: W.Pose) -> dict:
    tx, ty, tz = tool.R.as_euler("xyz", degrees=True)
    x, y, z = tool.t
    return {"ip": "fake", "joint_angles_deg": [0.0] * 6,
            "tool_pose": {"x": x, "y": y, "z": z, "theta_x": tx, "theta_y": ty, "theta_z": tz}}


def look_at(cam_pos, target, roll_deg) -> Rotation:
    z = target - cam_pos
    z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1.0], z) if abs(z[2]) < 0.99 else np.array([1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return Rotation.from_matrix(np.c_[x, y, z]) * Rotation.from_euler("z", roll_deg, degrees=True)


def synthetic_views(n=16, noise_px=0.2, seed=0):
    rng = np.random.default_rng(seed)
    obj_all = np.asarray(W.Chart().board().getChessboardCorners(), float)
    centre = CHART_TRUE.apply(obj_all).mean(0)
    views, statuses = [], []
    import cv2
    while len(views) < n:
        d, tilt, az = rng.uniform(0.28, 0.45), rng.uniform(5, 35), rng.uniform(0, 360)
        dirn = np.array([np.sin(np.radians(tilt)) * np.cos(np.radians(az)), np.sin(np.radians(tilt)) * np.sin(np.radians(az)),
                         np.cos(np.radians(tilt))])
        aim = centre + rng.uniform(-0.05, 0.05, 3) * [1, 1, 0]
        cam = W.Pose(look_at(centre + d * dirn, aim, rng.uniform(-60, 60)), centre + d * dirn)
        flange = cam * MOUNT_TRUE.inv()
        tool = flange * W.Pose(Rotation.identity(), np.array([0, 0, W.TOOL_Z]))
        c_from_chart = cam.inv() * CHART_TRUE
        p, _ = cv2.projectPoints(obj_all, c_from_chart.R.as_rotvec(), c_from_chart.t, K_TRUE, D_TRUE)
        p = p.reshape(-1, 2) + rng.normal(0, noise_px, (len(obj_all), 2))
        ok = (p[:, 0] > 5) & (p[:, 0] < 1275) & (p[:, 1] > 5) & (p[:, 1] < 715)
        if ok.sum() < 12:
            continue
        views.append((obj_all[ok], p[ok], np.flatnonzero(ok)))
        statuses.append(status_of(tool))
    return views, statuses


def test_chain_round_trip():
    st = status_of(W.Pose(Rotation.from_euler("xyz", [152.6, 10.9, 137.5], degrees=True), np.array([0.12, 0.06, 0.30])))
    cam = W.camera_pose(st)
    back = cam * W.MOUNT_MODEL.inv() * W.Pose(Rotation.identity(), np.array([0, 0, W.TOOL_Z]))
    assert np.allclose(back.t, W.tool_pose(st).t, atol=1e-12)
    assert np.allclose(cam.R.apply([0, 0, 1]), W.tool_pose(st).R.apply([0, 0, 1]))     # camera looks along the tool axis


def test_calibration_recovers_intrinsics_mount_and_table():
    views, statuses = synthetic_views()
    intr = W.calibrate_intrinsics(views, fix_k3=True)
    assert intr.rms_px < 0.4
    assert intr.K[0, 0] == pytest.approx(K_TRUE[0, 0], abs=4) and intr.K[1, 2] == pytest.approx(K_TRUE[1, 2], abs=4)
    he = W.solve_hand_eye(views, statuses, intr)
    assert he.rms_px < 0.4
    assert np.linalg.norm(he.mount.t - MOUNT_TRUE.t) < 1.5e-3
    assert np.degrees((MOUNT_TRUE.R.inv() * he.mount.R).magnitude()) < 0.3
    corners = he.chart_in_base.apply(W.Chart().board().getChessboardCorners())
    assert corners[:, 2].mean() == pytest.approx(-0.0018, abs=1e-3)


def test_joint_fit_recovers_scale_intrinsics_and_mount_translation():
    views, statuses = synthetic_views()
    s_true = 0.987                                           # chart printed 1.3 % small
    views = [(o / s_true, i, ids) for o, i, ids in views]   # nominal-size object points, true-size images
    fit = W.solve_joint(views, statuses)
    assert fit.chart_scale == pytest.approx(s_true, abs=0.002)
    assert fit.K[0, 0] == pytest.approx(K_TRUE[0, 0], abs=8)
    assert np.linalg.norm(fit.mount.t - MOUNT_TRUE.t) < 2e-3
    assert np.degrees((MOUNT_TRUE.R.inv() * fit.mount.R).magnitude()) < 0.5
    assert fit.rms_px < 0.4


def test_calibration_round_trips_through_the_file(tmp_path):
    views, statuses = synthetic_views(n=10, seed=1)
    intr = W.calibrate_intrinsics(views)
    he = W.solve_hand_eye(views, statuses, intr)
    W.save_calibration(intr, he, {"session": "synthetic"}, tmp_path / "c.json")
    K, d, m = W.load_calibration(tmp_path / "c.json")
    assert np.allclose(K, intr.K) and np.allclose(m.t, he.mount.t)
