"""Board model: marker layout from wrist views, board frame, and replaying taught knob grasps (no arm, no camera)."""
import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from etch_sim.real import board_model as B
from etch_sim.real import view_plan as V
from etch_sim.real import wrist_camera as W

K = np.array([[1290.0, 0, 650.0], [0, 1290.0, 350.0], [0, 0, 1]])
D = np.zeros(5)
LAYOUT = {0: (-0.0938, 0.0738, 0.02), 1: (0.0937, 0.0740, -0.01), 2: (0.0640, -0.0738, 0.0), 3: (-0.0640, -0.0740, 0.03)}


def scene(board: W.Pose, z=0.010):
    return {i: B.flat_pose(*board.apply([x, y, 0])[:2], z, board.R.as_euler("zyx")[0] + a) for i, (x, y, a) in LAYOUT.items()}


def views_of(markers, n=4, seed=0):
    rng = np.random.default_rng(seed)
    c = np.mean([p.t for p in markers.values()], 0)
    out = []
    for k in range(n):
        az, tilt = 2 * np.pi * k / n, np.radians(rng.uniform(8, 25))
        p = c + 0.40 * np.array([np.sin(tilt) * np.cos(az), np.sin(tilt) * np.sin(az), np.cos(tilt)])
        cam = W.Pose(V.look_at(p, c, rng.uniform(-30, 30)), p)
        tool = V.tool_from_camera(cam)
        tx, ty, tz = tool.R.as_euler("xyz", degrees=True)
        status = {"tool_pose": dict(zip(("x", "y", "z", "theta_x", "theta_y", "theta_z"), (*tool.t, tx, ty, tz)))}
        seen = {}
        for i, mp in markers.items():
            P = cam.inv() * mp
            uv, _ = cv2.projectPoints(B.LOCAL, P.R.as_rotvec(), P.t, K, D)
            seen[i] = uv.reshape(-1, 2) + rng.normal(0, 0.2, (4, 2))
        out.append((seen, status))
    return out


def test_layout_and_board_frame_are_recovered():
    board = B.flat_pose(0.30, 0.15, 0.010, np.radians(25))
    fit = B.fit_markers(views_of(scene(board)), K, D, W.MOUNT_MODEL)
    assert fit.rms_px < 0.5 and abs(fit.z - 0.010) < 1e-3
    est = B.board_frame(fit.markers, fit.z)
    lay = B.layout_in_board(fit.markers, est)
    cen = np.mean([v[:2] for v in LAYOUT.values()], 0)
    for i, (x, y, a) in LAYOUT.items():                       # same layout, up to the centroid convention
        assert np.hypot(lay[i][0] / 1000 - (x - cen[0]), lay[i][1] / 1000 - (y - cen[1])) < 1e-3


def test_fit_board_recovers_a_moved_board_from_the_known_layout():
    taught = B.flat_pose(0.30, 0.15, 0.010, np.radians(25))
    fit = B.fit_markers(views_of(scene(taught)), K, D, W.MOUNT_MODEL)
    est = B.board_frame(fit.markers, fit.z)
    layout = B.layout_in_board(fit.markers, est)
    moved = B.flat_pose(0.27, 0.19, 0.010, np.radians(5))                # shifted 5 cm and turned 20 deg
    new_views = views_of(scene(moved), n=5, seed=3)
    true_board = B.board_frame({i: (p.t[0], p.t[1], p.R.as_euler("zyx")[0]) for i, p in scene(moved).items()}, 0.010)
    bf = B.fit_board(new_views, layout, 0.010, K, D, W.MOUNT_MODEL)
    assert bf.rms_px < 0.5
    assert np.linalg.norm(bf.board.t[:2] - true_board.t[:2]) < 5e-4
    assert abs(np.degrees((true_board.R.inv() * bf.board.R).magnitude())) < 0.1
    # a view that sees only two markers still contributes
    two = [({i: q for i, q in m.items() if i in (2, 3)}, s) for m, s in new_views[:2]] + new_views[2:]
    assert np.linalg.norm(B.fit_board(two, layout, 0.010, K, D, W.MOUNT_MODEL).board.t[:2] - true_board.t[:2]) < 1e-3


def test_taught_grasp_follows_the_board():
    board1 = B.flat_pose(0.30, 0.15, 0.010, np.radians(25))
    board2 = B.flat_pose(0.26, 0.20, 0.010, np.radians(-10))      # the toy moved and turned
    taught = W.Pose(Rotation.from_euler("xyz", [180, 0, 111], degrees=True), board1.apply([-0.09, -0.07, 0.04]))
    tx, ty, tz = taught.R.as_euler("xyz", degrees=True)
    rel = B.relative_grasp(board1, {"tool_pose": dict(zip(("x", "y", "z", "theta_x", "theta_y", "theta_z"), (*taught.t, tx, ty, tz)))})
    new = B.grasp_in_base(board2, rel)
    assert np.allclose(new[:3], board2.apply([-0.09, -0.07, 0.04]), atol=1e-4)    # same spot on the toy
    assert abs(Rotation.from_euler("xyz", new[3:], degrees=True).apply([0, 0, 1])[2] + 1) < 1e-6   # still pointing down


def test_hand_corrections_average_into_the_model_and_grip_raise_lifts_it(tmp_path):
    import json
    f = tmp_path / "board_model.json"
    down = Rotation.from_euler("xyz", [180, 0, 100], degrees=True)
    taught = W.Pose(down, np.array([-0.0938, -0.0738, 0.035]))
    B.save_model(f, {0: [0, 0, 0], 1: [1, 0, 0], 2: [0, 1, 0], 3: [1, 1, 0]}, 0.02, {"x": taught, "y": taught}, {})
    B.add_sample(f, "x", W.Pose(down, taught.t + [0.0, 0.0, -0.0033]), "fix 1")     # hand grip 3.3 mm lower
    B.add_sample(f, "x", W.Pose(down, taught.t + [0.0, 0.0, +0.0009]), "fix 2")     # and 0.9 mm higher
    d = B.load_model(f)
    assert d["grasps"]["x"].t[2] == pytest.approx(0.035 + (-0.0033 + 0.0009) / 3, abs=1e-6)   # mean of 3 samples
    assert np.allclose(d["grasps"]["y"].t, taught.t)                                        # untouched knob
    data = json.loads(f.read_text())
    data["grip_raise_mm"] = 3.0
    f.write_text(json.dumps(data))
    assert B.load_model(f)["grasps"]["y"].t[2] == pytest.approx(0.038, abs=1e-6)
