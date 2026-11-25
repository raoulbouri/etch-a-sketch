"""Wrist-camera view planner: views are safe, see the target, are diverse, and moves are short (no arm)."""
import numpy as np
from scipy.spatial.transform import Rotation

from etch_sim.real import view_plan as V
from etch_sim.real import wrist_camera as W

CHART = W.Pose(Rotation.from_euler("z", 20, degrees=True), np.array([0.30, 0.17, -0.0018]))


def chart_points():
    ch = W.Chart()
    w, h = ch.squares[0] * ch.square_m, ch.squares[1] * ch.square_m
    return CHART.apply(np.vstack([ch.board().getChessboardCorners(), [[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]]]))


def test_planned_views_are_safe_diverse_and_see_the_chart():
    lim = V.ViewLimits()
    pts = chart_points()
    start = W.Pose(Rotation.from_euler("xyz", [152.6, 10.9, 137.5], degrees=True), np.array([0.12, 0.06, 0.30]))
    views = V.order(V.select(V.candidates(pts, lim), 20, start), start)
    assert len(views) == 20
    for v in views:
        x, y, z = v.tool.t
        assert z >= lim.tool_z_min_m and np.hypot(x, y) >= lim.rho_min_m and np.linalg.norm(v.tool.t) <= lim.r_max_m
        pc = v.cam.inv().apply(pts)                                # chart in front of the camera
        assert (pc[:, 2] > 0).all()
        assert np.allclose(V.tool_from_camera(v.cam).t, v.tool.t)
    tilts = [V.tilt_deg(v.tool) for v in views]
    assert min(tilts) < 10 and max(tilts) > 25                      # straight-down and oblique views both present
    dists = [np.linalg.norm(v.cam.t - pts.mean(0)) for v in views]
    assert max(dists) - min(dists) > 0.1
    moves = V.with_waypoints(views, start, lim.max_step_m)
    prev = start
    for tool, _ in moves:
        assert np.linalg.norm(tool.t - prev.t) <= lim.max_step_m + 1e-9
        prev = tool
    assert sum(cap for _, cap in moves) == 20


def test_kortex_pose_round_trip_and_envelope():
    cam = W.Pose(V.look_at(np.array([0.3, 0.2, 0.4]), np.array([0.3, 0.17, 0.0]), 30.0), np.array([0.3, 0.2, 0.4]))
    tool = V.tool_from_camera(cam)
    p = V.kortex_pose(tool)
    back = Rotation.from_euler("xyz", p[3:], degrees=True)
    assert (back.inv() * tool.R).magnitude() < 1e-3
    env = V.envelope_for([tool], "t", "")
    assert all(lo < v < hi for v, lo, hi in zip(tool.t, env["box_min"], env["box_max"]))
    assert env["max_tilt_deg"] >= V.tilt_deg(tool)
