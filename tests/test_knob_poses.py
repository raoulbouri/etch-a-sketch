"""Knob poses from hand-guided arm readings (scripts/knob_poses.py). No arm, no server."""
import json
import math

import numpy as np
import pytest

from etch_sim.real import knob_poses as K

READ = "hardware/arm_readings/20261122_blue"
PLACEHOLDER_ENV = {"box_min": [0.30, -0.16, 0.07], "box_max": [0.50, 0.16, 0.25], "max_tilt_deg": 5.0, "max_step_m": 0.25}


def reading_for(knob_xyz, theta):
    """The tool pose an arm would report with its pinch point on knob_xyz at Kortex angles theta."""
    z = K.tool_rotation(theta)[:, 2]
    return K.Reading([0.0] * 6, list(np.asarray(knob_xyz) - K.TOOL_TO_PINCH_M * z) + list(theta))


def test_parses_web_app_export_and_cli_status(tmp_path):
    r = K.parse_reading(f"{READ}/y_closed.json")
    assert r.tool == pytest.approx([0.408, -0.087, 0.056, 178.944, 3.007, 111.23])
    assert r.joints_deg[2] == pytest.approx(245.862 - 360)                      # wrapped to (-180, 180]
    p = tmp_path / "status.json"
    p.write_text(json.dumps({"joint_angles_deg": [-16.6, 5.9, -114.1, 2.1, -58.8, 93.3],
                             "tool_pose": {"x": 0.3164, "y": 0.0827, "z": 0.2522, "theta_x": 178.95,
                                           "theta_y": 3.07, "theta_z": 111.2}}))
    assert K.parse_reading(p).tool[2] == pytest.approx(0.2522)


def test_tilted_readings_give_the_knob_centres():
    kx, ky = np.array([0.34, 0.10, 0.018]), np.array([0.41, -0.085, 0.020])
    p = K.placement(reading_for(kx, (178.9, 4.4, 111.2)), reading_for(ky, (179.5, -3.0, 108.0)))
    assert np.allclose(p.knob["x"], kx, atol=1e-6) and np.allclose(p.knob["y"], ky, atol=1e-6)
    assert p.spacing_mm == pytest.approx(np.hypot(*(ky - kx)[:2]) * 1000)
    assert p.toy_yaw_offset_deg == pytest.approx(math.degrees(math.atan2(-0.185, 0.07)) + 90)


def test_poses_are_vertical_at_the_measured_height():
    p = K.placement(K.parse_reading(f"{READ}/x_closed.json"), K.parse_reading(f"{READ}/y_closed.json"))
    poses = K.knob_poses(p, lift_m=0.05)
    for ax in ("x", "y"):
        g, a = poses[ax]["grasp"], poses[ax]["above"]
        assert K.tilt_from_down_deg(*g[3:5]) == pytest.approx(0, abs=1e-6)
        assert np.allclose(K.pinch_point(g), p.knob[ax], atol=1e-4)            # fingers on the knob centre
        assert a[:2] == g[:2] and a[2] == pytest.approx(g[2] + 0.05)
    assert 190 < p.spacing_mm < 200                                             # 196.6 mm on 2026-11-22


def test_checks_catch_the_placeholder_envelope_and_the_proposal_fits():
    p = K.placement(K.parse_reading(f"{READ}/x_closed.json"), K.parse_reading(f"{READ}/y_closed.json"))
    poses = K.knob_poses(p)
    assert any("envelope: z=" in e for e in K.check_pose(poses["x"]["grasp"], PLACEHOLDER_ENV))
    prop = K.propose_envelope(poses, base=PLACEHOLDER_ENV)
    assert prop["max_tilt_deg"] == 5.0                                          # other fields kept
    for ax in poses:
        for kind in ("above", "grasp"):
            assert K.check_pose(poses[ax][kind], prop) == []
    assert K.check_pose(poses["x"]["above"], prop, current=poses["y"]["above"]) == []   # knob switch step fits
    assert K.check_pose([0.30, 0.0, 0.01, 180, 0, 90]) == ["z=0.010 m < Z_MIN=0.02"]
    assert any("rho" in e for e in K.check_pose([0.05, 0.05, 0.3, 180, 0, 90]))


def test_model_reaches_both_grasps_within_joint_limits():
    p = K.placement(K.parse_reading(f"{READ}/x_closed.json"), K.parse_reading(f"{READ}/y_closed.json"))
    for ax, poses in K.knob_poses(p).items():
        joints, errs = K.ik_check(poses["grasp"])
        assert joints is not None and errs == [], (ax, errs)
