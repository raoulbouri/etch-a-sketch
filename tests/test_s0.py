import numpy as np
import pytest

from etch_sim import (AxisParams, Calib, Faults, KernelBackend, OpenLoopDrawer, ToyKernel, ToyParams,
                      chamfer, fit_axis, polyline, square, sweep_axis)


def hard_toy(half_dead=10.0, k=0.085):
    ax = dict(k_mm_per_deg=k, half_dead_deg=half_dead, transition_deg=0.0)
    return ToyKernel(ToyParams(x=AxisParams(travel_mm=153.0, **ax), y=AxisParams(travel_mm=107.0, **ax)))


def test_dead_zone_then_linear():
    toy = hard_toy()
    assert toy.turn("x", 10.0) == 0.0                       # starts mid-gap: first half_dead does nothing
    assert toy.turn("x", 100.0) == pytest.approx(8.5)
    assert toy.turn("x", -20.0) == pytest.approx(0.0)       # reversal: full 2*half_dead dead zone
    assert toy.turn("x", -100.0) == pytest.approx(-8.5)


def test_one_knob_moves_one_axis():
    toy = hard_toy()
    x0, y0 = toy.stylus
    toy.turn("y", 200.0)
    assert toy.stylus[0] == x0 and toy.stylus[1] > y0


def test_end_stop_strain_and_snap():
    toy = hard_toy()
    toy.turn("x", 2000.0)                                   # far past the right edge
    assert toy.stylus[0] == pytest.approx(153.0)
    assert 0 < toy.strain["x"] and toy.snapped["x"]
    toy2 = hard_toy()
    toy2.turn("x", 1000.0)                                  # ~80 deg into the stop: strained, not snapped
    assert not toy2.snapped["x"] and toy2.strain["x"] > 0
    assert toy2.turn("x", -toy2.strain["x"]) == 0.0         # unwinding the strain moves nothing


def test_fit_recovers_hidden_parameters():
    toy = ToyKernel(ToyParams(x=AxisParams(k_mm_per_deg=0.091, half_dead_deg=7.0, transition_deg=3.0)))
    robot = KernelBackend(toy, faults=Faults(pen_noise_mm=0.1), seed=1)
    theta, pen = sweep_axis(robot, "x")
    fit = fit_axis(theta, pen)
    assert fit.k_mm_per_deg == pytest.approx(0.091, rel=0.01)
    assert fit.delta_deg == pytest.approx(10.0, abs=1.0)
    assert fit.rms_mm < 0.3


def test_chamfer():
    sq = polyline(square(40.0), (10, 10))
    assert chamfer(sq, sq)["chamfer_mm"] == pytest.approx(0.0, abs=1e-9)
    assert chamfer(sq + [2.0, 0.0], sq)["chamfer_mm"] == pytest.approx(1.0, abs=0.3)


def test_compensated_square_closes():
    toy = hard_toy()
    robot = KernelBackend(toy, faults=Faults(pen_noise_mm=0.0))
    drawer = OpenLoopDrawer(robot, Calib({"x": 0.085, "y": 0.085}, {"x": 10.0, "y": 10.0}))
    drawer.preload()
    mark, start = toy.mark(), toy.stylus
    drawer.draw(square(40.0))
    path = toy.path_since(mark)
    assert np.linalg.norm(path[-1] - start) < 0.05
    assert chamfer(path, polyline(square(40.0), start))["chamfer_mm"] < 0.05
    assert robot.n_switches == 5                            # x->y in preload, y->x to start, then 3 corners


def test_erase_and_fault_truth_log():
    toy = hard_toy()
    robot = KernelBackend(toy, faults=Faults(slip_frac={"x": 0.05, "y": 0.0}))
    robot.grasp_knob("x")
    robot.rotate(500.0)
    assert any(e["fault"] == "slip" for e in robot.truth_log)
    assert all("fault" not in e for e in robot.log)
    pen = toy.stylus
    res = robot.erase(peak_accel_ms2=9.0, duration_s=60.0)
    assert res.ink_remaining < 0.05 and toy.stylus == pen   # erasing does not move the stylus
    weak = toy.shake(peak_accel_ms2=9.0, duration_s=60.0, inverted=False)
    assert weak == 0.0 or weak == pytest.approx(1.0)
