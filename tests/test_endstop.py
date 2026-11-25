"""End-stop guard on the toy kernel, which winds string strain into an end stop and snaps it after snap_strain_deg."""
import pytest

from etch_sim.draw import Calib, OpenLoopDrawer, square
from etch_sim.endstop import EndStopRisk, GuardedDrawer, GuardParams, TravelGuard
from etch_sim.kernel import ToyKernel, ToyParams
from etch_sim.kernel_backend import KernelBackend


def rig(start=(12.0, 50.0), k_err=0.03):
    toy = ToyKernel(ToyParams(), start_mm=start)
    p = toy.p
    calib = Calib({a: p.axis(a).k_mm_per_deg * (1 + k_err) for a in "xy"}, {a: p.axis(a).delta_deg for a in "xy"})
    return toy, KernelBackend(toy), calib


def test_unguarded_drawer_snaps_the_string_guarded_one_refuses():
    toy, robot, calib = rig()
    OpenLoopDrawer(robot, calib).move("x", -40.0)            # 28 mm past the left edge: ~290 deg into the stop
    assert toy.snapped["x"]

    toy, robot, calib = rig()
    g = TravelGuard()
    g.anchor("x", 12.0, 2.0)
    g.anchor("y", 50.0, 2.0)
    with pytest.raises(EndStopRisk, match="allowed"):
        GuardedDrawer(robot, calib, g).move("x", -40.0)
    assert not toy.snapped["x"] and toy.strain["x"] == 0.0   # nothing was turned


def test_uncertainty_grows_and_unknown_position_is_refused():
    g = TravelGuard(GuardParams(k_rel_err=0.02, backlash_mm=1.0))
    with pytest.raises(EndStopRisk, match="unknown"):
        g.check("x", 5.0)
    g.anchor("x", 76.0, 1.5)
    for mm in (30.0, -30.0, 30.0):
        g.check("x", mm)
        g.commit("x", mm)
    assert g.unc["x"] == pytest.approx(1.5 + 90 * 0.02 + 2 * 1.0)
    lo, hi = g.allowed("x")
    assert lo < 0 < hi and hi < 153 - 5 - 106 + 1e-9


def test_torque_stop_pins_the_axis_and_allows_only_backing_off():
    g = TravelGuard()
    g.anchor("y", 100.0, 1.5)
    g.hit_stop("y", +1.0)
    assert g.pos["y"] == 107.0 and g.pinned["y"] == "high"
    with pytest.raises(EndStopRisk, match="only moves away"):
        g.check("y", 1.0)
    g.check("y", -10.0)
    g.commit("y", -10.0)
    assert g.pinned["y"] is None and g.pos["y"] == pytest.approx(97.0)


def test_a_square_near_the_edge_is_refused_before_any_turn_but_fits_further_in():
    toy, robot, calib = rig(start=(140.0, 50.0))
    g = TravelGuard()
    g.anchor("x", 140.0, 2.0)
    g.anchor("y", 50.0, 2.0)
    d = GuardedDrawer(robot, calib, g)
    with pytest.raises(EndStopRisk):
        d.draw(square(30.0))                                  # first side would run into the right edge
    assert toy.strain["x"] == 0.0
    toy, robot, calib = rig(start=(60.0, 40.0))
    g = TravelGuard()
    g.anchor("x", 60.0, 2.0)
    g.anchor("y", 40.0, 2.0)
    GuardedDrawer(robot, calib, g).draw(square(30.0))
    assert not any(toy.snapped.values()) and all(abs(s) < 1e-9 for s in toy.strain.values())


def test_near_an_edge_the_knob_turns_in_small_slow_steps():
    toy, robot, calib = rig(start=(25.0, 50.0))
    calls = []
    orig = robot.rotate
    robot.rotate = lambda deg: calls.append(deg) or orig(deg)
    robot.speed_dps_override = None                          # as RealBackend has
    g = TravelGuard()
    g.anchor("x", 25.0, 2.0)
    g.anchor("y", 50.0, 2.0)
    d = GuardedDrawer(robot, calib, g, slow_zone_mm=15.0, edge_step_deg=15.0)
    d.move("x", -6.0)                                        # ends at 19 +- 2.1 mm: inside the 15 mm slow zone? no
    assert len(calls) == 1 and d.slow_steps == 0
    d.move("x", -3.0)                                        # ends at 16 - 2.2 = 13.8 < 15 -> slow mode
    assert d.slow_steps >= 1 and all(abs(c) <= 15.0 + 1e-9 for c in calls[1:])
    assert robot.speed_dps_override is None                  # restored afterwards
    assert not toy.snapped["x"] and toy.strain["x"] == 0.0


def test_a_torque_stop_pins_the_axis_and_stops_the_drawing():
    toy, robot, calib = rig(start=(20.0, 50.0))
    g = TravelGuard()
    g.anchor("x", 20.0, 2.0)
    g.anchor("y", 50.0, 2.0)
    n = {"k": 0}
    orig = robot.rotate

    def rotate(deg):                                          # the second slow step "feels" the stop
        n["k"] += 1
        if n["k"] == 2:
            raise RuntimeError("rotate: TORQUE_LIMIT (J6 0.75 Nm)")
        return orig(deg)
    robot.rotate = rotate
    with pytest.raises(RuntimeError, match="TORQUE_LIMIT"):
        GuardedDrawer(robot, calib, g).move("x", -3.0)
    assert g.pinned["x"] == "low" and g.pos["x"] == 0.0
