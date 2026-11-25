"""Shape move lists for the base test (scripts/draw_shapes.py)."""
import numpy as np
import pytest

from etch_sim.draw import merge, polyline, rectangle, stepped_circle


def test_merge_joins_same_axis_and_drops_zero_moves():
    assert merge([("x", 2.0), ("x", 3.0), ("y", 0.0), ("y", 1.0), ("x", -1.0)]) == [("x", 5.0), ("y", 1.0), ("x", -1.0)]


@pytest.mark.parametrize("moves", [rectangle(40, 25), stepped_circle(15, 4), stepped_circle(10, 3)])
def test_shapes_close(moves):
    pts = polyline(moves, (50.0, 20.0))
    assert np.allclose(pts[-1], pts[0], atol=1e-9)


def test_stepped_circle_corners_lie_on_the_circle():
    r, n = 15.0, 4
    moves = stepped_circle(r, n)
    assert len(moves) == 8 * n                       # alternating x/y, one switch per line
    assert all(a != b for (a, _), (b, _) in zip(moves, moves[1:]))
    pts = polyline(moves, (0.0, 0.0))[::2]           # every second point is a chord end on the circle
    assert np.allclose(np.hypot(pts[:, 0], pts[:, 1] - r), r)
    assert moves[0][0] == "x" and moves[0][1] > 0    # starts at the bottom going right


def test_spiral_out_starts_in_the_middle_and_fills_its_box():
    from etch_sim.draw import spiral_out
    moves = spiral_out(60.0, 6, aspect=0.75)
    assert len(moves) == 24 and [a for a, _ in moves[:4]] == ["x", "y", "x", "y"]
    lengths = [abs(mm) for _, mm in moves]
    assert lengths[0] == pytest.approx(5.0)                                          # first line: one pitch
    assert all(b >= a - 1e-9 for a, b in zip(lengths[0::2], lengths[2::2]))       # each x line longer than the last
    pts = polyline(moves, (0.0, 0.0))
    assert pts[:, 0].max() - pts[:, 0].min() == pytest.approx(60.0)
    assert pts[:, 1].max() - pts[:, 1].min() == pytest.approx(45.0)
    assert abs(pts[:, 0].max() + pts[:, 0].min()) < 5.0                             # roughly centred on the start


def test_spiral_out_draws_safely_through_the_guard_on_the_toy():
    from etch_sim.draw import Calib, spiral_out
    from etch_sim.endstop import GuardedDrawer, GuardParams, TravelGuard
    from etch_sim.kernel import ToyKernel, ToyParams
    from etch_sim.kernel_backend import KernelBackend
    toy = ToyKernel(ToyParams(), start_mm=(76.0, 52.0))
    p = toy.p
    calib = Calib({a: p.axis(a).k_mm_per_deg * 1.03 for a in "xy"}, {a: p.axis(a).delta_deg for a in "xy"})  # 3 % off
    g = TravelGuard(GuardParams(travel_mm={"x": 152.5, "y": 103.3}))
    g.anchor("x", 76.0, 5.0)
    g.anchor("y", 52.0, 5.0)
    GuardedDrawer(KernelBackend(toy), calib, g).draw(spiral_out(60.0, 6))
    assert not any(toy.snapped.values()) and all(abs(s) < 1e-9 for s in toy.strain.values())
    ink = np.concatenate([np.asarray(s.points) for s in toy.strokes if len(s.points) > 1])
    assert np.ptp(ink[:, 0]) == pytest.approx(60.0 / 1.03, rel=0.05)                # drawn ~ 3 % small, as calibrated
