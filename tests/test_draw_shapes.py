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
