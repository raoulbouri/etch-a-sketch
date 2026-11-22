"""S3: image -> route planner."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from etch_sim import Calib, KernelBackend, OpenLoopDrawer, ToyKernel, ToyParams, chamfer
from etch_sim.planner import PlanParams, plan_for_budget, plan_route, simplify_manhattan


def _canvas():
    im = Image.new("L", (400, 300), 255)
    return im, ImageDraw.Draw(im)


def _square():
    im, d = _canvas()
    d.rectangle((100, 50, 300, 250), outline=0, width=4)
    return im


def _circle():
    im, d = _canvas()
    d.ellipse((60, 30, 330, 270), outline=0, width=4)
    return im


def _pieces():
    im, d = _canvas()
    d.rectangle((30, 40, 130, 140), outline=0, width=4)
    d.ellipse((230, 40, 370, 160), outline=0, width=4)
    d.line([(60, 260), (340, 220)], fill=0, width=4)
    return im


def _manhattan(poly):
    d = np.diff(poly, axis=0)
    return bool(np.all((np.abs(d[:, 0]) < 1e-9) | (np.abs(d[:, 1]) < 1e-9)))


def test_square_is_cheap_and_starts_at_pen():
    pen = (20.3, 17.9)
    r = plan_route(_square(), pen_mm=pen)
    assert np.allclose(r.start_mm, pen) and np.allclose(r.polyline[0], pen)
    assert _manhattan(r.polyline)
    assert r.stats["switches"] <= 6                      # pen connector + 4 sides
    assert r.stats["target_to_route_mm"] < 1.0


def test_route_stays_on_screen_inside_margin():
    p = PlanParams()
    r = plan_route(_circle(), p)                          # no pen: nothing outside the drawing
    lo = p.margin_mm - p.pitch_mm - p.tol
    assert r.polyline.min() >= lo
    assert r.polyline[:, 0].max() <= p.screen_mm[0] - lo
    assert r.polyline[:, 1].max() <= p.screen_mm[1] - lo


def test_separate_pieces_become_one_route():
    r = plan_route(_pieces(), pen_mm=(20.0, 20.0))
    assert r.stats["pieces"] == 4                         # 3 pieces + the pen
    assert r.stats["connector_mm"] > 0
    assert _manhattan(r.polyline)
    assert r.stats["target_to_route_mm"] < 2.5            # nothing in the picture is missed


def test_tolerance_trades_switches_for_error():
    s = [plan_route(_circle(), PlanParams(tol_mm=t)).stats for t in (0.0, 3.0, 6.0)]
    assert s[0]["switches"] > s[1]["switches"] > s[2]["switches"]
    assert s[0]["target_to_route_mm"] < s[2]["target_to_route_mm"]


def test_simplify_keeps_ends_and_axis_alignment():
    stair = np.array([[i // 2 * 2, (i + 1) // 2 * 2] for i in range(21)], float)   # 2 mm stairs
    out = simplify_manhattan(stair, 3.0)
    assert np.allclose(out[0], stair[0]) and np.allclose(out[-1], stair[-1])
    assert _manhattan(out) and len(out) < len(stair)
    assert np.allclose(simplify_manhattan(stair, 0.0), stair)


def test_budget_is_met():
    r = plan_for_budget(_circle(), 420.0)
    assert r.stats["est_time_s"] <= 420.0
    assert r.stats["tol_mm"] > 0


def test_drawn_on_kernel_matches_plan():
    tp = ToyParams()
    toy = ToyKernel(tp, start_mm=(20.0, 20.0))
    robot = KernelBackend(toy, seed=0)
    calib = Calib({a: tp.axis(a).k_mm_per_deg for a in "xy"}, {a: tp.axis(a).delta_deg for a in "xy"})
    drawer = OpenLoopDrawer(robot, calib)
    drawer.preload()
    r = plan_route(_square(), PlanParams(calib=calib), pen_mm=robot.observe_pen())
    mark, sw0 = toy.mark(), robot.n_switches
    drawer.draw(r.moves)
    assert chamfer(toy.path_since(mark), r.polyline)["chamfer_mm"] < 0.5
    assert abs((robot.n_switches - sw0) - r.stats["switches"]) <= 1   # +1 if the first move is not on the held knob


def test_empty_image_raises():
    with pytest.raises(ValueError):
        plan_route(Image.new("L", (50, 50), 255))


def test_planner_imports_no_backend():
    src = (Path(__file__).resolve().parent.parent / "src/etch_sim/planner.py").read_text()
    for name in ("kernel_backend", "from .mj", "from .real", "from .kernel import"):
        assert name not in src
