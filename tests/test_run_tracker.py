"""Run-long pen tracking on the toy kernel: lines from before the run, transient lines and lines away from
the pen must not move the pen estimate; lines hidden by the arm must not be lost."""
import cv2
import numpy as np

from etch_sim import run_tracker as R
from etch_sim import turn_measure as T
from etch_sim.kernel import AxisParams, ToyKernel, ToyParams

PPM = 4.0
H, W = round(107 * PPM), round(153 * PPM)


def toy_at(start=(20.0, 20.0), k=0.095, half_dead=6.0, trans=4.5) -> ToyKernel:
    ap = dict(k_mm_per_deg=k, half_dead_deg=half_dead, transition_deg=trans)
    return ToyKernel(ToyParams(x=AxisParams(travel_mm=153.0, **ap), y=AxisParams(travel_mm=107.0, **ap)), start_mm=start)


def mask_of(toy: ToyKernel, extra: list | None = None) -> np.ndarray:
    """Line mask as the camera pipeline gives it (6 px = 1.5 mm wide, round ends), plus extra segments in mm."""
    m = np.zeros((H, W), np.uint8)
    segs = [st.points for st in toy.strokes if len(st.points) > 1] + (extra or [])
    for pts_mm in segs:
        pts = np.array([(x * PPM, H - 1 - y * PPM) for x, y in pts_mm]).round().astype(np.int32)
        cv2.polylines(m, [pts], False, 1, 6)
    return m > 0


def tracker(toy: ToyKernel, **kw) -> R.RunTracker:
    return R.RunTracker(R.knobs_from_toy(toy.p), R.TrackParams(px_per_mm=PPM, **kw))


def err(tr: R.RunTracker, toy: ToyKernel) -> float:
    return float(np.hypot(*np.subtract(tr.pen_mm, toy.stylus)))


def test_line_from_before_the_run_is_ignored():
    toy = toy_at()
    old = [[(30.0, 10.0), (30.0, 60.0)], [(10.0, 40.0), (120.0, 40.0)]]     # crosses the pen's path
    tr = tracker(toy)
    tr.start(mask_of(toy, old), pen_mm=toy.stylus)
    for ax, deg in [("x", 360), ("y", 360), ("x", -360)]:
        toy.turn(ax, deg)
        r = tr.update(mask_of(toy, old), [(ax, deg)])
        assert r["pen_mm"] is not None and err(tr, toy) < 1.0


def test_line_away_from_the_pen_is_ignored():
    toy = toy_at()
    tr = tracker(toy)
    tr.start(mask_of(toy), pen_mm=toy.stylus)
    toy.turn("x", 360)
    streak = [[(60.0, 70.0), (140.0, 90.0)]]                 # longer than the real stroke, far from the pen
    tr.update(mask_of(toy, streak), [("x", 360)])
    assert err(tr, toy) < 1.0


def test_transient_line_is_dropped_and_the_pen_revised():
    toy = toy_at()
    tr = tracker(toy)
    tr.start(mask_of(toy), pen_mm=toy.stylus)
    toy.turn("x", 360)
    tip = toy.stylus
    shadow = [[tip, (tip[0] + 6.0, tip[1])]]                 # carries on from the real line: plausible
    tr.update(mask_of(toy, shadow), [("x", 360)])
    assert tr.pen_mm[0] > tip[0] + 3.0                       # fooled for now
    for _ in range(2):                                       # shadow gone in the next two captures
        toy.turn("y", 120)
        tr.update(mask_of(toy), [("y", 120)])
    assert any(q.gone for q in tr.pieces)                    # the shadow's pixels were dropped
    first = tr.results[0]["pen_mm"]
    assert abs(first[0] - tip[0]) < 2.0                      # revised from 6 mm off: end cap + present_tol_mm remain
    assert err(tr, toy) < 1.0


def test_line_hidden_by_the_arm_is_kept():
    toy = toy_at()
    tr = tracker(toy)
    tr.start(mask_of(toy), pen_mm=toy.stylus)
    toy.turn("x", 360)
    tr.update(mask_of(toy), [("x", 360)])
    vis = np.ones((H, W), bool)
    vis[:, : round(45 * PPM)] = False                        # arm over the left part, where the new line is
    seen = mask_of(toy) & vis
    for _ in range(3):
        tr.update(seen, [], visible=vis)
    assert not any(q.transient for q in tr.pieces)
    drawn = tr.final_drawing(mask_of(toy))
    assert drawn.sum() > 0.8 * (mask_of(toy) & ~mask_of(toy_at())).sum()


def test_retrace_draws_nothing_and_the_pen_is_predicted():
    toy = toy_at()
    tr = tracker(toy)
    tr.start(mask_of(toy), pen_mm=toy.stylus)
    for ax, deg in [("x", 360), ("x", -180)]:                # second move runs back over its own line
        toy.turn(ax, deg)
        r = tr.update(mask_of(toy), [(ax, deg)])
    assert r["predicted"] and r["pen_mm"] is None
    assert err(tr, toy) < 2.0                                # prediction includes the reversal's backlash


def test_calibration_session_still_recovers_k_and_backlash():
    toy = toy_at(start=(15.0, 15.0), k=0.09, half_dead=7.0, trans=3.0)
    prior = toy_at(k=0.0951, half_dead=6.1, trans=4.5)       # current ToyParams values, not the truth
    tr = R.RunTracker(R.knobs_from_toy(prior.p), R.TrackParams(px_per_mm=PPM, **R.CALIBRATION))
    tr.start(mask_of(toy))
    plan = T.turn_plan("x")
    for ax, d in plan:
        toy.turn(ax, d * 360.0)
        tr.update(mask_of(toy), [(ax, d * 360.0)])
    f = T.fit_session(plan, tr.public_results(), "x").fit
    assert abs(f.k_mm_per_deg - 0.09) / 0.09 < 0.01
    assert abs(abs(f.delta_deg) - 10.0) < 3.0
