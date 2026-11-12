"""Full-turn hand calibration on the toy kernel: does measuring the pen from line masks recover
the hidden mm-per-degree and backlash?"""
import cv2
import numpy as np

from etch_sim import turn_measure as T
from etch_sim.kernel import AxisParams, ToyKernel, ToyParams

PPM = 4.0


def mask_of(toy: ToyKernel) -> np.ndarray:
    """Line mask as the camera pipeline gives it: wider than the ink (6 px = 1.5 mm), round ends."""
    h = round(toy.p.y.travel_mm * PPM)
    m = np.zeros((h, round(toy.p.x.travel_mm * PPM)), np.uint8)
    for st in toy.strokes:
        if len(st.points) > 1:
            pts = np.array([(x * PPM, h - 1 - y * PPM) for x, y in st.points]).round().astype(np.int32)
            cv2.polylines(m, [pts], False, 1, 6)      # thick OpenCV lines have round caps
    return m > 0


def run_session(axis: str, k: float, half_dead: float, trans: float, angle_noise_deg: float = 0.0,
                seed: int = 0, start=(15.0, 15.0)):
    rng = np.random.default_rng(seed)
    ap = dict(k_mm_per_deg=k, half_dead_deg=half_dead, transition_deg=trans)
    toy = ToyKernel(ToyParams(x=AxisParams(travel_mm=153.0, **ap), y=AxisParams(travel_mm=107.0, **ap)), start_mm=start)
    plan = T.turn_plan(axis)
    track = T.PenTrack(px_per_mm=PPM)
    track.update(mask_of(toy))                                  # baseline capture before the first turn
    results: list = []
    for ax, d in plan:
        toy.turn(ax, d * 360.0 + rng.normal(0, angle_noise_deg))
        r = track.update(mask_of(toy))
        if r["resolved_prev_pen_mm"] is not None:
            results[-1]["pen_mm"] = r["resolved_prev_pen_mm"]
        results.append(r)
    return T.fit_session(plan, results, axis), toy


def test_recovers_k_and_backlash_exact_turns():
    for axis in ("x", "y"):
        tf, _ = run_session(axis, k=0.085, half_dead=9.0, trans=4.5)
        s = tf.summary()
        assert abs(s["k_mm_per_deg"] - 0.085) / 0.085 < 0.01, s
        assert abs(s["backlash_deg"] - 27.0) < 2.0, s            # 2 * (half_dead + transition)
        assert s["excluded_steps"] == [], s
        assert s["cw_moves"] == ("right" if axis == "x" else "up")


def test_hand_turn_errors_and_other_toy():
    """+-6 deg of error on every hand turn: k stays within a few %, backlash only within ~10 deg."""
    for seed in range(6):
        tf, _ = run_session("x", k=0.11, half_dead=6.0, trans=3.0, angle_noise_deg=6.0, seed=seed)
        s = tf.summary()
        assert abs(s["k_mm_per_deg"] - 0.11) / 0.11 < 0.03, (seed, s)
        assert abs(s["backlash_deg"] - 18.0) < 10.0, (seed, s)


def test_turn_plan_shape():
    p = T.turn_plan("x")
    assert p.count(("x", 1)) == 6 and p.count(("x", -1)) == 3 and p.count(("y", 1)) == 2
    q = T.turn_plan("y")
    assert q.count(("y", 1)) == 4 and q.count(("y", -1)) == 4 and q.count(("x", 1)) == 3


def test_camera_path_session():
    """Same session drawn as a faint, noisy camera image through the saved-style screen corners,
    then extracted by sketch_extract: the whole real pipeline except the camera itself."""
    from etch_sim import sketch_extract as S

    corners = np.float32([[637, 395], [925, 395], [953, 596], [633, 588]])
    p = S.ExtractParams(rotate_180=True)
    rng = np.random.default_rng(1)
    ap = dict(k_mm_per_deg=0.09, half_dead_deg=8.0, transition_deg=4.0)
    toy = ToyKernel(ToyParams(x=AxisParams(travel_mm=153.0, **ap), y=AxisParams(travel_mm=107.0, **ap)), start_mm=(15, 15))

    def camera_mask():
        scr = np.asarray(toy.render(px_per_mm=8).convert("L")).astype(np.float32)
        scr = cv2.rotate(150 + (scr - 188) * (25 / 133), cv2.ROTATE_180)    # faint line, toy upside down
        h, w = scr.shape
        M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]), corners)
        warped = cv2.warpPerspective(scr, M, (1280, 720), flags=cv2.INTER_AREA)
        inside = cv2.warpPerspective(np.ones_like(scr), M, (1280, 720)) > 0.5
        frames = []
        for _ in range(3):
            cam = np.full((720, 1280), 180, np.float32)
            cam[inside] = warped[inside]
            cam += rng.normal(0, 2.5, cam.shape)
            frames.append(np.clip(cam, 0, 255).astype(np.uint8))
        return S.extract(frames, corners, p).mask

    plan = T.turn_plan("x")
    track = T.PenTrack(px_per_mm=p.px_per_mm)
    track.update(camera_mask())
    results: list = []
    for ax, d in plan:
        toy.turn(ax, d * 360.0)
        r = track.update(camera_mask())
        if r["resolved_prev_pen_mm"] is not None:
            results[-1]["pen_mm"] = r["resolved_prev_pen_mm"]
        results.append(r)
    s = T.fit_session(plan, results, "x").summary()
    assert abs(s["k_mm_per_deg"] - 0.09) / 0.09 < 0.01, s
    assert abs(s["backlash_deg"] - 24.0) < 5.0, s               # camera sees ~0.5 mm per pixel: ~3 deg floor
