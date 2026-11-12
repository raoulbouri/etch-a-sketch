"""Sketch extraction on synthetic camera images that mimic the real setup (no camera needed).

The synthetic view: faint line (contrast ~25 grey levels), screen at ~1.5 px/mm seen at an angle,
uneven lighting, a saturated glare blob, a glare streak and sensor noise.
"""
import cv2
import numpy as np

from etch_sim import sketch_extract as S
from etch_sim.mj.toy_model import ToyGeometry

G = ToyGeometry()
SHAPE_MM = [[(30, 30), (90, 30), (90, 80), (30, 80), (30, 30)], [(100, 20), (140, 90)], [(20, 95), (60, 95)]]


def truth_mask(px_per_mm: float) -> np.ndarray:
    w, h = round(G.screen_mm[0] * px_per_mm), round(G.screen_mm[1] * px_per_mm)
    m = np.zeros((h, w), np.uint8)
    for line in SHAPE_MM:
        pts = np.array([(x * px_per_mm, (G.screen_mm[1] - y) * px_per_mm) for x, y in line], np.int32)
        cv2.polylines(m, [pts], False, 1, max(1, round(0.6 * px_per_mm)), cv2.LINE_8)
    return m > 0


def synthetic_view(seed: int = 0, n: int = 6, contrast: float = 25.0):
    """A burst of camera-like frames and the true screen corners in that image."""
    rng = np.random.default_rng(seed)
    ppm_hi = 10.0                                            # draw finely, then let the warp blur it
    w, h = round(G.screen_mm[0] * ppm_hi), round(G.screen_mm[1] * ppm_hi)
    ink = cv2.GaussianBlur(truth_mask(ppm_hi).astype(np.float32), (0, 0), 2.0)
    ink /= ink.max()
    yy, xx = np.mgrid[0:h, 0:w]
    light = 150 + 30 * xx / w - 15 * yy / h                  # uneven lighting
    glare = 120 * np.exp(-(((xx - 0.75 * w) / (0.06 * w)) ** 2 + ((yy - 0.3 * h) / (0.08 * h)) ** 2))
    streak = 40 * np.exp(-(((xx - 0.45 * w) - 0.3 * (yy - 0.5 * h)) / (0.01 * w)) ** 2)
    screen = np.clip(light + glare + streak - contrast * ink, 0, 255).astype(np.float32)

    W, H = 1920, 1080                                        # camera image: screen ~230 px wide
    corners = np.array([[820, 430], [1050, 440], [1060, 600], [810, 590]], np.float32)
    M = cv2.getPerspectiveTransform(np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32), corners)
    frames = []
    for _ in range(n):
        cam = np.full((H, W), 200.0, np.float32)            # light table
        frame_poly = np.array([[790, 405], [1080, 415], [1090, 630], [780, 618]], np.int32)
        cv2.fillConvexPoly(cam, frame_poly, 60.0)            # dark toy frame
        warped = cv2.warpPerspective(screen, M, (W, H), flags=cv2.INTER_AREA)
        inside = cv2.warpPerspective(np.ones_like(screen), M, (W, H)) > 0.5
        cam[inside] = warped[inside]
        cam += rng.normal(0, 3.0, cam.shape)
        g = np.clip(cam, 0, 255).astype(np.uint8)
        frames.append(cv2.cvtColor(g, cv2.COLOR_GRAY2RGB))
    return frames, corners


def scores(mask: np.ndarray, truth: np.ndarray, ppm: float, tol_mm: float = 1.0) -> tuple[float, float]:
    dt = cv2.distanceTransform((~truth).astype(np.uint8), cv2.DIST_L2, 5) / ppm
    dm = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5) / ppm if mask.any() else np.full(truth.shape, 1e9)
    precision = float((dt[mask] <= tol_mm).mean()) if mask.any() else 0.0
    recall = float((dm[truth] <= tol_mm).mean())
    return precision, recall


def test_find_screen_on_synthetic_view():
    frames, corners = synthetic_view()
    found = S.find_screen(cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY))
    assert found is not None
    assert np.abs(found - S.order_corners(corners)).max() < 6.0


def test_extracts_faint_line_through_glare_and_noise():
    frames, corners = synthetic_view()
    p = S.ExtractParams()
    ex = S.extract(frames, corners, p)
    truth = truth_mask(p.px_per_mm)
    truth &= ~ex.unknown                                     # glare-hidden pixels cannot be scored
    prec, rec = scores(ex.mask, truth, p.px_per_mm)
    assert prec > 0.9, prec
    assert rec > 0.8, rec
