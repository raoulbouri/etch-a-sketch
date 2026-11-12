"""Extract the etched line from a real camera image of the Etch-a-Sketch screen.

Real images differ from the simulator's: the line is a faint dark-grey on light-grey, the screen
reflects lights (glare blobs and streaks), lighting is uneven, and with the current tripod the screen
is only ~1.5 px per mm. The pipeline:

1. Screen corners: found automatically (the light screen inside the dark frame) or given by hand
   (`hardware/scene_camera_screen.json`) until ArUco markers are on the toy.
2. Rectify to a front-on view of the drawable area at `px_per_mm` (same frame as the simulator).
3. Combine a burst of frames: median removes sensor noise.
4. Background brightness: a closing larger than the line. Used to normalise the response (a
   line has the same score in bright and dim areas) and to show a lighting-flattened image.
5. Line response: an oriented two-sided test. A pixel is line only if it is darker than BOTH
   sides across the line direction; the dip is then averaged along the line direction (a matched
   filter). The edge of a glare streak or of a shadow is darker on one side only, so it scores ~0;
   round dots and dust gain little from the averaging along a direction.
6. Saturated pixels (specular glare) are marked unknown, never line.
7. Hysteresis threshold relative to the robust noise level of the response, then drop small
   components.

Every threshold below is a PLACEHOLDER until tuned on labelled real images.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .mj.toy_model import ToyGeometry

SCREEN_FILE = Path(__file__).resolve().parents[2] / "hardware" / "scene_camera_screen.json"


@dataclass
class ExtractParams:
    px_per_mm: float = 4.0             # rectified resolution (upsampled from ~1.5 real)
    line_width_mm: float = 0.6         # PLACEHOLDER: measure on the real toy
    bg_kernel_mm: float = 3.0          # closing size for the background; > line width, < glare size
    side_mm: float = 1.0               # PLACEHOLDER: where the two sides are sampled, > blurred half-width
    along_mm: float = 2.0              # PLACEHOLDER: averaging length along the line
    n_angles: int = 12
    high_k: float = 6.0                # PLACEHOLDER: seed threshold, in robust noise sigmas
    low_k: float = 3.0                 # PLACEHOLDER: grow threshold
    min_len_mm: float = 2.0            # PLACEHOLDER: drop components shorter than this
    saturation: int = 245              # raw value counted as glare
    border_mm: float = 2.5             # PLACEHOLDER: ignore a margin inside the screen edge (frame shadow, bevel)
    burst: str = "median"              # median | min
    min_strength_k: float = 8.0        # PLACEHOLDER: a short component must average this many noise sigmas...
    long_mm: float = 20.0              # ...unless it is at least this long (real faint lines are long)
    rotate_180: bool = False           # toy seen upside down (knobs at the far side of the image)
    toy_order: bool = False            # corners are given in the toy's own order (BL, BR, TR, TL; see below)


@dataclass
class Extraction:
    rectified: np.ndarray              # RGB or grey, front-on screen
    flat: np.ndarray                   # lighting-normalised grey (~1 background)
    response: np.ndarray               # line strength
    mask: np.ndarray                   # bool line pixels
    unknown: np.ndarray                # bool pixels hidden by glare
    stats: dict = field(default_factory=dict)


# ---- screen corners ------------------------------------------------------------------------------
def order_corners(pts: np.ndarray) -> np.ndarray:
    """Top-left, top-right, bottom-right, bottom-left (image coordinates)."""
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]], np.float32)


def find_screen(gray: np.ndarray, g: ToyGeometry | None = None) -> np.ndarray | None:
    """Corners (TL, TR, BR, BL) of the light screen inside the toy's dark frame, or None.

    Looks for a convex quadrilateral that is lighter inside than in a ring around it and whose
    aspect ratio is near the screen's. Fragile by design: confirm it on the debug image, and
    save hand-picked corners if it picks the wrong thing.
    """
    g = g or ToyGeometry()
    want = g.screen_mm[0] / g.screen_mm[1]
    img = cv2.GaussianBlur(gray, (5, 5), 0)
    H, W = img.shape
    best, best_score = None, 0.0
    for thr in np.linspace(np.percentile(img, 1) + 5, np.percentile(img, 99) - 5, 20):
        bw = (img > thr).astype(np.uint8) * 255
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cs, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            area = cv2.contourArea(c)
            if not (0.003 * H * W < area < 0.5 * H * W):
                continue
            q = cv2.approxPolyDP(cv2.convexHull(c), 0.04 * cv2.arcLength(c, True), True)
            if len(q) != 4:
                continue
            q = order_corners(q)
            wq = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
            hq = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
            ar = wq / max(hq, 1)
            if not (0.6 * want < ar < 1.6 * want):
                continue
            fill = area / max(cv2.contourArea(q), 1)
            inside = np.zeros_like(img)
            cv2.fillConvexPoly(inside, q.astype(np.int32), 1)
            ring = cv2.dilate(inside, np.ones((25, 25), np.uint8)) - inside
            contrast = img[inside > 0].mean() - img[ring > 0].mean()
            score = contrast * fill * np.sqrt(area)
            if score > best_score:
                best, best_score = q, score
    return best


TOY_ORDER = "BL BR TR TL"   # bottom-left (nearest the left knob), bottom-right (nearest the right knob), ...


def load_screen_corners(path: Path = SCREEN_FILE) -> np.ndarray | None:
    """Saved corners. In toy order (as clicked) they are returned as they are; older files hold
    image-order corners (TL, TR, BR, BL in the image), which are sorted."""
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    c = np.array(d["corners_px"], np.float32)
    return c if d.get("order") == TOY_ORDER else order_corners(c)


def screen_params(path: Path = SCREEN_FILE) -> "ExtractParams":
    """ExtractParams matching the saved corners file (toy order, or image order + rotate_180)."""
    if not path.exists():
        return ExtractParams()
    d = json.loads(path.read_text())
    if d.get("order") == TOY_ORDER:
        return ExtractParams(toy_order=True)
    return ExtractParams(rotate_180=bool(d.get("rotate_180", False)))


def save_toy_corners(corners: np.ndarray, image_size: tuple[int, int], source: str, path: Path = SCREEN_FILE) -> None:
    """Corners clicked in the toy's own order, which fixes the orientation whatever the camera's view."""
    path.write_text(json.dumps({"corners_px": np.asarray(corners).round(1).tolist(), "order": TOY_ORDER,
                                "image_wh": list(image_size), "source": source,
                                "note": "drawable-screen corners, seen from the knob side: bottom-left (near the left "
                                        "knob), bottom-right, top-right, top-left. Re-pick if the camera or toy moves."},
                               indent=2))


def save_screen_corners(corners: np.ndarray, image_size: tuple[int, int], path: Path = SCREEN_FILE) -> None:
    path.write_text(json.dumps({"corners_px": order_corners(corners).tolist(), "image_wh": list(image_size),
                                "note": "TL, TR, BR, BL of the drawable screen in the scene camera image"}, indent=2))


def rectify_quad(img: np.ndarray, corners: np.ndarray, g: ToyGeometry, px_per_mm: float,
                 toy_order: bool = False) -> np.ndarray:
    """Front-on view of the screen. Image-order corners give the camera's orientation; toy-order
    corners (BL, BR, TR, TL) give the toy's own frame directly (left knob bottom-left, y up)."""
    w, h = round(g.screen_mm[0] * px_per_mm), round(g.screen_mm[1] * px_per_mm)
    if toy_order:
        src = np.asarray(corners, np.float32)
        dst = np.array([[0, h - 1], [w - 1, h - 1], [w - 1, 0], [0, 0]], np.float32)
    else:
        src = order_corners(corners)
        dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_CUBIC)


# ---- line extraction -----------------------------------------------------------------------------
def _gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.float32)
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)


def _shift(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, img.shape[::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _two_sided_dip(flat: np.ndarray, side_px: float, along_px: float, n_angles: int) -> np.ndarray:
    """max over directions of: min(both sides) - centre, averaged along the line direction."""
    s = cv2.GaussianBlur(flat, (0, 0), 0.8)
    best = np.zeros_like(s)
    L = max(3, int(round(along_px)) | 1)
    for th in np.linspace(0, np.pi, n_angles, endpoint=False):
        nx, ny = np.cos(th), np.sin(th)                           # normal to the line
        a = _shift(s, -side_px * nx, -side_px * ny)
        b = _shift(s, side_px * nx, side_px * ny)
        dip = np.clip(np.minimum(a, b) - s, 0, None)
        k = np.zeros((L, L), np.float32)                          # averaging kernel along the tangent
        c = L // 2
        cv2.line(k, (int(round(c - c * ny)), int(round(c + c * nx))), (int(round(c + c * ny)), int(round(c - c * nx))), 1.0, 1)
        k /= k.sum()
        best = np.maximum(best, cv2.filter2D(dip, -1, k, borderType=cv2.BORDER_REPLICATE))
    return best


def _robust_sigma(x: np.ndarray) -> float:
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med))) or 1e-6


def extract(frames: list[np.ndarray] | np.ndarray, corners: np.ndarray | None = None,
            p: ExtractParams | None = None, g: ToyGeometry | None = None) -> Extraction:
    """frames: one RGB/grey image or a burst of them (camera view, or already-rectified if corners is None)."""
    p, g = p or ExtractParams(), g or ToyGeometry()
    frames = [frames] if isinstance(frames, np.ndarray) else list(frames)
    if corners is not None:
        frames = [rectify_quad(f, corners, g, p.px_per_mm, p.toy_order) for f in frames]
        if p.rotate_180 and not p.toy_order:                                    # into the toy's own frame: knobs at the bottom
            frames = [cv2.rotate(f, cv2.ROTATE_180) for f in frames]
    st = np.stack(frames)
    comb = (np.median(st, axis=0) if p.burst == "median" else st.min(axis=0)).astype(np.uint8)
    raw_max = comb if comb.ndim == 2 else comb.max(axis=2)
    gray = _gray(comb)

    k = max(3, int(round(p.bg_kernel_mm * p.px_per_mm)) | 1)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(cv2.GaussianBlur(gray, (0, 0), 0.7), cv2.MORPH_CLOSE, ker)
    bg = cv2.GaussianBlur(bg, (0, 0), k / 2)
    flat = gray / np.maximum(bg, 1.0)

    # dip measured on the raw grey image (the flattening itself darkens the edges of bright glare),
    # then divided by local brightness so the same line scores the same in bright and dim areas
    resp = _two_sided_dip(gray, p.side_mm * p.px_per_mm, p.along_mm * p.px_per_mm, p.n_angles) / np.maximum(bg, 1.0)

    unknown = cv2.dilate((raw_max >= p.saturation).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    b = int(round(p.border_mm * p.px_per_mm))
    valid = ~unknown
    if b:
        valid[:b], valid[-b:], valid[:, :b], valid[:, -b:] = False, False, False, False

    med = float(np.median(resp[valid])) if valid.any() else 0.0
    sig = _robust_sigma(resp[valid]) if valid.any() else 1.0
    hi = (resp > med + p.high_k * sig) & valid
    lo = (resp > med + p.low_k * sig) & valid
    n, lab, stats, _ = cv2.connectedComponentsWithStats(lo.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[hi])] = True
    keep[0] = False
    min_len = p.min_len_mm * p.px_per_mm
    extent = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    keep &= extent >= min_len
    # clutter (scratches, texture, specks) is short AND weak; drawn lines are long or strong
    strength = np.zeros(n)
    np.add.at(strength, lab.ravel(), resp.ravel())
    strength = (strength / np.maximum(stats[:, cv2.CC_STAT_AREA], 1) - med) / sig
    keep &= (strength >= p.min_strength_k) | (extent >= p.long_mm * p.px_per_mm)
    mask = keep[lab]
    return Extraction(comb, flat, resp, mask, unknown,
                      {"noise_sigma": sig, "line_frac": float(mask.mean()), "unknown_frac": float(unknown.mean()),
                       "components": int(keep.sum())})


def debug_panel(ex: Extraction) -> np.ndarray:
    """BGR image: rectified | flattened | response | mask overlay (red line, blue glare)."""
    def to_bgr(x, lo=None, hi=None):
        x = x.astype(np.float32)
        lo = np.percentile(x, 1) if lo is None else lo
        hi = np.percentile(x, 99.5) if hi is None else hi
        u = np.clip((x - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        return cv2.cvtColor(u, cv2.COLOR_GRAY2BGR)
    rect = ex.rectified if ex.rectified.ndim == 2 else cv2.cvtColor(ex.rectified, cv2.COLOR_RGB2GRAY)
    over = to_bgr(rect, 0, 255)
    over[ex.unknown] = (255, 120, 0)
    over[ex.mask] = (0, 0, 255)
    top = np.hstack([to_bgr(rect, 0, 255), to_bgr(ex.flat)])
    bot = np.hstack([to_bgr(ex.response), over])
    return np.vstack([top, bot])
