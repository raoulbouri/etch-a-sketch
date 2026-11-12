"""Camera image -> screen frame. Works on any RGB image, simulated or real.

Board localization uses the four ArUco markers stuck on the toy's frame; their positions come from
ToyGeometry. (S1 scope: localize, rectify, extract the drawn line. Pen-tip tracking comes in S2.)
"""
from __future__ import annotations

import cv2
import numpy as np

from .mj.toy_model import ARUCO_DICT, ToyGeometry

_DETECTOR = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), cv2.aruco.DetectorParameters())


def detect_markers(rgb: np.ndarray) -> dict[int, np.ndarray]:
    """{marker id: 4 x 2 pixel corners} for the board's markers found in the image."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    corners, ids, _ = _DETECTOR.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners)}


def board_homography(markers: dict[int, np.ndarray], g: ToyGeometry) -> np.ndarray | None:
    """3 x 3 homography from image pixels to the screen frame (mm). Needs at least 2 markers."""
    known = [i for i in markers if i in g.marker_xy_mm]
    if len(known) < 2:
        return None
    src = np.vstack([markers[i] for i in known]).astype(np.float32)
    dst = np.vstack([g.marker_corners_screen_mm(i) for i in known]).astype(np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H


def rectify(rgb: np.ndarray, H: np.ndarray, g: ToyGeometry, px_per_mm: float = 4.0) -> np.ndarray:
    """Front-on view of the drawable area, same layout as ToyKernel.render() (y up = image up)."""
    w, h = round(g.screen_mm[0] * px_per_mm), round(g.screen_mm[1] * px_per_mm)
    to_px = np.array([[px_per_mm, 0, 0], [0, -px_per_mm, h - 1], [0, 0, 1.0]])
    return cv2.warpPerspective(rgb, to_px @ H, (w, h), flags=cv2.INTER_LINEAR)


def line_mask(screen_rgb: np.ndarray, rel_contrast: float = 0.12, max_width_px: int = 9, border_px: int = 8) -> np.ndarray:
    """Boolean mask of drawn-line pixels in a rectified screen image.

    Keeps only thin dark structures (black top-hat), so shadow edges and large dark objects such as
    the gripper are not mistaken for line. A border of border_px is ignored: a rectification that is
    off by a pixel lets the red frame in there.
    """
    gray = cv2.cvtColor(screen_rgb, cv2.COLOR_RGB2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max_width_px, max_width_px))
    hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k).astype(np.float32)
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k).astype(np.float32)
    mask = hat > rel_contrast * np.maximum(bg, 1.0)
    if border_px:
        mask[:border_px], mask[-border_px:], mask[:, :border_px], mask[:, -border_px:] = False, False, False, False
    return mask


def board_pose_in_camera(markers: dict[int, np.ndarray], g: ToyGeometry, K: np.ndarray):
    """Toy frame in the camera frame (OpenCV convention: x right, y down, z forward) as (R, t), or None."""
    known = [i for i in markers if i in g.marker_xy_mm]
    if len(known) < 2:
        return None
    obj = np.vstack([g.marker_corners_toy_m(i) for i in known]).astype(np.float64)
    img = np.vstack([markers[i] for i in known]).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    return cv2.Rodrigues(rvec)[0], tvec.ravel()


class PenTracker:
    """Follows the stylus by watching where new line appears between photos.

    The stylus itself is assumed NOT to be visible (to be checked on the real toy), so the pen is
    taken as the end of the newest line. Retracing an existing line draws nothing new, and the
    estimate then stays where it was: plan calibration moves so that each one draws fresh line.
    """

    def __init__(self, px_per_mm: float = 4.0, line_width_mm: float = 0.6, band_mm: float = 1.2):
        self.s = px_per_mm
        self.half_w = line_width_mm / 2        # PLACEHOLDER: a line end sits half a line width beyond the stylus
        self.band = band_mm
        self.max_gap = 1.0                     # mm of missing line tolerated (e.g. where a shadow edge crosses it)
        self.prev: np.ndarray | None = None
        self.pen: tuple[float, float] | None = None

    def _largest_blob(self, new: np.ndarray, h: int) -> np.ndarray:
        n, lab, stats, _ = cv2.connectedComponentsWithStats(new.astype(np.uint8), connectivity=8)
        if n < 2 or stats[1:, cv2.CC_STAT_AREA].max() < 10:
            return np.empty((0, 2))
        ys, xs = np.nonzero(lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))
        return np.column_stack([xs / self.s, (h - 1 - ys) / self.s])

    def update(self, screen_rgb: np.ndarray, moves: list[tuple[str, float]]) -> tuple[float, float] | None:
        """screen_rgb: rectified screen. moves: (axis, sign) of each knob turn since the last update, in order."""
        mask = line_mask(screen_rgb)
        h = mask.shape[0]
        if self.prev is None:
            new = mask
        else:
            new = mask & ~cv2.dilate(self.prev.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        self.prev = mask
        ys, xs = np.nonzero(mask)
        pts = np.column_stack([xs / self.s, (h - 1 - ys) / self.s])        # every line pixel, screen mm
        fresh = new[ys, xs]
        for axis, sign in moves:
            i = 0 if axis == "x" else 1
            if self.pen is None:
                cand = self._largest_blob(new, h)
            else:
                # Walk along the line from the pen in the direction of the turn until it breaks. A shadow
                # edge or the gripper elsewhere in the band is separated from the pen by a gap. The walk
                # must pass over some line that was not in the last photo, otherwise nothing was drawn.
                along = sign * (pts[:, i] - self.pen[i])
                sel = (np.abs(pts[:, 1 - i] - self.pen[1 - i]) < self.band) & (along > -0.5)
                order = np.argsort(along[sel])
                cand, cand_fresh = pts[sel][order], fresh[sel][order]
                reach = np.concatenate([[0.0], along[sel][order]])
                gaps = np.nonzero(np.diff(reach) > self.max_gap)[0]
                if len(gaps):
                    cand, cand_fresh = cand[:gaps[0]], cand_fresh[:gaps[0]]
                if cand_fresh.sum() < 3:
                    continue
            if len(cand) < 3:
                continue
            ext = (sign * cand[:, i]).max()
            tip = cand[sign * cand[:, i] > ext - 2 / self.s]
            end = [0.0, 0.0]
            end[i] = sign * ext - sign * self.half_w
            end[1 - i] = float(np.median(tip[:, 1 - i]))
            self.pen = (float(end[0]), float(end[1]))
        return self.pen


def mask_distance_mm(a: np.ndarray, b: np.ndarray, px_per_mm: float = 4.0) -> dict:
    """Mean and max distance from each line pixel of one mask to the nearest of the other, in mm."""
    if not a.any() or not b.any():
        return {"a_to_b_mm": float("inf"), "b_to_a_mm": float("inf"), "max_mm": float("inf")}
    da = cv2.distanceTransform((~a).astype(np.uint8), cv2.DIST_L2, 5)
    db = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, 5)
    return {"a_to_b_mm": float(db[a].mean() / px_per_mm), "b_to_a_mm": float(da[b].mean() / px_per_mm),
            "max_mm": float(max(db[a].max(), da[b].max()) / px_per_mm)}
