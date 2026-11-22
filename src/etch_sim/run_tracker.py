"""Pen tracking over a whole run (all the moves of one drawing or calibration session, up to the next erase).

The pen is invisible; only the line it leaves is seen. Three rules decide which line counts:

1. Before the run: anything on the screen when the run starts (the baseline) is never the pen's line.
2. Appears, then disappears: a drawn line stays until the next erase. A piece of line that appears and is
   later gone while that part of the screen is visible (a shadow, reflection, someone passing) is transient:
   it is dropped, and pen estimates that used it are recomputed. A line hidden by the arm is not "gone".
3. Where the pen can be: after a knob turn, the new line must start at the pen, run along that knob's axis,
   in the direction the knob moves the pen, about as long as the turn predicts (k, backlash). Anything
   else is ignored, and the pen is then put where the turn predicts ("predicted").

Masks are in the screen frame at px_per_mm (row 0 = top, as sketch_extract produces them).
Positions are screen millimetres (x right, y up).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .kernel import ToyParams
from .turn_measure import Stroke, _line_center, _tip


@dataclass
class Knob:
    k_mm_per_deg: float                # pen travel per knob degree once slack is taken up
    delta_deg: float                   # half backlash: a reversal loses 2 * delta_deg of knob turn
    sign: int = 1                      # +1: clockwise moves the pen +x (right) / +y (up)


def knobs_from_toy(toy: ToyParams | None = None, signs: dict[str, int] | None = None) -> dict[str, Knob]:
    toy, signs = toy or ToyParams(), signs or {}
    return {ax: Knob(getattr(toy, ax).k_mm_per_deg, getattr(toy, ax).delta_deg, signs.get(ax, 1)) for ax in ("x", "y")}


@dataclass
class TrackParams:
    px_per_mm: float = 4.0
    gap_mm: float = 1.0                # known lines are grown by this much before looking for new line
    min_len_mm: float = 1.5            # PLACEHOLDER: shorter new pieces are noise
    start_tol_mm: float = 3.5          # PLACEHOLDER: new line must begin within this of the pen (half the old line + gap_mm;
                                       # real masks were 1.5-2 mm wide on 2026-11-22)
    side_tol_mm: float = 1.5           # PLACEHOLDER: and stay this close to the pen's line across the turn axis
    wide_mm: float = 1.5               # the accepted line's pixels are collected this much wider (its full width)
    max_gap_mm: float = 2.0            # PLACEHOLDER: break in the line tolerated while walking out from the pen
    len_window: tuple[float, float] = (0.7, 1.3)   # PLACEHOLDER: accepted length / predicted length (drawing)
    len_slack_mm: float = 1.0          # PLACEHOLDER: plus this, for short moves (~0.6 mm per camera pixel)
    backlash_tol_deg: float = 10.0     # PLACEHOLDER: hand-measured backlash is +-8 deg
    bridge_mm: float = 6.0             # PLACEHOLDER: a longer break is allowed where an older line crosses the path
    gone_after: int = 2                # PLACEHOLDER: captures a pixel must be visible and missing to count as gone
    present_tol_mm: float = 1.0        # PLACEHOLDER: still there if line is detected this close; > 1 camera px (0.6 mm on 2026-11-22)
    occluder_mm: float = 6.0           # dark areas wider than this are arm, gripper or shadow: not visible
    occluder_flat: float = 0.8         # "dark": lighting-normalised brightness below this


CALIBRATION = dict(len_window=(0.5, 2.0))      # calibration measures k: only reject gross mismatches


def visible_from(flat: np.ndarray, unknown: np.ndarray, p: TrackParams) -> np.ndarray:
    """Screen pixels the camera can judge: not glare, not under a large dark occluder (arm, gripper,
    their shadow). Drawn lines are thin, so the opening removes them and keeps only wide dark areas."""
    dark = (flat < p.occluder_flat).astype(np.uint8)
    r = max(1, int(round(p.occluder_mm * p.px_per_mm / 2)))
    big = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    big = cv2.dilate(big, np.ones((5, 5), np.uint8))
    return ~(big.astype(bool) | unknown)


@dataclass
class Piece:
    pixels: np.ndarray                 # N x 2 (row, col) still believed to be line
    first: int                         # observation index where it first appeared
    absent: np.ndarray | None = None   # per pixel: consecutive visible captures in which it was missing
    gone: list = field(default_factory=list)   # pixels dropped as transient (rule 2)

    def __post_init__(self):
        if self.absent is None:
            self.absent = np.zeros(len(self.pixels), int)

    @property
    def transient(self) -> bool:
        return len(self.pixels) == 0


@dataclass
class _Obs:
    moves: list[tuple[str, float]]     # (axis, knob degrees, cw +) since the previous observation
    pieces: list[int]                  # pieces first seen in this observation


class RunTracker:
    """Follows the pen through one run. start() with the first capture, then update() after each set of moves."""

    def __init__(self, knobs: dict[str, Knob], p: TrackParams | None = None, screen_mm: tuple[float, float] = (153.0, 107.0)):
        self.knobs, self.p, self.screen_mm = knobs, p or TrackParams(), screen_mm
        self.baseline: np.ndarray | None = None
        self.pieces: list[Piece] = []
        self.obs: list[_Obs] = []
        self.results: list[dict] = []                     # one per update(), revised when a piece turns out transient
        self.pen_px: np.ndarray | None = None
        self._start_pen: np.ndarray | None = None
        self._last_dir: dict[str, int] = {}
        self._shape: tuple[int, int] | None = None

    # ---- frame helpers -------------------------------------------------------------------------
    def to_mm(self, rc: np.ndarray) -> tuple[float, float]:
        return float(rc[1] / self.p.px_per_mm), float((self._shape[0] - 1 - rc[0]) / self.p.px_per_mm)

    def to_px(self, xy: tuple[float, float]) -> np.ndarray:
        return np.array([self._shape[0] - 1 - xy[1] * self.p.px_per_mm, xy[0] * self.p.px_per_mm])

    def _grow(self, m: np.ndarray) -> np.ndarray:
        r = max(1, int(round(self.p.gap_mm * self.p.px_per_mm)))
        return cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0

    def _paint(self, ids) -> np.ndarray:
        m = np.zeros(self._shape, bool)
        for i in ids:
            r, c = self.pieces[i].pixels.T
            m[r, c] = True
        return m

    @property
    def pen_mm(self) -> tuple[float, float] | None:
        return None if self.pen_px is None else self.to_mm(self.pen_px)

    # ---- run ----------------------------------------------------------------------------------
    def start(self, masks: np.ndarray | list[np.ndarray], pen_mm: tuple[float, float] | None = None) -> None:
        """Rule 1: the baseline is everything seen in the starting capture(s). pen_mm if known (else found
        from the first stroke)."""
        masks = [masks] if isinstance(masks, np.ndarray) else list(masks)
        self._shape = masks[0].shape
        self.baseline = np.logical_or.reduce(masks)
        self.pieces, self.obs, self.results, self._last_dir = [], [], [], {}
        self._start_pen = None if pen_mm is None else self.to_px(pen_mm)
        self.pen_px = self._start_pen

    def update(self, mask: np.ndarray, moves: list[tuple[str, float]], visible: np.ndarray | None = None) -> dict:
        """mask after the moves; moves = (axis, knob degrees, clockwise +) in order. Returns this observation's
        result: pen_mm (measured, else None), pen_pred_mm (where the turn alone puts it), stroke details."""
        if self.baseline is None:
            raise RuntimeError("start() the run first")
        visible = np.ones(self._shape, bool) if visible is None else visible
        idx = len(self.obs)
        revise_from = self._judge(mask, visible)          # rule 2 on the pieces seen so far
        known = self._grow(self.baseline | self._paint(i for i, q in enumerate(self.pieces) if not q.transient))
        new = mask & ~known & visible
        n, lab, st, _ = cv2.connectedComponentsWithStats(new.astype(np.uint8), connectivity=8)
        ids = []
        for j in range(1, n):
            if max(st[j, cv2.CC_STAT_WIDTH], st[j, cv2.CC_STAT_HEIGHT]) >= self.p.min_len_mm * self.p.px_per_mm:
                self.pieces.append(Piece(np.argwhere(lab == j), idx))
                ids.append(len(self.pieces) - 1)
        self.obs.append(_Obs(list(moves), ids))
        self._estimate_from(min(revise_from, idx))
        return self.results[idx]

    def _judge(self, mask: np.ndarray, visible: np.ndarray) -> int:
        """Rule 2, per pixel: a pixel of new line that is visible and missing in gone_after captures in a row
        is dropped, so a shadow touching a real line goes without the line. Hidden pixels (arm, glare) are
        not evidence either way. Returns the first observation whose estimate must be redone."""
        r = max(1, int(round(self.p.present_tol_mm * self.p.px_per_mm)))
        present = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
        redo = len(self.obs)
        for q in self.pieces:
            if q.transient:
                continue
            r, c = q.pixels.T
            miss = visible[r, c] & ~present[r, c]
            q.absent = np.where(miss, q.absent + 1, np.where(present[r, c], 0, q.absent))
            drop = q.absent >= self.p.gone_after
            if drop.any():
                q.gone.extend(q.pixels[drop].tolist())
                q.pixels, q.absent = q.pixels[~drop], q.absent[~drop]
                redo = min(redo, q.first)
        return redo

    # ---- rule 3 -------------------------------------------------------------------------------
    def _expected(self, axis: str, deg: float) -> tuple[float, float, float]:
        """(predicted, shortest, longest) pen travel in mm along +axis for one move, from k and backlash."""
        kn, p = self.knobs[axis], self.p
        d = 1 if deg > 0 else -1
        last = self._last_dir.get(axis)
        full = 2 * kn.delta_deg
        if last is None:                                   # slack state unknown at the start of a run
            lo_loss, hi_loss = 0.0, full + p.backlash_tol_deg
        elif last != d:                                    # reversal
            lo_loss, hi_loss = max(0.0, full - p.backlash_tol_deg), full + p.backlash_tol_deg
        else:
            lo_loss = hi_loss = 0.0
        a = abs(deg)
        sgn = kn.sign * d
        travel = lambda loss: kn.k_mm_per_deg * max(0.0, a - loss)
        mid = travel((lo_loss + hi_loss) / 2)
        lo = max(0.0, travel(hi_loss) * p.len_window[0] - p.len_slack_mm)
        hi = travel(lo_loss) * p.len_window[1] + p.len_slack_mm
        return sgn * mid, lo, hi

    def _walk(self, cand: np.ndarray, axis: str, sgn: int, known: np.ndarray) -> np.ndarray | None:
        """Candidate pixels that start at the pen and run continuously from it in direction sgn along axis.

        Whether a line starts at the pen and how far it runs is decided in a narrow band (side_tol_mm); the
        pixels returned are the line's full width over that stretch, so its centre is not biased by the band.
        Sorted by distance from the pen; self._tail_from indexes the part after the last bridged crossing."""
        j = 1 if axis == "x" else 0                        # pixel column for x, row for y
        dirn = sgn if axis == "x" else -sgn                # +y (up) is decreasing row
        ppm = self.p.px_per_mm
        across = np.abs(cand[:, 1 - j] - self.pen_px[1 - j])
        along = dirn * (cand[:, j] - self.pen_px[j])
        sel = (across <= self.p.side_tol_mm * ppm) & (along >= -self.p.start_tol_mm * ppm)
        if not sel.any():
            return None
        a = np.sort(along[sel])
        if a[0] > self.p.start_tol_mm * ppm:               # does not begin at the pen
            return None
        pts_n = cand[sel][np.argsort(along[sel])]
        end, tail_at = a[-1], a[0]
        for b in np.nonzero(np.diff(a) > self.p.max_gap_mm * ppm)[0]:
            if not self._bridged(pts_n[b], pts_n[b + 1], a[b + 1] - a[b], known):
                end = a[b]
                break
            tail_at = a[b + 1]
        wide = (across <= self.p.side_tol_mm * ppm + self.p.wide_mm * ppm) & (along >= a[0]) & (along <= end)
        order = np.argsort(along[wide])
        pts, aw = cand[wide][order], along[wide][order]
        self._tail_from = int(np.searchsorted(aw, tail_at))
        return pts

    def _bridged(self, p0: np.ndarray, p1: np.ndarray, gap_px: float, known: np.ndarray) -> bool:
        """A break in the new line that an older line fills (the new line crossed it): not a real break."""
        if gap_px > self.p.bridge_mm * self.p.px_per_mm:
            return False
        n = int(gap_px) + 1
        rr = np.clip(np.round(np.linspace(p0[0], p1[0], n)).astype(int), 0, self._shape[0] - 1)
        cc = np.clip(np.round(np.linspace(p0[1], p1[1], n)).astype(int), 0, self._shape[1] - 1)
        inside = known[rr, cc]
        # the gap minus what the older line covers must itself be a small break
        return (n - inside.sum()) <= self.p.max_gap_mm * self.p.px_per_mm

    def _first_stroke(self, cand: np.ndarray, axis: str) -> np.ndarray | None:
        """No pen yet: the largest new piece lying along the turned knob's axis."""
        n, lab, st, _ = cv2.connectedComponentsWithStats(self._mask_of(cand).astype(np.uint8), connectivity=8)
        best, size = None, 0
        for j in range(1, n):
            w, h = st[j, cv2.CC_STAT_WIDTH], st[j, cv2.CC_STAT_HEIGHT]
            if ("x" if w >= h else "y") == axis and st[j, cv2.CC_STAT_AREA] > size:
                best, size = j, st[j, cv2.CC_STAT_AREA]
        return None if best is None else np.argwhere(lab == best)

    def _mask_of(self, pts: np.ndarray) -> np.ndarray:
        m = np.zeros(self._shape, bool)
        if len(pts):
            m[pts[:, 0], pts[:, 1]] = True
        return m

    def _estimate_from(self, start: int) -> None:
        """(Re)compute pen estimates for observations start.. from the stored moves and pieces."""
        if start == 0:
            self.pen_px, self._last_dir = (None if self._start_pen is None else self._start_pen.copy()), {}
        else:
            prev = self.results[start - 1]
            self.pen_px = None if prev["_pen_px"] is None else prev["_pen_px"].copy()
            self._last_dir = dict(prev["_last_dir"])
        del self.results[start:]
        ppm = self.p.px_per_mm
        for n_ob, ob in enumerate(self.obs[start:], start=start):
            live = [i for i in ob.pieces if not self.pieces[i].transient]
            cand = np.concatenate([self.pieces[i].pixels for i in live]) if live else np.empty((0, 2), int)
            older = self._grow(self.baseline | self._paint(i for i, q in enumerate(self.pieces) if q.first < n_ob))
            out = {"stroke_mm": None, "stroke_axis": None, "pen_mm": None, "pen_pred_mm": None,
                   "line_center_mm": None, "predicted": False, "rejected": None, "resolved_prev_pen_mm": None}
            for axis, deg in ob.moves:
                if deg == 0:
                    continue
                pred, lo, hi = self._expected(axis, deg)
                sgn = self.knobs[axis].sign * (1 if deg > 0 else -1)   # pen moves +axis (1) or -axis (-1)
                self._tail_from = 0
                pts = (self._walk(cand, axis, sgn, older) if self.pen_px is not None else self._first_stroke(cand, axis)) if len(cand) else None
                j = 1 if axis == "x" else 0
                dirn = sgn if axis == "x" else -sgn
                accepted = False
                if pts is not None and len(pts):
                    extent = (pts[:, j].max() - pts[:, j].min() + 1) / ppm
                    s = Stroke(pts, extent, axis)
                    a = pts[:, j]
                    far = pts[a == (a.max() if dirn > 0 else a.min())].mean(axis=0)
                    near = pts[a == (a.min() if dirn > 0 else a.max())].mean(axis=0)
                    # length check: from the pen to the far end (first stroke: the stroke itself)
                    if self.pen_px is not None:
                        length = max(0.0, dirn * (far[j] - self.pen_px[j]) / ppm)
                    else:
                        length = extent
                    if lo <= length <= hi or self.pen_px is None:
                        tail = pts[self._tail_from:]       # _tip walks out from the middle: not across a crossing
                        if len(tail) >= 3 * ppm:
                            ta = tail[:, j]
                            tnear = tail[ta == (ta.min() if dirn > 0 else ta.max())].mean(axis=0)
                            tip = _tip(Stroke(tail, (ta.max() - ta.min() + 1) / ppm, axis), far, tnear, ppm=ppm)
                        else:
                            tip = far
                        self.pen_px = tip
                        out.update(stroke_mm=round(extent, 2), stroke_axis=axis,
                                   line_center_mm=self.to_mm(_line_center(s)), pen_mm=self.to_mm(tip))
                        accepted = True
                    else:
                        out["rejected"] = f"length {length:.1f} mm outside {lo:.1f}-{hi:.1f}"
                if not accepted and self.pen_px is not None:   # rule 3 fallback: where the turn puts the pen
                    moved = self.pen_px.copy()
                    moved[j] += (pred if axis == "x" else -pred) * ppm
                    h, w = self._shape
                    moved[0], moved[1] = np.clip(moved[0], 0, h - 1), np.clip(moved[1], 0, w - 1)
                    self.pen_px = moved
                    out["predicted"] = True
                    out["pen_mm"] = None
                self._last_dir[axis] = 1 if deg > 0 else -1
            out["pen_pred_mm"] = None if self.pen_px is None else self.to_mm(self.pen_px)
            out["_pen_px"] = None if self.pen_px is None else self.pen_px.copy()
            out["_last_dir"] = dict(self._last_dir)
            self.results.append(out)

    def public_results(self) -> list[dict]:
        """results without the internal fields (for JSON)."""
        return [{k: v for k, v in r.items() if not k.startswith("_")} for r in self.results]

    # ---- end of run ---------------------------------------------------------------------------
    def final_drawing(self, mask: np.ndarray) -> np.ndarray:
        """What this run drew, from one capture with the arm parked: line present now, not in the baseline,
        not where a transient line was."""
        real = self._grow(self._paint(range(len(self.pieces))))
        fake = np.zeros(self._shape, bool)
        for q in self.pieces:
            if q.gone:
                r, c = np.array(q.gone).T
                fake[r, c] = True
        return mask & ~self._grow(self.baseline) & ~(self._grow(fake) & ~real)
