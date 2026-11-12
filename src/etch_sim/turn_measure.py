"""Measure the pen after each knob turn from before/after line masks (real or simulated).

The pen is invisible; only the line it leaves is seen. After each turn the new piece of line is
found by subtracting the previous mask (grown by a small margin), and the pen is the end of that
new piece farthest from where the pen was before. The first stroke has no "before", so its tip is
settled by the next stroke, which starts there.

Masks are in the screen frame at px_per_mm (row 0 = top, as produced by sketch_extract with the
toy's orientation applied). Pen positions are returned in screen millimetres (x right, y up).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .calibration import AxisFit, fit_axis


@dataclass
class Stroke:
    pixels: np.ndarray          # N x 2 (row, col)
    length_mm: float            # extent along its main direction
    axis: str                   # "x" if mainly horizontal, else "y"


def new_stroke(prev: np.ndarray | None, cur: np.ndarray, px_per_mm: float,
               gap_mm: float = 1.0, min_len_mm: float = 3.0) -> Stroke | None:
    """The largest piece of line in `cur` that was not in `prev` (None if nothing new and long enough)."""
    new = cur.copy()
    if prev is not None and prev.any():
        r = max(1, int(round(gap_mm * px_per_mm)))
        grown = cv2.dilate(prev.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        new &= grown == 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(new.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    w, h = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
    if max(w, h) < min_len_mm * px_per_mm:
        return None
    return Stroke(np.argwhere(lab == i), max(w, h) / px_per_mm, "x" if w >= h else "y")


def _ends(s: Stroke) -> tuple[np.ndarray, np.ndarray]:
    """The two ends of a stroke along its main direction, as (row, col)."""
    j = 1 if s.axis == "x" else 0
    a = s.pixels[:, j]
    lo, hi = s.pixels[a == a.min()], s.pixels[a == a.max()]
    return lo.mean(axis=0), hi.mean(axis=0)


def _line_center(s: Stroke, keep: float = 0.6) -> np.ndarray:
    """(row, col) of a stroke's middle, from its central `keep` fraction only, so the corners where
    it joins other lines do not pull its position across."""
    j = 1 if s.axis == "x" else 0
    a = s.pixels[:, j]
    lo, hi = np.quantile(a, [(1 - keep) / 2, (1 + keep) / 2])
    mid = s.pixels[(a >= lo) & (a <= hi)]
    return np.median(mid if len(mid) else s.pixels, axis=0)


def _tip(s: Stroke, far: np.ndarray, near: np.ndarray, band_mm: float = 1.0, ppm: float = 4.0) -> np.ndarray:
    """Far end of a stroke: walking out from its middle, the last place where the line is still at
    least half its usual thickness.

    Thin spurs that noise grows off the end of a faint line are ignored. The result still overshoots
    the pen centre by part of the round end cap, in the direction of travel; fit_session measures
    that overshoot from the connector lines and removes it.
    """
    c = _line_center(s)
    j = 1 if s.axis == "x" else 0                      # coordinate along the stroke
    band = s.pixels[np.abs(s.pixels[:, 1 - j] - c[1 - j]) <= band_mm * ppm]
    if len(band) == 0:
        return far
    sign = 1.0 if far[j] >= near[j] else -1.0
    a = band[:, j].astype(int)
    lo = a.min()
    cover = np.bincount(a - lo)                        # line thickness at each position along it
    mid = int(round(c[j])) - lo
    typical = np.median(cover[cover > 0])
    pos = int(np.clip(mid, 0, len(cover) - 1))
    step = 1 if sign > 0 else -1
    while 0 <= pos + step < len(cover) and cover[pos + step] >= 0.5 * typical:
        pos += step
    out = c.copy()
    out[j] = pos + lo
    return out


@dataclass
class PenTrack:
    """Follows the pen through a sequence of masks, one per turn."""
    px_per_mm: float = 4.0
    height_px: int | None = None
    pen_px: np.ndarray | None = None              # (row, col) of the pen, once known
    _pending: Stroke | None = None                # first stroke, tip not yet known
    prev_mask: np.ndarray | None = None
    log: list[dict] = field(default_factory=list)

    def to_mm(self, rc: np.ndarray) -> tuple[float, float]:
        return float(rc[1] / self.px_per_mm), float((self.height_px - 1 - rc[0]) / self.px_per_mm)

    def update(self, mask: np.ndarray) -> dict:
        """Feed the mask after a turn. Returns {stroke_mm, stroke_axis, pen_mm (or None), resolved_prev}."""
        self.height_px = mask.shape[0]
        s = new_stroke(self.prev_mask, mask, self.px_per_mm)
        self.prev_mask = mask
        out: dict = {"stroke_mm": None, "stroke_axis": None, "pen_mm": None, "resolved_prev_pen_mm": None,
                     "line_center_mm": None}
        if s is None:
            self.log.append(out)
            return out
        out["stroke_mm"], out["stroke_axis"] = round(s.length_mm, 2), s.axis
        out["line_center_mm"] = self.to_mm(_line_center(s))   # where the line runs: no end-cap bias
        e1, e2 = _ends(s)
        if self.pen_px is None and self._pending is None:
            self._pending = s                              # first stroke: wait for the next one
        else:
            if self.pen_px is None:                        # the previous (first) stroke's tip is where this one starts
                p1, p2 = _ends(self._pending)
                near = lambda e: np.min(np.linalg.norm(s.pixels - e, axis=1))
                far, near_end = (p1, p2) if near(p1) < near(p2) else (p2, p1)
                self.pen_px = _tip(self._pending, far, near_end, ppm=self.px_per_mm)
                out["resolved_prev_pen_mm"] = self.to_mm(self.pen_px)
                self._pending = None
            d1, d2 = np.linalg.norm(e1 - self.pen_px), np.linalg.norm(e2 - self.pen_px)
            far, near_end = (e1, e2) if d1 > d2 else (e2, e1)
            self.pen_px = _tip(s, far, near_end, ppm=self.px_per_mm)
            out["pen_mm"] = self.to_mm(self.pen_px)
        self.log.append(out)
        return out


# ---- full-turn session -------------------------------------------------------------------------
DEFAULT_PLAN = {"x": (3, 3), "y": (2, 4)}      # (turns per row, rows): fits the 153 x 107 mm screen at ~30-40 mm/turn


def turn_plan(axis: str, turns_per_row: int | None = None, rows: int | None = None) -> list[tuple[str, int]]:
    """Knob turns for one calibration session: rows of full turns on `axis`, alternating direction,
    with one full turn of the other knob between rows ("connector"), so every row draws a new line.
    Entries: (knob axis, +1 / -1 full turn clockwise / counter-clockwise)."""
    n_default, r_default = DEFAULT_PLAN[axis]
    turns_per_row, rows = turns_per_row or n_default, rows or r_default
    other = "y" if axis == "x" else "x"
    plan: list[tuple[str, int]] = []
    for r in range(rows):
        plan += [(axis, 1 if r % 2 == 0 else -1)] * turns_per_row
        if r < rows - 1:
            plan.append((other, 1))
    return plan


@dataclass
class TurnFit:
    axis: str
    fit: AxisFit                        # pen = k * (theta - s * delta) + c, theta = knob angle (cw +)
    tip_bias_mm: list[float]            # per bracketed row: tip overshoot past the pen, along the turn direction
    n_turns: int
    excluded: list[int]                 # plan steps (1-based) that moved far less than the rest: end stop, slip

    @property
    def backlash_deg(self) -> float:
        return 2 * abs(self.fit.delta_deg)

    @property
    def cw_moves(self) -> str:
        pos = self.fit.k_mm_per_deg > 0
        return ("right" if pos else "left") if self.axis == "x" else ("up" if pos else "down")

    def summary(self) -> dict:
        k = abs(self.fit.k_mm_per_deg)
        return {"axis": self.axis, "k_mm_per_deg": round(k, 5), "mm_per_turn": round(360 * k, 2),
                "backlash_deg": round(self.backlash_deg, 2), "delta_deg": round(abs(self.fit.delta_deg), 2),
                "fit_rms_mm": round(self.fit.rms_mm, 3), "turns_used": self.fit.n_used,
                "tip_bias_mm": [round(b, 2) for b in self.tip_bias_mm],
                "excluded_steps": self.excluded, "cw_moves": self.cw_moves}


def fit_session(plan: list[tuple[str, int]], results: list[dict], axis: str, min_move_frac: float = 0.6,
                screen_mm: tuple[float, float] = (153.0, 107.0), edge_mm: float = 4.0) -> TurnFit:
    """plan[i] is the turn made before measurement results[i] (PenTrack.update output, with the
    first stroke's tip filled in). Units: screen mm, x right, y up.

    The measured tip overshoots the pen by an end-cap bias in the direction of travel; uncorrected,
    it flips sign on each reversal and reads as backlash. The connector line drawn after each row
    starts exactly at the row's last pen position, and its centre along this axis has no end-cap
    bias, so (last tip - connector centre), signed by the turn direction, measures the overshoot.
    Its mean over rows is removed from every tip, and all turns go into one least-squares fit of
    pen = k (theta - s delta) + c (calibration.fit_axis). Tips and connectors within edge_mm of the
    screen edge are left out: the extraction ignores a border, which clips them.
    """
    j = 0 if axis == "x" else 1
    rows, cur = [], []
    for i, (ax, _) in enumerate(plan):
        if ax == axis:
            cur.append(i)
        elif cur:
            rows.append((cur, i))
            cur = []
    if cur:
        rows.append((cur, None))                      # last row: no connector after it

    def inside(p) -> bool:                            # clear of the screen edge (border margin + line)
        return p is not None and edge_mm <= p[0] <= screen_mm[0] - edge_mm and edge_mm <= p[1] <= screen_mm[1] - edge_mm
    theta, th_at = 0.0, {}
    for i, (ax, d) in enumerate(plan):
        if ax == axis:
            theta += 360.0 * d
        th_at[i] = theta
    # overshoot along the direction of travel, from each bracketed row: (last tip - connector) * direction
    biases, measured = [], []
    for steps, after in rows:
        measured.append(steps)
        if after is None or after >= len(results):
            continue
        last, conn = results[steps[-1]].get("pen_mm"), results[after].get("line_center_mm")
        if inside(last) and inside(conn):
            biases.append((last[j] - conn[j]) * plan[steps[0]][1])
    overshoot = float(np.mean(biases)) if biases else 0.0
    thetas, pens, step_of = [], [], []
    for steps in measured:
        d = plan[steps[0]][1]
        for i in steps:
            if i < len(results) and inside(results[i].get("pen_mm")):
                thetas.append(th_at[i])
                pens.append(results[i]["pen_mm"][j] - d * overshoot)
                step_of.append(i + 1)                 # 1-based, as printed during the session
    thetas, pens = np.array(thetas), np.array(pens)
    dth, dy = np.diff(thetas), np.diff(pens)
    rate = np.abs(dy / np.where(dth == 0, np.nan, dth))
    typical = np.nanmedian(rate)
    excluded = [step_of[i + 1] for i in np.flatnonzero(rate < min_move_frac * typical)]
    f = fit_axis(thetas, pens, min_move_frac=min_move_frac)
    return TurnFit(axis, f, biases, len(thetas), excluded)
