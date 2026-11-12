"""Toy kernel: a pure-Python model of the Etch-a-Sketch itself (no robot).

Knob angle -> stylus position with backlash, end stops and string strain, plus a
screen that records what was drawn and a proxy for shake-to-erase.

Conventions (VERIFY ON THE REAL TOY in Phase 0):
  axis "x" = left knob  = horizontal, positive knob angle moves the stylus right
  axis "y" = right knob = vertical,   positive knob angle moves the stylus up
  screen frame: origin at the bottom-left of the drawable area, millimetres.

Backlash model per axis. theta is the knob angle, phi the angle of the internal
pulley that actually drives the stylus (stylus = k * phi), e = theta - phi the gap:
  |e| <  half_dead                      -> stylus does not move (dead zone)
  half_dead <= |e| < half_dead + trans  -> reduced motion, gain ramps 0 -> 1
  |e| =  half_dead + trans              -> rigid, stylus follows the knob
Once slack is taken up in direction s this reduces to the proposal's
  y = k * (theta - s * delta) + c   with delta = half_dead + trans,
and the transition zone is what leaves a structured residual in that fit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

AXES = ("x", "y")
G = 9.81


@dataclass
class AxisParams:
    k_mm_per_deg: float = 0.085      # PLACEHOLDER: ~5 turns across the screen (inferred from Cornell 2009)
    half_dead_deg: float = 9.0       # PLACEHOLDER: Strawprojects ~10 steps of 1.8 deg dead on reversal
    transition_deg: float = 4.5      # PLACEHOLDER: Strawprojects ~5 steps of reduced motion
    travel_mm: float = 153.0         # drawable extent along this axis

    @property
    def delta_deg(self) -> float:
        """Effective half backlash once slack is fully taken up."""
        return self.half_dead_deg + self.transition_deg


@dataclass
class ToyParams:
    x: AxisParams = field(default_factory=lambda: AxisParams(travel_mm=153.0))
    y: AxisParams = field(default_factory=lambda: AxisParams(travel_mm=107.0))
    line_width_mm: float = 0.6
    snap_strain_deg: float = 180.0   # PLACEHOLDER: knob rotation into an end stop before the string snaps
    # Erase proxy (PLACEHOLDER, fit to the Phase 0 shake test): while inverted,
    # ink *= exp(-erase_rate * duration * max(0, peak_accel/g - erase_accel_threshold_g))
    erase_rate_per_s: float = 0.12
    erase_accel_threshold_g: float = 0.3

    def axis(self, name: str) -> AxisParams:
        return self.x if name == "x" else self.y


@dataclass
class Stroke:
    points: list[tuple[float, float]]
    ink: float = 1.0                 # 1 = freshly drawn, 0 = fully erased


class ToyKernel:
    def __init__(self, params: ToyParams | None = None, start_mm: tuple[float, float] | None = None):
        self.p = params or ToyParams()
        if start_mm is None:
            start_mm = (self.p.x.travel_mm / 2, self.p.y.travel_mm / 2)
        self.theta = {a: 0.0 for a in AXES}                     # knob angle, deg
        self._phi = {a: start_mm[i] / self.p.axis(a).k_mm_per_deg for i, a in enumerate(AXES)}
        self._theta_off = {a: -self._phi[a] for a in AXES}      # so the gap starts at 0 (mid dead zone)
        self.strain = {a: 0.0 for a in AXES}                    # deg wound into an end stop
        self.snapped = {a: False for a in AXES}
        self.strokes: list[Stroke] = [Stroke([self.stylus])]

    # ---- state -----------------------------------------------------------------
    @property
    def stylus(self) -> tuple[float, float]:
        return (self._phi["x"] * self.p.x.k_mm_per_deg, self._phi["y"] * self.p.y.k_mm_per_deg)

    def gap(self, axis: str) -> float:
        """Signed backlash gap in deg (ground truth, not observable on the real toy)."""
        return self.theta[axis] - self._theta_off[axis] - self.strain[axis] - self._phi[axis]

    # ---- knobs -----------------------------------------------------------------
    def turn(self, axis: str, dtheta_deg: float, step_deg: float = 0.25) -> float:
        """Turn one knob by dtheta (deg). Returns the stylus displacement along that axis (mm)."""
        ax = self.p.axis(axis)
        before = self._phi[axis]
        n = max(1, math.ceil(abs(dtheta_deg) / step_deg))
        d = dtheta_deg / n
        for _ in range(n):
            self._step(axis, ax, d)
        moved = (self._phi[axis] - before) * ax.k_mm_per_deg
        if moved != 0.0:
            self.strokes[-1].points.append(self.stylus)
        return moved

    def _step(self, axis: str, ax: AxisParams, d: float) -> None:
        self.theta[axis] += d
        if self.snapped[axis] or d == 0.0:
            return
        s = 1.0 if d > 0 else -1.0
        # Unwind string strain first when backing off an end stop.
        strain = self.strain[axis]
        if strain != 0.0 and (strain > 0) != (d > 0):
            u = min(abs(d), abs(strain))
            self.strain[axis] -= math.copysign(u, strain)
            d -= s * u
            if d == 0.0:
                return
        phi = self._phi[axis]
        theta_eff = self.theta[axis] - self._theta_off[axis] - self.strain[axis]
        e_dir = s * (theta_eff - phi)                       # gap measured in the direction of motion
        if ax.transition_deg > 0 and e_dir > ax.half_dead_deg:
            phi += s * abs(d) * min(1.0, (e_dir - ax.half_dead_deg) / ax.transition_deg)
        full = ax.delta_deg                                 # hard bound: pulley can never lag more than this
        phi = min(max(phi, theta_eff - full), theta_eff + full)
        phi_max = ax.travel_mm / ax.k_mm_per_deg
        if phi > phi_max or phi < 0.0:                      # end stop: excess rotation stretches the string
            clipped = min(max(phi, 0.0), phi_max)
            self.strain[axis] += phi - clipped
            phi = clipped
            if abs(self.strain[axis]) > self.p.snap_strain_deg:
                self.snapped[axis] = True
        self._phi[axis] = phi

    # ---- erase -----------------------------------------------------------------
    def shake(self, peak_accel_ms2: float, duration_s: float, inverted: bool) -> float:
        """Erase proxy. Returns the fraction of ink remaining (1 = nothing erased).

        Erasing does not move the stylus. Real erasing is spatially patchy; this is uniform.
        """
        before = self.ink()
        if inverted:
            drive = max(0.0, peak_accel_ms2 / G - self.p.erase_accel_threshold_g)
            factor = math.exp(-self.p.erase_rate_per_s * duration_s * drive)
            for st in self.strokes:
                st.ink *= factor
        self.strokes = [st for st in self.strokes if st.ink > 0.01 and len(st.points) > 1]
        self.strokes.append(Stroke([self.stylus]))
        return self.ink() / before if before > 0 else 0.0

    def ink(self) -> float:
        """Total ink on screen: stroke length (mm) weighted by how dark it still is."""
        return sum(st.ink * _length(st.points) for st in self.strokes)

    # ---- read-out --------------------------------------------------------------
    def mark(self) -> tuple[int, int]:
        """Bookmark the current end of the drawing, for path_since()."""
        return (len(self.strokes) - 1, len(self.strokes[-1].points) - 1)

    def path_since(self, mark: tuple[int, int]) -> np.ndarray:
        """Ground-truth stylus polyline (N x 2, mm) drawn since mark, in the current stroke."""
        si, pi = mark
        if si != len(self.strokes) - 1:
            pi = 0
        return np.array(self.strokes[-1].points[pi:], dtype=float)

    def render(self, px_per_mm: float = 4.0, overlay: np.ndarray | None = None) -> Image.Image:
        """Ideal front-on view of the screen. overlay: optional target polyline (mm) drawn in red."""
        w = round(self.p.x.travel_mm * px_per_mm)
        h = round(self.p.y.travel_mm * px_per_mm)
        bg, dark = 188, 55
        img = Image.new("RGB", (w, h), (bg, bg, bg - 6))
        dr = ImageDraw.Draw(img)

        def px(pt):
            return (pt[0] * px_per_mm, h - 1 - pt[1] * px_per_mm)

        lw = max(1, round(self.p.line_width_mm * px_per_mm))
        for st in self.strokes:
            if len(st.points) > 1:
                v = round(bg + (dark - bg) * st.ink)
                dr.line([px(q) for q in st.points], fill=(v, v, v), width=lw, joint="curve")
        if overlay is not None and len(overlay) > 1:
            dr.line([px(q) for q in overlay], fill=(220, 40, 40), width=1)
        return img


def _length(points) -> float:
    if len(points) < 2:
        return 0.0
    a = np.asarray(points, dtype=float)
    return float(np.linalg.norm(np.diff(a, axis=0), axis=1).sum())
