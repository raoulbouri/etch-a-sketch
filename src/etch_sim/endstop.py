"""End-stop protection before a knob is turned: never drive the pen into the edge of the screen.

When the stylus reaches an edge the knob can still turn, but it winds the drive string against the stop, and a few
more turns can snap it. At an edge only the direction away from it is safe.

TravelGuard keeps, per axis, the pen position (mm from the left / bottom edge of the drawable area) and how uncertain
it is. The uncertainty starts from how the position was obtained (by eye ~5 mm, camera ~1.5 mm) and grows with every
move (mm-per-degree error) and every direction change (backlash). A move is refused if its end could come within
`margin_mm` of an edge. After a torque stop (the arm felt the edge) the axis is pinned to that edge and only the way
back is allowed. GuardedDrawer applies it to every move of OpenLoopDrawer.

Why the D435 matters here (not built yet; the plan):
Open loop, the uncertainty grows ~2 % of every mm drawn plus ~1 mm per direction change, so the guard has to refuse
large drawings (a 60 mm spiral from the middle ends at +-24 mm; an 84 mm one is refused). The torque backstop
(aims-kinova, 0.7 Nm) only trips at roughly 100 N of string tension and the wrist camera saturates on the screen, so
neither can tell where the pen is. The scene camera can: on real D435 frames sketch_extract + run_tracker located the
pen after each knob turn to 0.2-0.8 mm rms (hand calibration, 2026-11-22). Wiring:
  1. D435 on a Linux laptop (never on the Mac: CLAUDE.md rule 15) running `sh scripts/rs_capture.sh serve`; from the
     Mac `ssh -N -L 8766:127.0.0.1:8766 <user>@<laptop>`, so RealSenseServer (Harish's PR #1, real/camera.py) reads
     frames at localhost:8766. Screen corners clicked once per tripod position (analyze_sketch.py --pick).
  2. After every line (or every few), grab a burst; run_tracker puts the pen at the tip of the new line, in the same
     screen frame as this guard (mm from the left / bottom edge); call guard.anchor(axis, pos, ~1.5 mm). The
     uncertainty then stays small and the whole screen becomes usable.
  3. The same check detects trouble the arm cannot feel: a knob turn whose line did not grow as predicted (run_tracker
     falls back to "predicted") means the fingers slipped or the pen is at a stop; treat it like hit_stop and end
     the drawing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .draw import Calib, OpenLoopDrawer, grasp_or_stop, turn_or_stop


class EndStopRisk(RuntimeError):
    """A move was refused because it could reach an end stop. Nothing was turned."""


@dataclass
class GuardParams:
    travel_mm: dict = field(default_factory=lambda: {"x": 153.0, "y": 107.0})   # drawable area (kernel.ToyParams)
    margin_mm: float = 5.0             # keep this far from an edge even in the worst case
    k_rel_err: float = 0.02            # mm-per-degree uncertainty: hand calibration ~1 %, doubled
    backlash_mm: float = 1.0           # extra uncertainty per direction change (backlash compensation residual)


class TravelGuard:
    def __init__(self, params: GuardParams | None = None):
        self.p = params or GuardParams()
        self.pos: dict = {}
        self.unc: dict = {}
        self.last_dir: dict = {"x": 0.0, "y": 0.0}
        self.pinned: dict = {"x": None, "y": None}      # "low" / "high" after a torque stop at that edge

    def anchor(self, axis: str, pos_mm: float, unc_mm: float) -> None:
        """Pen position measured (by eye, camera, or homing) with its uncertainty."""
        self.pos[axis], self.unc[axis] = float(pos_mm), float(unc_mm)
        self.pinned[axis] = None

    def _after(self, axis: str, mm: float) -> tuple[float, float]:
        s = 1.0 if mm > 0 else -1.0
        unc = self.unc[axis] + abs(mm) * self.p.k_rel_err
        if self.last_dir[axis] and s != self.last_dir[axis]:
            unc += self.p.backlash_mm
        return self.pos[axis] + mm, unc

    def check(self, axis: str, mm: float) -> None:
        """Raise EndStopRisk if moving `mm` along `axis` could reach an edge."""
        if mm == 0:
            return
        if axis not in self.pos:
            raise EndStopRisk(f"{axis}: pen position unknown; anchor it (by eye, camera or homing) before drawing")
        pin = self.pinned[axis]
        if (pin == "low" and mm < 0) or (pin == "high" and mm > 0):
            raise EndStopRisk(f"{axis}: at the {pin} end stop; only moves away from it are allowed")
        end, unc = self._after(axis, mm)
        lo, hi = self.p.margin_mm + unc, self.p.travel_mm[axis] - self.p.margin_mm - unc
        if not lo <= end <= hi:
            raise EndStopRisk(f"{axis}: move {mm:+.1f} mm would end at {end:.1f} +- {unc:.1f} mm; allowed "
                              f"{lo:.1f}..{hi:.1f} mm (margin {self.p.margin_mm:g} mm)")

    def allowed(self, axis: str) -> tuple[float, float]:
        """Largest safe move each way (mm, <= 0 and >= 0) from here, ignoring the backlash term."""
        k = 1 + self.p.k_rel_err
        lo = -(self.pos[axis] - self.p.margin_mm - self.unc[axis]) / k
        hi = (self.p.travel_mm[axis] - self.p.margin_mm - self.unc[axis] - self.pos[axis]) / k
        if self.pinned[axis] == "low":
            lo = 0.0
        if self.pinned[axis] == "high":
            hi = 0.0
        return min(lo, 0.0), max(hi, 0.0)

    def commit(self, axis: str, mm: float) -> None:
        if mm == 0:
            return
        self.pos[axis], self.unc[axis] = self._after(axis, mm)
        self.last_dir[axis] = 1.0 if mm > 0 else -1.0
        if self.pinned[axis] is not None and ((self.pinned[axis] == "low") == (mm > 0)):
            self.pinned[axis] = None                    # moved away from the stop

    def hit_stop(self, axis: str, direction: float) -> None:
        """The arm felt the end stop while turning in `direction` (+1 / -1): the pen is at that edge."""
        high = direction > 0
        self.pos[axis] = self.p.travel_mm[axis] if high else 0.0
        self.unc[axis] = self.p.backlash_mm
        self.pinned[axis] = "high" if high else "low"
        self.last_dir[axis] = 1.0 if high else -1.0


class GuardedDrawer(OpenLoopDrawer):
    """OpenLoopDrawer that checks every move with a TravelGuard before turning a knob.

    Near an edge (the move's end could come within `slow_zone_mm` of it) the knob turns in steps of at most
    `edge_step_deg` at `slow_dps` (if the robot has a `speed_dps_override`, as RealBackend does), so a stop reached
    unexpectedly is met slowly and by at most one step. A torque stop reported by the arm (TORQUE_LIMIT) pins the
    axis to that edge and ends the drawing."""

    def __init__(self, robot, calib: Calib, guard: TravelGuard, slow_zone_mm: float = 15.0, edge_step_deg: float = 15.0,
                 slow_dps: float = 5.0, **kw):
        super().__init__(robot, calib, **kw)
        self.guard = guard
        self.slow_zone_mm, self.edge_step_deg, self.slow_dps = slow_zone_mm, edge_step_deg, slow_dps
        self.slow_steps = 0                               # knob steps taken in slow mode (for logs and tests)

    def near_edge(self, axis, mm: float) -> bool:
        end, unc = self.guard._after(axis, mm)
        return end - unc < self.slow_zone_mm or end + unc > self.guard.p.travel_mm[axis] - self.slow_zone_mm

    def preload(self) -> None:
        for axis in ("x", "y"):
            self.guard.check(axis, (2 * self.calib.delta_deg[axis] + self.margin) * self.calib.k_mm_per_deg[axis])
        super().preload()
        for axis in ("x", "y"):                          # preload moves the pen by up to this much in +
            self.guard.commit(axis, (2 * self.calib.delta_deg[axis] + self.margin) * self.calib.k_mm_per_deg[axis])

    def move(self, axis, mm: float) -> None:
        if mm == 0:
            return
        self.guard.check(axis, mm)
        if not self.near_edge(axis, mm):
            super().move(axis, mm)
            self.guard.commit(axis, mm)
            return
        s = 1.0 if mm > 0 else -1.0
        deg = mm / self.calib.k_mm_per_deg[axis]
        if self.compensate and s != self.last_dir[axis]:
            deg += s * 2 * self.calib.delta_deg[axis]
        n = max(1, math.ceil(abs(deg) / self.edge_step_deg))
        grasp_or_stop(self.robot, axis)
        had = hasattr(self.robot, "speed_dps_override")
        old = getattr(self.robot, "speed_dps_override", None)
        if had:
            self.robot.speed_dps_override = self.slow_dps
        try:
            for _ in range(n):
                try:
                    turn_or_stop(self.robot, deg / n)
                except Exception as e:                      # the arm felt the stop: pin the axis, then stop drawing
                    if "TORQUE_LIMIT" in str(e):
                        self.guard.hit_stop(axis, s)
                    raise
                self.guard.commit(axis, mm / n)
                self.slow_steps += 1
        finally:
            if had:
                self.robot.speed_dps_override = old
        self.last_dir[axis] = s
