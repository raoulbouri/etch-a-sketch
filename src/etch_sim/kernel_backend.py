"""Skills backend on top of the toy kernel: no arm, no camera, runs in milliseconds.

Adds what sits between a command and the toy: gripper slip on the knob, missed
grasps, pen-measurement noise, timing, and injectable faults with a ground-truth
log (truth_log) that is kept apart from the log the agent is allowed to read (log).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .interface import Axis, EraseResult, RotateResult
from .kernel import ToyKernel


@dataclass
class Timing:                          # PLACEHOLDERS, measure in Phase 0
    switch_s: float = 8.0              # release, move to the other knob, regrasp
    grasp_s: float = 3.0               # first grasp
    knob_speed_dps: float = 30.0       # repo cap KINOVA_MAX_ANG_SPEED; joint 6 URDF limit is 70 deg/s
    photo_s: float = 0.5
    erase_overhead_s: float = 20.0     # pick up the frame, flip, put back


@dataclass
class Faults:
    slip_frac: dict = field(default_factory=lambda: {"x": 0.0, "y": 0.0})   # fraction of rotation lost
    slip_noise_frac: float = 0.0       # random extra slip, std as a fraction of each rotation
    miss_grasp_prob: float = 0.0       # gripper closes but does not engage the knob
    erase_scale: float = 1.0           # <1 makes shaking less effective (incomplete erase)
    pen_noise_mm: float = 0.2          # std of the scene-camera pen measurement


class KernelBackend:
    def __init__(self, toy: ToyKernel | None = None, timing: Timing | None = None,
                 faults: Faults | None = None, seed: int = 0):
        self.toy = toy or ToyKernel()
        self.timing = timing or Timing()
        self.faults = faults or Faults()
        self.rng = np.random.default_rng(seed)
        self.clock_s = 0.0
        self.n_switches = 0
        self.held: Axis | None = None
        self._engaged = False
        self._last_axis: Axis | None = None
        self.log: list[dict] = []          # what the planner/agent may see
        self.truth_log: list[dict] = []    # injected faults, for grading diagnoses

    def grasp_knob(self, axis: Axis) -> bool:
        if self.held == axis:
            return True
        if self.held is not None:
            self.release()
        switching = self._last_axis is not None and self._last_axis != axis
        self.clock_s += self.timing.switch_s if switching else self.timing.grasp_s
        self.n_switches += int(switching)
        self.held, self._last_axis = axis, axis
        self._engaged = self.rng.random() >= self.faults.miss_grasp_prob
        if not self._engaged:
            self.truth_log.append({"t": self.clock_s, "fault": "missed_grasp", "axis": axis})
        self.log.append({"t": self.clock_s, "skill": "grasp_knob", "axis": axis})
        return True                        # the gripper cannot tell that it missed

    def rotate(self, deg: float) -> RotateResult:
        if self.held is None:
            return RotateResult(deg, 0.0, ok=False, note="no knob held")
        axis = self.held
        self.clock_s += abs(deg) / self.timing.knob_speed_dps
        knob_deg = 0.0
        if self._engaged:
            slip = self.faults.slip_frac[axis] + abs(self.rng.normal(0.0, self.faults.slip_noise_frac))
            knob_deg = deg * (1.0 - min(1.0, slip))
            if slip > 0:
                self.truth_log.append({"t": self.clock_s, "fault": "slip", "axis": axis,
                                       "lost_deg": deg - knob_deg})
        self.toy.turn(axis, knob_deg)
        if self.toy.snapped[axis]:
            self.truth_log.append({"t": self.clock_s, "fault": "string_snapped", "axis": axis})
        self.log.append({"t": self.clock_s, "skill": "rotate", "axis": axis, "deg": deg})
        return RotateResult(deg, deg)

    def release(self) -> None:
        if self.held is not None:
            self.log.append({"t": self.clock_s, "skill": "release", "axis": self.held})
        self.held, self._engaged = None, False

    def observe_pen(self) -> tuple[float, float]:
        self.clock_s += self.timing.photo_s
        x, y = self.toy.stylus
        n = self.rng.normal(0.0, self.faults.pen_noise_mm, 2)
        return (x + n[0], y + n[1])

    def photo(self, camera: str = "scene") -> Image.Image:
        self.clock_s += self.timing.photo_s
        return self.toy.render()           # ideal front-on view; real camera effects come with MuJoCo (S1)

    def locate_board(self) -> bool:
        return True

    def erase(self, peak_accel_ms2: float = 9.0, duration_s: float = 30.0) -> EraseResult:
        self.release()
        self._last_axis = None
        self.clock_s += duration_s + self.timing.erase_overhead_s
        if self.faults.erase_scale < 1.0:
            self.truth_log.append({"t": self.clock_s, "fault": "weak_erase", "scale": self.faults.erase_scale})
        remaining = self.toy.shake(peak_accel_ms2, duration_s * self.faults.erase_scale, inverted=True)
        self.log.append({"t": self.clock_s, "skill": "erase", "ink_remaining": remaining})
        return EraseResult(remaining, duration_s)
