"""The skill interface every backend implements (toy kernel, MuJoCo, real arm).

Planner, calibration, closed-loop and agent code talk only to this, so the same
script runs in simulation and on hardware.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from PIL import Image

Axis = Literal["x", "y"]          # "x" = left knob (horizontal), "y" = right knob (vertical)


@dataclass
class RotateResult:
    commanded_deg: float
    wrist_deg: float              # what the wrist joint actually turned (encoder); the knob may have slipped
    ok: bool = True
    note: str = ""


@dataclass
class EraseResult:
    ink_remaining: float          # fraction of pre-erase ink still visible (from the scene camera on hardware)
    duration_s: float


class Skills(Protocol):
    clock_s: float                # elapsed task time
    n_switches: int               # number of knob changes so far

    def grasp_knob(self, axis: Axis) -> bool: ...
    def rotate(self, deg: float) -> RotateResult: ...
    def release(self) -> None: ...
    def observe_pen(self) -> tuple[float, float]:
        """Stylus position in the screen frame (mm), as the scene camera would measure it."""
        ...
    def photo(self, camera: Literal["scene", "wrist"] = "scene") -> Image.Image: ...
    def locate_board(self) -> bool: ...
    def erase(self, peak_accel_ms2: float = 9.0, duration_s: float = 30.0) -> EraseResult: ...
