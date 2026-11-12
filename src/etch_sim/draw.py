"""Open-loop drawing of Manhattan paths through the Skills interface, plus test shapes."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .interface import Axis, Skills

Move = tuple[Axis, float]         # (axis, signed distance in mm)


@dataclass
class Calib:
    """What the controller believes about the toy (from calibration), not the truth."""
    k_mm_per_deg: dict = field(default_factory=lambda: {"x": 0.085, "y": 0.085})
    delta_deg: dict = field(default_factory=lambda: {"x": 13.5, "y": 13.5})


class OpenLoopDrawer:
    def __init__(self, robot: Skills, calib: Calib, compensate_backlash: bool = True,
                 preload_margin_deg: float = 10.0):
        self.robot, self.calib = robot, calib
        self.compensate = compensate_backlash
        self.margin = preload_margin_deg
        self.last_dir: dict = {"x": 0.0, "y": 0.0}

    def preload(self) -> None:
        """Take up the slack in + on both knobs so the backlash state is known.

        Leaves a short mark; the drawing is defined to start where the pen ends up.
        """
        for axis in ("x", "y"):
            self.robot.grasp_knob(axis)
            self.robot.rotate(2 * self.calib.delta_deg[axis] + self.margin)
            self.last_dir[axis] = 1.0

    def move(self, axis: Axis, mm: float) -> None:
        if mm == 0:
            return
        s = 1.0 if mm > 0 else -1.0
        deg = mm / self.calib.k_mm_per_deg[axis]
        if self.compensate and s != self.last_dir[axis]:
            deg += s * 2 * self.calib.delta_deg[axis]
        self.robot.grasp_knob(axis)
        self.robot.rotate(deg)
        self.last_dir[axis] = s

    def draw(self, moves: list[Move]) -> None:
        for axis, mm in moves:
            self.move(axis, mm)
        self.robot.release()


def polyline(moves: list[Move], start_mm) -> np.ndarray:
    """Ideal polyline (N x 2, mm) for a move list starting at start_mm."""
    pts = [np.array(start_mm, float)]
    for axis, mm in moves:
        step = np.array([mm, 0.0]) if axis == "x" else np.array([0.0, mm])
        pts.append(pts[-1] + step)
    return np.array(pts)


def square(side_mm: float = 40.0) -> list[Move]:
    return [("x", side_mm), ("y", side_mm), ("x", -side_mm), ("y", -side_mm)]


def staircase(step_mm: float = 8.0, n: int = 5) -> list[Move]:
    """Up a staircase, then straight back down and along the base: a closed outline."""
    up = [m for _ in range(n) for m in (("x", step_mm), ("y", step_mm))]
    return up + [("y", -step_mm * n), ("x", -step_mm * n)]


def spiral(outer_mm: float = 48.0, pitch_mm: float = 8.0) -> list[Move]:
    """Rectangular spiral winding inwards."""
    moves: list[Move] = []
    dirs = [("x", 1), ("y", 1), ("x", -1), ("y", -1)]
    length, i = outer_mm, 0
    while length > 0:
        axis, s = dirs[i % 4]
        moves.append((axis, s * length))
        if i % 2 == 1:
            length -= pitch_mm
        i += 1
    return moves
