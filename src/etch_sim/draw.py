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


class GraspFailed(RuntimeError):
    """A knob was not gripped or did not turn: the drawing stops (nothing is drawn with the wrist turning in air)."""


def grasp_or_stop(robot: Skills, axis: Axis) -> None:
    if not robot.grasp_knob(axis):
        raise GraspFailed(f"{axis} knob not gripped (missed, or the toy moved since the last re-locate)")


def turn_or_stop(robot: Skills, deg: float) -> None:
    r = robot.rotate(deg)
    if not getattr(r, "ok", True):
        raise GraspFailed(f"knob turn of {deg:+.1f} deg did not happen: {getattr(r, 'note', '')}")


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
            grasp_or_stop(self.robot, axis)
            turn_or_stop(self.robot, 2 * self.calib.delta_deg[axis] + self.margin)
            self.last_dir[axis] = 1.0

    def move(self, axis: Axis, mm: float) -> None:
        if mm == 0:
            return
        s = 1.0 if mm > 0 else -1.0
        deg = mm / self.calib.k_mm_per_deg[axis]
        if self.compensate and s != self.last_dir[axis]:
            deg += s * 2 * self.calib.delta_deg[axis]
        grasp_or_stop(self.robot, axis)
        turn_or_stop(self.robot, deg)
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


def rectangle(w_mm: float = 40.0, h_mm: float = 25.0) -> list[Move]:
    return [("x", w_mm), ("y", h_mm), ("x", -w_mm), ("y", -h_mm)]


def merge(moves: list[Move]) -> list[Move]:
    """Join consecutive moves on the same axis (each knob switch costs a release and a grasp)."""
    out: list[Move] = []
    for axis, mm in moves:
        if out and out[-1][0] == axis:
            out[-1] = (axis, out[-1][1] + mm)
        else:
            out.append((axis, mm))
    return [m for m in out if abs(m[1]) > 1e-9]


def stepped_circle(radius_mm: float = 15.0, steps_per_quadrant: int = 4) -> list[Move]:
    """Closed staircase through points on a circle, starting at its bottom and going anticlockwise.

    One knob at a time can only draw axis-aligned lines, so each chord becomes an x move then a y move.
    The circle's centre is radius_mm above the start.
    """
    n = 4 * steps_per_quadrant
    a = -np.pi / 2 + 2 * np.pi * np.arange(n + 1) / n
    pts = radius_mm * np.c_[np.cos(a), np.sin(a)]
    d = np.diff(pts, axis=0)
    d[np.abs(d) < 1e-9] = 0.0
    return merge([m for dx, dy in d for m in (("x", float(dx)), ("y", float(dy)))])


def staircase(step_mm: float = 8.0, n: int = 5) -> list[Move]:
    """Up a staircase, then straight back down and along the base: a closed outline."""
    up = [m for _ in range(n) for m in (("x", step_mm), ("y", step_mm))]
    return up + [("y", -step_mm * n), ("x", -step_mm * n)]


def spiral_out(outer_mm: float = 60.0, loops: int = 6, aspect: float = 0.75) -> list[Move]:
    """Rectangular spiral from the pen outwards (start in the middle, wind anticlockwise, end at the bottom-left corner).

    One knob at a time, so every line is horizontal or vertical: x +p, y +p*a, x -2p, y -2p*a, x +3p, ... with
    p = outer_mm / (2 * loops). After `loops` turns the outer box is outer_mm wide and outer_mm * aspect high, centred
    on the start. 4 * loops lines, each a knob switch.
    """
    p = outer_mm / (2 * loops)
    moves: list[Move] = []
    for n in range(1, 2 * loops + 1):
        s = 1.0 if n % 2 else -1.0
        moves += [("x", s * n * p), ("y", s * n * p * aspect)]
    return moves


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
