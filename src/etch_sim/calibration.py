"""Per-axis knob calibration: fit pen = k * (theta - s * delta) + c by least squares.

fit_axis() works on any (theta, pen) record, so the numbers measured by hand in
Phase 0 and the sweeps run by the arm later go through the same code.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .interface import Axis, Skills


@dataclass
class AxisFit:
    k_mm_per_deg: float
    delta_deg: float              # half the backlash; a reversal costs 2 * delta of extra rotation
    c_mm: float
    rms_mm: float
    max_abs_mm: float
    n_used: int
    n_dropped: int                # samples inside the dead zone / transition, excluded from the fit


def fit_axis(theta_deg, pen_mm, min_move_frac: float = 0.9) -> AxisFit:
    """theta_deg: cumulative knob angle at each sample; pen_mm: pen coordinate along that axis.

    A sample is kept only if the pen moved at least min_move_frac of the typical
    mm-per-degree since the previous sample, i.e. the slack was already taken up.
    """
    th = np.asarray(theta_deg, float)
    y = np.asarray(pen_mm, float)
    dth, dy = np.diff(th), np.diff(y)
    moving = dth != 0
    rate = np.zeros_like(dth)
    rate[moving] = dy[moving] / dth[moving]
    typical = np.median(np.abs(rate[moving]))
    keep = moving & (np.abs(rate) >= min_move_frac * typical)
    s = np.sign(dth[keep])
    A = np.column_stack([th[1:][keep], -s, np.ones(keep.sum())])
    (k, k_delta, c), *_ = np.linalg.lstsq(A, y[1:][keep], rcond=None)
    res = A @ np.array([k, k_delta, c]) - y[1:][keep]
    return AxisFit(float(k), float(k_delta / k), float(c), float(np.sqrt(np.mean(res**2))),
                   float(np.abs(res).max()), int(keep.sum()), int(len(dth) - keep.sum()))


def sweep_axis(robot: Skills, axis: Axis, step_deg: float = 15.0, n_steps: int = 16, n_cycles: int = 2,
               shift_deg: float = 60.0, preload_deg: float = 45.0) -> tuple[np.ndarray, np.ndarray]:
    """Zigzag sweep on one knob, measuring the pen after every step. Returns (theta_deg, pen_mm).

    The pen is only visible as the end of a fresh line, so a plain back-and-forth sweep (which
    retraces its own line) cannot be measured by a camera. After each leg the other knob is turned
    by shift_deg, so every leg draws a new line. preload_deg makes a first visible mark.
    Uses roughly step_deg * n_steps * k mm along the axis and (2 * n_cycles - 1) * shift_deg * k across it.
    """
    i = 0 if axis == "x" else 1
    other: Axis = "y" if axis == "x" else "x"
    robot.grasp_knob(axis)
    robot.rotate(preload_deg)
    theta, thetas, pens = 0.0, [0.0], [robot.observe_pen()[i]]
    legs = [d for _ in range(n_cycles) for d in (+1.0, -1.0)]
    for n, direction in enumerate(legs):
        for _ in range(n_steps):
            r = robot.rotate(direction * step_deg)
            theta += r.wrist_deg
            thetas.append(theta)
            pens.append(robot.observe_pen()[i])
        if shift_deg and n < len(legs) - 1:
            robot.grasp_knob(other)
            robot.rotate(shift_deg)
            robot.observe_pen()
            robot.grasp_knob(axis)
    return np.array(thetas), np.array(pens)
