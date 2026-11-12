"""Skills backend on the MuJoCo scene: the arm really grasps and turns the knobs, and the pen is
measured from the scene camera. Nothing here reads the toy's ground truth: the board pose comes
from the markers, the pen from the image, the knob angle from the wrist joint.

Control is joint position targets, like the real arm's high-level mode; it does not model the
Kortex controller's own trajectory generation.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from PIL import Image

from .. import perception as P
from ..interface import Axis, EraseResult, RotateResult
from . import ik
from .scene import SCREEN_PX_PER_MM, EtchScene
from .toy_model import KNOB_AXES

READY_DEG = [0.0, 20.0, -110.0, 0.0, -50.0, 90.0]     # tool down, wrist behind the board, clear of the cameras


@dataclass
class MotionParams:                    # PLACEHOLDERS until measured on the arm
    joint_speed_dps: float = 40.0      # free-space moves (URDF limits: 80 deg/s large, 70 deg/s small actuators)
    knob_speed_dps: float = 30.0       # aims-kinova angular speed cap
    safe_height_m: float = 0.045       # fingertip height above the knob top while travelling
    tip_clearance_m: float = 0.0025    # fingertip height above the toy face while gripping
    pregrasp_margin_m: float = 0.012   # finger gap minus knob diameter before descending
    settle_s: float = 0.25
    grip_s: float = 0.6


class MujocoBackend:
    def __init__(self, scene: EtchScene | None = None, camera: str = "scene_opposite",
                 params: MotionParams | None = None):
        self.s = scene or EtchScene()
        self.m, self.d = self.s.model, self.s.data
        self.camera = camera
        self.p = params or MotionParams()
        self.g = self.s.cfg.toy
        self.n_switches = 0
        self.held: Axis | None = None
        self._last_axis: Axis | None = None
        self._board: tuple[np.ndarray, np.ndarray] | None = None     # toy frame in world (R, t), from the camera
        self._tracker = P.PenTracker(SCREEN_PX_PER_MM)
        self._moves: list[tuple[str, float]] = []
        self._pinch = self.m.site("gripper_pinch").id
        self._pads = [self.m.geom(f"gripper_{side}_pad1").id for side in ("left", "right")]
        self.log: list[dict] = []
        self.frame_hook = None             # optional callable, run every 10 sim steps (e.g. a video recorder)
        self.s.reset(np.deg2rad(READY_DEG))
        self._grip_table = self._measure_gripper()
        self._t0 = self.d.time

    @property
    def clock_s(self) -> float:
        return float(self.d.time - self._t0)

    # ---- low level -------------------------------------------------------------
    def _run(self, seconds: float) -> None:
        n = max(1, round(seconds / self.m.opt.timestep))
        for k in range(n):
            mujoco.mj_step(self.m, self.d)
            if k % 10 == 9:
                self._tick()
        self.s.sync_toy()

    def _tick(self) -> None:
        self.s.sync_toy()
        if self.frame_hook is not None:
            self.frame_hook()

    def _q(self) -> np.ndarray:
        return self.d.qpos[self.s.arm_qadr].copy()

    def _move_joints(self, q_goal: np.ndarray, speed_dps: float | None = None, settle: bool = True) -> None:
        """Straight line in joint space from the current target, speed-limited, then settle."""
        q0 = self.d.ctrl[self.s.arm_act].copy()
        speed = np.deg2rad(speed_dps or self.p.joint_speed_dps)
        T = max(float(np.abs(q_goal - q0).max()) / speed, self.m.opt.timestep)
        n = max(1, round(T / self.m.opt.timestep))
        prev = q0
        for k in range(1, n + 1):
            a = k / n
            a = a * a * (3 - 2 * a) if T > 0.3 else a            # ease in and out on longer moves
            target = q0 + a * (q_goal - q0)
            self.d.ctrl[self.s.arm_act] = target
            self.d.ctrl[self.s.arm_vel_act] = (target - prev) / self.m.opt.timestep
            prev = target
            mujoco.mj_step(self.m, self.d)
            if k % 10 == 0:
                self._tick()
        self.d.ctrl[self.s.arm_vel_act] = 0.0
        if settle:
            self._run(self.p.settle_s)

    def _move_pinch(self, pos, step_m: float = 0.01) -> bool:
        """Move the fingertip reference point in a straight line to a world position, tool pointing down.

        The line is cut into short pieces because a joint-space move bows sideways by several
        millimetres, enough to land a finger on the knob.
        """
        pos = np.asarray(pos, float)
        d = mujoco.MjData(self.m)                                   # where the arm is currently told to be
        d.qpos[self.s.arm_qadr] = self.d.ctrl[self.s.arm_act]
        mujoco.mj_kinematics(self.m, d)
        start = d.site_xpos[self._pinch].copy()
        n = max(1, int(np.ceil(np.linalg.norm(pos - start) / step_m)))
        for k in range(1, n + 1):
            q, pe, ae = ik.solve(self.m, start + (pos - start) * k / n,
                                 seeds_deg=[np.rad2deg(self.d.ctrl[self.s.arm_act])], iters=60)
            if q is None:
                self.log.append({"t": self.clock_s, "event": "ik_failed", "pos": [float(v) for v in pos], "pos_err_mm": pe})
                return False
            self._move_joints(q, settle=(k == n))
        return True

    def _pinch_pos(self) -> np.ndarray:
        return self.d.site_xpos[self._pinch].copy()

    def _set_gripper(self, ctrl: float, seconds: float) -> None:
        self.d.ctrl[self.s.gripper_act] = ctrl
        self._run(seconds)

    def _pad_state(self) -> tuple[float, float]:
        """(gap between pad faces, how far the fingertips reach beyond the pinch point along the tool axis), m."""
        z = self.d.site_xmat[self._pinch].reshape(3, 3)[:, 2]
        c = [self.d.geom_xpos[i] for i in self._pads]
        gap = float(np.linalg.norm(c[0] - c[1])) - 2 * float(self.m.geom_size[self._pads[0]][1])
        tip = max(float((self.d.geom_xpos[i] - self._pinch_pos()) @ z) + float(self.m.geom_size[i][2]) for i in self._pads)
        return gap, tip

    def _measure_gripper(self) -> np.ndarray:
        """Table of (ctrl, finger gap, fingertip reach) for the free gripper, measured in the sim."""
        rows = []
        for ctrl in np.linspace(0, 255, 18):
            self._set_gripper(ctrl, 0.35)
            rows.append((ctrl, *self._pad_state()))
        self._set_gripper(0.0, 0.5)
        return np.array(rows)

    def _ctrl_for_gap(self, gap: float) -> float:
        t = self._grip_table
        return float(np.interp(gap, t[::-1, 1], t[::-1, 0]))

    def _tip_reach(self, ctrl: float) -> float:
        t = self._grip_table
        return float(np.interp(ctrl, t[:, 0], t[:, 2]))

    # ---- skills ----------------------------------------------------------------
    def locate_board(self) -> bool:
        """Toy pose in the world from the scene camera's view of the markers.

        Uses the camera's pose in the world, which in the lab comes from an extrinsic calibration.
        """
        img = self.s.render(self.camera)
        pose = P.board_pose_in_camera(P.detect_markers(img), self.g, self.s.intrinsics(self.camera))
        if pose is None:
            return False
        R_cb, t_cb = pose
        cid = self.m.camera(self.camera).id
        R_wc = self.d.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])     # OpenCV camera axes in the world
        self._board = (R_wc @ R_cb, R_wc @ t_cb + self.d.cam_xpos[cid])
        self.log.append({"t": self.clock_s, "skill": "locate_board", "toy_pos": [float(v) for v in self._board[1]]})
        return True

    def _knob_top(self, axis: Axis) -> np.ndarray:
        R, t = self._board
        return R @ self.g.knob_top_toy_m(KNOB_AXES[axis]) + t

    def grasp_knob(self, axis: Axis) -> bool:
        if self.held == axis:
            return True
        if self.held is not None:
            self.release()
        if self._board is None and not self.locate_board():
            return False
        switching = self._last_axis is not None and self._last_axis != axis
        top = self._knob_top(axis)
        knob_h, knob_d = self.g.knob_height_mm / 1000, self.g.knob_radius_mm / 500
        pre = self._ctrl_for_gap(knob_d + self.p.pregrasp_margin_m)
        closed = self._ctrl_for_gap(knob_d)
        self._set_gripper(pre, 0.0)
        travel_z = top[2] + self.p.safe_height_m + self._tip_reach(pre)
        here = self._pinch_pos()
        ok = self._move_pinch([here[0], here[1], max(here[2], travel_z)])
        ok = ok and self._move_pinch([top[0], top[1], travel_z])
        grasp_z = top[2] - knob_h + self.p.tip_clearance_m + self._tip_reach(closed)
        ok = ok and self._move_pinch([top[0], top[1], grasp_z])
        if not ok:
            return False
        self._set_gripper(255.0, self.p.grip_s)
        self.held, self._last_axis = axis, axis
        self.n_switches += int(switching)
        self.log.append({"t": self.clock_s, "skill": "grasp_knob", "axis": axis, "finger_gap_mm": self._pad_state()[0] * 1000})
        return True

    def rotate(self, deg: float) -> RotateResult:
        if self.held is None:
            return RotateResult(deg, 0.0, ok=False, note="no knob held")
        # Which way does joint 6 turn the tool about the world vertical? (kernel-positive = clockwise from above)
        jr = np.zeros((3, self.m.nv))
        mujoco.mj_jacSite(self.m, self.d, None, jr, self._pinch)
        yaw_per_q6 = jr[2, self.s.arm_dadr[5]]
        cw = -1.0 if self.g.knob_cw_positive else 1.0
        dq6 = np.deg2rad(deg) * cw * np.sign(yaw_per_q6)
        q_start = self._q()[5]
        goal = self.d.ctrl[self.s.arm_act].copy()
        goal[5] += dq6
        self._move_joints(goal, self.p.knob_speed_dps)
        wrist = float(np.rad2deg(self._q()[5] - q_start)) * cw * np.sign(yaw_per_q6)
        self._moves.append((self.held, 1.0 if deg > 0 else -1.0))
        self.log.append({"t": self.clock_s, "skill": "rotate", "axis": self.held, "deg": deg, "wrist_deg": wrist})
        return RotateResult(deg, wrist)

    def release(self) -> None:
        if self.held is None:
            return
        knob_d = self.g.knob_radius_mm / 500
        pre = self._ctrl_for_gap(knob_d + self.p.pregrasp_margin_m)
        self._set_gripper(pre, 0.4)
        top = self._knob_top(self.held)
        self._move_pinch([top[0], top[1], top[2] + self.p.safe_height_m + self._tip_reach(pre)])
        self.log.append({"t": self.clock_s, "skill": "release", "axis": self.held})
        self.held = None

    def photo(self, camera: str = "scene") -> Image.Image:
        return Image.fromarray(self.s.render(self.camera if camera == "scene" else camera))

    def rectified(self) -> np.ndarray | None:
        """Front-on screen image from the scene camera, or None if the board is not visible."""
        img = self.s.render(self.camera)
        H = P.board_homography(P.detect_markers(img), self.g)
        return None if H is None else P.rectify(img, H, self.g, SCREEN_PX_PER_MM)

    def observe_pen(self) -> tuple[float, float]:
        """Pen position (mm) from the scene camera; (nan, nan) until a first line has been seen."""
        rect = self.rectified()
        if rect is not None:
            self._tracker.update(rect, self._moves)
            self._moves = []
        pen = self._tracker.pen
        self.log.append({"t": self.clock_s, "skill": "observe_pen", "pen": pen})
        return pen if pen is not None else (float("nan"), float("nan"))

    def erase(self, peak_accel_ms2: float = 9.0, duration_s: float = 30.0) -> EraseResult:
        raise NotImplementedError("flip-and-shake in MuJoCo is a later stage; use KernelBackend.erase for now")
