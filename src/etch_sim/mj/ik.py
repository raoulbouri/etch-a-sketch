"""Damped least-squares IK for the arm: put a site at a point with its z axis along a direction.

Rotation about the tool axis is left free (5 constraints, 6 joints); for a tool pointing straight
down that free rotation is joint 6, which is the joint that spins the knob.
"""
from __future__ import annotations

import mujoco
import numpy as np

from . import gen3

SEEDS_DEG = [gen3.HOME_DEG, [0, 20, -110, 0, -50, 90], [0, -10, -120, 0, -70, 90], [0, 40, -90, 0, -50, 90]]


def solve(model, pos, z_axis=(0.0, 0.0, -1.0), site: str = "gripper_pinch", seeds_deg=SEEDS_DEG,
          tol_mm: float = 0.2, tol_deg: float = 0.2, iters: int = 300):
    """Returns (q_rad, pos_err_mm, axis_err_deg) for the best seed, or (None, ...) if none converged."""
    d = mujoco.MjData(model)
    sid = model.site(site).id
    qadr = np.array([model.joint(j).qposadr[0] for j in gen3.JOINTS])
    dadr = np.array([model.joint(j).dofadr[0] for j in gen3.JOINTS])
    lo, hi = model.jnt_range[[model.joint(j).id for j in gen3.JOINTS]].T
    limited = model.jnt_limited[[model.joint(j).id for j in gen3.JOINTS]].astype(bool)
    z_t = np.asarray(z_axis, float) / np.linalg.norm(z_axis)
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    best = (None, np.inf, np.inf)
    for seed in seeds_deg:
        q = np.deg2rad(seed).astype(float)
        for _ in range(iters):
            d.qpos[qadr] = q
            mujoco.mj_kinematics(model, d)
            mujoco.mj_comPos(model, d)
            z = d.site_xmat[sid].reshape(3, 3)[:, 2]
            e_p = np.asarray(pos) - d.site_xpos[sid]
            e_r = np.cross(z, z_t)
            if np.dot(z, z_t) < 0:                      # more than 90 deg off: keep the error from vanishing
                e_r = e_r / (np.linalg.norm(e_r) + 1e-9)
            err = np.concatenate([e_p, 0.3 * e_r])
            mujoco.mj_jacSite(model, d, jp, jr, sid)
            J = np.vstack([jp[:, dadr], 0.3 * jr[:, dadr]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
            q = q + np.clip(dq, -0.2, 0.2)
            q = np.where(limited, np.clip(q, lo, hi), q)
        d.qpos[qadr] = q
        mujoco.mj_kinematics(model, d)
        z = d.site_xmat[sid].reshape(3, 3)[:, 2]
        pe = float(np.linalg.norm(np.asarray(pos) - d.site_xpos[sid]) * 1000)
        ae = float(np.rad2deg(np.arccos(np.clip(np.dot(z, z_t), -1, 1))))
        if pe < tol_mm and ae < tol_deg:
            return q, pe, ae
        if pe + ae < best[1] + best[2]:
            best = (None, pe, ae)
    return best
