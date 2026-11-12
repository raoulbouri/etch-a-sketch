"""Deterministic drawing score: symmetric chamfer distance between polylines, in mm."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def resample(poly, spacing_mm: float = 0.25) -> np.ndarray:
    """Points every spacing_mm along a polyline (N x 2)."""
    p = np.asarray(poly, float)
    if len(p) < 2:
        return p.reshape(-1, 2)
    out = []
    for a, b in zip(p[:-1], p[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / spacing_mm)))
        out.append(a + (b - a) * np.linspace(0, 1, n, endpoint=False)[:, None])
    out.append(p[-1:])
    return np.vstack(out)


def chamfer(drawn, target, spacing_mm: float = 0.25) -> dict:
    """Mean symmetric nearest-point distance (mm), plus each direction and the worst point.

    drawn_to_target is high when stray lines were drawn; target_to_drawn when parts are missing.
    """
    d, t = resample(drawn, spacing_mm), resample(target, spacing_mm)
    d2t = cKDTree(t).query(d)[0]
    t2d = cKDTree(d).query(t)[0]
    return {
        "chamfer_mm": float((d2t.mean() + t2d.mean()) / 2),
        "drawn_to_target_mm": float(d2t.mean()),
        "target_to_drawn_mm": float(t2d.mean()),
        "max_mm": float(max(d2t.max(), t2d.max())),
    }
