"""Image-to-route planner: a picture in, one continuous Manhattan move list out.

The toy draws with one pen that never lifts, and the arm turns one knob at a time, so a drawing is a
single path of horizontal ("x") and vertical ("y") runs. Every change of knob is a release, move and
regrasp (seconds each), so the planner keeps the number of axis changes low.

Pipeline:
  1. image -> line mask (threshold, or Canny edges for photos), fitted into the screen inside a margin
  2. mask -> cells on a grid of `pitch_mm`, thinned to 1 cell, made 4-connected
  3. separate pieces (and the pen's start point) joined by the shortest connector lines (MST)
  4. cells -> graph of chains between junctions; each chain's staircases coarsened within `tol_mm`
  5. edges retraced where needed so one open path covers everything (Chinese postman, greedy matching)
  6. Euler path (Hierholzer) that prefers going straight, then merged into moves

Output is a list of `draw.Move` for `OpenLoopDrawer.draw`. This module only uses the Skills-level
types (`Move`, `Calib`); it never imports a backend (CLAUDE.md rule 6).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree
from scipy.spatial import cKDTree

from .draw import Calib, Move
from .scoring import resample

_FOUR = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass
class PlanParams:
    screen_mm: tuple = (153.0, 107.0)  # PLACEHOLDER screen size (ToyParams travel); re-measure with a ruler
    margin_mm: float = 8.0             # keep clear of the ~4 mm edge strip the camera cannot see, plus slack
    pitch_mm: float = 2.0              # grid spacing: the smallest step the route takes
    tol_mm: float | None = None        # staircase coarsening tolerance; None -> 1.5 * pitch (below 1 pitch it does nothing)
    edges: bool = False                # True for photos: Canny edges instead of a dark-line threshold
    threshold: int | None = None       # grey level for line art; None -> Otsu
    min_piece_cells: int = 3           # drop specks smaller than this (cells) before joining pieces
    # Time estimate only (the planner does not drive anything). PLACEHOLDERS, same values as the toy backend's Timing.
    switch_s: float = 8.0              # release, move to the other knob, regrasp
    knob_speed_dps: float = 30.0       # aims-kinova angular speed cap
    calib: Calib = field(default_factory=lambda: Calib({"x": 0.0951, "y": 0.1001}, {"x": 10.6, "y": 8.4}))

    @property
    def tol(self) -> float:
        return 1.5 * self.pitch_mm if self.tol_mm is None else self.tol_mm


@dataclass
class Route:
    moves: list[Move]                  # for OpenLoopDrawer.draw
    start_mm: tuple[float, float]      # where the pen must be when the moves start
    polyline: np.ndarray               # N x 2 mm, the ideal path (start + every corner)
    target_mm: np.ndarray              # M x 2 mm, the fitted line pixels (what the picture asked for)
    stats: dict


# ---- 1. image -> mask ------------------------------------------------------------------------------
def load_mask(image, params: PlanParams | None = None) -> np.ndarray:
    """Boolean line mask (True = line) from a path, PIL image or array. A bool array passes through."""
    p = params or PlanParams()
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        if image.mode in ("RGBA", "LA", "P"):
            image = image.convert("RGBA")
            bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(bg, image)
        image = np.asarray(image.convert("L"))
    a = np.asarray(image)
    if a.dtype == bool:
        return a.copy()
    if a.ndim == 3:
        a = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    a = a.astype(np.uint8)
    if p.edges:
        blur = cv2.GaussianBlur(a, (5, 5), 0)
        med = float(np.median(blur))
        return cv2.Canny(blur, int(max(0, 0.66 * med)), int(min(255, 1.33 * med))) > 0
    if p.threshold is None:
        _, m = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        m = m > 0
    else:
        m = a < p.threshold
    if m.mean() > 0.5:                 # light lines on a dark background
        m = ~m
    return m


# ---- 2. mask -> thin, 4-connected grid -------------------------------------------------------------
def _thin(img: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning (8-connected skeleton) of a small bool array."""
    img = np.pad(img.astype(np.uint8), 1)
    while True:
        changed = False
        for step in (0, 1):
            P = [np.roll(np.roll(img, -dy, 0), -dx, 1) for dy, dx in
                 ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))]
            B = sum(P)
            seq = P + [P[0]]
            A = sum(((seq[k] == 0) & (seq[k + 1] == 1)).astype(np.uint8) for k in range(8))
            p2, p4, p6, p8 = P[0], P[2], P[4], P[6]
            if step == 0:
                c = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                c = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            kill = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & c
            if kill.any():
                img[kill] = 0
                changed = True
        if not changed:
            return img[1:-1, 1:-1].astype(bool)


def _four_connect(g: np.ndarray) -> np.ndarray:
    """Fill one corner cell wherever two cells touch only diagonally, so the grid is 4-connected."""
    g = g.copy()
    h, w = g.shape
    for i in range(h - 1):
        for j in range(w):
            if not g[i, j]:
                continue
            if j + 1 < w and g[i + 1, j + 1] and not g[i + 1, j] and not g[i, j + 1]:
                g[i, j + 1] = True
            if j - 1 >= 0 and g[i + 1, j - 1] and not g[i + 1, j] and not g[i, j - 1]:
                g[i, j - 1] = True
    return g


def _fit(mask: np.ndarray, p: PlanParams, origin: np.ndarray):
    """Scale the mask's bounding box into the margin box. Returns (grid, target points in mm).

    grid[i, j] is the cell at x = origin[0] + i * pitch, y = origin[1] + j * pitch.
    """
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        raise ValueError("image has no line pixels")
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    W, H = p.screen_mm
    bw, bh = W - 2 * p.margin_mm, H - 2 * p.margin_mm
    s = min(bw / (c1 - c0), bh / (r1 - r0))             # mm per image pixel
    x0 = p.margin_mm + (bw - s * (c1 - c0)) / 2
    y0 = p.margin_mm + (bh - s * (r1 - r0)) / 2
    xs = x0 + (cols - c0 + 0.5) * s
    ys = y0 + (r1 - rows - 0.5) * s                      # image rows go down, screen y goes up
    nx = int(np.floor((W - origin[0]) / p.pitch_mm)) + 1
    ny = int(np.floor((H - origin[1]) / p.pitch_mm)) + 1
    gi = np.clip(np.rint((xs - origin[0]) / p.pitch_mm).astype(int), 0, nx - 1)
    gj = np.clip(np.rint((ys - origin[1]) / p.pitch_mm).astype(int), 0, ny - 1)
    grid = np.zeros((nx, ny), bool)
    grid[gi, gj] = True
    grid = ndimage.binary_closing(grid, np.ones((2, 2), bool)) | grid   # bridge 1-cell rounding gaps
    target = np.column_stack([xs, ys])
    if len(target) > 20000:
        target = target[np.random.default_rng(0).choice(len(target), 20000, replace=False)]
    return grid, target


def _drop_specks(g: np.ndarray, min_cells: int, keep: tuple | None) -> np.ndarray:
    lab, n = ndimage.label(g, _FOUR)
    if n == 0:
        return g
    sizes = ndimage.sum(g, lab, range(1, n + 1))
    keep_ids = {i + 1 for i, s in enumerate(sizes) if s >= min_cells}
    if keep is not None and lab[keep]:
        keep_ids.add(lab[keep])
    return np.isin(lab, list(keep_ids))


# ---- 3. join pieces --------------------------------------------------------------------------------
def _join_pieces(g: np.ndarray) -> tuple[np.ndarray, int]:
    """Add L-shaped connector cells along a minimum spanning tree of the pieces. Returns (grid, n_added)."""
    lab, n = ndimage.label(g, _FOUR)
    if n <= 1:
        return g, 0
    pts = [np.argwhere(lab == k + 1) for k in range(n)]
    trees = [cKDTree(q) for q in pts]
    D = np.zeros((n, n))
    pair = {}
    for a in range(n):
        for b in range(a + 1, n):
            small, big = (a, b) if len(pts[a]) <= len(pts[b]) else (b, a)
            d, idx = trees[big].query(pts[small], p=1)
            k = int(np.argmin(d))
            D[a, b] = D[b, a] = d[k]
            pair[(small, big)] = (pts[small][k], pts[big][idx[k]])
    mst = minimum_spanning_tree(csr_matrix(D)).tocoo()
    g = g.copy()
    before = int(g.sum())
    for a, b in zip(mst.row, mst.col):
        u, v = pair.get((a, b)) or pair[(b, a)]
        i0, j0 = u
        i1, j1 = v
        for i in range(min(i0, i1), max(i0, i1) + 1):   # along x first, then y
            g[i, j0] = True
        for j in range(min(j0, j1), max(j0, j1) + 1):
            g[i1, j] = True
    return g, int(g.sum()) - before


# ---- 4. cells -> chain graph -----------------------------------------------------------------------
def _chains(g: np.ndarray, force_nodes: set):
    """Split the 4-connected cells into chains between junctions. Returns (nodes, edges as cell lists)."""
    cells = {tuple(c) for c in np.argwhere(g)}

    def nbrs(c):
        return [(c[0] + dx, c[1] + dy) for dx, dy in _DIRS if (c[0] + dx, c[1] + dy) in cells]

    deg = {c: len(nbrs(c)) for c in cells}
    nodes = {c for c in cells if deg[c] != 2} | (force_nodes & cells)
    used: set = set()
    chains = []

    def walk(a, b):
        path = [a, b]
        used.add(frozenset((a, b)))
        while path[-1] not in nodes:
            nxt = [c for c in nbrs(path[-1]) if frozenset((path[-1], c)) not in used]
            if not nxt:
                break
            used.add(frozenset((path[-1], nxt[0])))
            path.append(nxt[0])
        return path

    for a in sorted(nodes):
        for b in nbrs(a):
            if frozenset((a, b)) not in used:
                chains.append(walk(a, b))
    for c in sorted(cells):                     # loops with no junction on them
        for b in nbrs(c):
            if frozenset((c, b)) not in used:
                nodes.add(c)
                chains.append(walk(c, b))
    return nodes, chains


def _corners(pts: np.ndarray) -> np.ndarray:
    """Keep only the ends and the points where the direction changes."""
    if len(pts) <= 2:
        return pts
    d = np.diff(pts, axis=0)
    d = d / np.maximum(np.abs(d).max(axis=1, keepdims=True), 1e-12)
    keep = [0] + [k for k in range(1, len(pts) - 1) if not np.allclose(d[k - 1], d[k])] + [len(pts) - 1]
    return pts[keep]


def _seg_dist(q: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Distance from each point in q to polyline poly."""
    q = np.atleast_2d(q)
    best = np.full(len(q), np.inf)
    for a, b in zip(poly[:-1], poly[1:]):
        ab = b - a
        L = ab @ ab
        t = np.zeros(len(q)) if L == 0 else np.clip(((q - a) @ ab) / L, 0, 1)
        best = np.minimum(best, np.linalg.norm(q - (a + t[:, None] * ab), axis=1))
    return best


def simplify_manhattan(pts: np.ndarray, tol: float) -> np.ndarray:
    """Coarsen staircases: greedily replace runs of corners by one L while every corner stays within tol.

    Ends are kept. Output is still axis-aligned. tol = 0 keeps the input (corners only).
    """
    v = _corners(np.asarray(pts, float))
    if tol <= 0 or len(v) <= 3:
        return v
    out = [v[0]]
    i = 0
    while i < len(v) - 1:
        best = None
        for j in range(len(v) - 1, i, -1):
            a, b = v[i], v[j]
            sub = v[i:j + 1]
            for corner in (np.array([b[0], a[1]]), np.array([a[0], b[1]])):
                L = np.array([a, corner, b])
                dev = max(_seg_dist(sub[1:-1], L).max() if j > i + 1 else 0.0,
                          float(_seg_dist(corner, sub)[0]))
                if dev <= tol and (best is None or dev < best[1]):
                    best = (corner, dev)
            if best is not None:
                break
        corner = best[0]
        if not (np.allclose(corner, out[-1]) or np.allclose(corner, v[j])):
            out.append(corner)
        out.append(v[j])
        i = j
    return _corners(np.array(out))


# ---- 5 + 6. postman and Euler path -----------------------------------------------------------------
def _euler_route(nodes: list, edges: list, start: int | None):
    """edges: list of (u, v, pts). Returns (sequence of point arrays, retrace length in mm)."""
    n = len(nodes)
    deg = np.zeros(n, int)
    best: dict = {}
    for k, (u, v, pts) in enumerate(edges):
        deg[u] += 1
        deg[v] += 1
        L = float(np.abs(np.diff(pts, axis=0)).sum())
        key = (min(u, v), max(u, v))
        if u != v and (key not in best or L < best[key][1]):
            best[key] = (k, L)
    W = np.zeros((n, n))
    for (u, v), (_, L) in best.items():
        W[u, v] = W[v, u] = max(L, 1e-6)
    odd = [i for i in range(n) if deg[i] % 2]
    dist, pred = (dijkstra(csr_matrix(W), indices=odd, return_predecessors=True) if odd else (None, None))
    row = {o: r for r, o in enumerate(odd)}

    if start is not None and start in odd:
        free = [o for o in odd if o != start]
    else:
        free = list(odd)
    pairs = sorted((dist[row[a], b], a, b) for ai, a in enumerate(free) for b in free[ai + 1:])
    matched, chosen = set(), []
    for d, a, b in pairs:
        if a not in matched and b not in matched:
            matched |= {a, b}
            chosen.append((d, a, b))
    if start is None and chosen:                 # open path: leave the most expensive pair unmatched
        chosen.sort()
        _, a, b = chosen.pop()
        start = a
    if start is None:
        start = 0

    extra = list(edges)
    retrace = 0.0
    for d, a, b in chosen:
        retrace += d
        r = row[a]
        cur = b
        while cur != a:
            prv = pred[r, cur]
            k, _ = best[(min(prv, cur), max(prv, cur))]
            extra.append(edges[k])
            cur = prv

    adj: list[list[int]] = [[] for _ in range(n)]
    for k, (u, v, _) in enumerate(extra):
        adj[u].append(k)
        if v != u:
            adj[v].append(k)
    used = np.zeros(len(extra), bool)

    def oriented(k, frm):
        u, v, pts = extra[k]
        return (pts, v) if u == frm else (pts[::-1], u)

    def heading(pts):
        d = pts[-1] - pts[-2]
        return d / max(np.abs(d).max(), 1e-12)

    stack = [(start, None)]                       # (node, points walked to reach it)
    out = []
    while stack:
        v, _ = stack[-1]
        free_e = [k for k in adj[v] if not used[k]]
        if free_e:
            arrived = stack[-1][1]
            h = heading(arrived) if arrived is not None else None

            def score(k):
                pts, _ = oriented(k, v)
                d0 = pts[1] - pts[0]
                d0 = d0 / max(np.abs(d0).max(), 1e-12)
                return 0 if h is not None and np.allclose(d0, h) else 1
            k = min(free_e, key=score)
            used[k] = True
            pts, w = oriented(k, v)
            stack.append((w, pts))
        else:
            _, pts = stack.pop()
            if pts is not None:
                out.append(pts)
    out.reverse()
    return out, retrace, start


def _to_moves(poly: np.ndarray) -> list[Move]:
    moves: list[Move] = []
    for a, b in zip(poly[:-1], poly[1:]):
        dx, dy = b - a
        if abs(dx) > 1e-9 and abs(dy) > 1e-9:
            raise AssertionError(f"non-Manhattan step {a} -> {b}")
        axis, mm = ("x", float(dx)) if abs(dx) > 1e-9 else ("y", float(dy))
        if mm == 0:
            continue
        if moves and moves[-1][0] == axis and np.sign(moves[-1][1]) == np.sign(mm):
            moves[-1] = (axis, moves[-1][1] + mm)
        else:
            moves.append((axis, mm))
    return moves


def route_stats(moves: list[Move], p: PlanParams) -> dict:
    """Knob switches, reversals and a time estimate for a move list (same model as KernelBackend)."""
    switches = sum(1 for a, b in zip(moves[:-1], moves[1:]) if a[0] != b[0])
    last = {"x": 0.0, "y": 0.0}
    reversals = {"x": 0, "y": 0}
    deg = 0.0
    for axis, mm in moves:
        s = np.sign(mm)
        d = abs(mm) / p.calib.k_mm_per_deg[axis]
        if last[axis] != 0 and s != last[axis]:
            reversals[axis] += 1
            d += 2 * p.calib.delta_deg[axis]
        last[axis] = s
        deg += d
    return {
        "n_moves": len(moves),
        "switches": switches,
        "reversals": reversals["x"] + reversals["y"],
        "knob_deg": deg,
        "est_time_s": switches * p.switch_s + deg / p.knob_speed_dps,
    }


# ---- entry points ----------------------------------------------------------------------------------
def plan_route(image, params: PlanParams | None = None, pen_mm: tuple[float, float] | None = None) -> Route:
    """Plan one continuous Manhattan route for `image`.

    pen_mm: where the pen is now (e.g. `Skills.observe_pen()` after `OpenLoopDrawer.preload()`). The
    route then starts exactly there and the line from the pen to the drawing is part of the plan.
    Without it the route starts at one end of the drawing and the caller must get the pen there.
    """
    p = params or PlanParams()
    mask = load_mask(image, p)
    pen = None if pen_mm is None else np.asarray(pen_mm, float)
    origin = np.zeros(2) if pen is None else np.mod(pen, p.pitch_mm)   # pen sits exactly on a grid point
    grid, target = _fit(mask, p, origin)
    grid = _four_connect(_thin(grid))
    pen_cell = None
    if pen is not None:
        pen_cell = tuple(np.rint((pen - origin) / p.pitch_mm).astype(int))
        if not (0 <= pen_cell[0] < grid.shape[0] and 0 <= pen_cell[1] < grid.shape[1]):
            raise ValueError(f"pen {pen_mm} is outside the screen")
        grid[pen_cell] = True
    grid = _drop_specks(grid, p.min_piece_cells, pen_cell)
    n_pieces = ndimage.label(grid, _FOUR)[1]
    grid, n_conn = _join_pieces(grid)
    grid = _four_connect(grid)

    node_set, chains = _chains(grid, {pen_cell} if pen_cell is not None else set())
    nodes = sorted(node_set)
    nid = {c: i for i, c in enumerate(nodes)}

    def mm(cells):
        return origin + np.asarray(cells, float) * p.pitch_mm

    edges = [(nid[c[0]], nid[c[-1]], simplify_manhattan(mm(c), p.tol)) for c in chains]
    ink = float(sum(np.abs(np.diff(e[2], axis=0)).sum() for e in edges))
    pieces, retrace, s_node = _euler_route(nodes, edges, nid[pen_cell] if pen_cell is not None else None)

    poly = [pieces[0][0]] if pieces else [mm(nodes[s_node])]
    for pts in pieces:
        poly.extend(pts[1:])
    poly = _corners(np.array(poly))
    moves = _to_moves(poly)

    stats = route_stats(moves, p)
    dense = resample(poly, 0.25)
    stats.update({
        "pitch_mm": p.pitch_mm, "tol_mm": p.tol,
        "pieces": int(n_pieces), "connector_mm": n_conn * p.pitch_mm,
        "ink_mm": ink, "retrace_mm": float(retrace), "route_mm": float(np.abs(np.diff(poly, axis=0)).sum()),
        "target_to_route_mm": float(cKDTree(dense).query(target)[0].mean()),
        "route_to_target_mm": float(cKDTree(target).query(dense)[0].mean()),
    })
    return Route(moves, tuple(map(float, poly[0])), poly, target, stats)


def plan_for_budget(image, budget_s: float, params: PlanParams | None = None,
                    pen_mm: tuple[float, float] | None = None,
                    tols_mm=(0.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0)) -> Route:
    """Smallest staircase tolerance whose estimated time fits budget_s, at params.pitch_mm.

    Raising the tolerance cuts knob switches far more than coarsening the grid for the same error
    (circle, 2 mm pitch: tol 0 -> 109 switches, tol 3 mm -> 36 at 1.2 mm error; 4 mm pitch, tol 0 -> 51).
    If nothing fits, returns the coarsest plan; check route.stats["est_time_s"].
    """
    p = params or PlanParams()
    route = None
    for tol in tols_mm:
        route = plan_route(image, PlanParams(**{**p.__dict__, "tol_mm": tol}), pen_mm)
        if route.stats["est_time_s"] <= budget_s:
            return route
    return route
