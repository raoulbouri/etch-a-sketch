"""The Etch-a-Sketch as a MuJoCo body: frame, screen, two hinged knobs, four ArUco markers.

Toy frame: origin at the centre of the body's bottom face, x to the screen's right, y to the
screen's top, z out of the screen. All dimensions are PLACEHOLDERS until the real toy is measured
(hardware/phase0_measurements.yaml).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

GENERATED = Path(__file__).resolve().parents[3] / "models" / "generated"
ARUCO_DICT = cv2.aruco.DICT_4X4_50
KNOB_AXES = {"x": "left", "y": "right"}       # kernel axis -> knob


@dataclass
class ToyGeometry:
    body_mm: tuple = (229.0, 187.0, 40.0)      # PLACEHOLDER: retail listing 229 x 187 x 44 incl. knobs
    mass_kg: float = 0.454                     # retail listing
    screen_mm: tuple = (153.0, 107.0)          # drawable area (Micah Tilton's measurement)
    screen_centre_mm: tuple = (0.0, 13.0)      # PLACEHOLDER: screen sits above the body centre
    knob_radius_mm: float = 14.0               # PLACEHOLDER
    knob_height_mm: float = 14.0               # PLACEHOLDER
    knob_xy_mm: dict = field(default_factory=lambda: {"left": (-90.0, -68.0), "right": (90.0, -68.0)})  # PLACEHOLDER
    knob_friction_nm: float = 0.03             # PLACEHOLDER: torque to keep a knob turning
    knob_cw_positive: bool = True              # PLACEHOLDER: clockwise (seen from the front) = kernel-positive
    marker_mm: float = 20.0                    # black square side; printed with a one-cell white border
    marker_xy_mm: dict = field(default_factory=lambda: {0: (-97.0, 78.0), 1: (97.0, 78.0),
                                                        2: (40.0, -72.0), 3: (-40.0, -72.0)})

    @property
    def screen_origin_mm(self) -> tuple:
        """Toy-frame xy of the screen frame's origin (bottom-left of the drawable area)."""
        return (self.screen_centre_mm[0] - self.screen_mm[0] / 2, self.screen_centre_mm[1] - self.screen_mm[1] / 2)

    def marker_corners_screen_mm(self, marker_id: int) -> np.ndarray:
        """4 x 2 corners of a marker's black square in the SCREEN frame, in ArUco order
        (top-left, top-right, bottom-right, bottom-left as printed)."""
        cx, cy = self.marker_xy_mm[marker_id]
        h = self.marker_mm / 2
        ox, oy = self.screen_origin_mm
        return np.array([[cx - h, cy + h], [cx + h, cy + h], [cx + h, cy - h], [cx - h, cy - h]]) - [ox, oy]

    def marker_corners_toy_m(self, marker_id: int) -> np.ndarray:
        """4 x 3 corners of a marker's black square in the TOY frame (metres), same order."""
        xy = (self.marker_corners_screen_mm(marker_id) + self.screen_origin_mm) / 1000
        return np.column_stack([xy, np.full(4, self.body_mm[2] / 1000)])

    def knob_top_toy_m(self, side: str) -> np.ndarray:
        x, y = self.knob_xy_mm[side]
        return np.array([x, y, self.body_mm[2] + self.knob_height_mm]) / 1000


def marker_png(marker_id: int, cell_px: int = 40) -> Path:
    """ArUco 4x4 marker with a one-cell white quiet zone (8 x 8 cells in total)."""
    GENERATED.mkdir(parents=True, exist_ok=True)
    path = GENERATED / f"aruco_4x4_{marker_id}.png"
    if not path.exists():
        d = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        inner = cv2.aruco.generateImageMarker(d, marker_id, 6 * cell_px)
        img = np.full((8 * cell_px, 8 * cell_px), 255, np.uint8)
        img[cell_px:-cell_px, cell_px:-cell_px] = inner
        cv2.imwrite(str(path), img)
    return path


def toy_mjcf(g: ToyGeometry, pos, yaw_deg: float, screen_px: tuple[int, int]) -> dict[str, str]:
    """MJCF fragments {'asset', 'body'} for a free-floating toy resting at world pos (bottom-face centre)."""
    bx, by, bz = (v / 1000 for v in g.body_mm)
    sx, sy = (v / 1000 for v in g.screen_mm)
    scx, scy = (v / 1000 for v in g.screen_centre_mm)
    patch = g.marker_mm / 1000 * 8 / 6
    assets = [f'<texture name="screen" type="2d" builtin="flat" width="{screen_px[0]}" height="{screen_px[1]}" '
              f'rgb1="0.74 0.74 0.71" nchannel="3"/>',
              '<material name="screen" texture="screen" specular="0.25" shininess="0.6"/>',
              '<material name="toy_red" rgba="0.80 0.08 0.08 1" specular="0.3" shininess="0.4"/>',
              '<material name="knob_white" rgba="0.95 0.95 0.93 1"/>']
    half_yaw = np.deg2rad(yaw_deg) / 2
    body = [f'<body name="toy" pos="{pos[0]} {pos[1]} {pos[2]}" quat="{np.cos(half_yaw):.9g} 0 0 {np.sin(half_yaw):.9g}">',
            '<freejoint name="toy_free"/>',
            f'<geom name="toy_frame" type="box" size="{bx/2} {by/2} {bz/2}" pos="0 0 {bz/2}" material="toy_red" '
            f'mass="{g.mass_kg}" friction="0.8"/>',
            f'<geom name="toy_screen" type="box" size="{sx/2} {sy/2} 0.0005" pos="{scx} {scy} {bz + 0.0005}" '
            f'material="screen" contype="0" conaffinity="0" mass="0"/>',
            f'<site name="screen_origin" pos="{scx - sx/2} {scy - sy/2} {bz + 0.001}" size="0.002" group="4"/>']
    for mid, (mx, my) in g.marker_xy_mm.items():
        assets.append(f'<texture name="aruco{mid}" type="2d" file="{marker_png(mid)}"/>')
        assets.append(f'<material name="aruco{mid}" texture="aruco{mid}" specular="0" shininess="0"/>')
        body.append(f'<geom name="marker{mid}" type="box" size="{patch/2} {patch/2} 0.0003" '
                    f'pos="{mx/1000} {my/1000} {bz + 0.0003}" material="aruco{mid}" contype="0" conaffinity="0" mass="0"/>')
    r, h = g.knob_radius_mm / 1000, g.knob_height_mm / 1000
    for side, (kx, ky) in g.knob_xy_mm.items():
        body += [f'<body name="knob_{side}" pos="{kx/1000} {ky/1000} {bz}">',
                 f'<joint name="knob_{side}" axis="0 0 1" damping="0.002" frictionloss="{g.knob_friction_nm}" armature="1e-4"/>',
                 f'<geom name="knob_{side}" type="cylinder" size="{r} {h/2}" pos="0 0 {h/2}" material="knob_white" '
                 f'mass="0.01" friction="1.0"/>',
                 f'<geom type="box" size="0.001 {r*0.8} 0.0005" pos="0 0 {h + 0.0003}" rgba="0.2 0.2 0.2 1" '
                 f'contype="0" conaffinity="0" mass="0"/>',
                 f'<site name="knob_{side}_top" pos="0 0 {h}" size="0.002" group="4"/>',
                 '</body>']
    body.append("</body>")
    return {"asset": "\n".join(assets), "body": "\n".join(body)}
