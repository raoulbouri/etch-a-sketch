"""Printable sheet of the board's fiducial markers: ArUco DICT_4X4_50, IDs 0-3, 30 mm cut-out.

    uv run python scripts/make_marker_sheet.py                    # both papers -> hardware/aruco_markers_30mm_{a4,letter}.pdf
    uv run python scripts/make_marker_sheet.py --cut-mm 25 --paper a4

Same dictionary and IDs as the simulation and perception code (etch_sim.mj.toy_model.ARUCO_DICT,
ToyGeometry.marker_xy_mm). The bit patterns come from OpenCV's dictionary and are drawn as vector
rectangles, so the printed squares are exact (no resampled image).

Each tile is cut_mm square: the black square (3/4 of the tile, 22.5 mm for a 30 mm tile) plus a one-cell white
quiet zone, which the detector needs, so cut on the dashed line and not inside it. A small grey "TOP" mark sits
in the quiet zone. Stick every marker with TOP towards the top of the screen: the board homography assumes
upright markers. Print at 100 % ("actual size", no "fit to page") on matte paper, and check the 100 mm bar.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DICT = cv2.aruco.DICT_4X4_50          # = etch_sim.mj.toy_model.ARUCO_DICT
IDS = (0, 1, 2, 3)
PT = 72 / 25.4                        # PDF points per mm
PAPERS = {"a4": (210.0, 297.0), "letter": (215.9, 279.4)}   # mm


def black_mm(cut_mm: float) -> float:
    """Black square side for a tile of cut_mm: 6 marker cells plus one quiet-zone cell each side = 8 cells."""
    return cut_mm * 6 / 8


def marker_cells(marker_id: int) -> np.ndarray:
    """6 x 6 bool grid, True = black, row 0 at the top as printed (includes the black border)."""
    img = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(DICT), marker_id, 6)
    return img < 128


def _rect(x, y, w, h) -> str:
    return f"{x * PT:.3f} {y * PT:.3f} {w * PT:.3f} {h * PT:.3f} re"


def _text(x, y, size_pt, s, centre=True) -> str:
    w = 0.5 * size_pt * len(s) / PT if centre else 0.0     # rough Helvetica width, mm
    s = s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return f"BT /F1 {size_pt} Tf {(x - w / 2) * PT:.3f} {y * PT:.3f} Td ({s}) Tj ET"


def tile(cx: float, cy: float, marker_id: int, cut: float) -> list[str]:
    """Drawing commands for one cut_mm tile centred at (cx, cy) mm, page origin bottom-left."""
    size = black_mm(cut)
    cell = size / 6
    x0, y0 = cx - size / 2, cy - size / 2
    ops = ["0 g", _rect(x0, y0, size, size), "f", "1 g"]
    for r, row in enumerate(marker_cells(marker_id)):
        for c, black in enumerate(row):
            if not black:
                ops += [_rect(x0 + c * cell, y0 + (5 - r) * cell, cell, cell), "f"]
    h = cut / 2                                             # cut line = outer edge of the quiet zone
    ops += ["0.5 G 0.3 w [2 1.5] 0 d", _rect(cx - h, cy - h, cut, cut), "S", "[] 0 d"]
    # orientation mark in the top quiet zone: small light-grey triangle + "TOP"
    ty, th = cy + size / 2 + cell * 0.25, cell * 0.45
    ops += ["0.72 g",
            f"{(cx - 5.0) * PT:.3f} {ty * PT:.3f} m {(cx - 3.4) * PT:.3f} {ty * PT:.3f} l "
            f"{(cx - 4.2) * PT:.3f} {(ty + th) * PT:.3f} l f",
            _text(cx + 0.8, ty, 4.5, f"TOP {marker_id}")]
    ops += ["0 g", _text(cx, cy - h - 4.5, 8, f"ID {marker_id}")]
    return ops


def page_ops(cut: float, paper: str) -> list[str]:
    W, H = PAPERS[paper]
    size = black_mm(cut)
    pitch = cut + 20.0
    xs = [W / 2 + (i - 1.5) * pitch for i in range(4)]     # one row of four tiles
    y = H - 50 - cut / 2
    ops = ["0 g", _text(W / 2, H - 22, 14, "Etch-a-Sketch board markers"),
           _text(W / 2, H - 30, 9, f"ArUco DICT_4X4_50, IDs 0-3. Cut-out {cut:g} mm, black square {size:g} mm. "
                                   f"{paper.upper() if paper == 'a4' else 'US Letter'}, print at 100 %.")]
    for x, mid in zip(xs, IDS):
        ops += tile(x, y, mid, cut)
    bx, by = W / 2 - 50, y - cut / 2 - 28                   # scale bar: 100 mm, 10 mm ticks
    ops += ["0 g", _rect(bx, by, 100, 0.6), "f"]
    for k in range(11):
        ops += [_rect(bx + 10 * k - 0.15, by, 0.3, 4.0 if k % 5 == 0 else 2.5), "f"]
    ops += [_text(W / 2, by - 6, 9, f"This bar must measure 100 mm, each cut-out {cut:g} mm and each black "
                                    f"square {size:g} mm. If not, reprint at 100 %.")]
    notes = ["Cut on the dashed line. Keep the white border around the black square: the detector needs it.",
             "Stick all four flat, with TOP towards the top of the screen (rotated markers give a wrong board pose).",
             "Suggested spots: near the four corners of the screen, all on the same flat face of the toy.",
             f"Afterwards measure each marker centre in the screen frame (mm) for ToyGeometry.marker_xy_mm; "
             f"marker_mm = {size:g}."]
    for j, s in enumerate(notes):
        ops.append(_text(18, by - 18 - 5.5 * j, 8.5, s, centre=False))
    return ops


def write_pdf(path: Path, ops: list[str], paper: str) -> None:
    W, H = (v * PT for v in PAPERS[paper])
    stream = "\n".join(ops).encode("latin-1")
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W:.3f} {H:.3f}] "
            f"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>".encode(),
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"]
    out, offs = bytearray(b"%PDF-1.4\n"), []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offs)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cut-mm", type=float, default=30.0, help="side of the cut-out tile (black square = 3/4 of it)")
    ap.add_argument("--paper", choices=[*PAPERS, "both"], default="both")
    a = ap.parse_args()
    for paper in PAPERS if a.paper == "both" else [a.paper]:
        out = ROOT / "hardware" / f"aruco_markers_{a.cut_mm:g}mm_{paper}.pdf"
        write_pdf(out, page_ops(a.cut_mm, paper), paper)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
