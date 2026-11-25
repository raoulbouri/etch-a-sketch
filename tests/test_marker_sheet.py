"""Printable marker sheet (scripts/make_marker_sheet.py): right dictionary, IDs and size."""
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

from etch_sim.mj.toy_model import ARUCO_DICT, ToyGeometry

spec = importlib.util.spec_from_file_location("sheet", Path(__file__).resolve().parents[1] / "scripts" / "make_marker_sheet.py")
sheet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sheet)


def test_sheet_markers_match_the_perception_code():
    assert sheet.DICT == ARUCO_DICT and set(sheet.IDS) == set(ToyGeometry().marker_xy_mm)
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT), cv2.aruco.DetectorParameters())
    for mid in sheet.IDS:
        img = np.full((8, 8), 255, np.uint8)
        img[1:7, 1:7] = np.where(sheet.marker_cells(mid), 0, 255)   # one-cell quiet zone, as printed
        _, ids, _ = det.detectMarkers(cv2.resize(img, (160, 160), interpolation=cv2.INTER_NEAREST))
        assert ids is not None and ids.ravel().tolist() == [mid]


@pytest.mark.parametrize("paper", sorted(sheet.PAPERS))
def test_cut_out_and_black_square_sizes(tmp_path, paper):
    out = tmp_path / "m.pdf"
    sheet.write_pdf(out, sheet.page_ops(30.0, paper), paper)
    pdf = out.read_bytes()
    assert pdf.startswith(b"%PDF") and pdf.rstrip().endswith(b"%%EOF")
    sq = lambda mm: f" {mm * sheet.PT:.3f} {mm * sheet.PT:.3f} re".encode()
    assert pdf.count(sq(22.5)) == 4                                   # black squares
    assert pdf.count(sq(30.0)) == 4                                   # cut lines
    W, H = (v * sheet.PT for v in sheet.PAPERS[paper])
    assert f"/MediaBox [0 0 {W:.3f} {H:.3f}]".encode() in pdf
