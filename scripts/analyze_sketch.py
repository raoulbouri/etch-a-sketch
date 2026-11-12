"""Extract the etched sketch from a RealSense capture (scripts/rs_capture.sh). No arm, no camera access.

    uv run python scripts/analyze_sketch.py                          # newest capture, RGB frames
    uv run python scripts/analyze_sketch.py test_images/rs/<dir> --source ir
    uv run python scripts/analyze_sketch.py --pick                    # click the 4 screen corners once
    uv run python scripts/analyze_sketch.py --image photo.png         # any single image

Screen corners: hand-picked ones from hardware/scene_camera_screen.json when they were picked on the
same source (rgb/ir) at the same image size, otherwise found automatically. Check corners.png either way.
Output: test_images/analysis/<capture>_<source>/ (corners.png, panel.png, mask.png, stats.json).
panel.png: rectified | lighting-flattened | line response | mask (red = line, blue = glare, ignored).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from etch_sim import sketch_extract as S

ROOT = Path(__file__).resolve().parents[1] / "test_images"


def pick_corners(img_bgr: np.ndarray) -> np.ndarray:
    """Click TL, TR, BR, BL of the drawable screen. r = restart, Enter = accept, q = abort."""
    scale = min(1.0, 1600 / img_bgr.shape[1])
    show = cv2.resize(img_bgr, None, fx=scale, fy=scale)
    pts: list[tuple[float, float]] = []

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x / scale, y / scale))

    win = "click TL, TR, BR, BL of the screen; r restart, Enter accept, q abort"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_click)
    while True:
        v = show.copy()
        for i, (x, y) in enumerate(pts):
            cv2.circle(v, (int(x * scale), int(y * scale)), 4, (0, 0, 255), -1)
            cv2.putText(v, "TL TR BR BL".split()[i], (int(x * scale) + 6, int(y * scale) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow(win, v)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("r"):
            pts.clear()
        elif k == ord("q"):
            raise SystemExit("aborted")
        elif k in (13, 10) and len(pts) == 4:
            break
    cv2.destroyAllWindows()
    return S.order_corners(np.array(pts))


def load_frames(src: Path, source: str) -> list[np.ndarray]:
    """RGB (or grey for IR) frames of a burst, or one image."""
    files = [src] if src.is_file() else sorted(src.glob(f"{source}_[0-9][0-9].png"))
    if not files:
        raise SystemExit(f"no {source}_NN.png frames in {src}")
    out = []
    for f in files:
        im = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        out.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im.ndim == 3 else im)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", nargs="?", type=Path, help="capture folder (default: newest in test_images/rs)")
    ap.add_argument("--image", type=Path)
    ap.add_argument("--source", choices=["rgb", "ir"], default="rgb")
    ap.add_argument("--pick", action="store_true", help="click the screen corners and save them")
    ap.add_argument("--auto", action="store_true", help="ignore saved corners, detect automatically")
    a = ap.parse_args()

    if a.image:
        src, name = a.image, a.image.stem
    else:
        src = a.capture or max((ROOT / "rs").glob("*/"), key=lambda p: p.name, default=None)
        if src is None:
            raise SystemExit("no captures in test_images/rs; run: sudo sh scripts/rs_capture.sh")
        name = f"{src.name}_{a.source}"
    frames = load_frames(src, a.source)
    first = frames[0]
    h, w = first.shape[:2]
    first_bgr = cv2.cvtColor(first, cv2.COLOR_RGB2BGR) if first.ndim == 3 else cv2.cvtColor(first, cv2.COLOR_GRAY2BGR)

    corners, how = None, ""
    if a.pick:
        corners, how = pick_corners(first_bgr), "picked"
        S.save_screen_corners(corners, (w, h))
        d = json.loads(S.SCREEN_FILE.read_text())
        S.SCREEN_FILE.write_text(json.dumps({**d, "source": a.source}, indent=2))   # RGB and IR views differ
        print(f"saved corners to {S.SCREEN_FILE}")
    elif not a.auto and S.SCREEN_FILE.exists():
        saved = json.loads(S.SCREEN_FILE.read_text())
        if tuple(saved.get("image_wh", ())) == (w, h) and saved.get("source", "rgb") == a.source:
            corners, how = S.load_screen_corners(), "saved"
    if corners is None:
        gray = first if first.ndim == 2 else cv2.cvtColor(first, cv2.COLOR_RGB2GRAY)
        corners, how = S.find_screen(gray), "auto"
    out = ROOT / "analysis" / name
    out.mkdir(parents=True, exist_ok=True)
    if corners is None:
        cv2.imwrite(str(out / "corners.png"), first_bgr)
        raise SystemExit(f"screen not found automatically; run with --pick. Image: {out / 'corners.png'}")

    view = first_bgr.copy()
    cv2.polylines(view, [corners.astype(np.int32)], True, (0, 255, 0), 2)
    cv2.imwrite(str(out / "corners.png"), view)
    p = S.ExtractParams()
    if S.SCREEN_FILE.exists():
        p.rotate_180 = bool(json.loads(S.SCREEN_FILE.read_text()).get("rotate_180", False))
    ex = S.extract(frames, corners, p)
    cv2.imwrite(str(out / "panel.png"), S.debug_panel(ex))
    cv2.imwrite(str(out / "mask.png"), ex.mask.astype(np.uint8) * 255)
    side = float(np.linalg.norm(corners[1] - corners[0]))
    stats = {"frames": len(frames), "corners": how, "corners_px": corners.round(1).tolist(),
             "screen_width_px": round(side, 1), "camera_px_per_mm": round(side / S.ToyGeometry().screen_mm[0], 2),
             **{k: round(v, 5) if isinstance(v, float) else v for k, v in ex.stats.items()}}
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
