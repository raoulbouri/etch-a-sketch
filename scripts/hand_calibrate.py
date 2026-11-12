"""Hand calibration of one knob with full turns only: measures mm per degree and backlash from the camera.

You turn the knobs by hand (a tape mark on each knob is enough to count full turns); after each turn
the scene camera takes a burst and the new piece of line is measured.

    uv run python scripts/hand_calibrate.py x            # left knob (horizontal)
    uv run python scripts/hand_calibrate.py y            # right knob (vertical)
    uv run python scripts/hand_calibrate.py x --rows 4   # more rows = better backlash (if the screen has room)
    uv run python scripts/hand_calibrate.py --replay test_images/calib/<session>   # re-measure saved captures
    ... --save                                           # also write the result to hardware/phase0_measurements.yaml

Before starting: erase the screen, and put the pen ~1.5 cm from the bottom-left corner (looking at the
toy with the knobs toward you). The plan assumes clockwise moves the pen right (left knob) and up (right
knob), as on the classic toy; if one knob is the other way round, add --flip x or --flip y.
Default plans: left knob 3 rows x 3 turns, right knob 4 rows x 2 turns.
The screen corners must be saved (scripts/analyze_sketch.py --pick) for the current camera position.

Captures come from the RealSense camera server, running for the whole session in a second terminal:
    sh scripts/rs_capture.sh serve        (Linux; see README "Real scene camera")
It opens the camera once and serves a 10-frame colour burst per request. On macOS this is blocked:
librealsense froze the Mac (see README).
"""
from __future__ import annotations

import argparse
import json
import re
import socket
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from etch_sim import sketch_extract as S
from etch_sim import turn_measure as T

ROOT = Path(__file__).resolve().parents[1]
CALIB = ROOT / "test_images" / "calib"
YAML = ROOT / "hardware" / "phase0_measurements.yaml"
KNOB = {"x": "LEFT knob (horizontal)", "y": "RIGHT knob (vertical)"}


SERVER = ("127.0.0.1", 8766)
START_SERVER = "sh scripts/rs_capture.sh serve"


def server_call(req: dict, timeout_s: float = 30.0) -> dict:
    with socket.create_connection(SERVER, timeout=timeout_s) as s, s.makefile("rw") as f:
        f.write(json.dumps(req) + "\n")
        f.flush()
        line = f.readline()
    if not line:
        raise RuntimeError("camera server closed the connection")
    return json.loads(line)


def check_server() -> None:
    try:
        server_call({"cmd": "ping"}, timeout_s=3.0)
    except OSError:
        sys.exit(f"The camera server is not running. In a second terminal, from the repo root, run:\n"
                 f"    {START_SERVER}\nwait for 'camera streaming', leave it open, then rerun this.")


def capture(out: Path) -> None:
    """Ask the camera server (already holding the camera) for a 10-frame burst."""
    try:
        r = server_call({"cmd": "grab", "out": str(out.resolve()), "n": 10})
    except OSError as e:
        raise RuntimeError(f"camera server not reachable ({e}); is `{START_SERVER}` still running?")
    if not r.get("ok") or not list(out.glob("rgb_*.png")):
        raise RuntimeError(f"capture failed: {r.get('error')}")


def mask_of(step_dir: Path, corners: np.ndarray, p: S.ExtractParams) -> np.ndarray:
    frames = [cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB) for f in sorted(step_dir.glob("rgb_*.png"))]
    return S.extract(frames, corners, p).mask


def load_corners() -> tuple[np.ndarray, S.ExtractParams]:
    if not S.SCREEN_FILE.exists():
        sys.exit("no saved screen corners: capture once and run scripts/analyze_sketch.py --pick")
    return S.load_screen_corners(), S.screen_params()


def turn_words(ax: str, d: int) -> str:
    return f"{KNOB[ax]}: 1 full turn {'CLOCKWISE' if d > 0 else 'COUNTER-clockwise'} (seen from above)"


def save_overlay(step_dir: Path, mask: np.ndarray, pen_mm) -> None:
    v = np.full(mask.shape + (3,), 200, np.uint8)
    v[mask] = (0, 0, 255)
    if pen_mm is not None:
        c = (int(round(pen_mm[0] * 4)), int(round(mask.shape[0] - 1 - pen_mm[1] * 4)))
        cv2.circle(v, c, 6, (0, 160, 0), 2)
    cv2.imwrite(str(step_dir / "mask_pen.png"), v)


def write_yaml(tf: T.TurnFit) -> None:
    s = tf.summary()
    text = YAML.read_text()
    travel = 153.0 if tf.axis == "x" else 107.0
    line = (f"  {tf.axis}: {{k_mm_per_deg: {s['k_mm_per_deg']}, delta_deg: {s['delta_deg']}, "
            f"fit_rms_mm: {s['fit_rms_mm']}, turns_across_screen: {round(travel / s['mm_per_turn'], 2)}}}"
            f"   # hand full turns {time.strftime('%Y-%m-%d')}: backlash {s['backlash_deg']} deg")
    text, n = re.subn(rf"^  {tf.axis}: \{{k_mm_per_deg:.*$", line, text, flags=re.M)
    key = "left_knob_cw_moves" if tf.axis == "x" else "right_knob_cw_moves"
    text = re.sub(rf"^(\s*{key}:) \S+", rf"\1 {s['cw_moves']}", text, flags=re.M)
    if n != 1:
        sys.exit("could not find the knob line in phase0_measurements.yaml; not written")
    YAML.write_text(text)
    print(f"wrote {YAML.relative_to(ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("axis", nargs="?", choices=["x", "y"])
    ap.add_argument("--turns-per-row", type=int, help="default: 3 for x, 2 for y")
    ap.add_argument("--rows", type=int, help="default: 3 for x, 4 for y")
    ap.add_argument("--flip", action="append", choices=["x", "y"], default=[],
                    help="this knob moves the pen left/down when turned clockwise: turn it the other way")
    ap.add_argument("--replay", type=Path, help="re-measure a saved session folder (no camera)")
    ap.add_argument("--save", action="store_true", help="write the result to hardware/phase0_measurements.yaml")
    a = ap.parse_args()
    corners, p = load_corners()

    if a.replay:
        meta = json.loads((a.replay / "session.json").read_text())
        axis, plan, session, live = meta["axis"], [tuple(x) for x in meta["plan"]], a.replay, False
    else:
        if not a.axis:
            ap.error("give the knob to calibrate: x or y")
        plan = [(ax, -d if ax in a.flip else d) for ax, d in T.turn_plan(a.axis, a.turns_per_row, a.rows)]
        axis, live = a.axis, True
        check_server()
        session = CALIB / f"{time.strftime('%Y%m%d-%H%M%S')}_{axis}"
        session.mkdir(parents=True)
        (session / "session.json").write_text(json.dumps({"axis": axis, "plan": plan}, indent=1))
        print(f"Calibrating the {KNOB[axis]}: {len(plan)} turns. Session folder: {session.relative_to(ROOT)}")
        print("Erase the screen and put the pen ~1.5 cm from the bottom-left corner (knobs toward you).")
        input("Press Enter to take the starting capture... ")
        capture(session / "step_00")

    track = T.PenTrack(px_per_mm=p.px_per_mm)
    track.update(mask_of(session / "step_00", corners, p))
    results: list = []
    for i, (ax, d) in enumerate(plan, start=1):
        step = session / f"step_{i:02d}"
        if live:
            while True:
                ans = input(f"[{i}/{len(plan)}] {turn_words(ax, d)}. Then Enter (q = stop here): ").strip().lower()
                if ans == "q":
                    break
                try:
                    capture(step)
                    break
                except RuntimeError as e:
                    print(f"  {e}. Press Enter to capture again WITHOUT turning, or q to stop.")
            if ans == "q":
                plan = plan[: i - 1]
                break
        elif not step.exists():
            plan = plan[: i - 1]
            break
        mask = mask_of(step, corners, p)
        r = track.update(mask)
        if r["resolved_prev_pen_mm"] is not None and results:
            results[-1]["pen_mm"] = r["resolved_prev_pen_mm"]
        results.append(r)
        save_overlay(step, mask, r["pen_mm"])
        msg = f"  new line {r['stroke_mm']} mm along {r['stroke_axis']}" if r["stroke_mm"] else "  NO new line found"
        if live and i <= 2 and (not r["stroke_mm"] or r["stroke_mm"] < 10):
            msg += (f"\n  The pen barely moved: it may be against the edge. If clockwise moves it left/down on your"
                    f" toy, press q, erase, and rerun with --flip {ax}.")
        if r["stroke_axis"] and r["stroke_axis"] != ax:
            msg += f"  (expected along {ax}: wrong knob?)"
        if r["pen_mm"]:
            x, y = r["pen_mm"]
            msg += f"; pen at ({x:.1f}, {y:.1f}) mm"
            if min(x, y, 153 - x, 107 - y) < 8:
                msg += "  WARNING: pen within 8 mm of the edge (end stop ahead; points within 4 mm are not used)"
        print(msg)

    (session / "results.json").write_text(json.dumps({"plan": plan, "results": results}, indent=1))
    if sum(1 for (ax, _), r in zip(plan, results) if ax == axis and r["pen_mm"]) < 4:
        sys.exit("too few measured turns to fit")
    tf = T.fit_session(plan, results, axis)
    s = tf.summary()
    (session / "fit.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))
    print(f"-> one full turn moves the pen {s['mm_per_turn']} mm; a reversal loses {s['backlash_deg']} deg of knob turn; "
          f"clockwise moves the pen {s['cw_moves']}.")
    if s["excluded_steps"]:
        print(f"   steps {s['excluded_steps']} moved far less than the others and were left out (end stop or slip?)")
    if a.save:
        write_yaml(tf)


if __name__ == "__main__":
    main()
