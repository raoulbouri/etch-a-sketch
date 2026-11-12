"""Stage S0 demo: calibrate, erase, then draw square / staircase / spiral open loop on the toy kernel.

Compares controller configurations and one injected fault. Writes PNGs (target in red) and a
results table to out/s0/. Run:  uv run python scripts/s0_demo.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from etch_sim import (Calib, Faults, KernelBackend, OpenLoopDrawer, ToyKernel, ToyParams, chamfer,
                      fit_axis, polyline, spiral, square, staircase, sweep_axis)

OUT = Path(__file__).resolve().parent.parent / "out" / "s0"
SHAPES = {"square": square(), "staircase": staircase(), "spiral": spiral()}
START = (45.0, 25.0)        # pen position before each run (mm)


def calibrate(robot) -> tuple[Calib, dict]:
    fits = {}
    for axis in ("x", "y"):
        theta, pen = sweep_axis(robot, axis)
        fits[axis] = fit_axis(theta, pen)
    calib = Calib({a: f.k_mm_per_deg for a, f in fits.items()}, {a: f.delta_deg for a, f in fits.items()})
    return calib, fits


def run(shape: str, calib: Calib, compensate: bool, faults: Faults, seed: int = 0):
    toy = ToyKernel(ToyParams(), start_mm=START)
    robot = KernelBackend(toy, faults=faults, seed=seed)
    drawer = OpenLoopDrawer(robot, calib, compensate_backlash=compensate)
    drawer.preload()
    mark, start = toy.mark(), toy.stylus     # ground-truth start; perception replaces this in S2
    moves = SHAPES[shape]
    drawer.draw(moves)
    drawn, target = toy.path_since(mark), polyline(moves, start)
    score = chamfer(drawn, target)
    score["end_error_mm"] = float(np.linalg.norm(drawn[-1] - target[-1]))
    score["time_s"] = robot.clock_s
    score["switches"] = robot.n_switches
    return score, toy.render(overlay=target)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # 1. Calibrate on a fresh toy with hidden default parameters, then erase.
    toy = ToyKernel(ToyParams())
    robot = KernelBackend(toy, seed=42)
    calib, fits = calibrate(robot)
    truth = toy.p
    print("Calibration (fit vs hidden truth)")
    for a, f in fits.items():
        ax = truth.axis(a)
        print(f"  {a}: k={f.k_mm_per_deg:.4f} (true {ax.k_mm_per_deg:.4f}) mm/deg   "
              f"delta={f.delta_deg:.2f} (true {ax.delta_deg:.2f}) deg   "
              f"rms={f.rms_mm:.2f} mm  max={f.max_abs_mm:.2f} mm  used {f.n_used}, dropped {f.n_dropped}")
    toy.render().save(OUT / "calibration_sweeps.png")
    erased = robot.erase(peak_accel_ms2=9.0, duration_s=60.0)
    print(f"Erase proxy: {erased.ink_remaining:.1%} ink left after 60 s at 9 m/s^2 (placeholder model)\n")

    # 2. Draw each shape under each configuration.
    off = Calib({a: k * 1.05 for a, k in calib.k_mm_per_deg.items()}, dict(calib.delta_deg))
    configs = {
        "no backlash comp": (calib, False, Faults()),
        "calibrated": (calib, True, Faults()),
        "k 5% wrong": (off, True, Faults()),
        "3% slip on x": (calib, True, Faults(slip_frac={"x": 0.03, "y": 0.0})),
    }
    results, tiles = {}, []
    print(f"{'shape':<10} {'config':<18} {'chamfer':>8} {'max':>6} {'end err':>8} {'time':>6} {'switches':>8}")
    for shape in SHAPES:
        for name, (c, comp, faults) in configs.items():
            score, img = run(shape, c, comp, faults)
            results[f"{shape} | {name}"] = score
            ImageDraw.Draw(img).text((8, 6), f"{shape}: {name}  chamfer {score['chamfer_mm']:.2f} mm",
                                     fill=(20, 20, 120))
            tiles.append(img)
            print(f"{shape:<10} {name:<18} {score['chamfer_mm']:>7.2f}  {score['max_mm']:>5.2f} "
                  f"{score['end_error_mm']:>7.2f}  {score['time_s']:>5.0f}s {score['switches']:>7d}")
    w, h = tiles[0].size
    sheet = Image.new("RGB", (w * len(configs), h * len(SHAPES)), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % len(configs)) * w, (i // len(configs)) * h))
    sheet.save(OUT / "shapes.png")
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT}/shapes.png, calibration_sweeps.png, results.json")


if __name__ == "__main__":
    main()
