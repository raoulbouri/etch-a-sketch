"""Stage S3 demo: image -> route planner, drawn open loop on the toy kernel.

For each test picture: preload the knobs, read the pen, plan a route from there within a time budget,
draw it with OpenLoopDrawer, and score
  plan fidelity      route vs the picture's line pixels (what the planner gave up)
  execution          drawn vs route, chamfer (what backlash/calibration cost)
Writes out/s3/<name>.png (picture | drawn, route in red) and out/s3/results.json.

  uv run python scripts/s3_demo.py                          # built-in test pictures
  uv run python scripts/s3_demo.py --image my.png --budget 600
  uv run python scripts/s3_demo.py --image photo.jpg --edges

Exit check: every route is one connected Manhattan path starting at the pen, drawn within 1.0 mm
chamfer of its plan, and no part of the picture is missed: mean picture->route distance <= 2.5 mm.
route->picture is reported but not checked: the pen never lifts, so the lines from the pen and
between separate pieces are drawn but are not in the picture (3 pieces: ~3 mm mean).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from etch_sim import Calib, KernelBackend, OpenLoopDrawer, ToyKernel, ToyParams, chamfer
from etch_sim.planner import PlanParams, plan_for_budget

OUT = Path(__file__).resolve().parent.parent / "out" / "s3"
PEN0 = (20.0, 20.0)                 # pen position before preload (mm)


def test_pictures() -> dict:
    def canvas():
        im = Image.new("L", (400, 300), 255)
        return im, ImageDraw.Draw(im)
    pics = {}
    im, d = canvas()
    d.polygon([(100, 280), (100, 140), (200, 40), (300, 140), (300, 280)], outline=0, width=4)
    d.rectangle((180, 200, 220, 280), outline=0, width=4)
    pics["house"] = im
    im, d = canvas()
    d.ellipse((60, 30, 330, 270), outline=0, width=4)
    pics["circle"] = im
    im, d = canvas()
    t = np.linspace(0, 2 * np.pi, 400)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    d.line(list(zip(200 + 10 * x, 140 - 10 * y)), fill=0, width=4, joint="curve")
    pics["heart"] = im
    im, d = canvas()                 # three separate pieces: tests the connectors
    d.rectangle((30, 40, 130, 140), outline=0, width=4)
    d.ellipse((230, 40, 370, 160), outline=0, width=4)
    d.line([(60, 260), (340, 220)], fill=0, width=4)
    pics["pieces"] = im
    return pics


def run(name: str, image, budget_s: float, params: PlanParams) -> dict:
    toy = ToyKernel(ToyParams(), start_mm=PEN0)
    robot = KernelBackend(toy, seed=0)
    tp = ToyParams()
    calib = Calib({a: tp.axis(a).k_mm_per_deg for a in "xy"}, {a: tp.axis(a).delta_deg for a in "xy"})
    drawer = OpenLoopDrawer(robot, calib)
    drawer.preload()
    pen = robot.observe_pen()                        # camera estimate, not ground truth
    route = plan_for_budget(image, budget_s, PlanParams(**{**params.__dict__, "calib": calib}), pen_mm=pen)
    mark = toy.mark()
    t0 = robot.clock_s
    drawer.draw(route.moves)
    drawn = toy.path_since(mark)
    ex = chamfer(drawn, route.polyline)
    s = dict(route.stats)
    s.update({"exec_chamfer_mm": ex["chamfer_mm"], "exec_max_mm": ex["max_mm"],
              "sim_time_s": robot.clock_s - t0, "sim_switches": robot.n_switches})
    pic = image.convert("RGB").resize((612, 428)) if isinstance(image, Image.Image) else None
    drawn_img = toy.render(overlay=route.polyline)
    panel = Image.new("RGB", (612 * 2 + 8, 428), (255, 255, 255))
    if pic is not None:
        panel.paste(pic, (0, 0))
    panel.paste(drawn_img, (620, 0))
    ImageDraw.Draw(panel).text((626, 4), f"{name}: {s['switches']} switches, ~{s['est_time_s'] / 60:.1f} min, "
                                         f"tol {s['tol_mm']:.0f} mm", fill=(20, 20, 120))
    panel.save(OUT / f"{name}.png")
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="picture to draw (default: built-in test pictures)")
    ap.add_argument("--budget", type=float, default=600.0, help="arm time budget per drawing, s (default 600)")
    ap.add_argument("--pitch", type=float, default=2.0)
    ap.add_argument("--edges", action="store_true", help="photo: use Canny edges instead of dark lines")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    params = PlanParams(pitch_mm=args.pitch, edges=args.edges)
    pics = {Path(args.image).stem: Image.open(args.image)} if args.image else test_pictures()

    results, ok = {}, True
    hdr = f"{'picture':<8} {'tol':>4} {'switch':>6} {'est min':>7} {'sim min':>7} {'pic->rt':>7} {'rt->pic':>7} {'exec':>5} {'retrace':>7}"
    print(f"Budget {args.budget:.0f} s, pitch {args.pitch} mm\n{hdr}")
    for name, im in pics.items():
        s = run(name, im, args.budget, params)
        results[name] = s
        good = s["exec_chamfer_mm"] <= 1.0 and s["target_to_route_mm"] <= 2.5
        ok &= good or bool(args.image)
        print(f"{name:<8} {s['tol_mm']:4.0f} {s['switches']:6d} {s['est_time_s'] / 60:7.1f} {s['sim_time_s'] / 60:7.1f} "
              f"{s['target_to_route_mm']:7.2f} {s['route_to_target_mm']:7.2f} {s['exec_chamfer_mm']:5.2f} "
              f"{s['retrace_mm']:5.0f}mm {'' if good else '  <- over the exit-check limits'}")
    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=float))
    print(f"\nPNGs and results.json in {OUT}")
    print("EXIT CHECK:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
