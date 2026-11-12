"""Record a video of the simulated arm drawing on the Etch-a-Sketch.

    uv run python scripts/make_video.py                       # spiral, 4x speed -> out/video/spiral.mp4
    uv run python scripts/make_video.py --shape square --speed 2
    uv run python scripts/make_video.py --shape staircase --view scene_overhead

Main picture: a free camera on the arm and toy. Insets: the scene camera, and the toy's screen.
The drawing is open loop and uses the toy's true mm-per-degree and backlash (no calibration step),
so this shows the motion, not calibration accuracy; scripts/s2_demo.py covers that.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from etch_sim import Calib, OpenLoopDrawer, ToyKernel, chamfer, polyline, spiral, square, staircase
from etch_sim.mj import EtchScene
from etch_sim.mj.backend import MujocoBackend

OUT = Path(__file__).resolve().parent.parent / "out" / "video"
SHAPES = {"square": lambda: square(40.0), "staircase": lambda: staircase(8.0, 5), "spiral": lambda: spiral(40.0, 8.0)}
W, H, FPS = 1280, 720, 25


class Recorder:
    def __init__(self, backend: MujocoBackend, path: Path, speed: float, inset_cam: str, title: str):
        self.b, self.speed, self.inset_cam, self.title = backend, speed, inset_cam, title
        self.writer = imageio.get_writer(path, fps=FPS, codec="libx264", quality=7, macro_block_size=16,
                                         pixelformat="yuv420p")
        # Shared with the scene so that screen-texture updates reach this renderer too.
        self.renderer = backend.s._renderers.get((W, H)) or mujoco.Renderer(backend.m, H, W)
        backend.s._renderers[(W, H)] = self.renderer
        backend.s._paint_screen()
        self.font = ImageFont.load_default(size=18)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:], self.cam.distance, self.cam.azimuth, self.cam.elevation = (0.42, 0.0, 0.09), 0.80, 152, -36
        self.next_t = backend.d.time
        self.frames = 0
        self.caption = ""

    def __call__(self) -> None:
        if self.b.d.time < self.next_t:
            return
        self.next_t += self.speed / FPS
        self.renderer.update_scene(self.b.d, self.cam)
        frame = Image.fromarray(self.renderer.render())
        scene = Image.fromarray(self.b.s.render(self.inset_cam)).resize((400, 225), Image.LANCZOS)
        screen = self.b.s.kernel.render(2.0)                          # 306 x 214
        frame.paste(scene, (W - 400 - 16, 16))
        frame.paste(screen, (W - screen.width - 16, H - screen.height - 16))
        d = ImageDraw.Draw(frame)
        d.rectangle((W - 416, 16, W - 16, 241), outline=(255, 255, 255), width=2)
        d.rectangle((W - screen.width - 16, H - screen.height - 16, W - 16, H - 16), outline=(255, 255, 255), width=2)
        d.text((W - 414, 246), "scene camera", fill=(255, 255, 255), font=self.font)
        d.text((W - screen.width - 14, H - screen.height - 42), "toy screen (front-on)", fill=(255, 255, 255), font=self.font)
        d.text((16, 12), self.title, fill=(255, 255, 255), font=self.font)
        d.text((16, 38), f"t = {self.b.clock_s:5.1f} s   {self.speed:g}x speed   knob switches: {self.b.n_switches}", fill=(255, 255, 255), font=self.font)
        d.text((16, 64), self.caption, fill=(255, 235, 150), font=self.font)
        self.writer.append_data(np.asarray(frame))
        self.frames += 1

    def close(self) -> None:
        self.writer.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", choices=SHAPES, default="spiral")
    ap.add_argument("--speed", type=float, default=4.0, help="playback speed-up")
    ap.add_argument("--view", default="scene_opposite", help="scene camera shown in the inset")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = args.out or OUT / f"{args.shape}.mp4"

    toy = ToyKernel(start_mm=(52.0, 28.0))
    b = MujocoBackend(EtchScene(kernel=toy))
    rec = Recorder(b, path, args.speed, args.view, f"Kinova Gen3 6-DoF drawing a {args.shape} (MuJoCo, placeholder toy parameters)")
    b.frame_hook = rec

    class Narrated:                                               # captions for what the arm is doing
        clock_s = property(lambda self: b.clock_s)
        n_switches = property(lambda self: b.n_switches)

        def grasp_knob(self, axis):
            rec.caption = f"grasp {'left' if axis == 'x' else 'right'} knob"
            return b.grasp_knob(axis)

        def rotate(self, deg):
            rec.caption = f"turn {'left' if b.held == 'x' else 'right'} knob {deg:+.0f} deg"
            return b.rotate(deg)

        def release(self):
            rec.caption = "release"
            return b.release()

    rec.caption = "locate board from the markers"
    assert b.locate_board()
    b._run(0.6)
    calib = Calib({a: toy.p.axis(a).k_mm_per_deg for a in "xy"}, {a: toy.p.axis(a).delta_deg for a in "xy"})
    drawer = OpenLoopDrawer(Narrated(), calib)
    drawer.preload()
    mark, start = toy.mark(), toy.stylus
    moves = SHAPES[args.shape]()
    drawer.draw(moves)
    rec.caption = "done"
    b._run(1.5 * args.speed)
    rec.close()
    score = chamfer(toy.path_since(mark), polyline(moves, start))
    print(f"{path}  ({rec.frames} frames, {rec.frames / FPS:.0f} s of video for {b.clock_s:.0f} s of arm time, "
          f"{b.n_switches} knob switches, chamfer {score['chamfer_mm']:.2f} mm)")


if __name__ == "__main__":
    main()
