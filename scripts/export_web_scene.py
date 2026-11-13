"""Export the MuJoCo scene for the browser demo in docs/ (served by GitHub Pages).

    uv run python scripts/export_web_scene.py

Writes docs/scene/scene.xml, the meshes and marker images it refers to, and docs/scene/meta.json
with the toy geometry, toy kernel parameters and motion parameters, so the JavaScript port in
docs/js/ uses the same numbers as the Python code. Run it again whenever those numbers change.
"""
from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict
from pathlib import Path

from etch_sim.kernel import ToyParams
from etch_sim.mj import EtchScene
from etch_sim.mj.backend import READY_DEG, MotionParams
from etch_sim.mj.scene import SCREEN_PX_PER_MM

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "scene"


def main() -> None:
    scene = EtchScene()
    xml = scene.spec.to_xml()
    OUT.mkdir(parents=True, exist_ok=True)
    names = []
    for ref in sorted(set(re.findall(r'file="([^"]+)"', xml))):
        src = next(c for c in [Path(ref), *ROOT.joinpath("models").rglob(Path(ref).name)] if c.is_file())
        shutil.copy(src, OUT / src.name)
        names.append(src.name)
        xml = xml.replace(f'file="{ref}"', f'file="{src.name}"')
    xml = re.sub(r'\s(meshdir|texturedir|assetdir)="[^"]*"', "", xml)
    (OUT / "scene.xml").write_text(xml)
    toy, p = scene.cfg.toy, ToyParams()
    meta = {
        "files": names,
        "ready_deg": READY_DEG,
        "motion": asdict(MotionParams()),
        "screen_px_per_mm": SCREEN_PX_PER_MM,
        "toy": {
            "body_mm": toy.body_mm, "screen_mm": toy.screen_mm, "screen_centre_mm": toy.screen_centre_mm,
            "knob_radius_mm": toy.knob_radius_mm, "knob_height_mm": toy.knob_height_mm,
            "knob_xy_mm": toy.knob_xy_mm, "knob_cw_positive": toy.knob_cw_positive,
            "marker_mm": toy.marker_mm, "marker_xy_mm": {str(k): v for k, v in toy.marker_xy_mm.items()},
        },
        "kernel": {
            "x": asdict(p.x), "y": asdict(p.y), "line_width_mm": p.line_width_mm,
            "snap_strain_deg": p.snap_strain_deg,
        },
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    size = sum(f.stat().st_size for f in OUT.iterdir()) / 1e6
    print(f"{OUT}: scene.xml, meta.json and {len(names)} asset files, {size:.1f} MB")


if __name__ == "__main__":
    main()
