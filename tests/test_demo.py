"""demo.py's handover: for every board placement in the re-locate workspace, the pose above the left knob at the
handover height lies inside BOTH envelopes, so the demo can switch routines without a move outside an approval."""
import importlib.util
import json
from pathlib import Path

from etch_sim.real import board_model as B

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_handover_fits_both_envelopes_for_every_workspace_placement():
    demo, rl = load("demo"), load("relocate")
    env_r = json.loads(demo.RELOCATE_ENV.read_text())
    env_d = json.loads(demo.DRAW_ENV.read_text())
    ws = json.loads(rl.WORKSPACE_FILE.read_text())
    grasp_x = B.load_model()["grasps"]["x"]
    boards = rl.placements(ws["region"])
    assert len(boards) == 27
    for b in boards:
        g = B.grasp_in_base(b, grasp_x)
        handover = demo.handover_pose(g, env_r, env_d)
        assert demo.inside(handover, env_r) and demo.inside(handover, env_d), handover
        approach = [g[0], g[1], g[2] + 0.05]                  # the drawer's first move: handover -> above the knob
        assert sum((u - v) ** 2 for u, v in zip(handover, approach)) ** 0.5 <= env_d["max_step_m"]
