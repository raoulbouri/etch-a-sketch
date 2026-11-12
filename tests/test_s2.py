import numpy as np

from etch_sim.mj.backend import MujocoBackend


def test_arm_turns_knobs_and_camera_tracks_pen():
    b = MujocoBackend()
    toy = b.s.kernel
    pos0 = b.d.body("toy").xpos.copy()
    assert b.locate_board()
    assert np.linalg.norm(b._board[1] - pos0) < 0.002                  # board found within 2 mm

    assert b.grasp_knob("x")
    k0 = b.s.knob_deg("x")
    r = b.rotate(300.0)
    assert abs(r.wrist_deg - 300.0) < 1.0
    assert abs((b.s.knob_deg("x") - k0) / 300.0 - 1.0) < 0.02          # knob follows the wrist, under 2% slip
    assert np.hypot(*np.subtract(b.observe_pen(), toy.stylus)) < 0.6   # pen seen by the camera within 0.6 mm

    assert b.grasp_knob("y") and b.n_switches == 1
    b.rotate(250.0)
    assert np.hypot(*np.subtract(b.observe_pen(), toy.stylus)) < 0.6

    b.release()
    assert np.linalg.norm(b.d.body("toy").xpos - pos0) < 0.001         # the toy stayed put
    assert not [e for e in b.log if e.get("event") == "ik_failed"]
    b.s.close()
