import mujoco
import numpy as np
import pytest

from etch_sim import perception as P
from etch_sim.mj import EtchScene, ik


@pytest.fixture(scope="module")
def scene():
    s = EtchScene()
    yield s
    s.close()


def test_model_layout(scene):
    m = scene.model
    assert m.nu == 13                                       # 6 position + 6 velocity servos + gripper
    assert [m.jnt_limited[m.joint(f"joint_{i}").id] for i in range(1, 7)] == [0, 1, 1, 0, 1, 0]


def test_fk_matches_real_arm_home(scene):
    # Green arm at Home reported tool (0.574, -0.004, 0.426); its tool frame is 0.12 m beyond the flange.
    scene.set_arm(np.deg2rad([-0.022, 15.089, -129.961, -0.014, 55.126, 89.992]))
    ee = scene.model.site("ee").id
    z = scene.data.site_xmat[ee].reshape(3, 3)[:, 2]
    tool = scene.data.site_xpos[ee] + 0.12 * z
    assert np.linalg.norm(tool - [0.574, -0.004, 0.426]) < 0.015


def test_both_knobs_reachable_tool_down(scene):
    scene.reset()
    for side in ("left", "right"):
        top = scene.data.site_xpos[scene.model.site(f"knob_{side}_top").id]
        q, pe, ae = ik.solve(scene.model, top - [0, 0, 0.007])
        assert q is not None and pe < 0.2 and ae < 0.2


def test_knob_drives_kernel_and_camera_sees_it(scene):
    scene.reset(np.deg2rad([5.8, -14.8, -141.3, 5.2, -61.2, 93.1]))      # parked, clear of the board
    x0 = scene.kernel.stylus[0]
    adr = scene._knob_qadr["x"]
    for _ in range(20):
        scene.data.qpos[adr] -= np.deg2rad(20)             # clockwise seen from the front
        mujoco.mj_forward(scene.model, scene.data)
        scene.sync_toy()
    assert scene.kernel.stylus[0] > x0 + 20                # clockwise left knob moves the stylus right
    g = scene.cfg.toy
    img = scene.render("scene_overhead")
    markers = P.detect_markers(img)
    assert sorted(markers) == [0, 1, 2, 3]
    rect = P.rectify(img, P.board_homography(markers, g), g)
    dist = P.mask_distance_mm(P.line_mask(rect), P.line_mask(np.asarray(scene.kernel.render())))
    assert dist["a_to_b_mm"] < 0.5 and dist["b_to_a_mm"] < 0.5
