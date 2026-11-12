"""MuJoCo scene: Gen3 6-DoF + Robotiq 2F-85 on a table, the toy, a wrist camera and scene cameras.

The toy's screen is not simulated by MuJoCo. Knob joint angles are fed to the toy kernel
(etch_sim.kernel), and the kernel's picture is written into the screen texture, so the cameras
see exactly what the kernel drew.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from ..kernel import ToyKernel
from . import gen3
from .toy_model import KNOB_AXES, ToyGeometry, toy_mjcf

GRIPPER_XML = gen3.VENDOR / "robotiq_2f85" / "2f85.xml"
SCREEN_PX_PER_MM = 4.0


@dataclass
class SceneConfig:
    toy: ToyGeometry = field(default_factory=ToyGeometry)
    # Toy lies flat on the table, knob edge towards the robot: toy +y points along world +x.
    toy_pos: tuple = (0.45, 0.0, 0.0)          # PLACEHOLDER: bottom-face centre, world metres
    toy_yaw_deg: float = -90.0
    scene_cam_res: tuple = (1920, 1080)        # PLACEHOLDER: plain 1080p USB webcam
    scene_cam_fovy: float = 45.0               # PLACEHOLDER
    # Candidate scene-camera mounts: (position, point looked at, image-up hint) in world metres. PLACEHOLDERS.
    # A 0.6 mm line needs roughly 3 px/mm, so the toy has to fill most of a 1080p frame.
    scene_cams: dict = field(default_factory=lambda: {
        "scene_overhead": ((0.463, 0.0, 0.45), (0.463, 0.0, 0.04), (1.0, 0.0, 0.0)),
        "scene_opposite": ((0.75, 0.0, 0.40), (0.463, 0.0, 0.04), (0.0, 0.0, 1.0)),
    })


def _look_at(pos, target, up_hint) -> str:
    """xyaxes for a camera at pos looking at target; up_hint is roughly 'image up' in the world."""
    fwd = np.subtract(target, pos, dtype=float)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up_hint)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return " ".join(f"{v:.9g}" for v in (*right, *up))


def build_spec(cfg: SceneConfig | None = None) -> mujoco.MjSpec:
    cfg = cfg or SceneConfig()
    arm = gen3.arm_mjcf()
    screen_px = (round(cfg.toy.screen_mm[0] * SCREEN_PX_PER_MM), round(cfg.toy.screen_mm[1] * SCREEN_PX_PER_MM))
    toy = toy_mjcf(cfg.toy, cfg.toy_pos, cfg.toy_yaw_deg, screen_px)
    w, h = cfg.scene_cam_res
    cams = "\n".join(
        f'<camera name="{name}" pos="{p[0]} {p[1]} {p[2]}" xyaxes="{_look_at(p, t, up)}" '
        f'fovy="{cfg.scene_cam_fovy}" resolution="{w} {h}"/>' for name, (p, t, up) in cfg.scene_cams.items())
    xml = f"""
<mujoco model="etch_scene">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <visual>
    <global offwidth="1920" offheight="1080"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.3 0.3 0.3" specular="0 0 0"/>
    <quality shadowsize="2048"/>
  </visual>
  <default>
    <default class="gen3_visual"><geom contype="0" conaffinity="0" group="2" rgba="0.75 0.75 0.75 1" mass="0"/></default>
    <default class="gen3_collision"><geom group="3" mass="0"/></default>
  </default>
  <asset>
    <texture type="skybox" builtin="flat" width="32" height="192" rgb1="0.78 0.80 0.82"/>
    <texture name="table" type="2d" builtin="flat" width="64" height="64" rgb1="0.86 0.84 0.80"/>
    <material name="table" texture="table" specular="0.05"/>
    {arm["asset"]}
    {toy["asset"]}
  </asset>
  <worldbody>
    <light pos="0.3 -0.4 1.4" dir="0.1 0.3 -1" diffuse="0.35 0.35 0.35" specular="0.2 0.2 0.2"/>
    <geom name="table" type="plane" size="1.2 1.2 0.05" material="table"/>
    {cams}
    {arm["body"]}
    {toy["body"]}
  </worldbody>
  <contact>
    <!-- base_link is fixed to the world, so MuJoCo's parent-child filter does not cover this pair -->
    <exclude body1="base_link" body2="shoulder_link"/>
  </contact>
  <actuator>
    {arm["actuator"]}
  </actuator>
</mujoco>"""
    spec = mujoco.MjSpec.from_string(xml)
    gripper = mujoco.MjSpec.from_file(str(GRIPPER_XML))
    frame = spec.body("bracelet_link").add_frame(pos=gen3.EE_POS, quat=gen3.EE_QUAT)
    spec.attach(gripper, prefix="gripper_", frame=frame)
    for body in spec.bodies:                    # the real arm compensates its own weight
        if body.name and not body.name.startswith(("toy", "knob", "world")):
            body.gravcomp = 1.0
    return spec


class EtchScene:
    """Compiled scene plus the knob-to-kernel coupling and the live screen texture."""

    def __init__(self, cfg: SceneConfig | None = None, kernel: ToyKernel | None = None):
        self.cfg = cfg or SceneConfig()
        self.spec = build_spec(self.cfg)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.kernel = kernel or ToyKernel()
        m = self.model
        self.arm_qadr = np.array([m.joint(j).qposadr[0] for j in gen3.JOINTS])
        self.arm_dadr = np.array([m.joint(j).dofadr[0] for j in gen3.JOINTS])
        self.arm_act = np.array([m.actuator(j).id for j in gen3.JOINTS])
        self.arm_vel_act = np.array([m.actuator(f"{j}_vel").id for j in gen3.JOINTS])
        self.gripper_act = m.actuator("gripper_fingers_actuator").id
        self._knob_qadr = {ax: m.joint(f"knob_{side}").qposadr[0] for ax, side in KNOB_AXES.items()}
        self._knob_last = {ax: 0.0 for ax in KNOB_AXES}
        self._tex = m.texture("screen").id
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}
        self._screen_dirty = True
        self.reset()

    # ---- state -----------------------------------------------------------------
    def reset(self, arm_q: np.ndarray | None = None) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.set_arm(gen3.home_qpos() if arm_q is None else arm_q)
        self._knob_last = {ax: 0.0 for ax in KNOB_AXES}
        self._paint_screen()

    def set_arm(self, q: np.ndarray, gripper: float = 0.0) -> None:
        """Teleport the arm to joint angles q (rad) and hold there. gripper: 0 open .. 1 closed."""
        self.data.qpos[self.arm_qadr] = q
        self.data.qvel[self.arm_dadr] = 0.0
        self.data.ctrl[self.arm_act] = q
        self.data.ctrl[self.arm_vel_act] = 0.0
        self.data.ctrl[self.gripper_act] = 255.0 * gripper
        mujoco.mj_forward(self.model, self.data)

    def step(self, seconds: float = 0.0, n: int | None = None) -> None:
        for _ in range(n if n is not None else max(1, round(seconds / self.model.opt.timestep))):
            mujoco.mj_step(self.model, self.data)
        self.sync_toy()

    # ---- knob -> kernel -> screen ---------------------------------------------
    def knob_deg(self, axis: str) -> float:
        """Knob angle in kernel sign convention (deg)."""
        s = -1.0 if self.cfg.toy.knob_cw_positive else 1.0      # MuJoCo +z rotation is counter-clockwise
        return s * float(np.rad2deg(self.data.qpos[self._knob_qadr[axis]]))

    def sync_toy(self) -> None:
        """Feed knob rotation since the last call to the kernel; repaint the screen if it drew."""
        for ax in KNOB_AXES:
            now = self.knob_deg(ax)
            d = now - self._knob_last[ax]
            if abs(d) > 1e-6:
                if self.kernel.turn(ax, d) != 0.0:
                    self._screen_dirty = True
                self._knob_last[ax] = now
        if self._screen_dirty:
            self._paint_screen()

    def _paint_screen(self) -> None:
        m = self.model
        w, h = int(m.tex_width[self._tex]), int(m.tex_height[self._tex])
        img = np.asarray(self.kernel.render(SCREEN_PX_PER_MM).resize((w, h)))
        adr = int(m.tex_adr[self._tex])
        m.tex_data[adr:adr + w * h * 3] = img.ravel()   # row 0 = top of the screen (checked against the cameras in s1_demo)
        for r in self._renderers.values():
            r._gl_context.make_current()
            mujoco.mjr_uploadTexture(m, r._mjr_context, self._tex)
        self._screen_dirty = False

    # ---- cameras ---------------------------------------------------------------
    def _renderer(self, camera: str) -> mujoco.Renderer:
        w, h = (int(v) for v in self.model.cam_resolution[self.model.camera(camera).id])
        if (w, h) not in self._renderers:
            self._renderers[(w, h)] = mujoco.Renderer(self.model, height=h, width=w)
        return self._renderers[(w, h)]

    def render(self, camera: str) -> np.ndarray:
        """RGB image (H x W x 3, uint8) from a named camera at its native resolution."""
        r = self._renderer(camera)
        r.update_scene(self.data, camera=camera)
        return r.render().copy()

    def visible_fraction(self, camera: str, geom: str = "toy_screen") -> float:
        """Share of a geom's unobstructed image area that is actually visible from a camera.

        Compares the segmentation image with one rendered with the arm and gripper hidden.
        """
        r = self._renderer(camera)
        gid = self.model.geom(geom).id

        def count() -> int:
            r.update_scene(self.data, camera=camera)
            seg = r.render()
            return int(((seg[..., 0] == gid) & (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))).sum())

        r.enable_segmentation_rendering()
        seen = count()
        saved = self.model.geom_rgba.copy()
        robot = [i for i in range(self.model.ngeom)
                 if self.model.body(self.model.geom_bodyid[i]).rootid[0] == self.model.body("base_link").id]
        self.model.geom_rgba[robot, 3] = 0.0
        clear = count()
        self.model.geom_rgba[:] = saved
        r.disable_segmentation_rendering()
        return seen / clear if clear else 0.0

    def intrinsics(self, camera: str) -> np.ndarray:
        cid = self.model.camera(camera).id
        w, h = (int(v) for v in self.model.cam_resolution[cid])
        f = h / 2 / np.tan(np.deg2rad(self.model.cam_fovy[cid]) / 2)
        return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])

    def close(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()
