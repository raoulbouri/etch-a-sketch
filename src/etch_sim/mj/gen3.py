"""Kinova Gen3 6-DoF (with vision bracelet) as MJCF, generated from Kinova's own xacro.

MuJoCo Menagerie only ships the 7-DoF arm, so the kinematics, inertias and meshes here are read
from models/vendor/kinova_gen3_6dof/gen3_macro.xacro (ros_kortex, noetic-devel). Nothing is typed in
by hand except actuator gains, which follow Menagerie's kinova_gen3 model.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

VENDOR = Path(__file__).resolve().parents[3] / "models" / "vendor"
XACRO = VENDOR / "kinova_gen3_6dof" / "gen3_macro.xacro"
MESHES = VENDOR / "kinova_gen3_6dof" / "meshes"
XACRO_NS = "{http://ros.org/wiki/xacro}"

JOINTS = [f"joint_{i}" for i in range(1, 7)]
# Real-arm Home (deg), read back from the green arm on 2026-11-09 (aims-kinova logs/selftest_A.log).
HOME_DEG = [0.0, 15.0, -130.0, 0.0, 55.0, 90.0]

# End-effector interface frame in bracelet_link (URDF `end_effector` joint): z points out of the wrist.
EE_POS = (0.0, 0.0, -0.061525)
EE_QUAT = (0.0, 0.0, 1.0, 0.0)                 # rpy (pi, 0, pi) = 180 deg about y
# Colour camera in bracelet_link, composed from the URDF `end_effector` and `color_module` joints.
# In MuJoCo's camera convention (looks along -z, +y up) this comes out as the identity rotation.
WRIST_CAM_POS = (0.0, 0.05639, -0.058475)
WRIST_CAM_QUAT = (1.0, 0.0, 0.0, 0.0)
WRIST_CAM_RES = (1280, 720)                    # live colour stream on the green arm
WRIST_CAM_FOVY = 34.7                          # PLACEHOLDER: 65 deg diagonal at 16:9; replace with `kinova intrinsics`

# Menagerie kinova_gen3 gains; force limits are the URDF effort limits (39 Nm large, 9 Nm small actuators).
GAINS = {"large": (2000.0, 100.0, 39.0), "small": (500.0, 50.0, 9.0)}


def _quat(rpy) -> str:
    x, y, z, w = R.from_euler("xyz", rpy).as_quat()
    return f"{w:.9g} {x:.9g} {y:.9g} {z:.9g}"


def _floats(s: str) -> list[float]:
    return [float(v) for v in s.split()]


def _load():
    text = XACRO.read_text().replace("${prefix}", "").replace("${dof}", "6")
    root = ET.fromstring(text)
    macro = root.find(f"{XACRO_NS}macro")
    elems = []
    for el in macro:
        if el.tag == f"{XACRO_NS}if":          # vision = true
            elems.extend(el)
        elif el.tag == f"{XACRO_NS}unless":
            continue
        else:
            elems.append(el)
    links = {e.get("name"): e for e in elems if e.tag == "link"}
    joints = {e.get("name"): e for e in elems if e.tag == "joint"}
    return links, joints


def arm_mjcf(base_pos=(0.0, 0.0, 0.0)) -> dict[str, str]:
    """Returns MJCF fragments: {'asset', 'body', 'actuator'}. Bodies and joints keep the URDF names."""
    links, joints = _load()
    child_of = {j.find("parent").get("link"): j for j in joints.values() if j.get("type") != "fixed"}
    assets, actuators = [], []

    def link_xml(name: str, pos, rpy, joint) -> str:
        link = links[name]
        inertial = link.find("inertial")
        io = inertial.find("origin").get("xyz")
        m = inertial.find("mass").get("value")
        i = inertial.find("inertia")
        full = " ".join(i.get(k) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"))
        mesh = re.search(r"meshes/(.+)\.STL", link.find("visual/geometry/mesh").get("filename")).group(1)
        assets.append(f'<mesh name="gen3_{mesh}" file="{MESHES / (mesh + ".STL")}"/>')
        out = [f'<body name="{name}" pos="{" ".join(map(str, pos))}" quat="{_quat(rpy)}">',
               f'<inertial pos="{io}" mass="{m}" fullinertia="{full}"/>']
        if joint is not None:
            jn = joint.get("name")
            lim = joint.find("limit")
            rng = f' range="{lim.get("lower")} {lim.get("upper")}"' if joint.get("type") == "revolute" else ""
            out.append(f'<joint name="{jn}" axis="0 0 1" armature="0.1"{rng}/>')
            kp, kv, f = GAINS["large" if float(lim.get("effort")) > 20 else "small"]
            ctrl = f' ctrlrange="{lim.get("lower")} {lim.get("upper")}"' if rng else ""
            # Position servo plus a velocity servo that takes the commanded joint speed as feedforward.
            # With damping on the measured speed alone the arm lags a moving target by kv/kp seconds
            # (2 to 4 deg at 40 deg/s), which the real arm's controller does not do.
            actuators.append(f'<position name="{jn}" joint="{jn}" kp="{kp}" forcerange="{-f} {f}"{ctrl}/>')
            actuators.append(f'<velocity name="{jn}_vel" joint="{jn}" kv="{kv}" forcerange="{-f} {f}"/>')
        out.append(f'<geom type="mesh" mesh="gen3_{mesh}" class="gen3_visual"/>')
        out.append(f'<geom type="mesh" mesh="gen3_{mesh}" class="gen3_collision"/>')
        if name == "bracelet_link":
            w, h = WRIST_CAM_RES
            out.append(f'<camera name="wrist" pos="{" ".join(map(str, WRIST_CAM_POS))}" '
                       f'quat="{" ".join(map(str, WRIST_CAM_QUAT))}" fovy="{WRIST_CAM_FOVY}" resolution="{w} {h}"/>')
            out.append(f'<site name="ee" pos="{" ".join(map(str, EE_POS))}" quat="{" ".join(map(str, EE_QUAT))}" size="0.004" group="4"/>')
        nxt = child_of.get(name)
        if nxt is not None:
            o = nxt.find("origin")
            out.append(link_xml(nxt.find("child").get("link"), _floats(o.get("xyz")), _floats(o.get("rpy")), nxt))
        out.append("</body>")
        return "\n".join(out)

    body = link_xml("base_link", base_pos, (0, 0, 0), None)
    return {"asset": "\n".join(assets), "body": body, "actuator": "\n".join(actuators)}


def home_qpos() -> np.ndarray:
    return np.deg2rad(HOME_DEG)
