// MuJoCo scene in the browser: loads docs/scene (exported by scripts/export_web_scene.py) into the
// official MuJoCo WebAssembly build and couples the knob joints to the toy kernel, like
// src/etch_sim/mj/scene.py. The single-threaded build is used because GitHub Pages cannot send the
// cross-origin isolation headers the multi-threaded one needs.
import loadMujoco from '../vendor/mujoco.js';
import { ToyKernel } from './kernel.js';

const JOINTS = [1, 2, 3, 4, 5, 6].map((i) => `joint_${i}`);
export const KNOB_AXES = { x: 'left', y: 'right' };
export const deg2rad = (d) => d * Math.PI / 180;
export const rad2deg = (r) => r * 180 / Math.PI;

export async function loadSim(onProgress = () => {}) {
  const base = new URL('../scene/', import.meta.url);
  const meta = await (await fetch(new URL('meta.json', base))).json();
  onProgress('Loading the physics engine');
  const mj = await loadMujoco();
  mj.FS.mkdir('/scene');
  const names = [...meta.files, 'scene.xml'];
  let done = 0;
  await Promise.all(names.map(async (f) => {
    const r = await fetch(new URL(f, base));
    if (!r.ok) throw new Error(`Could not load ${f} (${r.status})`);
    mj.FS.writeFile('/scene/' + f, new Uint8Array(await r.arrayBuffer()));
    onProgress(`Loading the robot model (${++done}/${names.length})`);
  }));
  onProgress('Building the scene');
  const model = mj.MjModel.from_xml_path('/scene/scene.xml');
  return new Sim(mj, model, meta);
}

export class Sim {
  constructor(mj, model, meta) {
    this.mj = mj; this.m = model; this.meta = meta;
    this.d = new mj.MjData(model);
    this.scratch = new mj.MjData(model);               // for kinematics that must not disturb the simulation
    const O = mj.mjtObj;
    const id = (type, name) => {
      const i = mj.mj_name2id(model, type.value, name);
      if (i < 0) throw new Error(`"${name}" is not in the scene`);
      return i;
    };
    this.id = {
      joint: (n) => id(O.mjOBJ_JOINT, n), geom: (n) => id(O.mjOBJ_GEOM, n), site: (n) => id(O.mjOBJ_SITE, n),
      body: (n) => id(O.mjOBJ_BODY, n), actuator: (n) => id(O.mjOBJ_ACTUATOR, n),
    };
    const jid = JOINTS.map(this.id.joint);
    this.armJid = jid;
    this.armQadr = jid.map((j) => model.jnt_qposadr[j]);
    this.armDadr = jid.map((j) => model.jnt_dofadr[j]);
    this.armAct = JOINTS.map(this.id.actuator);
    this.armVelAct = JOINTS.map((j) => this.id.actuator(j + '_vel'));
    this.gripperAct = this.id.actuator('gripper_fingers_actuator');
    this.knobQadr = {};
    for (const [ax, side] of Object.entries(KNOB_AXES)) this.knobQadr[ax] = model.jnt_qposadr[this.id.joint('knob_' + side)];
    this.toyBody = this.id.body('toy');
    this.dt = model.opt.timestep;
    this.nv = model.nv;
    this.onDraw = null;                                // called when the kernel has drawn something new
    this.reset();
  }

  reset(armQ) {
    const { mj, m, d } = this;
    mj.mj_resetData(m, d);
    this.kernel = new ToyKernel(this.meta.kernel);
    this.knobLast = { x: 0, y: 0 };
    const q = armQ ?? this.meta.ready_deg.map(deg2rad);
    const qpos = d.qpos, qvel = d.qvel, ctrl = d.ctrl;
    for (let i = 0; i < 6; i++) {
      qpos[this.armQadr[i]] = q[i]; qvel[this.armDadr[i]] = 0;
      ctrl[this.armAct[i]] = q[i]; ctrl[this.armVelAct[i]] = 0;
    }
    ctrl[this.gripperAct] = 0;
    mj.mj_forward(m, d);
  }

  step() { this.mj.mj_step(this.m, this.d); }

  get time() { return this.d.time; }

  /** Knob angle in the kernel's sign convention (deg). MuJoCo +z rotation is counter-clockwise. */
  knobDeg(axis) {
    const s = this.meta.toy.knob_cw_positive ? -1 : 1;
    return s * rad2deg(this.d.qpos[this.knobQadr[axis]]);
  }

  /** Feed knob rotation since the last call to the kernel. */
  syncToy() {
    let drew = false;
    for (const ax of ['x', 'y']) {
      const now = this.knobDeg(ax);
      const dd = now - this.knobLast[ax];
      if (Math.abs(dd) > 1e-6) {
        if (this.kernel.turn(ax, dd) !== 0) drew = true;
        this.knobLast[ax] = now;
      }
    }
    if (drew && this.onDraw) this.onDraw();
  }
}
