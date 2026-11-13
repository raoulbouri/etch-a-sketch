// Port of src/etch_sim/mj/backend.py: the arm really grasps and turns the knobs. Control is joint
// position targets, as in the Python backend.
//
// Every skill is a generator that yields once per physics step, so the page can run the simulation
// at any speed without blocking. `drain` runs one to completion for set-up work.
//
// One difference from Python: the board pose is read from the simulation state. The Python backend
// finds it from the ArUco markers in the scene camera image, which needs OpenCV.
import { IK } from './ik.js';
import { KNOB_AXES, deg2rad, rad2deg } from './sim.js';

export function drain(gen) {
  let r = gen.next();
  while (!r.done) r = gen.next();
  return r.value;
}

function interp(x, xs, ys) {                           // numpy.interp for ascending xs
  if (x <= xs[0]) return ys[0];
  for (let i = 1; i < xs.length; i++) {
    if (x <= xs[i]) return ys[i - 1] + (ys[i] - ys[i - 1]) * (x - xs[i - 1]) / (xs[i] - xs[i - 1]);
  }
  return ys[ys.length - 1];
}

export class Backend {
  constructor(sim) {
    this.s = sim; this.p = sim.meta.motion; this.g = sim.meta.toy;
    this.ik = new IK(sim);
    this.nSwitches = 0;
    this.held = null;
    this.lastAxis = null;
    this.board = null;                                 // toy frame in the world: {R (row-major 3x3), t}
    this.pinch = sim.id.site('gripper_pinch');
    this.pads = ['left', 'right'].map((side) => sim.id.geom(`gripper_${side}_pad1`));
    this.failure = null;
    sim.reset();
    this.gripTable = drain(this._measureGripper());
    this.t0 = sim.time;
  }

  get clock() { return this.s.time - this.t0; }

  // ---- low level ---------------------------------------------------------------
  *_run(seconds) {
    const n = Math.max(1, Math.round(seconds / this.s.dt));
    for (let k = 0; k < n; k++) {
      this.s.step();
      if (k % 10 === 9) this.s.syncToy();
      yield;
    }
    this.s.syncToy();
  }

  _ctrlArm() { const c = this.s.d.ctrl; return this.s.armAct.map((a) => c[a]); }
  _qArm() { const q = this.s.d.qpos; return this.s.armQadr.map((a) => q[a]); }

  /** Straight line in joint space from the current target, speed-limited, then settle. */
  *_moveJoints(qGoal, speedDps, settle = true) {
    const { s } = this, dt = s.dt;
    const q0 = this._ctrlArm();
    const speed = deg2rad(speedDps ?? this.p.joint_speed_dps);
    const T = Math.max(Math.max(...qGoal.map((v, i) => Math.abs(v - q0[i]))) / speed, dt);
    const n = Math.max(1, Math.round(T / dt));
    let prev = q0;
    for (let k = 1; k <= n; k++) {
      let a = k / n;
      if (T > 0.3) a = a * a * (3 - 2 * a);            // ease in and out on longer moves
      const target = q0.map((v, i) => v + a * (qGoal[i] - v));
      const ctrl = s.d.ctrl;
      for (let i = 0; i < 6; i++) {
        ctrl[s.armAct[i]] = target[i];
        ctrl[s.armVelAct[i]] = (target[i] - prev[i]) / dt;
      }
      prev = target;
      s.step();
      if (k % 10 === 0) s.syncToy();
      yield;
    }
    const ctrl = s.d.ctrl;
    for (let i = 0; i < 6; i++) ctrl[s.armVelAct[i]] = 0;
    if (settle) yield* this._run(this.p.settle_s);
  }

  /** Move the fingertip reference point in a straight line to a world position, tool pointing down.
   *  The line is cut into short pieces because a joint-space move bows sideways by several millimetres. */
  *_movePinch(pos, stepM = 0.01) {
    const start = this.ik.fk(this._ctrlArm()).pos;     // where the arm is currently told to be
    const dist = Math.hypot(pos[0] - start[0], pos[1] - start[1], pos[2] - start[2]);
    const n = Math.max(1, Math.ceil(dist / stepM));
    for (let k = 1; k <= n; k++) {
      const wp = start.map((v, i) => v + (pos[i] - v) * k / n);
      const r = this.ik.solve(wp, this._ctrlArm());
      if (r.q === null) {
        this.failure = `The arm cannot reach that point (IK error ${r.posErrMm.toFixed(1)} mm).`;
        return false;
      }
      yield* this._moveJoints(r.q, undefined, k === n);
    }
    return true;
  }

  _pinchPos() { const p = this.s.d.site_xpos, i = 3 * this.pinch; return [p[i], p[i + 1], p[i + 2]]; }

  *_setGripper(ctrl, seconds) {
    this.s.d.ctrl[this.s.gripperAct] = ctrl;
    yield* this._run(seconds);
  }

  /** [gap between pad faces, how far the fingertips reach beyond the pinch point along the tool axis], m. */
  _padState() {
    const d = this.s.d, m = this.s.m;
    const R = d.site_xmat, i = 9 * this.pinch;
    const z = [R[i + 2], R[i + 5], R[i + 8]];
    const gx = d.geom_xpos, size = m.geom_size;
    const c = this.pads.map((g) => [gx[3 * g], gx[3 * g + 1], gx[3 * g + 2]]);
    const gap = Math.hypot(c[0][0] - c[1][0], c[0][1] - c[1][1], c[0][2] - c[1][2]) - 2 * size[3 * this.pads[0] + 1];
    const pp = this._pinchPos();
    const tip = Math.max(...this.pads.map((g, k) =>
      (c[k][0] - pp[0]) * z[0] + (c[k][1] - pp[1]) * z[1] + (c[k][2] - pp[2]) * z[2] + size[3 * g + 2]));
    return [gap, tip];
  }

  /** Table of (ctrl, finger gap, fingertip reach) for the free gripper, measured in the simulation. */
  *_measureGripper() {
    const rows = [];
    for (let k = 0; k < 18; k++) {
      const ctrl = 255 * k / 17;
      yield* this._setGripper(ctrl, 0.35);
      rows.push([ctrl, ...this._padState()]);
    }
    yield* this._setGripper(0, 0.5);
    return rows;
  }

  _ctrlForGap(gap) {
    const t = [...this.gripTable].reverse();           // gap falls as ctrl rises
    return interp(gap, t.map((r) => r[1]), t.map((r) => r[0]));
  }

  _tipReach(ctrl) { return interp(ctrl, this.gripTable.map((r) => r[0]), this.gripTable.map((r) => r[2])); }

  // ---- skills ------------------------------------------------------------------
  locateBoard() {
    const d = this.s.d, b = this.s.toyBody;
    this.board = { R: Array.from(d.xmat.subarray(9 * b, 9 * b + 9)), t: Array.from(d.xpos.subarray(3 * b, 3 * b + 3)) };
    return true;
  }

  _knobTop(axis) {
    const [x, y] = this.g.knob_xy_mm[KNOB_AXES[axis]];
    const v = [x / 1000, y / 1000, (this.g.body_mm[2] + this.g.knob_height_mm) / 1000];
    const { R, t } = this.board;
    return [0, 1, 2].map((r) => R[3 * r] * v[0] + R[3 * r + 1] * v[1] + R[3 * r + 2] * v[2] + t[r]);
  }

  *graspKnob(axis) {
    if (this.held === axis) return true;
    if (this.held !== null) yield* this.release();
    if (this.board === null) this.locateBoard();
    const switching = this.lastAxis !== null && this.lastAxis !== axis;
    const top = this._knobTop(axis);
    const knobH = this.g.knob_height_mm / 1000, knobD = this.g.knob_radius_mm / 500;
    const pre = this._ctrlForGap(knobD + this.p.pregrasp_margin_m);
    const closed = this._ctrlForGap(knobD);
    yield* this._setGripper(pre, 0);
    const travelZ = top[2] + this.p.safe_height_m + this._tipReach(pre);
    const here = this._pinchPos();
    let ok = yield* this._movePinch([here[0], here[1], Math.max(here[2], travelZ)]);
    ok = ok && (yield* this._movePinch([top[0], top[1], travelZ]));
    const graspZ = top[2] - knobH + this.p.tip_clearance_m + this._tipReach(closed);
    ok = ok && (yield* this._movePinch([top[0], top[1], graspZ]));
    if (!ok) return false;
    yield* this._setGripper(255, this.p.grip_s);
    this.held = axis; this.lastAxis = axis;
    this.nSwitches += switching ? 1 : 0;
    return true;
  }

  *rotate(deg) {
    if (this.held === null) return { ok: false, wristDeg: 0 };
    const { s } = this;
    // Which way does joint 6 turn the tool about the world vertical? (kernel-positive = clockwise from above)
    s.mj.mj_jacSite(s.m, s.d, this.ik.jp, this.ik.jr, this.pinch);
    const yawPerQ6 = this.ik.jr.GetView()[2 * s.nv + s.armDadr[5]];
    const cw = this.g.knob_cw_positive ? -1 : 1;
    const sign = cw * Math.sign(yawPerQ6);
    const qStart = this._qArm()[5];
    const goal = this._ctrlArm();
    goal[5] += deg2rad(deg) * sign;
    yield* this._moveJoints(goal, this.p.knob_speed_dps);
    return { ok: true, wristDeg: rad2deg(this._qArm()[5] - qStart) * sign };
  }

  *release() {
    if (this.held === null) return;
    const knobD = this.g.knob_radius_mm / 500;
    const pre = this._ctrlForGap(knobD + this.p.pregrasp_margin_m);
    yield* this._setGripper(pre, 0.4);
    const top = this._knobTop(this.held);
    yield* this._movePinch([top[0], top[1], top[2] + this.p.safe_height_m + this._tipReach(pre)]);
    this.held = null;
  }
}
