// Port of src/etch_sim/mj/ik.py: damped least-squares IK that puts a site at a point with its z axis
// along a direction. Rotation about the tool axis is left free (5 constraints, 6 joints); for a tool
// pointing straight down that free rotation is joint 6, the joint that spins the knob.

function solve6(A, b) {                                // Gaussian elimination with partial pivoting, 6 x 6
  const n = 6, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
    x[r] = s / M[r][r];
  }
  return x;
}

export class IK {
  constructor(sim, site = 'gripper_pinch') {
    this.sim = sim;
    this.sid = sim.id.site(site);
    this.jp = new sim.mj.DoubleBuffer(3 * sim.nv);
    this.jr = new sim.mj.DoubleBuffer(3 * sim.nv);
    const m = sim.m;
    this.lo = sim.armJid.map((j) => m.jnt_range[2 * j]);
    this.hi = sim.armJid.map((j) => m.jnt_range[2 * j + 1]);
    this.limited = sim.armJid.map((j) => !!m.jnt_limited[j]);
  }

  /** Site position and z axis for arm joint angles q (rad), without touching the simulation. */
  fk(q) {
    const { mj, m, scratch: d, armQadr } = this.sim;
    const qpos = d.qpos;
    for (let i = 0; i < 6; i++) qpos[armQadr[i]] = q[i];
    mj.mj_kinematics(m, d);
    const p = d.site_xpos, R = d.site_xmat, s = this.sid;
    return { pos: [p[3 * s], p[3 * s + 1], p[3 * s + 2]], z: [R[9 * s + 2], R[9 * s + 5], R[9 * s + 8]] };
  }

  /** Returns {q, posErrMm, axisErrDeg}; q is null if the seed did not converge. */
  solve(pos, seedRad, { zAxis = [0, 0, -1], tolMm = 0.2, tolDeg = 0.2, iters = 60 } = {}) {
    const { mj, m, scratch: d, armDadr, nv } = this.sim;
    let q = [...seedRad];
    for (let it = 0; it < iters; it++) {
      const { pos: p, z } = this.fk(q);
      mj.mj_comPos(m, d);
      const dot = z[0] * zAxis[0] + z[1] * zAxis[1] + z[2] * zAxis[2];
      let er = [z[1] * zAxis[2] - z[2] * zAxis[1], z[2] * zAxis[0] - z[0] * zAxis[2], z[0] * zAxis[1] - z[1] * zAxis[0]];
      if (dot < 0) {                                   // more than 90 deg off: keep the error from vanishing
        const n = Math.hypot(...er) + 1e-9;
        er = er.map((v) => v / n);
      }
      const err = [pos[0] - p[0], pos[1] - p[1], pos[2] - p[2], 0.3 * er[0], 0.3 * er[1], 0.3 * er[2]];
      mj.mj_jacSite(m, d, this.jp, this.jr, this.sid);
      const jp = this.jp.GetView(), jr = this.jr.GetView();
      const J = [];
      for (let r = 0; r < 3; r++) J.push(armDadr.map((c) => jp[r * nv + c]));
      for (let r = 0; r < 3; r++) J.push(armDadr.map((c) => 0.3 * jr[r * nv + c]));
      const A = J.map((ri, i) => J.map((rj, j) => ri.reduce((s, v, k) => s + v * rj[k], 0) + (i === j ? 1e-4 : 0)));
      const y = solve6(A, err);
      for (let k = 0; k < 6; k++) {
        let dq = 0;
        for (let r = 0; r < 6; r++) dq += J[r][k] * y[r];
        q[k] += Math.min(0.2, Math.max(-0.2, dq));
        if (this.limited[k]) q[k] = Math.min(this.hi[k], Math.max(this.lo[k], q[k]));
      }
    }
    const { pos: p, z } = this.fk(q);
    const posErrMm = Math.hypot(pos[0] - p[0], pos[1] - p[1], pos[2] - p[2]) * 1000;
    const dot = Math.min(1, Math.max(-1, z[0] * zAxis[0] + z[1] * zAxis[1] + z[2] * zAxis[2]));
    const axisErrDeg = Math.acos(dot) * 180 / Math.PI;
    const ok = posErrMm < tolMm && axisErrDeg < tolDeg;
    return { q: ok ? q : null, posErrMm, axisErrDeg };
  }
}
