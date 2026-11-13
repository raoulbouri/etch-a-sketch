// Port of src/etch_sim/kernel.py: knob angle -> stylus with backlash, end stops and string strain.
// Erase is left out; the page resets the scene instead.
const AXES = ['x', 'y'];

export class ToyKernel {
  constructor(params, startMm) {
    this.p = params;                                   // {x, y: {k_mm_per_deg, half_dead_deg, transition_deg, travel_mm}, ...}
    const start = startMm ?? [params.x.travel_mm / 2, params.y.travel_mm / 2];
    this.theta = { x: 0, y: 0 };
    this.phi = {}; this.thetaOff = {};
    AXES.forEach((a, i) => {
      this.phi[a] = start[i] / params[a].k_mm_per_deg;
      this.thetaOff[a] = -this.phi[a];                 // gap starts at 0 (mid dead zone)
    });
    this.strain = { x: 0, y: 0 };
    this.snapped = { x: false, y: false };
    this.points = [this.stylus];
  }

  get stylus() {
    return [this.phi.x * this.p.x.k_mm_per_deg, this.phi.y * this.p.y.k_mm_per_deg];
  }

  delta(axis) { return this.p[axis].half_dead_deg + this.p[axis].transition_deg; }

  /** Turn one knob by dtheta (deg). Returns the stylus displacement along that axis (mm). */
  turn(axis, dthetaDeg, stepDeg = 0.25) {
    const ax = this.p[axis];
    const before = this.phi[axis];
    const n = Math.max(1, Math.ceil(Math.abs(dthetaDeg) / stepDeg));
    const d = dthetaDeg / n;
    for (let i = 0; i < n; i++) this._step(axis, ax, d);
    const moved = (this.phi[axis] - before) * ax.k_mm_per_deg;
    if (moved !== 0) this.points.push(this.stylus);
    return moved;
  }

  _step(axis, ax, d) {
    this.theta[axis] += d;
    if (this.snapped[axis] || d === 0) return;
    const s = d > 0 ? 1 : -1;
    const strain = this.strain[axis];
    if (strain !== 0 && (strain > 0) !== (d > 0)) {    // unwind string strain first when backing off an end stop
      const u = Math.min(Math.abs(d), Math.abs(strain));
      this.strain[axis] -= Math.sign(strain) * u;
      d -= s * u;
      if (d === 0) return;
    }
    let phi = this.phi[axis];
    const thetaEff = this.theta[axis] - this.thetaOff[axis] - this.strain[axis];
    const eDir = s * (thetaEff - phi);                 // gap measured in the direction of motion
    if (ax.transition_deg > 0 && eDir > ax.half_dead_deg) {
      phi += s * Math.abs(d) * Math.min(1, (eDir - ax.half_dead_deg) / ax.transition_deg);
    }
    const full = ax.half_dead_deg + ax.transition_deg; // the pulley can never lag more than this
    phi = Math.min(Math.max(phi, thetaEff - full), thetaEff + full);
    const phiMax = ax.travel_mm / ax.k_mm_per_deg;
    if (phi > phiMax || phi < 0) {                     // end stop: excess rotation stretches the string
      const clipped = Math.min(Math.max(phi, 0), phiMax);
      this.strain[axis] += phi - clipped;
      phi = clipped;
      if (Math.abs(this.strain[axis]) > this.p.snap_strain_deg) this.snapped[axis] = true;
    }
    this.phi[axis] = phi;
  }
}

/** Paints the kernel's line into a canvas (front-on screen, y up), adding only what is new. */
export class ScreenPainter {
  constructor(canvas, kernel, pxPerMm) {
    this.canvas = canvas; this.k = kernel; this.s = pxPerMm;
    canvas.width = Math.round(kernel.p.x.travel_mm * pxPerMm);
    canvas.height = Math.round(kernel.p.y.travel_mm * pxPerMm);
    this.ctx = canvas.getContext('2d');
    this.clear();
  }

  clear() {
    const c = this.ctx;
    c.fillStyle = 'rgb(188,188,182)';
    c.fillRect(0, 0, this.canvas.width, this.canvas.height);
    this.done = 0;
  }

  /** Returns true if anything was drawn. */
  paint() {
    const pts = this.k.points;
    if (pts.length - 1 <= this.done) return false;
    const c = this.ctx, h = this.canvas.height, s = this.s;
    c.strokeStyle = 'rgb(55,55,55)';
    c.lineWidth = Math.max(1, Math.round(this.k.p.line_width_mm * s));
    c.lineCap = 'round'; c.lineJoin = 'round';
    c.beginPath();
    c.moveTo(pts[this.done][0] * s, h - pts[this.done][1] * s);
    for (let i = this.done + 1; i < pts.length; i++) c.lineTo(pts[i][0] * s, h - pts[i][1] * s);
    c.stroke();
    this.done = pts.length - 1;
    return true;
  }
}
