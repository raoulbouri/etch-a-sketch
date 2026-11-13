// Port of src/etch_sim/draw.py and scoring.py: open-loop drawing of Manhattan paths, and the
// chamfer score. A move is [axis, signed distance in mm]; "x" is the left knob, "y" the right.

export class OpenLoopDrawer {
  /** calib: {k: {x, y} mm per deg, delta: {x, y} deg}: what the controller believes about the toy. */
  constructor(robot, calib, { preloadMarginDeg = 10, say = () => {} } = {}) {
    this.robot = robot; this.calib = calib; this.margin = preloadMarginDeg; this.say = say;
    this.lastDir = { x: 0, y: 0 };
  }

  /** Take up the slack in + on both knobs so the backlash state is known. Leaves a short mark. */
  *preload() {
    for (const axis of ['x', 'y']) {
      if (!(yield* this.robot.graspKnob(axis))) return false;
      yield* this.robot.rotate(2 * this.calib.delta[axis] + this.margin);
      this.lastDir[axis] = 1;
    }
    return true;
  }

  *move(axis, mm) {
    if (mm === 0) return true;
    const s = mm > 0 ? 1 : -1;
    let deg = mm / this.calib.k[axis];
    const reversing = s !== this.lastDir[axis];
    if (reversing) deg += s * 2 * this.calib.delta[axis];    // backlash compensation
    const knob = axis === 'x' ? 'left' : 'right';
    if (this.robot.held !== axis) this.say(`Grasping the ${knob} knob`);
    if (!(yield* this.robot.graspKnob(axis))) return false;
    const sign = deg > 0 ? '+' : '−';
    this.say(`Turning the ${knob} knob ${sign}${Math.abs(deg).toFixed(0)}°` + (reversing ? ' (includes backlash take-up)' : ''));
    this.lastDir[axis] = s;                            // set first: if the page stops the turn, the slack is mostly taken up this way
    yield* this.robot.rotate(deg);
    return true;
  }

  *draw(moves, onMove = () => {}) {
    for (let i = 0; i < moves.length; i++) {
      onMove(i);
      if (!(yield* this.move(moves[i][0], moves[i][1]))) return false;
    }
    this.say('Releasing the knob');
    yield* this.robot.release();
    return true;
  }
}

/** Ideal polyline (array of [x, y] mm) for a move list starting at startMm. */
export function polyline(moves, startMm) {
  const pts = [[...startMm]];
  for (const [axis, mm] of moves) {
    const p = [...pts[pts.length - 1]];
    p[axis === 'x' ? 0 : 1] += mm;
    pts.push(p);
  }
  return pts;
}

export function resample(poly, spacing = 0.25) {
  if (poly.length < 2) return poly.map((p) => [...p]);
  const out = [];
  for (let i = 0; i + 1 < poly.length; i++) {
    const [a, b] = [poly[i], poly[i + 1]];
    const n = Math.max(1, Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1]) / spacing));
    for (let k = 0; k < n; k++) out.push([a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n]);
  }
  out.push([...poly[poly.length - 1]]);
  return out;
}

function meanAndMaxNearest(from, to) {
  let sum = 0, max = 0;
  for (const p of from) {
    let best = Infinity;
    for (const q of to) {
      const d = (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2;
      if (d < best) best = d;
    }
    best = Math.sqrt(best);
    sum += best; if (best > max) max = best;
  }
  return [sum / from.length, max];
}

/** Mean symmetric nearest-point distance between two polylines (mm), and the worst point. */
export function chamfer(drawn, target, spacing = 0.25) {
  const d = resample(drawn, spacing), t = resample(target, spacing);
  const [d2t, m1] = meanAndMaxNearest(d, t);
  const [t2d, m2] = meanAndMaxNearest(t, d);
  return { chamferMm: (d2t + t2d) / 2, maxMm: Math.max(m1, m2) };
}
