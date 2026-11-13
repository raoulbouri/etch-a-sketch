// The drawing pad. An Etch A Sketch draws one unbroken line, and the arm has one gripper, so it can
// turn only one knob at a time. A drawable sketch is therefore a single path of horizontal and
// vertical segments. The pad snaps the pointer to a grid anchored at the pen and enforces a budget.

export const GRID_MM = 8.5;          // 100 degrees of knob at 0.085 mm/deg
export const MARGIN_MM = 5;          // keep the pen off the end stops
export const MAX_LENGTH_MM = 255;    // 30 grid steps: about 100 s of arm time
export const MAX_MOVES = 24;

/** Grid steps [[dx, dy], ...] -> moves [[axis, mm], ...], merging steps that continue straight on. */
export function stepsToMoves(steps) {
  const moves = [];
  for (const [dx, dy] of steps) {
    const axis = dx !== 0 ? 'x' : 'y', mm = (dx || dy) * GRID_MM;
    const last = moves[moves.length - 1];
    if (last && last[0] === axis && Math.sign(last[1]) === Math.sign(mm)) last[1] += mm;
    else moves.push([axis, mm]);
  }
  return moves;
}

export class Sketchpad {
  /** getState: () => ({pen: [x, y] mm, travel: [w, h] mm, source: canvas of what is already drawn}) */
  constructor(canvas, getState, onChange) {
    this.canvas = canvas; this.getState = getState; this.onChange = onChange;
    this.ctx = canvas.getContext('2d');
    this.steps = [];
    this.enabled = true;
    this.colors = { plan: '#d92d20', pen: '#15803d', grid: 'rgba(40,40,40,0.28)' };
    this.message = '';
    let drawing = false;
    canvas.addEventListener('pointerdown', (e) => {
      if (!this.enabled) return;
      drawing = true; canvas.setPointerCapture(e.pointerId); this._follow(e); e.preventDefault();
    });
    canvas.addEventListener('pointermove', (e) => { if (drawing) this._follow(e); });
    const end = () => { drawing = false; };
    canvas.addEventListener('pointerup', end);
    canvas.addEventListener('pointercancel', end);
    new ResizeObserver(() => this.redraw()).observe(canvas);
  }

  get moves() { return stepsToMoves(this.steps); }
  get lengthMm() { return this.steps.length * GRID_MM; }

  _node() {                          // current end of the path, in grid steps from the pen
    return this.steps.reduce((p, s) => [p[0] + s[0], p[1] + s[1]], [0, 0]);
  }

  _inside(i, j) {
    const { pen, travel } = this.getState();
    const x = pen[0] + i * GRID_MM, y = pen[1] + j * GRID_MM;
    return x >= MARGIN_MM && x <= travel[0] - MARGIN_MM && y >= MARGIN_MM && y <= travel[1] - MARGIN_MM;
  }

  /** Try to add one grid step. Stepping back along the last step removes it. */
  addStep(dx, dy) {
    const last = this.steps[this.steps.length - 1];
    if (last && last[0] === -dx && last[1] === -dy) { this.steps.pop(); this.message = ''; return true; }
    const [i, j] = this._node();
    if (!this._inside(i + dx, j + dy)) { this.message = 'That is the edge of the drawable area.'; return false; }
    if (this.steps.length * GRID_MM + GRID_MM > MAX_LENGTH_MM + 1e-9) { this.message = 'Line length limit reached. Undo or draw what you have.'; return false; }
    if (stepsToMoves([...this.steps, [dx, dy]]).length > MAX_MOVES) { this.message = 'Corner limit reached. Undo or draw what you have.'; return false; }
    this.steps.push([dx, dy]); this.message = '';
    return true;
  }

  _follow(e) {
    const r = this.canvas.getBoundingClientRect();
    const { pen, travel } = this.getState();
    const x = (e.clientX - r.left) / r.width * travel[0];
    const y = (1 - (e.clientY - r.top) / r.height) * travel[1];
    const ti = Math.round((x - pen[0]) / GRID_MM), tj = Math.round((y - pen[1]) / GRID_MM);
    for (let guard = 0; guard < 64; guard++) {
      const [i, j] = this._node();
      const di = ti - i, dj = tj - j;
      if (di === 0 && dj === 0) break;
      const step = Math.abs(di) >= Math.abs(dj) ? [Math.sign(di), 0] : [0, Math.sign(dj)];
      if (!this.addStep(...step)) break;
    }
    this.redraw(); this.onChange();
  }

  setSteps(steps) {
    this.steps = [];
    for (const s of steps) if (!this.addStep(...s)) break;
    this.message = '';
    this.redraw(); this.onChange();
  }

  undo() {
    const moves = this.moves;
    if (!moves.length) return;
    const n = Math.round(Math.abs(moves[moves.length - 1][1]) / GRID_MM);   // remove the last straight segment
    this.steps.length -= n; this.message = '';
    this.redraw(); this.onChange();
  }

  clear() { this.steps = []; this.message = ''; this.redraw(); this.onChange(); }

  redraw(target = null) {
    const { pen, travel, source } = this.getState();
    const c = this.canvas, dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.round(c.clientWidth * dpr), h = Math.round(c.clientWidth * travel[1] / travel[0] * dpr);
    if (!w) return;
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    const ctx = this.ctx, s = w / travel[0];
    const X = (mm) => mm * s, Y = (mm) => h - mm * s;
    ctx.drawImage(source, 0, 0, w, h);
    if (this.enabled) {
      ctx.fillStyle = this.colors.grid;
      for (let i = -20; i <= 20; i++) for (let j = -14; j <= 14; j++) {
        if (!this._inside(i, j)) continue;
        ctx.beginPath(); ctx.arc(X(pen[0] + i * GRID_MM), Y(pen[1] + j * GRID_MM), 1.3 * dpr, 0, 7); ctx.fill();
      }
    }
    const path = target ?? (() => {
      const pts = [[...pen]];
      for (const [dx, dy] of this.steps) { const p = pts[pts.length - 1]; pts.push([p[0] + dx * GRID_MM, p[1] + dy * GRID_MM]); }
      return pts;
    })();
    if (path.length > 1) {
      ctx.strokeStyle = this.colors.plan; ctx.lineWidth = (target ? 1.2 : 2.5) * dpr; ctx.lineJoin = 'round'; ctx.lineCap = 'round';
      ctx.globalAlpha = target ? 0.75 : 1;
      ctx.beginPath(); ctx.moveTo(X(path[0][0]), Y(path[0][1]));
      for (const p of path.slice(1)) ctx.lineTo(X(p[0]), Y(p[1]));
      ctx.stroke(); ctx.globalAlpha = 1;
    }
    ctx.fillStyle = this.colors.pen;
    ctx.beginPath(); ctx.arc(X(pen[0]), Y(pen[1]), 4 * dpr, 0, 7); ctx.fill();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5 * dpr; ctx.stroke();
  }
}
