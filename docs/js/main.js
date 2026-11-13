import { loadSim } from './sim.js';
import { Backend, drain } from './backend.js';
import { OpenLoopDrawer, polyline, chamfer } from './draw.js';
import { ScreenPainter } from './kernel.js';
import { View } from './view.js';
import { Sketchpad } from './sketchpad.js';

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);

// ---- theme: ?theme=dark|light, else the system setting; a parent page can post {theme} to switch ----
let theme = params.get('theme') || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
document.documentElement.dataset.theme = theme;

const SWITCH_S = 3.1;                // measured in this simulation: release one knob, move, grasp the other
const PRESETS = {
  square: [[1, 0], [0, 1], [-1, 0], [0, -1]].flatMap((s) => Array(5).fill(s)),
  stairs: [...Array(4).fill([[1, 0], [0, 1]]).flat(), ...Array(4).fill([0, -1]), ...Array(4).fill([-1, 0])],
  spiral: [[4, [1, 0]], [4, [0, 1]], [3, [-1, 0]], [3, [0, -1]], [2, [1, 0]], [2, [0, 1]], [1, [-1, 0]], [1, [0, -1]]]
    .flatMap(([n, s]) => Array(n).fill(s)),
};

let sim, view, backend, drawer, painter, pad;
let job = null, target = null, startIndex = 0, speed = 8, screenDirty = false, running = false;
const screenCanvas = document.createElement('canvas');

function say(text) { $('status').textContent = text; }

/** Fresh scene: arm at ready, gripper measured, backlash taken up on both knobs. Runs faster than real time. */
function setup() {
  backend = new Backend(sim);                          // resets the simulation and the toy
  painter = new ScreenPainter(screenCanvas, sim.kernel, sim.meta.screen_px_per_mm);
  sim.onDraw = () => { screenDirty = true; };
  const k = sim.meta.kernel;
  const calib = {
    k: { x: k.x.k_mm_per_deg, y: k.y.k_mm_per_deg },
    delta: { x: k.x.half_dead_deg + k.x.transition_deg, y: k.y.half_dead_deg + k.y.transition_deg },
  };
  drawer = new OpenLoopDrawer(backend, calib, { say });
  drain(drawer.preload());
  drain(backend.release());
  painter.paint();
  job = null; target = null; running = false;
}

function estimate(moves) {
  const { motion, kernel: k } = sim.meta;
  let t = 0, dir = { ...drawer.lastDir }, held = backend.lastAxis;
  for (const [axis, mm] of moves) {
    const s = Math.sign(mm);
    let deg = Math.abs(mm) / k[axis].k_mm_per_deg;
    if (s !== dir[axis]) deg += 2 * (k[axis].half_dead_deg + k[axis].transition_deg);
    t += deg / motion.knob_speed_dps + motion.settle_s + (axis !== held || t === 0 ? SWITCH_S : 0);
    dir[axis] = s; held = axis;
  }
  return t;
}

function refresh() {
  const moves = pad.moves, idle = !running;
  $('sLen').textContent = `${Math.round(pad.lengthMm)} mm`;
  $('sMoves').textContent = `${moves.length}`;
  $('sTime').textContent = moves.length ? `~${Math.round(estimate(moves))} s` : '0 s';
  $('go').textContent = running ? 'Stop' : 'Draw it';
  $('go').disabled = idle && moves.length === 0;
  $('undo').disabled = $('clear').disabled = !idle || moves.length === 0;
  document.querySelectorAll('[data-preset]').forEach((b) => { b.disabled = !idle; });
  $('pad').classList.toggle('locked', !idle);
  const msg = $('msg');
  if (idle && pad.message) { msg.textContent = pad.message; msg.className = 'msg warn'; }
}

function note(text, warn = false) { $('msg').textContent = text; $('msg').className = warn ? 'msg warn' : 'msg'; }

function start() {
  const moves = pad.moves;
  if (!moves.length) return;
  target = polyline(moves, sim.kernel.stylus);
  startIndex = sim.kernel.points.length - 1;
  running = true; pad.enabled = false;
  const n0 = backend.nSwitches, t0 = backend.clock;
  job = (function* () {
    const ok = yield* drawer.draw(moves);
    return { ok, switches: backend.nSwitches - n0, seconds: backend.clock - t0 };
  })();
  note('Red is your line. Dark grey is what the pen draws.');
  pad.redraw(target); refresh();
}

function finish(result) {
  job = null; running = false; pad.enabled = true;
  $('clock').hidden = true;
  if (result && result.ok) {
    const score = chamfer(sim.kernel.points.slice(startIndex), target);
    say('Done. Draw another line from where the pen stopped.');
    note(`Drawn in ${result.seconds.toFixed(0)} s of arm time with ${result.switches} knob changes. ` +
         `Mean error ${score.chamferMm.toFixed(2)} mm, worst ${score.maxMm.toFixed(1)} mm.`);
  } else if (result) {
    say('The arm stopped.');
    note((backend.failure || 'The move failed.') + ' Reset the scene to try again.', true);
  } else {
    say('Stopped. The pen stays where it is.');
    note('');
  }
  target = null; pad.steps = []; pad.redraw(); refresh();
}

function stop() {
  job = (function* () { say('Stopping: releasing the knob'); yield* backend.release(); return null; })();
}

function reset() {
  say('Resetting the scene');
  setup(); view.sync(); view.screenChanged();
  pad.enabled = true; pad.steps = []; pad.redraw();
  note(''); say('Ready. Draw a line on the pad.'); $('clock').hidden = true; refresh();
}

// ---- main loop: run as many physics steps as the chosen speed needs, within a time budget per frame ----
let last = performance.now(), owed = 0;
function frame(now) {
  const dtFrame = Math.min((now - last) / 1000, 1 / 20); last = now;
  if (job) {
    owed = Math.min(owed + speed * dtFrame / sim.dt, 2000);
    const deadline = performance.now() + 14;
    let r = null;
    while (owed >= 1 && performance.now() < deadline) {
      r = job.next(); owed -= 1;
      if (r.done) break;
    }
    if (performance.now() >= deadline) owed = 0;       // slow device: run slower instead of freezing
    view.sync();
    $('clock').hidden = false;
    $('clock').textContent = `t = ${backend.clock.toFixed(1)} s · ${speed}×`;
    if (r && r.done) finish(r.value);
  }
  if (screenDirty) {
    screenDirty = false;
    painter.paint(); view.screenChanged();
    pad.redraw(target);
  }
  view.render();
  requestAnimationFrame(frame);
}

function setSpeed(v) {
  speed = v;
  document.querySelectorAll('[data-speed]').forEach((b) => b.setAttribute('aria-pressed', String(+b.dataset.speed === v)));
}

async function main() {
  try {
    sim = await loadSim((t) => { $('loadText').textContent = t; });
    $('loadText').textContent = 'Warming up the arm';
    await new Promise((r) => setTimeout(r, 30));       // let the text paint before the blocking set-up
    setup();
    view = new View($('view'), sim, screenCanvas, theme);
    pad = new Sketchpad($('pad'), () => ({
      pen: sim.kernel.stylus, travel: [sim.meta.kernel.x.travel_mm, sim.meta.kernel.y.travel_mm], source: screenCanvas,
    }), refresh);
    $('go').onclick = () => (running ? stop() : start());
    $('undo').onclick = () => pad.undo();
    $('clear').onclick = () => pad.clear();
    $('reset').onclick = reset;
    document.querySelectorAll('[data-preset]').forEach((b) => { b.onclick = () => pad.setSteps(PRESETS[b.dataset.preset]); });
    document.querySelectorAll('[data-speed]').forEach((b) => { b.onclick = () => setSpeed(+b.dataset.speed); });
    setSpeed(speed);
    window.addEventListener('message', (e) => {
      const t = e.data && e.data.theme;
      if (t === 'dark' || t === 'light') { theme = t; document.documentElement.dataset.theme = t; view.setTheme(t); }
    });
    $('loading').hidden = true;
    say('Ready. Draw a line on the pad.');
    pad.redraw(); refresh();
    window.__demo = { sim, pad, get backend() { return backend; }, get running() { return running; } };
    requestAnimationFrame(frame);
  } catch (err) {
    console.error(err);
    $('loadText').textContent = 'The simulation could not start in this browser.';
    $('loadText').nextElementSibling.nextElementSibling.textContent = String(err.message || err);
  }
}
main();
