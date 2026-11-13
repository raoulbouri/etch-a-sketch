# Portfolio project page: Etch A Sketch robot artist

Draft content for a project page on raoulbouri.github.io. The site's source is not in this repository,
so this file holds the copy, the embed code and the media list, in the same order as the F1TENTH page.
This folder is not published; GitHub Pages serves only `docs/`.

## Card and header

- **Slug:** `etch-a-sketch-robot`
- **Title:** Etch A Sketch Robot Artist
- **Status:** in progress · **Code:** public, https://github.com/raoulbouri/etch-a-sketch
- **Dates:** Nov 2026 to present
- **Summary:** A Kinova Gen3 arm draws on a classic Etch A Sketch by gripping and turning its two knobs,
  one at a time. The same skill code runs on a toy model, in MuJoCo and on the real arm, and the MuJoCo
  version runs live in your browser below.
- **Tags:** Manipulation · MuJoCo · Sim-to-Real · Inverse Kinematics · Calibration · Computer Vision ·
  Kinova Gen3 · WebAssembly

## Live demo (top of the page, in place of a video)

Caption: *Live physics in your browser. Draw a line on the pad and the simulated arm plans the knob
turns and draws it.*

```tsx
// Click to load: the demo downloads about 20 MB, so it should not load for every visitor.
const [live, setLive] = useState(false);
const src = `https://raoulbouri.github.io/etch-a-sketch/?theme=${dark ? "dark" : "light"}`;

{live ? (
  <iframe
    src={src}
    title="Etch A Sketch robot: live MuJoCo simulation"
    className="w-full h-[1180px] md:h-[640px] rounded-lg border"
    allow="fullscreen"
  />
) : (
  <button onClick={() => setLive(true)} className="relative w-full">
    <img src="/projects/etch-a-sketch/sim_spiral.gif" alt="Simulated arm drawing a spiral" />
    <span>Load the live simulation (about 20 MB)</span>
  </button>
)}
```

The demo stacks its two panels below 760 px, which is why the frame is taller on phones. To follow the
site's theme toggle after loading, post a message to the frame:
`iframe.contentWindow.postMessage({ theme: "dark" }, "https://raoulbouri.github.io")`.

## At a glance

- **Built:** One skill interface (grasp a knob, rotate, release, observe the pen) with three backends:
  a millisecond toy model for planner tests, a MuJoCo scene where the arm really grips and turns the
  knobs, and a client for the real arm. Drawing code is written once and runs on all three.
- **Hardware:** Kinova Gen3 6-DoF arm · Robotiq 2F-85 gripper · Intel RealSense D435 scene camera ·
  wrist camera · a stock Etch A Sketch
- **Status:** Simulation draws closed shapes open loop to about 0.1 mm mean error on the simulated toy.
  The real-arm backend is built and tested against a simulated arm server. Line extraction works on
  real camera captures. Calibration of the real toy and the first drawing on hardware are next.

## Body

### Why this is a hard manipulation problem

An Etch A Sketch gives the robot almost nothing to work with.

- **One gripper, two knobs.** The arm can turn only one knob at a time, so every drawing is a single
  unbroken path of horizontal and vertical strokes. Each change of knob costs about three seconds.
- **The pen is invisible.** Only the line it leaves can be seen, and it is a faint grey line on a
  reflective grey screen.
- **The knobs have backlash.** After a reversal, the knob turns about 27 degrees before the pen moves
  again. A controller that ignores this draws every corner about 2 mm short.
- **Mistakes are permanent.** The only eraser is turning the toy over and shaking it.

### One interface, three backends

`interface.py` defines the skills. The toy kernel models the toy alone: knob angle to pen position
with backlash, end stops and string strain. It runs in milliseconds and can inject faults such as a
slipped grasp, which makes it the test bed for planning. The MuJoCo backend adds the arm, the
gripper, contact and cameras. The real backend sends the same calls to the arm's motion server,
inside a motion envelope a person approves.

### The backlash model

For each axis, the knob angle is θ and the internal pulley angle is φ. The pen sits at k·φ, and the
gap is e = θ − φ.

- |e| below the dead zone: the pen does not move.
- Through a short transition zone the gain ramps from 0 to 1.
- At the full gap δ the pulley follows the knob rigidly.

Once the slack is taken up in direction s, the pen position is y = k·(θ − s·δ) + c. The drawer
therefore adds 2δ of knob rotation to the first turn after each reversal. Calibration turns each knob
back and forth and fits k and δ from the pen positions the camera measures.

### Grasping and turning in MuJoCo

The 6-DoF Gen3 model is generated from Kinova's own xacro file, because MuJoCo Menagerie ships only
the 7-DoF arm. A damped least-squares IK solver places the fingertips over a knob with the tool
pointing straight down. It constrains five degrees of freedom and leaves rotation about the tool axis
free. For a tool pointing down, that free rotation is joint 6, which is the joint that then turns the
knob. Straight-line moves are cut into 1 cm pieces, because a single joint-space move bows sideways by
several millimetres, enough to land a finger on the knob.

### Seeing a grey line on a grey screen

Media: `media/sketch_extraction.png`. Caption: *Front-on screen, lighting flattened, line strength,
detected line.*

The extractor rectifies the screen from its four corners and takes the median of a 10-frame burst. It
keeps only pixels that are darker than the screen on both sides of a short line segment, so the edges
of glare patches do not count. It then divides by local brightness, marks saturated glare as unknown
and thresholds with hysteresis.

### The browser demo

The demo at the top of this page is the same MuJoCo scene compiled to WebAssembly. The drawer, the IK
solver, the motion code and the toy model were ported from Python to JavaScript, and the port reproduces
the Python backend's knob angles to within 0.0001 rad on the same command sequence. It is open
loop and uses the toy's nominal parameters, which are placeholders until the real toy is measured.

### What is next

- Measure the real toy's gear ratio and backlash, by hand and then with the arm.
- First drawing on hardware, then closed-loop correction from the scene camera.
- An LLM agent that plans drawings and corrects them from what the camera sees.

## Media to copy into the portfolio

- `media/sim_spiral.gif`: demo poster and project card image
- `media/sketch_extraction.png`: perception section
