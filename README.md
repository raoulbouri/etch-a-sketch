# Etch-a-Sketch robot artist

A Kinova Gen3 arm (6-DoF, Robotiq 2F-85 gripper) draws on a classic Etch A Sketch by turning its knobs one at a
time, watched by a scene camera (Intel RealSense D435) and the arm's wrist camera. An LLM agent will plan the
drawings and correct them from what the camera sees.

This repository holds the simulation and the code that runs unchanged on the real arm. Everything is written
against one skill interface (`src/etch_sim/interface.py`) with three backends:

| Backend | Status | Use |
|---|---|---|
| Toy kernel (`kernel.py`, `kernel_backend.py`) | done | planner and agent tests in milliseconds, with fault injection |
| MuJoCo (`src/etch_sim/mj/`) | done, except erase | grasping, reach, cameras, timing |
| Real arm (`src/etch_sim/real/`) | built; tested only against a simulated arm server | hardware |

## Simulation

The arm grasps a knob, turns it, releases, switches to the other knob, and the scene camera measures the line:

![Simulated arm drawing a spiral](media/sim_spiral.gif)

## Browser demo

`docs/` is a static page that runs the MuJoCo scene in the browser with the official MuJoCo WebAssembly
build: draw a line on a pad and the arm plans the knob turns and draws it. `docs/js/` is a JavaScript port
of `draw.py`, `mj/ik.py`, `mj/backend.py` and `kernel.py`; it reads the board pose from the simulation
instead of the markers. The page lives on the `web-demo` branch only.

```bash
uv run python scripts/export_web_scene.py   # regenerate docs/scene after changing the scene or toy numbers
python3 -m http.server -d docs 8000         # then open http://localhost:8000
```

`web/portfolio-page.md` is the draft text for the portfolio project page that embeds the demo.

## Sketch extraction from the real camera

The scene camera sees a faint grey line on a reflective grey screen. `sketch_extract.py` finds it through glare
and uneven light. Panels: front-on screen, lighting flattened, line strength, detected line (red):

![Sketch extraction on a real capture](media/sketch_extraction.png)

How it works: rectify the screen from its four corners, take the median of a 10-frame burst, keep only pixels
darker than the screen on *both* sides of a short line segment (so glare edges do not count), divide by local
brightness, mark saturated glare as unknown, then threshold with hysteresis and drop short, weak pieces.
Thresholds are tuned on two hand-drawn captures so far.

## Install

```bash
uv sync
uv run pytest -q
```

Python 3.12, dependencies in `pyproject.toml`. MuJoCo renders offscreen; no GPU needed.

## Run

Simulation (writes to `out/`, which git ignores):

```bash
uv run python scripts/s0_demo.py      # toy kernel: calibrate, erase, draw
uv run python scripts/s1_demo.py      # MuJoCo scene: cameras, markers, live screen
uv run python scripts/s2_demo.py      # arm turns the knobs, camera calibration, open-loop square (~30 s)
uv run python scripts/make_video.py   # video of the arm drawing; --shape square|staircase|spiral (~90 s)
uv run python scripts/fit_hand_measurements.py hardware/x_sweep.csv   # fit mm/deg and backlash from hand data
```

Real scene camera (read-only, no arm). Captures and results go to `test_images/`, which git ignores:

```bash
sudo sh scripts/rs_capture.sh grab               # 10 colour frames, 1280x720 (works over USB 2)
sudo sh scripts/rs_capture.sh --all              # colour + aligned depth + infrared (not yet run on the camera)
uv run python scripts/analyze_sketch.py --pick   # newest capture: click the 4 screen corners once
uv run python scripts/analyze_sketch.py          # newest capture -> test_images/analysis/<capture>_rgb/
```

On macOS the RealSense library needs `sudo`, and the wrapper briefly stops macOS's camera service; it always
restarts it, and gives up after 35 s. It uses the Python environment of the aims-kinova repo (`RS_PYTHON` to change).

Real arm: all motion goes through the motion server in the aims-kinova repo, inside a motion envelope that a
person approves when starting it. A refused or failed motion raises `ArmRefused`; nothing retries. Knob poses
are taught with `scripts/teach_pose.py` into `hardware/taught_poses.json`.

## Code layout

| Module | What it does |
|---|---|
| `interface.py` | `Skills`: grasp a knob, rotate, release, observe the pen, photo, locate the board, erase |
| `kernel.py` | The toy: knob angle to stylus with backlash, end stops, string strain, strokes, erase |
| `kernel_backend.py` | Skills on the kernel, with slip, missed grasps, pen noise, timing; ground truth kept apart |
| `calibration.py`, `draw.py`, `scoring.py` | Back-and-forth calibration sweep and fit; open-loop drawer; chamfer score |
| `perception.py` | ArUco board localisation, rectification, line mask and pen tracking (simulation) |
| `sketch_extract.py` | Line extraction for real camera images |
| `mj/` | 6-DoF Gen3 generated from Kinova's xacro, toy model, scene, IK, MuJoCo skills backend |
| `real/backend.py` | Client of the aims-kinova motion server |

`hardware/` holds the measurement template, taught-pose template and the scene camera's screen corners.

## Placeholders

Toy and timing numbers are guesses until measured on the real toy: mm per knob degree, backlash, string snap,
the erase model, switch time and knob speed, toy dimensions, camera field of view and mounts. They are marked
`PLACEHOLDER` in the code; measurements go in `hardware/phase0_measurements.yaml`.

## Branches

`main` is the stable branch. Development happens on `dev`, and `scripts/promote_main.sh` brings `main` up to date.

## Vendored models

- `models/vendor/kinova_gen3_6dof`: xacro and meshes from Kinovarobotics/ros_kortex (noetic-devel, c3280d8), BSD-3-Clause.
- `models/vendor/robotiq_2f85`: google-deepmind/mujoco_menagerie (98a70c5), BSD-2-Clause.

Licences are kept beside the files. These files are not edited; changes go in the scene builder code.
