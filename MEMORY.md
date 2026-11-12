# MEMORY.md: current state and durable facts for etch-sim

Update protocol is in `CLAUDE.md` (facts edited in place with a verified date; log is append-only,
newest first, timestamped from the system clock).

## Facts

### Project
- Team project, 4 people, 60 h, Nov 10 to Nov 30 2026. Proposal phases P0 to P4 with gates; sim stages S0 to S5
  run alongside. Proposal PDF: `etch_a_sketch_feasibility_plan.pdf` (not in the repo).
- Stages done: S0 (toy kernel), S1 (MuJoCo scene), S2 (arm skills in MuJoCo, pen tracking, calibration through the
  camera). Not started: real-arm backend, erase in MuJoCo, image-to-route planner, closed loop, agent. (verified 2026-11-09)
- Hardware: Kinova Gen3 6-DoF + Robotiq 2F-85 + wrist camera at AIMS (green arm). Real-arm code and rules live in
  the `aims-kinova` repo. No hardware Phase 0 measurement has been taken yet. (verified 2026-11-09)

### Environment
- macOS, `uv`, Python 3.12, MuJoCo 4.0.0, OpenCV 5.0.0 (ArUco is in the main package). Offscreen rendering works
  with no extra setup. (verified 2026-11-09)
- Git repo initialised locally, nothing committed, no remote. (verified 2026-11-09)

### Arm model
- MuJoCo Menagerie has only the 7-DoF Gen3. Our 6-DoF model is generated at load time from Kinova's xacro
  (`models/vendor/kinova_gen3_6dof`, ros_kortex noetic-devel @ c3280d8) by `src/etch_sim/mj/gen3.py`.
  Gripper is Menagerie `robotiq_2f85` (@ 98a70c5), attached with MjSpec at the URDF end-effector frame.
- Joints 1, 4, 6 are continuous; 2, 3, 5 limited. Joint 6 is the knob-turning joint (the proposal says "joint 7").
  URDF speed limit 70 deg/s on joints 4 to 6; the aims-kinova repo caps angular speed at 30 deg/s.
- Model checked against the real arm: forward kinematics vs three tool read-backs from the green arm agree within
  6 to 11 mm, with the real tool frame 0.12 m beyond the flange. (verified 2026-11-09)
- Positive joint 1 swings the tool towards world -y (same on the real arm).
- Gripper fingers open along the bracelet x axis; the wrist camera sits at bracelet +y. The sim wrist image shows
  the two fingertips in the bottom corners, as the real wrist frame does. (verified 2026-11-09)
- Wrist camera: 1280x720, vertical FOV 34.7 deg is a PLACEHOLDER (65 deg diagonal assumed); replace with
  `kinova intrinsics`.

### Toy and scene
- All toy numbers are PLACEHOLDERS: body 229 x 187 x 40 mm, screen 153 x 107 mm, knob radius 14 mm at (+-90, -68) mm,
  knob friction 0.03 Nm, mm per degree 0.085, dead zone 9 deg half + 4.5 deg transition, clockwise = positive,
  erase model, switch time 8 s. Record real values in `hardware/phase0_measurements.yaml`.
- The screen is not simulated by MuJoCo: knob joint angles drive the toy kernel, and the kernel's picture is
  uploaded into the screen texture (`EtchScene.sync_toy`). Texture row 0 is the top of the screen.
- Toy lies flat at world (0.45, 0, 0), knob edge towards the robot. Both knobs are reachable tool-down.
  Home pose puts the wrist directly over the toy and blocks an overhead camera; photograph from a knob or parked.
- Four ArUco 4x4 markers (ids 0 to 3, 20 mm) on the toy frame give the image-to-screen homography; at least 2 needed.
- Scene camera PLACEHOLDER: 1080p, 45 deg vertical FOV. A 0.6 mm line needs about 3 px/mm, so the toy must fill
  most of the frame (camera roughly 0.4 m away). At 720p from 0.6 m the line was under one pixel.

### S1 findings (sim, placeholder geometry) (verified 2026-11-09)
- Rectified camera image vs kernel picture: line offset about 0.1 mm for both camera mounts.
- With the gripper at a knob, over a full turn of joint 6: overhead camera sees at least 94% of the screen,
  the camera opposite the robot at least 99%; one bottom marker is hidden at some angles (3 of 4 remain).
- The arm casts a shadow across the screen under a single light. Line detection must cope with it (risk R7).
- `line_mask` ignores an 8 px border of the rectified screen, because a 1 px rectification error lets the red
  frame in and it was being counted as line.

### S2 findings (sim, placeholder geometry and friction) (verified 2026-11-09)
- `MujocoBackend` uses no ground truth: board pose from markers (0.7 mm error), pen from the scene camera
  (median 0.22 mm, 95th percentile 0.40 mm, max 1.3 mm), knob angle from joint 6.
- Calibration through the camera recovered hidden k within 0.4% and backlash within 1.2 deg on both axes; a 40 mm
  open-loop square drawn by the arm then had 0.25 mm chamfer and 0.33 mm worst side error.
- The stylus is assumed invisible, so the pen is the end of the newest line. Consequences: a back-and-forth sweep
  retraces its own line and cannot be measured, so `sweep_axis` is a zigzag (other knob turned 60 deg after each leg,
  13 knob switches for both axes); the tracker is blind during any retrace; a line end is half a line width beyond
  the stylus (`PenTracker.half_w`, PLACEHOLDER 0.3 mm), which biases backlash by about 3.5 deg if ignored.
- Pen tracker must walk along the line from the last pen position; picking any new dark pixel in the band locked
  onto the arm's moving shadow edge. `line_mask` is now a black top-hat (thin dark structures only).
- Fingertips are only 12 mm clear of each knob side at pre-grasp. A joint-space move bows sideways by several mm,
  so fingertip moves are cut into 10 mm straight pieces (`_move_pinch`).
- Sim arm servo: position + velocity feedforward. Position-only servos lagged a moving target by 2 to 4 deg
  (about 9 mm at the fingertip), which landed a finger on the knob. This is a sim controller choice, not a model
  of the Kortex controller. Arm and gripper bodies have gravity compensation.
- `base_link`-`shoulder_link` contact is excluded: base_link is fixed to the world, so MuJoCo's parent-child filter
  did not cover it and joint 1 was locked by the overlap.
- Sim numbers that are NOT predictions for hardware: knob slip 0.55% (placeholder friction), knob switch 3.1 s
  (ideal servo at 40 deg/s, no Kortex action overhead), toy drift 0.2 mm (perfect grasp alignment).
- Board pose in the world needs the scene camera's pose in the arm base frame. The sim reads it from the model;
  the lab needs an extrinsic calibration (new hardware task).

### Real arm interface (read from aims-kinova, `python/kinova`, 2026-11-09; nothing run on the arm)
- Motion goes through `kinova.motion` over one Kortex TCP session (`with KortexConnection(cfg) as k`), high-level
  mode only, 40 Hz. Each call blocks until ACTION_END / ACTION_ABORT, 30 s timeout, then `Base.Stop()`.
- Available: `move_to_pose` / `move_relative` (straight-line Cartesian, speed-capped 0.10 m/s and 30 deg/s),
  `move_joints` (absolute angles, NO speed argument), `run_stored_action`, `gripper(0..1)` with stall detection
  (a stall means something is held, and the stalled position gives its width), `stop`, `clear_faults`.
- Not available yet and needed for knob turning: a multi-turn, speed-limited rotation of joint 6. Joint angles are
  read back wrapped to (-180, 180], and absolute joint/pose targets cannot express more than half a turn or its direction.
- Safety checks act on the tool frame, which is 0.12 m beyond the flange (fingertip reference is 0.156 m):
  |p| <= 0.80 m, z >= 0.02 m, radial >= 0.15 m, step <= 0.30 m per command. The knob poses pass all four.
- That repo's rule 1 requires human approval of each specific motion command; a drawing is hundreds of commands.
  The approval model for a scripted routine is the developer's decision.
- Session inactivity timeout is 60 s; long pauses (an LLM call between drawings) need a reconnect or keep-alive.
- `kortex_api` 2.6.0 with protobuf 3.20.3 runs inside the Docker container (ROS Noetic, Python 3.8). etch-sim needs
  Python 3.12, so the real backend cannot simply import both in one interpreter; where it runs is undecided.
- Wrist camera frames come only through ROS `vision.launch` (RTSP) today; the Python package has intrinsics and
  the RTSP URLs. Measured colour intrinsics at 1280x720: fx 1298, fy 1299, cx 621, cy 238, i.e. vertical FOV 31.0 deg
  (the sim still uses the 34.7 deg placeholder) and a principal point 122 px above the image centre.
- Measured on the arm: Home and Retract about 2 s each; +-5 cm Cartesian error <= 0.5 mm.

### Real-arm backend (built 2026-11-09, tested only against a fake arm)
- aims-kinova, branch `etch-bounded-routine` (uncommitted): `kinova/envelope.py` (`Envelope`, `Routine`),
  `kinova/arms.py`, `kinova/server.py`, `motion.rotate_joint`, `envelopes/etch_draw.json` (PLACEHOLDER box),
  port 8765 published to localhost in `docker-compose.yml`, docs/05 section, bounded-routine clause under CLAUDE.md rule 1.
- etch-sim: `src/etch_sim/real/backend.py` (`ArmClient`, `RealBackend`, `webcam`), `hardware/taught_poses.json`
  (all poses null), `scripts/teach_pose.py`. Any refusal raises `ArmRefused`; nothing retries.
- Run without an arm: in aims-kinova `PYTHONPATH=python python3 -m kinova.server --envelope envelopes/etch_draw.json --fake --yes`,
  then use `RealBackend(ArmClient())`.
- First version uses taught knob poses with the toy in a fixture and keeps the wrist's current rotation between
  knobs. Not built: wrist-camera frames and board re-localization, erase routine, scene-camera capture is untested.
- Never run on the arm. Unverified there: speed constraint on joint moves, short-way rotation, gripper stall on
  the knob, the sign of joint 6 versus knob direction (`j6_deg_per_knob_deg`, PLACEHOLDER -1).

### Open decisions
- Decided 2026-11-09: bounded-routine approval (a person approves one envelope at the motion server); chunked
  joint 6 moves; a motion server in aims-kinova with etch-sim as client.
- One arm or two (blue 192.168.2.9 and green 192.168.2.10). Built for one so far.
- Scene camera hardware (plain USB webcam suggested) and mount.
- Which LLM API for the agent.

## Log

### 2026-11-12 17:32 EST | Raoul Bouri (+ Claude)
- First push to github.com/raoulbouri/etch-a-sketch (public). `main` = code without CLAUDE.md/MEMORY.md; `dev` =
  main + these two (force-added; both are in .gitignore). Rule 14 added; `scripts/promote_main.sh` updates main
  from dev in a temp worktree; CI `guard-main.yml` rejects the two files on main. Git identity set for this repo.
- Cleanup: removed `scripts/real_camera_check.py`, `ToyKernel.turn_both`, `out/`, `test_images/analysis/`, caches.
  Kept `test_images/rs/` (2 real captures, 21 MB, local only).
- README rewritten with `media/sim_spiral.gif` (1.5 MB) and `media/sketch_extraction.png` (real spiral capture).
### 2026-11-12 17:15 EST | Raoul Bouri (+ Claude)
- Capture 20261112-171326 (hand-drawn rectangular spiral, 7 turns, same camera pose: saved corners still fit).
  Whole spiral extracted as one component (24,986 px) + 3 small specks at the same screen spots as before
  (static scratches/edge). Inner turns ~2 mm apart stay separate except one short touch on the left.
- No target drawings (Raoul verifies by eye from panel.png). Ideas: blank-screen reference capture to cancel static
  clutter; axis-aligned shapes (one knob at a time) are easy to draw by hand and to score.
### 2026-11-12 17:01 EST | Raoul Bouri (+ Claude)
- First real colour capture (`rs_capture.sh grab`, D435 on USB 2, 1280x720 @ 6 fps, 10 frames): screen ~288 px wide
  = 1.88 px/mm; toy upside down in the view. Corners read by eye into `hardware/scene_camera_screen.json` (`rotate_180`);
  auto `find_screen` put the right edge ~8 px out on the bevel.
- `sketch_extract`: added component strength filter (short components need >= 8 noise sigma) and 2.5 mm border.
  Result on the drawn shape: whole shape found as one component, 97.6% of mask pixels; 3 small false specks
  near the bottom-left border and one short spur along a glare streak remain. No labelled truth yet, so no precision/recall.
- Next: centreline of the mask (for chamfer vs target), more drawings incl. faint/partly erased lines, labelled truth.
### 2026-11-12 16:56 EST | Raoul Bouri (+ Claude)
- First real rs_capture run: D435 fw 05.17.00.10 reports USB **2.1**; fixed 1920x1080 / 1280x720 @15 requests were
  refused ("Couldn't resolve requests"). USB 2 also explains why macOS shows only the D435 "Depth" (IR) webcam.
- Raoul reports the capture wrapper hangs the Mac. Likely cause: the root `killall UVCAssistant VDCAssistant` loop is a
  background job, which `sh` makes ignore Ctrl-C, so an interrupted capture left it running. Wrapper now: trap on exit,
  loop self-stops after 40 s, capture killed after 35 s (tested with a fake hanging capture, no camera).
  aims-kinova `scripts/realsense_mac.sh` has the same unbounded loop (not changed).
- rs_capture.py now picks modes from the device's own list (can_resolve), waits at most 3 s per frame, creates the
  output folder only after streaming starts. Not yet re-run on the camera. Test images deleted (Raoul: no test files kept).
### 2026-11-12 16:49 EST | Raoul Bouri (+ Claude)
- Constraints from Raoul: tripod cannot move closer (screen ~230 px wide, ~1.5 px/mm); RGB is the default capture,
  `--all` adds depth + IR; combining wrist + RealSense images is a later stage. All test images live in `test_images/`
  (git-ignored, Raoul cleans it); `out/real/` removed. Earlier real frames were already deleted.
- Added `scripts/rs_capture.py` + `rs_capture.sh` (sudo, aims-kinova `.venv` pyrealsense2 2.50; projector on only for depth),
  `src/etch_sim/sketch_extract.py`, `scripts/analyze_sketch.py`, `tests/test_sketch_extract.py` (18 tests pass).
- Synthetic camera view (1.5 px/mm, glare blob + streak, noise, line 10-25 grey levels): precision 0.96-0.985, recall 1.0
  within 1 mm. A first top-hat version scored precision 0.60 (glare-streak edges); fixed with a two-sided oriented test.
- Not yet run on a real capture: rs_capture never executed (needs Raoul's sudo). Thresholds are PLACEHOLDER.
### 2026-11-12 16:34 EST | Raoul Bouri (+ Claude)
- Real scene camera: Intel RealSense D435; macOS lists its colour stream as an ordinary webcam (no SDK/sudo needed for colour).
- Added `scripts/real_camera_check.py` (camera index, `--live`, or `--image`): markers, rectified screen, line mask -> `out/real/`.
  Verified only on a sim image (4 markers found). Real frames not yet captured: macOS denied camera access to the agent's app.
- Next: grant camera access or supply a photo, then judge line visibility on the real screen.
- 16:4x: first 3 real frames (out/real/20261112-163548_*) are black (99% of pixels = 0): taken from the D435 *Depth*
  UVC node. macOS now lists only "...435 with RGB Module Depth" at AVFoundation index 0; the "RGB" node seen earlier
  is gone. Line visibility still untested.
- 16:3752 frame: macOS "Depth" node delivers the left IR image (1280x720 grey) with the emitter's dot pattern on it.
  Board only ~230 px wide (~1.5 px/mm, sim used 4). Strokes faintly visible after CLAHE under dots + glare streaks.
  RGB/depth/aligned RGB-D need librealsense (pyrealsense2-macosx + sudo, as in aims-kinova docs/04). Frames deleted after analysis (user rule).

### 2026-11-09 22:57 EST | Raoul Bouri (+ Claude)
- Added `scripts/make_video.py` (`--shape square|staircase|spiral`, `--speed`, `--view`) and `MujocoBackend.frame_hook`
  (called every 10 sim steps). Dev deps: imageio, imageio-ffmpeg. A renderer that should show the live screen must be
  registered in `EtchScene._renderers`, otherwise its screen texture is never updated.
- The video draws open loop with the toy's true parameters (no calibration step); spiral: 151 s of arm time, 11 switches.

### 2026-11-09 22:50 EST | Raoul Bouri (+ Claude)
- With the developer's consent, applied the bounded-routine wording: `CLAUDE.md` rule 9 here, and a clause under
  rule 1 of aims-kinova's `CLAUDE.md`. Still open: `CLAUDE.md` "How to run" does not list `s2_demo.py`.

### 2026-11-09 22:15 EST | Raoul Bouri (+ Claude)
- Built the real-arm path with bounded-routine approval, in aims-kinova (branch `etch-bounded-routine`) and
  etch-sim `real/`. 20 offline tests pass there, 16 here; end-to-end run against the fake-arm server works,
  including refusal after an out-of-envelope command. No arm contact.
- Pending the developer's consent: wording changes to both CLAUDE.md files for the new approval model.

### 2026-11-09 21:14 EST | Raoul Bouri (+ Claude)
- Read the aims-kinova codebase to map it onto the `Skills` interface. No code changed, nothing run on the arm.
  Findings are under "Real arm interface"; three decisions added to "Open decisions".

### 2026-11-09 21:06 EST | Raoul Bouri (+ Claude)
- S2 built: `mj/backend.py` (`MujocoBackend`: locate board, grasp, rotate, release, observe pen), `PenTracker`,
  board pose from markers, zigzag `sweep_axis`, velocity-feedforward arm servos, gravity compensation.
- `scripts/s2_demo.py` exit check passes; 12 tests pass; S0 and S1 demos still pass. Findings above.
- `CLAUDE.md` "How to run" does not list `s2_demo.py` yet (needs the developer's consent to edit).
- Next options: real-arm backend (code only), or the image-to-route planner on the kernel.

### 2026-11-09 20:50 EST | Raoul Bouri (+ Claude)
- S1 built: 6-DoF Gen3 + 2F-85 scene, toy with hinged knobs and ArUco markers, wrist camera and two scene-camera
  mounts, live screen texture from the kernel, IK, `perception.py` (markers, homography, rectify, line mask).
- `scripts/s1_demo.py` exit check passes; 11 tests pass. Findings above.
- Added `CLAUDE.md` and this file. Next: S2 after go-ahead.

### 2026-11-09 ~20:20 EST (time approximate, entry written afterwards) | Raoul Bouri (+ Claude)
- S0 built: toy kernel, `Skills` interface, kernel backend with fault injection and `truth_log`, calibration fit,
  open-loop drawer, chamfer scoring. `scripts/s0_demo.py`: calibrated drawing 0.11 to 0.15 mm chamfer on
  placeholder parameters; a 40 mm square takes about 110 s of simulated time.
- The fit underestimates backlash (12.3 vs 13.5 deg) because the transition zone is not fully taken up in one step.
