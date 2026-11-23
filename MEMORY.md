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
- Repo on github.com/raoulbouri/etch-a-sketch; work on `dev` (rule 14). (verified 2026-11-22)
- Camera laptop: Ubuntu 22.04 (kernel 6.8), `uv sync --group realsense` (pyrealsense2 2.59.1); the D435 streams without
  Intel's udev rules; `MUJOCO_GL=egl uv run pytest -q` passes. (verified 2026-11-22)

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
  knob friction 0.03 Nm, transition 4.5 deg, clockwise = positive, erase model, switch time 8 s, except the measured
  knob k and backlash below (`ToyParams` defaults). Record real values in `hardware/phase0_measurements.yaml`.
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

### Real scene camera and knob calibration (verified 2026-11-12)
- Scene camera: Intel RealSense D435 (fw 05.17.00.10) on a tripod, on a USB 2 cable (no USB 3 cable available):
  colour 1280x720 @ 6 fps. Screen ~288 px wide = ~1.9 px/mm; extraction works (spiral and "5" captures).
- librealsense on macOS froze the Mac 3 times (kills macOS camera services); `rs_capture.sh` refuses on macOS
  (CLAUDE.md rule 15). Camera work runs on a teammate's Linux laptop: `uv sync --group realsense`.
- Raoul's toy, seen from the knob side: left knob cw = pen right, right knob cw = pen up (default plan, no --flip).
- Knob calibration by hand (README procedure), Raoul's toy, in the frame of the visible screen (verified 2026-11-22):
  left knob (x) 34.2 mm/turn = k 0.0951 mm/deg, backlash 21 deg (likely 20-30), rms 0.22 mm (calib/20261122-140755_x);
  right knob (y) 36.1 mm/turn = k 0.1001 mm/deg, backlash 17 deg (likely 15-25), rms 0.77 mm (calib/20261122-141546_y).
  k to ~1%; backlash +-8 deg (hand turns). In phase0_measurements.yaml and the `ToyParams` defaults.
- The 5% x/y difference in k may be a scale error: mm come from the PLACEHOLDER 153 x 107 mm screen. Measure the visible
  screen with a ruler and rescale k if it differs.
- End stops seen: left x ~3 mm, bottom y ~0-3 mm (under the frame lip / edge strip), right x >= 150 mm, top y ~104-105 mm.
- Session lessons: re-pick corners after every erase (lifting the toy moved it 3-12 mm); start the pen 15-20 mm in from
  both edges (a corner start puts row 1 in the ignored edge strip); the first turn off an end stop is mostly slack;
  the overhead light makes a diagonal band on the left that gives false short lines. Toy is not fixed to the table.
- Pen tracking on real images: `run_tracker.RunTracker` (one run = until the next erase): ignores lines from before the
  run, drops line that later disappears while visible (per pixel, 2 captures), accepts new line only where the turn
  puts the pen (k, backlash, direction). Used by hand_calibrate.py and RealBackend; MuJoCo still uses
  perception.PenTracker. Tolerances are PLACEHOLDER, set on the 2026-11-22 captures; rule 2 vs the arm hiding lines is
  untested until the arm draws. (verified 2026-11-22)
- Current camera pose: toy ~1.67 px/mm, corners in hardware/scene_camera_screen.json (capture 20261122-141512).

### Blue arm and the toy at the arm (verified 2026-11-22, Harish)
- Blue arm (192.168.2.9) from a Windows laptop on Ethernet (laptop 192.168.2.20/24). Readings are Web App data
  exports in hardware/arm_readings/20261122_blue/. Model FK vs the blue arm: 8-9 mm, ~1.2 deg at 5 poses; tool frame
  0.12 m beyond the flange (0 / 0.156 m give 126 / 31 mm). Kortex theta_x/y/z = fixed-axis XYZ Euler (0.7 deg vs the
  model; intrinsic is 120 deg off). Pinch point is 35.8 mm beyond the tool frame (model).
- aims-kinova runs natively on Windows without Docker for the CLI: `uv venv --python 3.10`, kortex_api 2.6.0.post3
  wheel with `--no-deps` (it pins protobuf 3.5.1), then protobuf 3.20.3, deprecated, pyyaml. Config comes only from
  environment variables (PYTHONPATH=python, KINOVA_ARM, KINOVA_USERNAME, KINOVA_PASSWORD). `kinova --json ping` and
  `status` work. The motion server has not been run natively yet; the arm repo says to run inside its container.
- Arm base sits on a ~2 cm adapter plate: the table is at about z = -0.02 m in the arm base frame (photo + readings).
- Toy taped flat next to the arm. Knob centres at grasp height (pinch, base frame): x (0.337, 0.095, 0.018),
  y (0.409, -0.089, 0.020); spacing 196.6 mm (ToyGeometry PLACEHOLDER 180; card photo 195-200); toy rotated +21.5 deg
  from square to the arm (finger yaw theta_z 111.2 at both knobs agrees). Which knob is x still to be confirmed by
  turning it by hand (the toy is turned sideways to the arm).
- Read knobs in Cartesian admittance with the gripper closed on the knob: the fingers pull the arm onto the knob's
  centre. A joystick-mode reading did not centre and was 7.9 mm off.
- `scripts/knob_poses.py` -> vertical approach (+5 cm) and grasp poses: grasp tool z 0.054-0.056 m, BELOW the
  placeholder envelope floor 0.07 (etch_draw.json). Proposed box: x 0.307-0.439, y -0.118-0.124, z 0.039-0.136.
- Envelope max_rotation_deg (1800) is per rotate command, not per routine. Xbox RB (the lab's "lock") puts the arm in
  ARMSTATE_IN_FAULT until LB clears it.
- First real run 2026-11-22 (native server, envelope etch_draw_blue_20261122 = box from knob_poses.py, ceiling 0.26):
  grasp of the x knob WORKED (computed poses put the fingers around the knob; gripper stalled at 0.712, held). The 30 deg
  rotate came back ACTION_ABORT with nothing moved; a later CLI move-rel aborted with abort_details 18 =
  ROBOT_MOVEMENT_IN_PROGRESS while arm_state was SERVOING_PLAYING_SEQUENCE / SERVOING_MANUALLY_CONTROLLED: the arm
  refuses API commands while the Web App's admittance actually holds the arm. arm_state SERVOING_MANUALLY_CONTROLLED
  by itself is harmless (any gripper command sets it; move-rel and rotate still work in it).
- Rotate aborts: TWO causes, both fixed (blue arm, 2026-11-22). (1) aims-kinova rotate_joint sent reach_joint_angles
  with JOINT_CONSTRAINT_SPEED: ACTION_ABORT, abort_details 1 (METHOD_FAILED) even in the air; the same joint targets via
  move_joints worked. Fix: JOINT_CONSTRAINT_DURATION = |chunk| / speed, min 1 s (patch on etch-bounded-routine, applied
  locally on Harish's laptop, NOT yet reviewed or pushed by Raoul). (2) Too fast: 30 deg in 1 s is refused the same way
  (code 1, no motion); 10 deg in 1 s and 30 deg in 2 s work. Fix: knob_speed_dps 15 in taught_poses.json. Every earlier
  "after a grasp" failure was a 30 deg turn and every success a 10 deg one, so the stalled gripper was most likely never
  the cause. The gripper-stall Base.Stop() patch (local only) is harmless but unproven as needed.
- Joint-6 sign: joint 6 +10 deg turned the x knob counter-clockwise seen from above, so knob clockwise = j6 negative;
  j6_deg_per_knob_deg x = -1 verified (y not yet checked). Gripper stalls on the x knob at 0.703-0.712.
- The pen is at the screen's bottom-left end stop after the session: do not turn the x knob counter-clockwise or the
  y knob down until it has moved away (end-stop strain on the string).
- One controller at a time: close the Web App's pose/admittance panel and leave the pad alone before scripts; check
  `kinova --json status` says ARMSTATE_SERVOING_READY. RB on the pad applies a stop (IN_FAULT); LB clears it.

### Open decisions
- Decided 2026-11-09: bounded-routine approval (a person approves one envelope at the motion server); chunked
  joint 6 moves; a motion server in aims-kinova with etch-sim as client.
- One arm or two (blue 192.168.2.9 and green 192.168.2.10). Built for one so far.
- Scene camera: D435 decided; mount and host (Linux laptop) settled 2026-11-12; USB 3 cable would give more modes.
- Which LLM API for the agent.

## Log

### 2026-11-22 20:33 EST | Harish Bhamitipadi Magesh (+ Claude)
- First full x-knob turn on the real arm: knob_dry_run x:30 at 15 dps -> grasp (stall 0.707), wrist +30.1 deg, release.
- Correction to 20:17: the remaining rotate abort was speed, not the gripper. x:10 then x:30 with the knob held, no new
  stall in between: 10 OK, 30 refused (code 1). At 15 dps (30 deg over 2 s) it works. knob_speed_dps 30 -> 15.

### 2026-11-22 20:17 EST | Harish Bhamitipadi Magesh (+ Claude)
- Correction to 20:03: holding the gripper at its stalled position does not stop the rotate abort. Test: grasp
  (stall + hold) -> `kinova stop` from a second session -> rotate -10 deg worked, knob turned clockwise with the wrist.
  So Base.Stop() ends the stalled gripper command and the grip stays. Fix moved to aims-kinova motion.gripper (stop on
  stall, 2 offline tests; local patch for Raoul). RealBackend hold removed again; 35 tests pass.
- Joint-6 sign confirmed twice for the x knob: j6 -10 deg = clockwise.

### 2026-11-22 20:03 EST | Harish Bhamitipadi Magesh (+ Claude)
- Rotate on the real arm diagnosed with in-air tests: move_joints (no constraint) turns joint 6, the server's rotate
  (JOINT_CONSTRAINT_SPEED) aborts with code 1; with a duration constraint it works (10.05 deg). Then rotate after a
  stalled grasp aborts (code 1) but works once the gripper is retargeted to its stalled position (10.06 deg, knob held).
- RealBackend.grasp_knob now holds the gripper at its stalled position (+1 test; 36 pass). taught_poses.json records
  the verified x-knob joint-6 sign. aims-kinova duration patch is local only, for Raoul to review.
- Next: knob_dry_run x:30 (clockwise, pen moves right off the corner) with both fixes; then y knob and its sign.

### 2026-11-22 19:32 EST | Harish Bhamitipadi Magesh (+ Claude)
- First real-arm run through RealBackend + the aims-kinova server (native Windows): grasp of the x knob worked, the
  30 deg rotate aborted with no motion. CLI recovery (gripper opened from the pad, move-rel up 5 cm) found abort
  code 18 (ROBOT_MOVEMENT_IN_PROGRESS) under manual control from the Web App/pad; fixed by turning admittance off.
- ArmClient refusals now include abort_details, rotated_deg, chunks (and explain code 18). Fixed
  test_untaught_poses_are_rejected, which read hardware/taught_poses.json and broke once real poses were written.
  35 tests pass. Next: retry x:30 with SERVOING_READY confirmed; if it aborts again, read its abort_details.

### 2026-11-22 18:59 EST | Harish Bhamitipadi Magesh (+ Claude)
- First session at the blue arm (Web App, Xbox, admittance; read-only CLI from a native Windows aims-kinova venv).
  Checked model FK, tool offset and Euler convention on the blue arm; placed and taped the toy; read both knobs.
- Added `src/etch_sim/real/knob_poses.py` and `scripts/knob_poses.py` (readings -> knob centres, toy placement,
  vertical approach/grasp poses, safety + envelope checks mirrored from aims-kinova, IK check on the model, envelope
  proposal) and `tests/test_knob_poses.py` (5 tests; 34 pass). taught_poses.json NOT written yet (knob labels unconfirmed).
- Proposed, not applied (rule 5): ToyGeometry knob spacing 196.6 mm; envelope floor 0.039 m (Raoul's file).

### 2026-11-22 14:42 EST | Yashasvi Sorapalli (+ Claude)
- Added `src/etch_sim/run_tracker.py` (Yashasvi's three rules for a whole run) and `tests/test_run_tracker.py` (6 tests);
  hand_calibrate.py and RealBackend.observe_pen use it (RealBackend now on sketch_extract masks, records wrist degrees).
  session.json now stores the corners and --flip used, so --replay of a later session uses its own corners. 29 tests pass.
- Replays (copies, with each session's own corners): x7 34.27 mm/turn (was 34.22), backlash 24.1 (21.1), 8 turns (step 1
  was at the wrong line end before); y 36.05 (same), backlash 18.5 (16.8). Bad sessions: x1 refused, x2 2 turns only,
  x6 34.4 mm/turn with the top-edge steps not found. Saved yaml / ToyParams values kept (differences within +-8 deg).
- Tuned on real masks: start_tol 3.5 mm (lines 1.5-2 mm wide), present_tol 1.0 mm (>1 camera px). Cost: a shadow
  touching a line end can leave ~1 mm of itself. Not done: MuJoCo backend, Skills start/end-run (decision B).
### 2026-11-22 14:19 EST | Yashasvi Sorapalli (+ Claude)
- Toy moved ~3-6 mm when lifted to erase; corners re-picked on 20261122-141512 (blank screen: 0 false lines).
- y session (test_images/calib/20261122-141546_y): clean, tracker on the line end at every step. Fit k 0.10014 mm/deg
  (36.05 mm/turn), backlash 16.8 deg (delta 8.4), rms 0.765 mm, 7 turns used, right knob cw = up. Saved to the yaml.
- Checks: same-direction turns 36.0, 34.75, 35.75, 36.25 mm (mean 35.7, k ~0.099); reversals lost ~0.7, 2.5, 3.0 mm
  (~7-30 deg). One overshoot estimate (-1.5 mm) used a connector at y 5.75 mm near the cut and is suspect.
- y k is ~5% above x k: real, or a screen-size error (scale assumes PLACEHOLDER 153 x 107 mm). Needs a ruler check.
  Camera server stopped. AxisParams / Facts updates proposed in chat, not applied.
### 2026-11-22 14:12 EST | Yashasvi Sorapalli (+ Claude)
- Corners re-picked on capture 20261122-140501 (erased screen). x session 7 (test_images/calib/20261122-140755_x) is the
  first clean x run: rows at y 5.8, 40.5, 76 mm, tracker on the line end at every step, fit k 0.09507 mm/deg (34.22 mm/turn),
  backlash 21.1 deg (delta 10.57), rms 0.224 mm, 7 turns used, cw = right. `--save` values kept in phase0_measurements.yaml.
- Checks: same-direction turns 35.0, 34.5, 34.0, 34.7, 34.0 mm (mean 34.4); reversals lost ~2.9 and ~2.4 mm (~25-30 deg if the
  end-cap bias is ~0), so backlash is ~20-30 deg (hand accuracy). Steps 1-2 split one turn unevenly (~20 + ~49 mm; pen started
  at x ~3 mm); their sum is right. Row 1 at y 5.8 mm is close to the 4 mm cut. Next: y session.
### 2026-11-22 14:02 EST | Yashasvi Sorapalli (+ Claude)
- Toy shifted twice more (4 px, then ~20 px); corners re-picked on capture 20261122-134903 (1.67 px/mm). Toy is not fixed down.
- x session 6 (test_images/calib/20261122-135447_x): steps 1-8 good, rows at y 40 and 76 mm. Step 5 is corrupt (Yashasvi turned
  cw to the right stop, then ccw). Steps 9-11 ran along the top edge (pen at y ~104 after step 8) and are lost. Session fit
  (k 0.0639, rms 27.9 mm) is invalid; yaml reverted.
- Left knob, from same-direction steps (end-cap bias cancels): cw 35.0, 35.0; ccw 34.75, 33.75 mm/turn -> 34.6 mm/turn,
  k ~0.096 mm/deg (+-1%, on the PLACEHOLDER 153 x 107 mm screen size). Repo fit_session on steps 1-8: k 0.0956, rms 0.28 mm.
  Backlash NOT measured (the only reversal is step 5). Stops: right >= 150 mm, top ~104-105 mm (visible-screen frame).
### 2026-11-22 13:11 EST | Yashasvi Sorapalli (+ Claude)
- Toy moved closer; corners re-picked on capture 20261122-130028: 1.69 px/mm, the light's white streak is off the screen.
  Drawn rows are now strong and continuous. A diagonal light band on the left third still gives false short lines.
- x session 2 (test_images/calib/20261122-130548_x): fit INVALID again (k 0.0899, backlash 1354 deg, tip bias -61 mm);
  yaml values reverted. Causes: row 1 again ran along the bottom edge (pen started at y ~0, x ~0-5 mm, not 15 mm in),
  so it is invisible; row 2 ran into the left end stop; the tracker picked clutter or the wrong line end on steps 1-3, 6, 8, 9.
- By eye from line ends, 6 clean turns: 32.5-34.5 mm/turn, ~33 mm/turn = ~0.092 mm/deg (about 2-3%). Backlash not measurable
  from this run. Left knob cw = pen right confirmed.
### 2026-11-22 12:47 EST | Yashasvi Sorapalli (+ Claude)
- Correction to 12:41: the "old vertical line" is a reflection of the overhead light, not a drawn line (Yashasvi).
- Why sketch_extract marks it as line: it peaks at ~200 grey, under the 245 glare threshold, so it is not masked; the
  bicubic upsampling (1.35 -> 4 px/mm) rings beside its sharp edge, a 4-6 level dark undershoot absent in the raw
  image, which the two-sided dip test accepts as a line. Drawn rows are only 1-5 levels deep (1-3%) and fade out where the
  light brightens the screen (grey ~160 vs ~140), which explains the gaps. Normalising by background is multiplicative;
  reflected light adds to line and screen alike. No code changed.
### 2026-11-22 12:41 EST | Yashasvi Sorapalli (+ Claude)
- New camera pose: corners re-picked on capture 20261122-123203 (screen ~207-242 px wide, 1.35 px/mm). Grab and serve
  work on Linux without Intel's udev rules.
- First x session (test_images/calib/20261122-123517_x) is INVALID: an old vertical line was not erased, and the pen
  started ~2.5 mm from the bottom edge, so row 1 (3 cw turns) left no visible line; the tracker then jumped to specks and
  to crossings with the old line. Fit used 3 turns: k 0.121 mm/deg, backlash 194 deg (nonsense). The yaml values
  `--save` wrote were reverted. By eye from the masks, two clean cw turns span ~68 mm (~34 mm/turn); not a measurement.
### 2026-11-22 12:30 EST | Yashasvi Sorapalli (+ Claude)
- Linux laptop setup for the hand knob calibration (Ubuntu 22.04, kernel 6.8): `dev` checked out, `uv sync --group
  realsense` (pyrealsense2 2.59.1), `MUJOCO_GL=egl uv run pytest -q`: 23 pass. D435 (fw 5.17.0.10, USB 2.1) is listed
  by pyrealsense2 without Intel's udev rules installed. Next: grab, corner pick, then `hand_calibrate.py x` and `y`.
### 2026-11-12 18:37 EST | Raoul Bouri (+ Claude)
- Reverted the macOS AVFoundation camera path (avcam.py, cam_grab.py removed; imageio-ffmpeg back to dev deps).
  Back to librealsense, but on Linux only: Raoul or a teammate will calibrate on a non-macOS laptop.
- `rs_capture.sh`: Linux runs as the user with no service-killing (pyrealsense2 via `uv sync --group realsense`,
  Linux-only dependency group); macOS still refuses. `hand_calibrate.py` uses the `serve` camera server again.
- Kept: toy-order corner picking (BL near left knob, BR, TR, TL). Fixed: `--save` used summary keys the fit no
  longer has (would have crashed at the end of a session). CLAUDE.md: rule 15 + camera commands (Raoul asked for
  the docs to carry the calibration process). Tested with a stand-in camera on the Mac only; 23 tests pass.
### 2026-11-12 18:30 EST | Raoul Bouri (+ Claude)
- The single-push-aside camera server ALSO froze the Mac (forced restart 18:22, before any capture). Decision:
  no librealsense on the Mac. `rs_capture.sh` now refuses on macOS (exit 3); rs_* scripts kept for a Linux host.
- New path: `etch_sim.avcam` + `scripts/cam_grab.py` read the D435 through AVFoundation (ffmpeg from imageio-ffmpeg,
  now a runtime dependency): no sudo, nothing killed. Over USB 2 that is the LEFT IR camera (grey, 1280x720) with
  the dot projector on: Raoul to tape over the projector. Raoul declined using the iPhone (Continuity Camera).
- Corners are now clicked in toy order (BL near left knob, BR, TR, TL; `order` in scene_camera_screen.json), so
  no rotate flag is needed; old image-order RGB corners are ignored. hand_calibrate uses avcam directly.
- Not yet run on the camera: my app has no camera permission; Raoul runs cam_grab.py from his terminal. 23 tests pass.
### 2026-11-12 18:17 EST | Raoul Bouri (+ Claude)
- Two hand-calibration runs froze the Mac at the first per-turn capture. Crash report (WindowServer watchdog,
  18:09): WindowServer main thread stuck in CoreMediaIO. Cause: a fresh `sudo rs_capture.sh grab` per turn, each
  killing UVCAssistant/VDCAssistant again while they were still respawning from the previous one.
- Fix: `scripts/rs_server.py` (`sudo sh scripts/rs_capture.sh serve`) opens the camera once per session and serves
  bursts over 127.0.0.1:8766; hand_calibrate.py now asks it. Wrapper kill loop stops as soon as the camera streams
  (ready file), max 20 s. Tested with a stand-in camera library only; not yet on the real camera.
- Raoul's toy (seen from the knob side): left knob cw = pen right, right knob cw = pen up (matches default plan, no --flip).
  Failed sessions deleted.
### 2026-11-12 17:58 EST | Raoul Bouri (+ Claude)
- Added hand calibration with full turns only (no angle dial): `src/etch_sim/turn_measure.py` (pen after each turn
  from before/after masks; fit), `scripts/hand_calibrate.py` (guided session, sudo grab per turn, `--replay`, `--save`,
  `--flip x|y`), `tests/test_turn_measure.py` (4 tests; 22 total pass). scikit-image was tried for thinning
  (centreline ends retreated too far) and removed again.
- Key finding: the measured line end overshoots the pen by ~half the line width in the direction of travel; that
  flips sign on reversal and reads as backlash (first version: 3 deg measured vs 24 true). Fixed by measuring the
  overshoot from the connector line drawn between rows. Backlash per row (360n - L/k) was abandoned: 1% k error = 11 deg.
- Synthetic camera sessions, exact turns: k within 0.2%, backlash within 2-3 deg (camera ~0.5 mm/px floor). With
  +-4-6 deg hand error per turn: k ~1-2%, backlash +-6-8 deg. Precise backlash should come from the arm (exact angles).
- Points within 4 mm of the screen edge are excluded (border margin clips them). Not yet run on the real toy.
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
