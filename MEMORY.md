# MEMORY.md: current state and durable facts for etch-sim

Update protocol is in `CLAUDE.md` (facts edited in place with a verified date; log is append-only,
newest first, timestamped from the system clock).

## Facts

### Project
- Team project, 4 people, 60 h, Nov 10 to Nov 30 2026. Proposal phases P0 to P4 with gates; sim stages S0 to S5
  run alongside. Proposal PDF: `etch_a_sketch_feasibility_plan.pdf` (not in the repo).
- Stages done: S0 (toy kernel), S1 (MuJoCo scene), S2 (arm skills in MuJoCo, pen tracking, calibration through the
  camera), S3 (image-to-route planner, toy kernel only). Real arm: base test passed (blue arm, 30 mm square open loop,
  2026-11-22). Not started: erase in MuJoCo, closed loop, agent; not run on the arm: other shapes, planner pictures,
  erase, camera feedback. (verified 2026-11-22)
- Hardware: Kinova Gen3 6-DoF + Robotiq 2F-85 + wrist camera at AIMS: blue arm (192.168.2.9, used for the base test)
  and green arm (192.168.2.10). Real-arm code and rules live in the `aims-kinova` repo. Measured so far: knob k and
  backlash by hand; knob speed limit and switch time on the arm. (verified 2026-11-22)

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
  `status` work, and so does the motion server (`python -m kinova.server --envelope ...`), used for the whole
  2026-11-22 session; the arm repo says to run inside its container (deviation, to clear with Raoul).
- Arm base sits on a ~2 cm adapter plate: the table is at about z = -0.02 m in the arm base frame (photo + readings).
- Toy taped flat next to the arm. Knob centres at grasp height (pinch, base frame): x (0.337, 0.095, 0.018),
  y (0.409, -0.089, 0.020); spacing 196.6 mm (ToyGeometry PLACEHOLDER 180; card photo 195-200); toy rotated +21.5 deg
  from square to the arm (finger yaw theta_z 111.2 at both knobs agrees). x = the left knob seen from the knob side, at
  base y +0.095 (confirmed by turning it). Photo of the setup: hardware/draw_runs/20261122_square30.jpg.
- Read knobs in Cartesian admittance with the gripper closed on the knob: the fingers pull the arm onto the knob's
  centre. A joystick-mode reading did not centre and was 7.9 mm off.
- `scripts/knob_poses.py` -> vertical approach (+5 cm) and grasp poses: grasp tool z 0.054-0.056 m, BELOW the
  placeholder envelope floor 0.07 (etch_draw.json). Envelope used: `hardware/envelopes/etch_draw_blue_20261122.json`,
  box x 0.307-0.439, y -0.118-0.124, z 0.039-0.26 (copy here; Raoul's repo still has the placeholder).
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
- Kortex can report a downward tool as (~0, ~180, theta_z + 180) instead of (180, 0, theta_z), e.g. after a 180 deg
  knob turn. Never feed a read-back theta_z into a (180, 0, .) target; RealBackend uses down_yaw_deg (tool heading).
- Measured on the blue arm: 180 deg of knob moved the pen ~15 mm (x) and ~16 mm (y) by eye, as ToyParams predicts;
  a knob switch (release + grasp) takes ~10 s; grasp stalls at 0.703-0.712 on both knobs.
- Joint-6 sign: joint 6 +10 deg turned the x knob counter-clockwise seen from above, so knob clockwise = j6 negative;
  j6_deg_per_knob_deg = -1 verified for x and y (clockwise moves the pen right / up).
- Pen position is only known by eye (draw_shapes.py --start). Before any shape, move the pen >= 5 mm off the stops;
  driving a knob into an end stop strains the string.
- One controller at a time: close the Web App's pose/admittance panel and leave the pad alone before scripts; check
  `kinova --json status` says ARMSTATE_SERVOING_READY. RB on the pad applies a stop (IN_FAULT); LB clears it.

### S3 planner (`src/etch_sim/planner.py`, kernel only) (verified 2026-11-22)
- `plan_route(image, PlanParams, pen_mm)` -> `Route(moves, start_mm, polyline, target_mm, stats)`; `plan_for_budget` picks the
  smallest staircase tolerance that fits a time budget. Grid pitch 2 mm, margin 8 mm, default tol 1.5 x pitch.
- Pen never lifts: separate pieces and the pen are joined by an MST of shortest L connectors; a Chinese-postman retrace
  (greedy matching) makes one open path from the pen; Euler path prefers going straight to save knob switches.
- Tolerance is the lever, not pitch: circle at 2 mm pitch, tol 0 -> 109 switches (~17 min), tol 3 mm -> 36 (~7 min,
  1.2 mm mean error); tol below one pitch does nothing. 4 mm pitch at tol 0 still needs 51.
- s3_demo at 600 s budget: house/circle/heart/3 pieces in 6.3-9.1 min est (kernel clock within 3%), picture->route
  1.1-1.6 mm, drawn vs plan 0.12-0.13 mm chamfer with true k/backlash. Timing PLACEHOLDERS (switch 8 s, 30 deg/s).
- Known cost: connectors join mid-line (T-junctions), so pieces force retrace (3 pieces: 200 mm). Joining at loose
  ends would cut it. Route->picture includes connector ink, so it is reported, not checked.

### Green arm board view and markers (verified 2026-11-24, Raoul)
- Board markers: ArUco DICT_4X4_50, black squares printed 21.6 mm (designed 22.5; the print was scaled ~96 %), from
  `scripts/make_marker_sheet.py`. Seen from the knob side: id0 top-left, id1 top-right, id2 bottom-right, id3 bottom-left
  (id2/id3 sit next to the knobs). Ruler centre distances in `hardware/phase0_measurements.yaml` (planar to 0.5 mm);
  one wrist frame with 21.6 mm reproduces them within 2 mm, so the Kortex focal length is fine.
- Green arm tool transform z 0.12 m (read 2026-11-24), same as blue.
- Wrist-camera board view, green arm (192.168.2.10): joints [-38.01, -50.723, -139.148, 3.737, -63.082, 102.073] deg,
  tool [0.1225, 0.0634, 0.3037 m, 152.58, 10.89, 137.51 deg]. All 4 markers seen at 50-65 px. Reading in
  `hardware/arm_readings/20261124_green/board_view.json`; reuse with `kinova move-joints` (motion: needs approval).
- Green wrist colour intrinsics (Kortex, 1280x720): fx 1297.7, fy 1298.6, cx 620.9, cy 238.3, zero distortion. Focal
  length confirmed (ruler, and the 20-view joint fit: 1285-1292). The 1920x1080 profile (cy 97) is wrong.
  CORRECTION (19:04): the earlier "cy 238 is wrong, cy 360 right" rested on assuming the table sits at -1.8 mm in the
  base frame. The 20-view arm run puts the chart at -15.4 mm in Kortex's base frame consistently (z std 2 mm), and its
  joint fit lands on cy ~238 with the camera tilted ~3 deg from Kinova's mount model. cy and mount tilt are nearly
  interchangeable in the images, so which is "right" is open; what matters is the combined chain. Absolute height in
  the base frame is unverified (13.6 mm off the pad measurement) until a touch/hover test. PLACEHOLDER.
- Wrist calibration run 2026-11-24 (green arm, `scripts/wrist_calib.py`, session test_images/wrist_calib/20261124-185020):
  20 views planned by the script and driven through the motion server (envelope hardware/envelopes/wrist_calib_10_20261124.json,
  approved by Raoul). Joint fit rms 4.0 px (target < 0.5): NOT saved. Chart-only fit rms 1.7 px and fx 1366: the paper
  chart is not flat (single views miss a plane by up to 2.8 px). Chart printed at 98.7 % (29.6 mm squares).
  Single-view repeatability of the chart centre in the base frame: std 1.0-2.0 mm, worst 4.8 mm, yaw 0.13 deg.
- Second run 2026-11-24 19:20 (chart re-taped, plan limited to tool 0.17-0.25 m from the base axis: Raoul says 0.44 m is
  too far on this table; prefers higher views): 25 moves, 20 views, no refusals. Taping helped little (plane-fit 1.9 -> 1.6 px).
- PROVISIONAL wrist calibration saved in hardware/wrist_camera_calib.json: both sessions (42 views) in one joint fit with a
  shared lens + mount: fx 1284.8 fy 1290.8 cx 670.4 cy 242.9, distortion ~0, mount 3.2 mm / 2.3 deg from Kinova's model,
  chart printed at 98.5 %. Joint rms 4.4 px vs 1.6 px for the chart alone: the extra ~4 px is pose/image disagreement
  (Kortex absolute accuracy). Our MuJoCo FK is worse (5-8 mm from Kortex, rms 29-41 px): use Kortex's reported pose.
  Single-view chart repeatability 1.0-2.0 mm std, worst 5 mm. Chart height -15.2 / -12.7 mm vs -1.8 expected: unverified.
- Hover test 2026-11-24 19:45 (green arm, approved moves; the first try of the approach aborted, ACTION_ABORT, no fault,
  likely the Xbox pad's manual-control input; the retry worked): tool at (0.2376, 0.1217, 0.0540) straight down over a chart
  corner the camera chain put at z -11.7 mm. Raoul measured: chart to the OPEN fingertips 46 mm; camera window to table
  ~20 cm (rough). With the table at -18 mm the open fingertips are ~146 mm below the flange (26 mm beyond the Kortex tool
  point). Camera-chain chart height (-11.7 to -15.2 mm) vs pad-based -18 mm: 3-6 mm residual, the camera reads high.
  Caveat: the 2F-85 fingertips move along an arc, so open and closed fingertip heights differ (closed knob grasps fit
  TOOL_TO_PINCH 0.0358 on the blue arm); do not use the open-fingertip number for grasp height.
- Height check 2026-11-24 ~20:00 (arm at tool (0.2377, 0.1218, 0.1241), straight down, flange z 0.2441 m in the base
  frame), Raoul's ruler: table -> camera lens 267 mm (predicted 265 mm with the table at -18 mm, 260 mm with the camera's
  table estimate); table -> robot base 18 mm (the pad); table -> a ring 180 mm up (~82 mm below the flange: the gripper
  body bottom, not the wrist ring). CONFIRMED: table at -18 mm in Kortex's base frame (+-2 mm), base origin at the bottom
  of the base. The wrist-camera chain reads table-level targets 3-6 mm HIGH (-11.7 to -15.2 mm): a known vision bias,
  on the safe side for clearance. Re-locate idea: a board lying on the table only needs x, y, yaw from the images; its
  height and tilt are known from the table (toy marker surface = table + 28 mm = +10 mm in the base frame).
- Board model (green arm, 2026-11-24 ~21:15, hardware/board_model.json, session test_images/teach/20261124-205648):
  knob grasps hand-taught by Raoul (gripper stalled 0.618 left / 0.583 right; left re-taught after a lift and return:
  0.1 mm from the first, so the toy had not moved), then 5 planned wrist views (routine board_views_10_20261124,
  approved; 6 moves, no refusals). Grasps are stored RELATIVE to the board frame (marker centroid, x from id3 to id2),
  so replay = re-located board * stored grasp (no pad offset needed). Knob spacing 186.5 mm, heights within 2 mm.
  Fit rms 3.35 px with the camera's own marker height (+20.5 mm in the base frame) vs 8.0 px at the table-based +10 mm;
  with the fitted height the layout is ~3 % smaller than the ruler (camera scale/height coupling, same chain bias as the
  provisional calibration). Teach and re-locate share the chain, so the bias should cancel: VERIFY with a moved-toy
  re-locate plus a hover over each knob. Knob-to-marker centre: left 32.7 mm (ruler 35), right 31.2 mm (ruler 31).
- FIRST RE-LOCATE TEST (2026-11-24 ~21:40, green arm, scripts/relocate.py): toy moved 8.4 cm and turned 7 deg; sighting +
  5 planned views (routine relocate_10_20261124_2122, approved; marker 0 was cut off at the image top in all views: the
  planner used K_GUESS (cy 360) instead of the calibrated cy 243, fixed). Fit 4.1 px. Raoul lowered + closed by hand from
  5 cm above each predicted grasp: sideways error 0.0 mm on BOTH knobs (grip 0.622 / 0.618 = taught), height -3.3 mm
  (left) / +0.9 mm (right). Hand grasp heights of one knob spread ~4 mm, no consistent "higher" bias, so height
  tolerance is 5 mm and grasps are learned: `relocate.py learn` adds each hand correction (in the board frame) and the
  model uses the per-knob mean; `grip_raise_mm` in board_model.json sets a deliberate offset (default 0).
- FIRST AUTONOMOUS GRASPS (2026-11-24 ~22:00, green arm): `relocate.py workspace` (3 survey poses + one envelope
  relocate_workspace_10 for board centre 0.30-0.40 m, azimuth 0-30 deg, yaw +-20 deg) and `relocate.py auto` (look -> survey
  until the tags are seen -> 5 views inside the envelope -> solve). Toy moved ~5 cm and turned 33 deg (beyond the +-20 deg
  workspace): start saw 1 tag, surveys saw 2/3/3, views 4/2/4/2/4; fit 5.65 px; 96 s. The arm then went on its own above
  each predicted knob and straight down to the exact grasp pose; left knob centred (Raoul checked), both closed and stalled
  at 0.630 (taught 0.583-0.622). Right knob done in one shot (open, lift, above, descend, close).
- Knob torque measurement (green arm, 2026-11-24 ~21:50, approved routine etch_draw_workspace_10, read-only logger at
  21 Hz -> kinova-gen3-docker/logs/j6_torque_20261124.csv): direction check OK (y +15 and x +15 knob deg = J6 -15 deg,
  signs -1/-1 confirmed on the green arm; grips 0.630/0.634). Joint-6 torque still -0.09 +- 0.17 Nm; turning a held knob
  +0.18 / -0.25..-0.33 Nm at 5 AND 15 deg/s, the same as the wrist turning empty (+0.26 / -0.16..-0.25): the load is
  joint-6 friction, the knob itself needs almost nothing. Moving readings scatter to ~0.5 Nm, so a torque alarm would
  sit near 0.7 Nm (~100+ N of string tension on a ~5.4 mm drum): a backstop only, not protection.
- Kortex GRIPPER_FORCE is "not implemented yet" (ros_kortex Base.proto): no software grip-force limit. A slip clutch
  needs a light position-mode grip (target near the 0.63 contact point) or a mechanical compliant coupling; untested.
- Session 2026-11-24 ~22:10-22:40 (green arm): (1) re-locate twice more, board unchanged -> repeatability 0.3-0.9 mm, knobs
  0.4-1.3 mm. (2) Light grip: target 0.610 (stops 0.606), 0.620, 0.627 ALL turned both knobs +-15 deg at 5 deg/s (Raoul
  watched the pen); J6 torque identical at every level (+0.35/-0.11 Nm), so torque cannot tell slip from turn. 0.610 is
  the drawing grip (gripper.light in taught_poses.json); the slip point is below 0.606, not yet found. (3) Guarded 10 mm
  test line on the left knob with the light grip, under etch_draw_workspace_10 with the 0.7 Nm backstop: 55 commands,
  no refusals, backstop never tripped. (4) C4 (pen from wrist frames) BLOCKED: rectified screens are 35-94 % saturated
  (wrist camera auto exposure on the light screen + glare); the new line is not in the images. Next: read the vision
  module's exposure options (read-only) and try fixed lower exposure / glare-free view angles, or use the D435.
- Xbox pad: input from the connected pad (thumb, stick not centring) aborts CLI moves (ACTION_ABORT after a few mm)
  and drifted the arm 7-18 mm between commands. Unplug the pad's USB during scripted sessions. The arm reads
  "SERVOING_MANUALLY_CONTROLLED" after any CLI command even with the pad unplugged: that state alone is harmless.
- Restart note: after a limp/e-stop restart the green arm sat with joint 3 at -158 deg; Cartesian moves were refused
  (ACTION_ABORT, INVALID_PARAM) until a joint move brought it back in range. The gripper reported closed (0.996) with
  the fingers apart until the arm was power-cycled.
- Planner lesson: a 2 cm move with a ~100 deg wrist turn at 0.19 m from the base axis timed out (30 s); views now keep
  >= 0.25 m from the base axis, roll relative to the current wrist, and split moves turning > 35 deg.
- Heights (Raoul, ruler 2026-11-24): toy marker surface ~28 mm above the table (said "2.8"; 28 mm fits the camera);
  the arm base stands on an 18 mm rubber pad (corrected 19:50; first noted as 1.8 mm), so the table is at about
  -18 mm in the base frame (taking the base frame origin at the bottom of the base).
- Camera mount (Kinova gen3_macro.xacro color_module): from the flange (= Kortex tool frame - 0.12 m along tool z)
  xyz (0, 0.05639, -0.00305) m, rpy (pi, pi, 0), camera z along tool z. Confirmed: board within 3-4 deg of level.
- ChArUco chart on the table: 8x5 squares, DICT_5X5_100, square 30 mm, marker 22.5 mm (ratio fitted from the image).
- Harish's knob poses and envelope are for the BLUE arm; they do not apply to the green arm.

### Open decisions
- Decided 2026-11-09: bounded-routine approval (a person approves one envelope at the motion server); chunked
  joint 6 moves; a motion server in aims-kinova with etch-sim as client.
- One arm or two (blue 192.168.2.9 and green 192.168.2.10). Built for one so far.
- Scene camera: D435 decided; mount and host (Linux laptop) settled 2026-11-12; USB 3 cable would give more modes.
- Which LLM API for the agent.

## Log
### 2026-11-24 23:26 EST | Raoul Bouri (+ Claude)
- scripts/demo.py: checks -> re-locate (relocate_workspace_10, operator types yes) -> handover pose -> draw
  (etch_draw_workspace_10 at KINOVA_ROTATE_MAX_DPS=30, operator types yes) -> park at z 0.20. The two envelopes did not
  overlap in height (0.188 vs 0.134 m): drawing envelope ceiling raised to 0.22 m for a handover at z 0.20, placed in the
  overlap of both boxes nearest above the left knob (straight above it is outside the re-locate box for 8 of 27
  workspace placements: tests/test_demo.py). Dry run catches an arm parked outside the re-locate box. 78 tests pass.
### 2026-11-24 23:23 EST | Raoul Bouri (+ Claude)
- First spiral on the arm: the toy had been shaken (re-locate found it moved 4 cm / 13 deg; the earlier attempt with old
  poses failed its grasp -> grasp now closes fully to detect the knob, then eases to the light grip; the drawer stops on
  any failed grasp/turn). After a fresh re-locate: 60 mm centre-out spiral, 24 lines, 631 s (estimate 6-7 min; slow
  15-deg steps near edges cost ~3.5 min of it), 306 commands, no refusals, no failed grasps, backstop never tripped.
  Knobs at up to 30 deg/s (KINOVA_ROTATE_MAX_DPS=30, green arm). J6 torque while turning: mean 0.24, p99 0.51, max
  0.64 Nm vs the 0.70 Nm backstop: margin is thin at 30 deg/s. J2 max 41.3 C. Drawing quality: awaiting Raoul's photo.
### 2026-11-24 22:54 EST | Raoul Bouri (+ Claude)
- draw.spiral_out (centre-out rectangular spiral) + draw_shapes.py `spiral`; draw_shapes now runs the WHOLE drawing
  through the end-stop guard before moving (open loop a 60 mm spiral from the middle passes, ending at +-24 mm; 84 mm is
  refused at line 24/26 instead of half-drawn). Light grip 0.610 in taught_poses.json. 75 tests pass.
- README "Draw after a re-locate, without snapping the string": the protection layers and the D435 plan (Linux laptop +
  SSH tunnel to RealSenseServer from PR #1, run_tracker re-anchors the guard to +-1.5 mm, a line that does not grow =
  slip or stop). Same plan in endstop.py's docstring.
### 2026-11-24 22:41 EST | Raoul Bouri (+ Claude)
- C1 (15 deg slow steps near edges), C2 (aims-kinova branch rotate-torque-guard: J6 torque backstop, envelope field
  max_joint_torque_nm), C3 (screen 152.5 x 103.3 mm in board_model.json) built; light grip 0.610 chosen; test line drawn.
- C4 blocked by wrist-camera saturation on the screen. Arm parked high (z 0.32) after the last re-locate.
### 2026-11-24 22:00 EST (approximate) | Raoul Bouri (+ Claude)
- Branch relocate-drawing (on PR #4): relocate.py solve writes hardware/knob_poses_live.json (RealBackend format,
  git-ignored); draw_shapes.py / knob_dry_run.py take --poses; `relocate.py draw-envelope` writes ONE drawing envelope
  hardware/envelopes/etch_draw_workspace_10.json (knob poses for any workspace placement, 0.19-0.41 m from the base
  axis, tilt 5, joint-6 turns). New src/etch_sim/endstop.py: TravelGuard + GuardedDrawer refuse any move that could end
  within 5 mm (+ growing uncertainty) of an edge, and allow only backing off after a torque stop; wired into
  draw_shapes.py (--start-unc). Kernel tests: unguarded 40 mm move snaps the string, guarded one is refused. 69 pass.
- aims-kinova PR #2 (merged): green arm J6 fine at 15-30 deg/s; J6 + = knob anticlockwise from above; J2 overheats
  when parked low (60 C limit; read temperature_motor). J2 read 39.8 C while holding a knob ~10 min.
- Next: arm session (approved per step) for the direction check and the J6 torque measurement, then the torque guard
  in aims-kinova.
### 2026-11-24 22:00 EST | Raoul Bouri (+ Claude)
- relocate.py auto + workspace (survey fallback, one envelope); first autonomous re-locate + grasps of both knobs succeeded.
- To do: widen the workspace yaw range (board turned 33 deg), speed up (96 s), more placements, then drawing from here.
### 2026-11-24 21:45 EST | Raoul Bouri (+ Claude)
- relocate.py built (sight/plan/run/solve/taught/compare/learn) + board_model.fit_board + view_run; tests pass (64).
- First moved-toy test: both knobs predicted to 0.0 mm sideways; heights within hand variation. Corrections learned.
- Next: more placements (shifted, rotated +-25 deg, after a real shake) to test out of sample.
### 2026-11-24 21:15 EST | Raoul Bouri (+ Claude)
- Gripper stuck (reported closed, fingers apart) -> Raoul power-cycled; elbow out of range -> joint moves back (approved).
- Knob grasps taught by hand and recorded read-only; toy-stillness check by lift-and-return (0.1 mm). scripts/teach_knobs.py
  (view/knob/plan/run/solve) + src/etch_sim/real/board_model.py + tests. 5-view board routine run (approved), board model saved.
- Next: re-locate command (5 views -> board x, y, yaw -> both knob grasps) and a moved-toy validation with a hover per knob.
### 2026-11-24 19:25 EST | Raoul Bouri (+ Claude)
- E-stop fault found after taping (EMERGENCY_STOP_ACTIVATED); Raoul released it, clear-faults (approved) -> ready.
- Planner got a max tool radius (--rho-max, default 0.25 m) and higher views (camera 40-60 cm). Approved move-in plus a
  second routine: 25 moves, 20 views, no refusals. Combined two-session fit saved as PROVISIONAL calibration.
- Next: board re-locate from 1-3 wrist frames of the toy markers (repeatability ~1-2 mm per frame); a hover test to settle
  absolute height; flatter or rigid chart if a better lens fit is needed.
### 2026-11-24 19:04 EST | Raoul Bouri (+ Claude)
- Built wrist-camera calibration: src/etch_sim/real/wrist_camera.py (frame chain, RTSP grabber, ChArUco, joint fit),
  view_plan.py (views from one sighting: safe, reachable, diverse, short moves), scripts/wrist_calib.py
  (capture/plan/run/solve); tests pass. First arm run: move 1 timed out (wrist turn near the base), routine ended,
  planner fixed. Second run (approved): 27 moves, 20 views, no refusals.
- Result not good enough to save (joint rms 4 px): paper chart not flat; absolute height 13.6 mm off the pad
  assumption. Repeatability 1-2 mm per view. Next: chart glued flat on a rigid board, rerun (~3 min); hover test.
### 2026-11-24 17:39 EST | Raoul Bouri (+ Claude)
- PR #3 (Harish, knob poses + first real square) merged into dev as bd72d39; local dev pulled; 54 tests pass.
- Markers stuck on the toy (layout in Facts). Green arm placed by Raoul for a wrist-camera view of the whole board;
  state saved (read-only) as the reusable board view. One wrist frame over RTSP: all 4 markers detected
  (test_images/wrist/20261124-173925/). RealSense frame not taken on the Mac (rule 15).
- Next: plan Stage A (board pose in the base frame from the wrist camera) with Raoul before building.
### 2026-11-22 21:45 EST | Harish Bhamitipadi Magesh (+ Claude)
- Correction to 21:26 entry: the aims-kinova changes are now committed locally on branch harish/rotate-fixes (bff79a7, on
  7d0dab2; docstrings say the gripper Stop is a precaution). Push refused: HBMatic has no write access to
  raoulbouri/aims-kinova; pushed to the fork HBMatic/aims-kinova instead, PR into etch-bounded-routine for Raoul.
- Project overview doc (Claude Doc) updated with today's status, tasks and plan.

### 2026-11-22 21:26 EST | Harish Bhamitipadi Magesh (+ Claude)
- Documentation pass for the day: stale facts updated (stages, hardware, native server, which knob is x, envelope,
  both joint-6 signs, pen-position rule, Euler-flip and arm timing facts). Envelope used copied to
  hardware/envelopes/. README has a Real arm section with the base-test photo.
- Still outside this repo: the aims-kinova duration fix (and the unproven gripper-stall Stop) are uncommitted on
  Harish's laptop, waiting for Raoul.

### 2026-11-22 21:15 EST | Harish Bhamitipadi Magesh (+ Claude)
- BASE TEST PASSED on the real arm (blue, toy taped): pen moved off the corner with knob_dry_run x:170 y:160 (~14-16 mm
  each), then `draw_shapes.py square --size 30 --start 14,14`: a closed square, corners meet, sides ~27-30 mm judged
  from an angled phone photo with the screen window as scale (+-2-3 mm). Photo hardware/draw_runs/20261122_square30.jpg.
- Next: triangle / circle / picture --image house; a square-on photo or D435 frame for a real error number.

### 2026-11-22 20:58 EST | Harish Bhamitipadi Magesh (+ Claude)
- Merged dev (Andrew's S3 planner) into harish/knob-poses; MEMORY conflict = both sides kept. 51 tests pass.
- draw_shapes.py `picture --image ... --budget`: plan_for_budget from the pen with real-arm timing (switch 10 s,
  15 dps, ToyParams calib). House at 600 s: tol 3 mm, 28 switches, ~9.9 min est; fake-server end to end OK.
- For the planner's timing PLACEHOLDERS (switch 8 s, 30 dps): blue arm measured ~10 s per switch (release + grasp,
  from knob_dry_run logs) and 15 dps max (30 deg in 1 s is refused).

### 2026-11-22 20:53 EST | Harish Bhamitipadi Magesh (+ Claude)
- Both knobs drive the pen on the real arm at 15 dps: x:180 -> ~15 mm right (by eye; model 0.0951 x (180 - 2x6.1)
  = 16 mm), y:180 -> ~16 mm up (model ~17 mm). y joint-6 sign -1 confirmed (pen moved off the bottom stop).
  Release after a 180 deg turn is clean with down_yaw_deg.
- New scripts/draw_shapes.py (base test): square / rectangle / triangle (staircase) / circle (stepped), OpenLoopDrawer
  with k and backlash from ToyParams, refuses shapes within 5 mm of an end stop from --start, --plan prints knob
  degrees and time. draw.py: rectangle, merge, stepped_circle; +5 tests (42 pass). Fake-server end to end OK; not yet on the arm.

### 2026-11-22 20:42 EST | Harish Bhamitipadi Magesh (+ Claude)
- knob_dry_run x:180: wrist +180.2 deg OK, then the release lift timed out (move_pose TIMEOUT, routine ended) halfway
  through an unintended wrist spin. After the turn Kortex reported the downward tool as (0.4, -179.8, tz+180) instead
  of (180, 0, tz); RealBackend fed that theta_z back with theta_x=180, theta_y=0 = a 180 deg spin. Fix: RealBackend
  takes the wrist yaw from the tool x-axis heading (down_yaw_deg), same in both Euler forms; +2 tests, 37 pass.
  Gripper was already open, knob released; no harm.

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

### 2026-11-22 17:53 EST | Andrew Haggstrom (+ Claude)
- Added S3: `src/etch_sim/planner.py` (image -> one continuous Manhattan route), `scripts/s3_demo.py`, `tests/test_planner.py`
  (9 tests). Exit check passes; full suite passes. Kernel only, not run in MuJoCo or on the arm.
- Next: join pieces at loose ends to cut retrace; try real line-art / LLM-generated pictures; MuJoCo run of a planned route.

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
### 2026-11-22 11:55 EST | Raoul Bouri (+ Claude)
- Blue arm: started IN_FAULT (left from last e-stop), Raoul restarted it -> SERVOING_READY. He then placed the gripper
  closed on the LEFT knob with the Xbox controller; the arm went IN_FAULT again there. Fault bank: base
  EMERGENCY_STOP_ACTIVATED (0x800000), no actuator faults; Raoul did not press the e-stop (bumped or not fully released?).
- Saved left-knob grasp to hardware/taught_poses.json: tool [0.3361, 0.1225, 0.0529 m, -175.38, 2.12, 107.60 deg],
  joints [-21.07, 34.97, -118.83, 1.70, -30.97, 84.38], gripper 0.6974 (stalled on knob: inside held_min..max).
- Model FK vs blue-arm read-backs: 8.5 mm (knob pose) and 9.9 mm (retracted), mostly along x; same size as green (6-11 mm).
- `kinova clear-faults` (approved) restored SERVOING; pose unchanged. Then `kinova gripper 0.45` (approved) did NOT
  open: STALLED at 0.693 (from 0.697) while opening. Lift not run (would pull the knob). `kinova stop` sent. Cause
  unknown: fingers jammed, or gripper not re-enabled after the e-stop, or the Xbox controller holding the gripper
  (arm state SERVOING_MANUALLY_CONTROLLED).
- Raoul then hit the e-stop; after a restart the gripper opened (knob released) and the arm was folded near
  rest; `kinova home` (approved) reached Home within 0.04 deg. Left-knob grasp saved; right knob not taught yet.
  No knob rotations done (Raoul: none until he says). Watch the e-stop cable: the first trip was likely a bump.
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
