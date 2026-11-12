# CLAUDE.md: working agreement for etch-sim

Read this before changing anything. Several developers and several coding agents work in this repo;
these rules keep their work compatible. Then read `MEMORY.md` (current state and measured facts)
and `README.md` (how to run).

## What this project is

An Etch-a-Sketch robot artist: a Kinova Gen3 (6-DoF, Robotiq 2F-85, wrist camera) plus a scene camera
turns the toy's knobs one at a time, erases by flipping and shaking, and an LLM agent plans drawings
and corrects them from scored results. This repo holds the simulation and the code that must run
unchanged on the real arm. The project runs Nov 10 to Nov 30, 2026 in phases with go/no-go gates;
hardware tests start as early as possible and their measurements flow back into the sim.

## Hard rules

1. **Do not edit `CLAUDE.md` without the coder's consent.** Propose the exact change in chat and wait
   for a yes from the person you are working with. This applies to agents and to scripts.
2. **Keep `MEMORY.md` current, with timestamps.** Follow the protocol below in the same change as the work.
3. **Git commits are authored by the human developer, never by Claude or any agent.**
   - No agent name or email as author or committer. Do not pass `--author` with an agent identity and
     do not set `user.name` / `user.email` to one.
   - No `Co-Authored-By`, "Generated with", or any other agent attribution in commit messages,
     PR titles or PR bodies.
   - Before the first commit in a session run `git config user.name` and `git config user.email`.
     If either is empty or names an agent, stop and ask the developer for their identity; do not invent one.
   - Commit, push, or open a PR only when the developer asks. Work on `dev` (or a branch off `dev`), never on
     `main` (see rule 14).
4. **One stage at a time.** Before building or running a new stage, say what it will do and wait for
   a go-ahead. Show results (numbers and images) before starting the next one.
5. **No unmeasured physical constant without a label.** Every toy, timing, camera or grasp number is
   either marked `PLACEHOLDER` in a comment or carries the measurement it came from.
   `hardware/phase0_measurements.yaml` is the single record of hardware measurements; when a value
   is measured, update the yaml, the code default and `MEMORY.md` together.
6. **The `Skills` interface (`src/etch_sim/interface.py`) is the contract.** Planner, calibration,
   closed-loop and agent code talk only to it and must not import backend internals, so the same
   script runs on the toy kernel, in MuJoCo and on the arm. Changing the interface needs the
   developer's agreement because it touches every backend.
7. **The LLM agent never sees ground truth.** `truth_log`, kernel internals (`gap`, `strain`, hidden
   parameters) and injected-fault settings are for grading only. Scoring, erase and safety code are
   read-only to the drawing agent.
8. **Do not hand-edit vendored models.** `models/vendor/` holds upstream files (Kinova `ros_kortex`,
   MuJoCo Menagerie) with their licences. Our changes live in the scene builder code, not in those files.
9. **Real arm:** all motion goes through the aims-kinova motion server, inside an envelope a human approved when
   starting it. Code here never opens its own arm session. A refused or failed motion (`ArmRefused`) stops the
   script; never catch it to retry. Everything in that repo's `CLAUDE.md` still applies.
10. **Report honestly.** A stage is done only when its tests and exit check pass. If something fails
    or was skipped, say so with the output. Never tune a check until it passes without saying why.
11. **Docs live in `README.md`, `CLAUDE.md` and `MEMORY.md`.** Ask before adding another markdown file.
12. **No secrets in the repo.** API keys and arm credentials stay in an untracked `.env`; never print them.
13. **Run locally with `uv`.** Do not use shared lab machines or GPUs for this project unless the
    developer says so, and do not leave shells or agents polling long jobs.
14. **`CLAUDE.md` and `MEMORY.md` live only on `dev`.** Work on `dev` (or branches off it). `main` never tracks
    these two files: they are in `.gitignore` and were force-added on `dev` only. Update `main` only with
    `scripts/promote_main.sh` (which strips them), never by checking out `main` here or merging by hand.
    CI (`.github/workflows/guard-main.yml`) rejects them on `main`.

## MEMORY.md protocol

`MEMORY.md` has two parts.

- **Facts**: short, current statements a new session needs (environment, model facts, measured values,
  open decisions). Edit in place. Each fact that can go stale ends with `(verified YYYY-MM-DD)`.
  A fact that is a guess says `PLACEHOLDER`.
- **Log**: append-only, newest entry at the top. One entry per working session or stage:

  ```
  ### 2026-11-09 20:44 EST | <developer name> (+ agent, if one was used)
  - what changed, what was found, what is next (3 to 6 lines)
  ```

  Take the timestamp from the system clock (`date "+%Y-%m-%d %H:%M %Z"`), not from memory.
  Never rewrite or delete another person's log entry; if it is wrong, add a new entry that corrects it.

Update `MEMORY.md` whenever a stage finishes, a measurement lands, a decision is made, or a
non-obvious bug is found and fixed. Keep narrative out of it; a fact is one or two lines.

## How to run

```bash
uv sync
uv run pytest -q
uv run python scripts/s0_demo.py     # toy kernel: calibrate, erase, draw; writes out/s0/
uv run python scripts/s1_demo.py     # MuJoCo scene: cameras, markers, live screen; writes out/s1/
uv run python scripts/s2_demo.py     # MuJoCo: arm turns knobs, camera calibration; writes out/s2/
uv run python scripts/make_video.py   # simulation video -> out/video/
sudo sh scripts/rs_capture.sh grab   # real camera: 10 colour frames -> test_images/rs/
uv run python scripts/analyze_sketch.py  # extract the sketch from the newest capture -> test_images/analysis/
```

`out/` and `test_images/` are ignored by git. Python 3.12, dependencies in `pyproject.toml`.

## Conventions

- Screen frame: millimetres, origin at the bottom-left of the drawable area, x right, y up.
- World frame (MuJoCo and arm): metres, arm base at the origin, z up. Knob angles in degrees.
- Axis `x` = left knob = horizontal; axis `y` = right knob = vertical.
- New code gets a test in `tests/`; a new stage gets a `scripts/sN_demo.py` that prints its exit check.
