#!/bin/sh
# macOS: run scripts/rs_capture.py with the access librealsense needs.
#   sudo sh scripts/rs_capture.sh grab          (minimal: 10 colour frames, scripts/rs_grab.py)
#   sudo sh scripts/rs_capture.sh [--all] [--frames N] [--exposure E] [--tag name]
#
# macOS's camera daemons (UVCAssistant/VDCAssistant) claim every USB camera and respawn within seconds,
# so they are killed in a loop while the capture runs. That loop runs as root and starves the Mac's
# camera stack, so it is fenced in three ways:
#   - a trap stops it on any exit, Ctrl-C included (a background job in `sh` ignores Ctrl-C itself,
#     which is how an earlier version could leave it running after the script was interrupted);
#   - it stops by itself after LOOP_MAX_S seconds;
#   - the capture is killed after CAPTURE_MAX_S seconds if it has not finished.
# Python: the aims-kinova venv that has pyrealsense2-macosx; override with RS_PYTHON=/path/to/python.
LOOP_MAX_S=40
CAPTURE_MAX_S=35

cd "$(dirname "$0")/.." || exit 1
if [ "$(id -u)" != 0 ]; then echo "run with sudo: sudo sh scripts/rs_capture.sh $*"; exit 1; fi
USER_HOME="/Users/${SUDO_USER:-$(logname)}"
PY="${RS_PYTHON:-$USER_HOME/Developer/kinova-gen3-docker/.venv/bin/python}"
[ -x "$PY" ] || { echo "no python at $PY (set RS_PYTHON)"; exit 1; }

LOOP=""; PYPID=""; DOG=""
cleanup() {
  [ -n "$DOG" ] && kill "$DOG" 2>/dev/null
  [ -n "$PYPID" ] && kill "$PYPID" 2>/dev/null
  [ -n "$LOOP" ] && kill "$LOOP" 2>/dev/null
  [ -n "$SUDO_USER" ] && [ -d test_images ] && chown -R "$SUDO_USER" test_images
}
trap cleanup EXIT
trap 'echo "interrupted"; exit 130' INT TERM HUP

( end=$(( $(date +%s) + LOOP_MAX_S ))
  while [ "$(date +%s)" -lt "$end" ]; do killall UVCAssistant VDCAssistant 2>/dev/null; sleep 0.3; done ) &
LOOP=$!
sleep 1

SCRIPT=scripts/rs_capture.py
if [ "$1" = grab ]; then SCRIPT=scripts/rs_grab.py; shift; fi      # minimal 10-frame colour grab
PYTHONUNBUFFERED=1 "$PY" "$SCRIPT" "$@" &
PYPID=$!
( sleep "$CAPTURE_MAX_S"; echo "capture took over ${CAPTURE_MAX_S}s: stopping it"; kill "$PYPID" 2>/dev/null ) &
DOG=$!
wait "$PYPID"
RC=$?
PYPID=""
exit $RC
