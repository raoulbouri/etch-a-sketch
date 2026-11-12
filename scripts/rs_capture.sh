#!/bin/sh
# Run the RealSense (librealsense / pyrealsense2) scripts.
#   sh scripts/rs_capture.sh serve         camera server for a whole session (scripts/rs_server.py):
#                                          start once in its own terminal and leave it running
#   sh scripts/rs_capture.sh grab [dir]    one-off: 10 colour frames (scripts/rs_grab.py)
#   sh scripts/rs_capture.sh [--all ...]   one-off: colour + depth + IR (scripts/rs_capture.py)
#
# LINUX (supported): runs as your user. Python: this repo's .venv after `uv sync --group realsense`
# (pyrealsense2 wheels exist for Linux x86_64/aarch64, Python 3.12), or RS_PYTHON=/path/to/python.
# If the camera is not found or permission is denied, install Intel's udev rules
# (https://github.com/IntelRealSense/librealsense/blob/master/config/99-realsense-libusb.rules) and replug.
#
# MACOS (blocked): librealsense can only open the camera after macOS's camera services
# (UVCAssistant/VDCAssistant) are killed, which needs sudo. On 2026-11-12 that froze the whole Mac three
# times (WindowServer stuck in CoreMediaIO, forced restarts): first with one capture per knob turn, then even
# with a single kill burst at server start. The old macOS path is kept below only behind an explicit override.
cd "$(dirname "$0")/.." || exit 1

MODE=oneoff
SCRIPT=scripts/rs_capture.py
case "${1:-}" in
  grab)  SCRIPT=scripts/rs_grab.py; shift ;;
  serve) SCRIPT=scripts/rs_server.py; MODE=serve; shift ;;
esac

if [ "$(uname)" != Darwin ]; then
  if [ -n "${RS_PYTHON:-}" ]; then PY="$RS_PYTHON"
  elif [ -x .venv/bin/python ] && .venv/bin/python -c "import pyrealsense2" 2>/dev/null; then PY=.venv/bin/python
  else PY=python3; fi
  "$PY" -c "import pyrealsense2" 2>/dev/null || {
    echo "no pyrealsense2 in $PY. Run: uv sync --group realsense   (or set RS_PYTHON)"; exit 1; }
  if [ "$MODE" = oneoff ]; then exec timeout 35 "$PY" "$SCRIPT" "$@"; fi
  exec "$PY" "$SCRIPT" "$@"
fi

# ---- macOS ------------------------------------------------------------------------------------
if [ "${RS_ALLOW_MAC:-}" != "i-accept-freezes" ]; then
  echo "rs_capture.sh is disabled on macOS: librealsense froze the Mac (WindowServer in CoreMediaIO)."
  echo "Run the camera on a Linux laptop instead (README, 'Real scene camera')."
  exit 3
fi
LOOP_MAX_S=20
CAPTURE_MAX_S=35
if [ "$(id -u)" != 0 ]; then echo "run with sudo"; exit 1; fi
USER_HOME="/Users/${SUDO_USER:-$(logname)}"
PY="${RS_PYTHON:-$USER_HOME/Developer/kinova-gen3-docker/.venv/bin/python}"
[ -x "$PY" ] || { echo "no python at $PY (set RS_PYTHON)"; exit 1; }
READY=$(mktemp -u "${TMPDIR:-/tmp}/rs_ready.XXXXXX")
export RS_READY_FILE="$READY"
LOOP=""; PYPID=""; DOG=""
cleanup() {
  [ -n "$DOG" ] && kill "$DOG" 2>/dev/null
  [ -n "$PYPID" ] && kill "$PYPID" 2>/dev/null
  [ -n "$LOOP" ] && kill "$LOOP" 2>/dev/null
  rm -f "$READY"
  [ -n "$SUDO_USER" ] && [ -d test_images ] && chown -R "$SUDO_USER" test_images
}
trap cleanup EXIT
trap 'echo "interrupted"; exit 130' INT TERM HUP
( end=$(( $(date +%s) + LOOP_MAX_S ))
  while [ ! -e "$READY" ] && [ "$(date +%s)" -lt "$end" ]; do
    killall UVCAssistant VDCAssistant 2>/dev/null; sleep 0.5
  done ) &
LOOP=$!
sleep 1
PYTHONUNBUFFERED=1 "$PY" "$SCRIPT" "$@" &
PYPID=$!
if [ "$MODE" = oneoff ]; then
  ( sleep "$CAPTURE_MAX_S"; echo "capture took over ${CAPTURE_MAX_S}s: stopping it"; kill "$PYPID" 2>/dev/null ) &
  DOG=$!
fi
wait "$PYPID"
RC=$?
PYPID=""
exit $RC
