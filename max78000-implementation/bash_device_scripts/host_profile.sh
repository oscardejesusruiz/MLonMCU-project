#!/usr/bin/env bash
# Terminal B — read single-inference profiling report from the MAX78000.
#
# Usage:  ./bash_device_scripts/host_profile.sh {baseline|improved|mininet|deeper|nascifarnet|ressimplenet} [/dev/cu.usbmodemXXXX]
#
# If no port is given, the first matching /dev/cu.usbmodem* or /dev/ttyACM* is used.
#
# Output:
#   - prints the ST.AI-style per-layer table to stdout
#   - writes a copy to  reports/profile_<variant>.txt
#
# Prerequisite: the device must have been flashed with device_profile.sh first.

. "$(dirname "$0")/_common.sh"

PORT="${2:-}"
if [ -z "$PORT" ]; then
  PORT=$(detect_port) || {
    echo "ERROR: no serial port detected (/dev/cu.usbmodem* or /dev/ttyACM*)" >&2
    echo "       plug the FTHR board and try again, or pass the port explicitly:" >&2
    echo "       $0 $VARIANT /dev/cu.usbmodemXXXX" >&2
    exit 1
  }
fi
echo "[host_profile] port: $PORT"
echo "[host_profile] variant: $VARIANT"
echo

# If the device booted a moment ago the report block is already on the UART
# buffer. If it timed out, prompt the user to hit reset.
uv run --project "$PC_DIR" python "$THIS_DIR/host/host_profile.py" \
    --port "$PORT" \
    --baud 115200 \
    --variant "$VARIANT" || {
  echo
  echo "If you saw a timeout: press the RESET button on the FTHR board"
  echo "and re-run this script within a second."
  exit 1
}

echo
echo "================================================================"
echo "Profile saved to: $THIS_DIR/reports/profile_${VARIANT}.txt"
echo "================================================================"
