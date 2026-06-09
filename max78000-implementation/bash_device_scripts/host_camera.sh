#!/usr/bin/env bash
# Live monitor for the camera-streaming inference firmware. Reads the
# UART packet stream from the MAX78000, computes softmax over the 10
# class logits per frame, and prints a continuously-updating distribution
# in the terminal. Also appends a CSV log under reports/camera_logs/.
#
# Usage:
#   ./bash_device_scripts/host_camera.sh <variant>
#   ./bash_device_scripts/host_camera.sh <variant> /dev/cu.usbmodemXXXX
#
# Variant is informational (the firmware streams the same packet format
# regardless of which model is loaded), but it's used to namespace the
# CSV log file.

. "$(dirname "$0")/_common.sh"

PORT="${2:-}"
if [ -z "$PORT" ]; then
    PORT=$(detect_port) || true
fi
if [ -z "$PORT" ]; then
    echo "ERROR: no /dev/cu.usbmodem* found and no port given as arg 2" >&2
    echo "       Pass it explicitly:" >&2
    echo "         ./bash_device_scripts/host_camera.sh $VARIANT /dev/cu.usbmodemXXXX" >&2
    exit 1
fi

LOG_DIR="$THIS_DIR/reports/camera_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${VARIANT}_$(date +%Y%m%d_%H%M%S).csv"

echo "================================================================"
echo "Live camera inference monitor"
echo "  variant : $VARIANT"
echo "  port    : $PORT"
echo "  log     : $LOG_FILE"
echo "  Ctrl-C  : stop"
echo "================================================================"
echo ""

# Use the PC implementation's venv (has pyserial + numpy already).
uv run --project "$PC_DIR" python "$THIS_DIR/host/host_camera.py" \
    --port "$PORT" \
    --log  "$LOG_FILE"
