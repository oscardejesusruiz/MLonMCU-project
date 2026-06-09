#!/usr/bin/env bash
# Live viewer for the camera-livestream firmware. Opens a matplotlib
# window showing the 32x32 image the CNN currently sees plus the live
# CIFAR-10 probability distribution. Also appends a CSV log under
# reports/camera_stream_logs/.
#
# Usage:
#   ./bash_device_scripts/host_camera_stream.sh <variant>
#   ./bash_device_scripts/host_camera_stream.sh <variant> /dev/cu.usbmodemXXXX
#
# Variant is informational (the firmware streams the same packet format
# regardless of which model is loaded) — used to namespace the log file.

. "$(dirname "$0")/_common.sh"

PORT="${2:-}"
if [ -z "$PORT" ]; then
    PORT=$(detect_port) || true
fi
if [ -z "$PORT" ]; then
    echo "ERROR: no /dev/cu.usbmodem* found and no port given as arg 2" >&2
    echo "       Pass it explicitly:" >&2
    echo "         ./bash_device_scripts/host_camera_stream.sh $VARIANT /dev/cu.usbmodemXXXX" >&2
    exit 1
fi

LOG_DIR="$THIS_DIR/reports/camera_stream_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${VARIANT}_$(date +%Y%m%d_%H%M%S).csv"

echo "================================================================"
echo "Live camera + image viewer"
echo "  variant : $VARIANT"
echo "  port    : $PORT"
echo "  log     : $LOG_FILE"
echo "  baud    : 115200 (matches firmware UART_BAUD)"
echo "  Ctrl-C  : stop"
echo "================================================================"
echo ""

# Use the PC implementation's venv — has pyserial + numpy + matplotlib.
uv run --project "$PC_DIR" python "$THIS_DIR/host/host_camera_stream.py" \
    --port "$PORT" \
    --baud 115200 \
    --log  "$LOG_FILE"
