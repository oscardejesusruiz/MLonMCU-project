#!/usr/bin/env bash
# Live profiler — reads per-inference packets from the MAX78000 (sent by
# `c_harness/profile_camera.c`) and prints a rolling-average dashboard
# of every device-side metric the report needs:
#
#   end-to-end latency, CNN latency, CNN cycles @ 50 MHz, MAC/cycle,
#   end-to-end energy, inference energy, power (µW/MHz, both clocks),
#   measured TOPS/W, peak TOPS/W (paper Table I).
#
# Each metric is computed from one paper-cited power constant (28 mW)
# plus the variant's MAC count. Pass `--power-mw N` to the underlying
# python script if you ever measure real power with an INA219/Joulescope.
#
# Usage:
#   ./bash_device_scripts/host_e2e_profile.sh <variant>
#   ./bash_device_scripts/host_e2e_profile.sh <variant> /dev/cu.usbmodemXXXX
#
# Variant must match the variant currently flashed (controls the MAC
# count used for MAC/cycle and TOPS/W) — it's also used to namespace the
# CSV log.

. "$(dirname "$0")/_common.sh"

PORT="${2:-}"
if [ -z "$PORT" ]; then
    PORT=$(detect_port) || true
fi
if [ -z "$PORT" ]; then
    echo "ERROR: no /dev/cu.usbmodem* found and no port given as arg 2" >&2
    echo "       Pass it explicitly:" >&2
    echo "         ./bash_device_scripts/host_e2e_profile.sh $VARIANT /dev/cu.usbmodemXXXX" >&2
    exit 1
fi

LOG_DIR="$THIS_DIR/reports/e2e_profile_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${VARIANT}_$(date +%Y%m%d_%H%M%S).csv"

echo "================================================================"
echo "End-to-end profiling monitor"
echo "  variant : $VARIANT"
echo "  port    : $PORT"
echo "  log     : $LOG_FILE"
echo "  window  : 10 inferences (rolling)"
echo "  P       : 28 mW (paper Table I — pass --power-mw N to override)"
echo "  Ctrl-C  : stop"
echo "================================================================"
echo ""

# Use the PC implementation's venv (pyserial + numpy).
uv run --project "$PC_DIR" python "$THIS_DIR/host/host_e2e_profile.py" \
    --port "$PORT" \
    --variant "$VARIANT" \
    --window 10 \
    --log "$LOG_FILE"
