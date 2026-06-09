#!/usr/bin/env bash
# Build + flash the camera-driven *end-to-end profiling* firmware. Same
# camera capture + inference path as device_camera.sh, but the firmware
# wraps every inference with the Cortex-M4 DWT cycle counter so the host
# can compute end-to-end latency, CNN latency, energy, MAC/cycle, and
# TOPS/W from one stream of packets.
#
# Usage:
#   ./bash_device_scripts/device_e2e_profile.sh <variant>
#
# Variants:  baseline | improved | mininet | deeper | nascifarnet | ressimplenet
#
# After flashing, run:
#   ./bash_device_scripts/host_e2e_profile.sh <variant>

. "$(dirname "$0")/_common.sh"

require_synth
swap_main_c "$THIS_DIR/c_harness/profile_camera.c"
build_and_flash

echo ""
echo "✓ Flashed $VARIANT — end-to-end profiling firmware"
echo ""
echo "  Next: rolling-average metrics in your terminal"
echo "      ./bash_device_scripts/host_e2e_profile.sh $VARIANT"
