#!/usr/bin/env bash
# Build + flash the camera-streaming inference firmware for the chosen
# variant. After flashing, run host_camera.sh to display the live
# probability distribution coming from the board.
#
# Usage:
#   ./bash_device_scripts/device_camera.sh <variant>
#
# Variants:  baseline | improved | mininet | deeper | nascifarnet | ressimplenet

. "$(dirname "$0")/_common.sh"

require_synth
swap_main_c "$THIS_DIR/c_harness/camera_inference.c"
build_and_flash

echo ""
echo "✓ Flashed $VARIANT — camera streaming firmware"
echo ""
echo "  Next: live monitor (probabilities + FPS in your terminal)"
echo "      ./bash_device_scripts/host_camera.sh $VARIANT"
