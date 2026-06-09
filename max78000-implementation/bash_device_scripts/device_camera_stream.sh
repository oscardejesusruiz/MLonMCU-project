#!/usr/bin/env bash
# Build + flash the camera-livestream firmware for the chosen variant.
# Same flow as device_camera.sh, but the firmware (camera_stream.c)
# additionally ships the 32x32 RGB888 image to the host so the viewer
# can render a live picture next to the probability bars.
#
# Usage:
#   ./bash_device_scripts/device_camera_stream.sh <variant>
#
# Variants:  baseline | improved | mininet | deeper | nascifarnet | ressimplenet

. "$(dirname "$0")/_common.sh"

require_synth
swap_main_c "$THIS_DIR/c_harness/camera_stream.c"
build_and_flash

echo ""
echo "✓ Flashed $VARIANT — camera livestream firmware (image + probs)"
echo ""
echo "  Next: live viewer window (matplotlib)"
echo "      ./bash_device_scripts/host_camera_stream.sh $VARIANT"
