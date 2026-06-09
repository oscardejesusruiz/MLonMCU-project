#!/usr/bin/env bash
# Terminal A — flash the per-layer profiling firmware to the MAX78000.
#
# Usage:  ./bash_device_scripts/device_profile.sh {baseline|improved|mininet|deeper|nascifarnet|ressimplenet}
#
# What it does:
#   1. Checks the synthesized C project exists.
#   2. Backs up the auto-generated main.c (once) as main.c.orig.
#   3. Copies c_harness/profile_layers.c → main.c.
#   4. make distclean && make && make flash.openocd.
#
# After this, run the matching Terminal B script:
#   ./bash_device_scripts/host_profile.sh {baseline|improved|mininet|deeper|nascifarnet|ressimplenet}

. "$(dirname "$0")/_common.sh"

require_synth
swap_main_c "$THIS_DIR/c_harness/profile_layers.c"
build_and_flash

echo
echo "================================================================"
echo "Device ready (variant: $VARIANT)"
echo "  Firmware: profile_layers.c"
echo "  UART baud: 115200"
echo "  Next: open another terminal and run"
echo "      cd $THIS_DIR"
echo "      ./bash_device_scripts/host_profile.sh $VARIANT"
echo "================================================================"
