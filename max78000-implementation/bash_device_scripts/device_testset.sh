#!/usr/bin/env bash
# Terminal A — flash the UART test-set streaming firmware to the MAX78000.
#
# Usage:  ./bash_device_scripts/device_testset.sh {baseline|improved|mininet|deeper|nascifarnet|ressimplenet}
#
# What it does:
#   1. Checks the synthesized C project exists.
#   2. Backs up the auto-generated main.c (once) as main.c.orig.
#   3. Copies c_harness/inference_test_set.c → main.c.
#   4. make distclean && make && make flash.openocd.
#
# After this, run the matching Terminal B script:
#   ./bash_device_scripts/host_testset.sh {baseline|improved|mininet|deeper|nascifarnet|ressimplenet}

. "$(dirname "$0")/_common.sh"

require_synth
swap_main_c "$THIS_DIR/c_harness/inference_test_set.c"
build_and_flash

echo
echo "================================================================"
echo "Device ready (variant: $VARIANT)"
echo "  Firmware: inference_test_set.c"
echo "  UART baud: 115200"
echo "  LED 0 will blink on every inference."
echo "  Next: open another terminal and run"
echo "      cd $THIS_DIR"
echo "      ./bash_device_scripts/host_testset.sh $VARIANT"
echo "================================================================"
