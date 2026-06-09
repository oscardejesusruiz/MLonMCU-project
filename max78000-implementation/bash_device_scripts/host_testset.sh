#!/usr/bin/env bash
# Terminal B — stream the CIFAR-10 test set to the MAX78000, collect predictions,
# regenerate per-model figures from the device-measured numerics.
#
# Usage:  ./bash_device_scripts/host_testset.sh {baseline|improved|mininet|deeper|nascifarnet|ressimplenet} [/dev/cu.usbmodemXXXX] [N_IMAGES]
#
# If no port is given, the first matching /dev/cu.usbmodem* or /dev/ttyACM* is used.
# N_IMAGES defaults to 10000 (full test set).
#
# Output:
#   reports/predictions/<variant>_qat_device.npz   (y_true + fp32/int8 preds + probs + cycles)
#   reports/figures/<variant>_qat_device/          (4 PNGs: confusion + ROC, fp32 and int8 each)
#
# Total time at 115200 baud + ~1ms inference: ~45 minutes for 10000 images.
#
# Prerequisite: the device must have been flashed with device_testset.sh first.

. "$(dirname "$0")/_common.sh"

PORT="${2:-}"
N_IMAGES="${3:-10000}"
TAG="${VARIANT}_qat_device"

if [ -z "$PORT" ]; then
  PORT=$(detect_port) || {
    echo "ERROR: no serial port detected (/dev/cu.usbmodem* or /dev/ttyACM*)" >&2
    echo "       plug the FTHR board and try again, or pass the port explicitly:" >&2
    echo "       $0 $VARIANT /dev/cu.usbmodemXXXX [N]" >&2
    exit 1
  }
fi

echo "[host_testset] port:    $PORT"
echo "[host_testset] variant: $VARIANT"
echo "[host_testset] images:  $N_IMAGES"
echo "[host_testset] tag:     $TAG"
echo

# ---------- stream test set, collect predictions ---------------------------

uv run --project "$PC_DIR" python "$THIS_DIR/host/host_test_set.py" \
    --port "$PORT" \
    --baud 115200 \
    --n "$N_IMAGES" \
    --tag "$TAG"

# ---------- regenerate figures from device predictions ---------------------

echo
echo "[host_testset] regenerating plots..."
uv run --project "$PC_DIR" python "$THIS_DIR/scripts/plot_models_report.py" \
    "$TAG" --root "$THIS_DIR" || true
uv run --project "$PC_DIR" python "$THIS_DIR/scripts/plot_models_report.py" \
    "$TAG" --variant int8 --root "$THIS_DIR" || true

echo
echo "================================================================"
echo "Done."
echo
echo "  predictions npz : $THIS_DIR/reports/predictions/${TAG}.npz"
echo "  figures         : $THIS_DIR/reports/figures/${TAG}/"
echo
echo "  Compare with PyTorch QAT predictions at:"
echo "      $THIS_DIR/reports/figures/${VARIANT}_qat/"
echo "  (device-side accuracy should match within ~0.3 pp)"
echo "================================================================"
