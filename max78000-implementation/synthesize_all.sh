#!/usr/bin/env bash
# Generate the C deployment project for each trained MAX78000 variant.
#
# For each <variant>_fp32.pth.tar in trained_models/, this:
#   1. Copies it to $AI/ai8x-synthesis/trained/<variant>_qat.pth.tar
#      (despite the name, ai8x-synthesis treats any checkpoint as the
#      starting point; PTQ-like quantization is applied next)
#   2. Quantizes to int8 via quantize.py             → <V>_q.pth.tar
#   3. Synthesizes a C project via ai8xize.py        → synthed_net_<V>/<V>/
#
# The resulting C project at
#   $AI/ai8x-synthesis/synthed_net_<V>/<V>/
# is ready to flash. See bash_device_scripts/device_profile.sh + bash_device_scripts/host_profile.sh
# for on-device measurement of MAC_Ops/Cycle, latency, and energy.
#
# Usage:
#   ./synthesize_all.sh                      # fp32 PTQ, all variants
#   ./synthesize_all.sh <variant>            # fp32 PTQ, one variant
#   ./synthesize_all.sh <variant> --from-qat # use <v>_qat_train.pth.tar instead
#   ./synthesize_all.sh all --from-qat       # QAT synth for all variants
#
# Variants:   baseline, improved, mininet, deeper, resnet8, wide_improved, ai85nascifarnet
# NOT supported: baseline_5x5 (5x5 kernels not on MAX78000)

set -uo pipefail
cd "$(dirname "$0")"

AI="${AI:-/Users/oscardejesusruiz/Desktop/project/max78000}"
THIS_DIR="$(pwd)"
export MAXIM_PATH="$AI/ai8x-synthesis/sdk"

TRAINED_DIR="$AI/ai8x-synthesis/trained"
SAMPLE_NPY="$AI/ai8x-synthesis/tests/sample_cifar10.npy"
LOG="$THIS_DIR/reports/_synthesize_all.log"

mkdir -p "$TRAINED_DIR" "$(dirname "$SAMPLE_NPY")" "$(dirname "$LOG")"
echo "" >> "$LOG"
echo "==== started $(date) ====" >> "$LOG"

# ---- parse args (variant and optional --from-qat flag) -----------------

FROM_QAT=0
VARIANT_ARG=""
for arg in "$@"; do
  case "$arg" in
    --from-qat) FROM_QAT=1 ;;
    *)          VARIANT_ARG="$arg" ;;
  esac
done

if [ "$FROM_QAT" = "1" ]; then
  CKPT_SUFFIX="_qat_train.pth.tar"
  FUSED_SUFFIX="_qat_fused.pth.tar"
  NORM_SUFFIX="_qat_norm.pth.tar"
  Q_SUFFIX="_qat_q.pth.tar"
  MODE_LABEL="QAT (from *_qat_train.pth.tar)"
else
  CKPT_SUFFIX="_fp32.pth.tar"
  FUSED_SUFFIX="_fused.pth.tar"
  NORM_SUFFIX="_qat.pth.tar"
  Q_SUFFIX="_q.pth.tar"
  MODE_LABEL="PTQ (from *_fp32.pth.tar)"
fi

# ---- pick variants -----------------------------------------------------

if [ -n "$VARIANT_ARG" ] && [ "$VARIANT_ARG" != "all" ]; then
  VARIANTS="$VARIANT_ARG"
else
  VARIANTS=""
  for ckpt in "trained_models/"*"${CKPT_SUFFIX}"; do
    [ -f "$ckpt" ] || continue
    v=$(basename "$ckpt" "${CKPT_SUFFIX}")
    case "$v" in *_OLD*) continue ;; esac
    VARIANTS="$VARIANTS $v"
  done
fi

echo "================================================================"
echo "MAX78000 synthesis pipeline — mode: $MODE_LABEL"
echo "  variants : $VARIANTS"
echo "  AI       : $AI"
echo "  output   : $AI/ai8x-synthesis/synthed_net_<variant>/<variant>/"
echo "================================================================"

# ---- per-variant pipeline --------------------------------------------

synth_one () {
  local V=$1
  # Most variants use the ai85net_cmsis_<V> arch name (our custom models in
  # ai8x-training/models/project_models.py), but two variants are Maxim's
  # own reference architectures with different names — keep this case
  # statement in sync with train_max78000_models.sh's ARCH selection.
  local ARCH
  case "$V" in
    nascifarnet)   ARCH="ai85nascifarnet" ;;
    ressimplenet)  ARCH="ai85ressimplenetbn" ;;   # BN-augmented; BN folds at synth
    *)             ARCH="ai85net_cmsis_$V" ;;
  esac
  local FUSED_ARCH="$ARCH"
  local YAML="$THIS_DIR/networks/network_${V}.yaml"
  local SRC_CKPT="$THIS_DIR/trained_models/${V}${CKPT_SUFFIX}"
  local FUSED_CKPT="$TRAINED_DIR/${V}${FUSED_SUFFIX}"   # BN-folded
  local NORM_CKPT="$TRAINED_DIR/${V}${NORM_SUFFIX}"     # quantize.py input
  local Q_CKPT="$TRAINED_DIR/${V}${Q_SUFFIX}"
  local SYNTH_DIR="$AI/ai8x-synthesis/synthed_net_${V}/${V}"

  echo ""                                                | tee -a "$LOG"
  echo "######## $V ($ARCH) [${MODE_LABEL}] $(date) ########" | tee -a "$LOG"

  if [ ! -f "$SRC_CKPT" ]; then
    echo "✗ skip $V: checkpoint not found at $SRC_CKPT"  | tee -a "$LOG"
    [ "$FROM_QAT" = "1" ] && \
      echo "  run: ./train_max78000_models.sh $V qat  first"      | tee -a "$LOG"
    return 1
  fi
  if [ ! -f "$YAML" ]; then
    echo "✗ skip $V: no synthesis YAML at $YAML"         | tee -a "$LOG"
    return 1
  fi

  # ---- 1. BN-fold --------------------------------------------------------
  #
  # For fp32-trained models BN is a separate layer — must be folded into the
  # preceding conv before quantization. For QAT-trained models ai8x already
  # fuses BN during QAT (via ai8x.fuse_bn_layers), so bn_fuser_v2.py is still
  # needed here to handle any remaining unfused BN keys gracefully.
  #
  # We use bn_fuser_v2.py (local) instead of ai8x-training's batchnormfuser.py
  # because the upstream version has a rsplit bug for nested names (resnet8).
  # NOTE: do NOT apply the 0.25 rescale — it is wrong for PTQ/QAT pipelines.
  source "$AI/ai8x-training/.venv/bin/activate"
  if [ ! -f "$FUSED_CKPT" ] || [ "$SRC_CKPT" -nt "$FUSED_CKPT" ]; then
    echo "[1a/4] BN-fold → $FUSED_CKPT"                  | tee -a "$LOG"
    python "$THIS_DIR/scripts/bn_fuser_v2.py" \
      -i "$SRC_CKPT" \
      -o "$FUSED_CKPT" \
      -oa "$FUSED_ARCH" 2>&1 | tee -a "$LOG"
    if [ ! -f "$FUSED_CKPT" ]; then
      echo "✗ BN-fold failed for $V — synthesis will not work"   | tee -a "$LOG"
      deactivate
      return 1
    fi
  else
    echo "[1a/4] BN-fold up-to-date → $FUSED_CKPT"       | tee -a "$LOG"
  fi
  deactivate

  # Copy fused ckpt under the conventional name expected by ai8x-synthesis
  if [ ! -f "$NORM_CKPT" ] || [ "$FUSED_CKPT" -nt "$NORM_CKPT" ]; then
    cp "$FUSED_CKPT" "$NORM_CKPT"
    echo "[1b/4] copied → $NORM_CKPT"                    | tee -a "$LOG"
  else
    echo "[1b/4] up-to-date → $NORM_CKPT"                | tee -a "$LOG"
  fi

  # ---- 2. quantize to int8 ----
  source "$AI/ai8x-synthesis/.venv/bin/activate"
  if [ ! -f "$Q_CKPT" ] || [ "$NORM_CKPT" -nt "$Q_CKPT" ]; then
    echo "[2/4] quantize → $Q_CKPT"                      | tee -a "$LOG"
    (cd "$AI/ai8x-synthesis" && \
       python quantize.py "$NORM_CKPT" "$Q_CKPT" --device MAX78000 -v) \
       2>&1 | tee -a "$LOG"
  else
    echo "[2/4] up-to-date → $Q_CKPT"                    | tee -a "$LOG"
  fi
  deactivate

  # ---- 3a. sample input (one-time per project) ----
  if [ ! -f "$SAMPLE_NPY" ]; then
    echo "[3/4] saving CIFAR-10 sample → $SAMPLE_NPY"    | tee -a "$LOG"
    source "$AI/ai8x-training/.venv/bin/activate"
    (cd "$AI/ai8x-training" && \
       python train.py --model "$ARCH" --dataset CIFAR10 --evaluate \
         --save-sample 42 \
         --exp-load-weights-from "$Q_CKPT" \
         --use-bias \
         -8 --device MAX78000) 2>&1 | tee -a "$LOG"
    cp "$AI/ai8x-training/sample_cifar10.npy" "$SAMPLE_NPY"
    deactivate
  fi

  # ---- 3b. synthesize C project ----
  source "$AI/ai8x-synthesis/.venv/bin/activate"
  if [ ! -f "$SYNTH_DIR/cnn.c" ] || [ "$Q_CKPT" -nt "$SYNTH_DIR/cnn.c" ]; then
    echo "[4/4] synthesize → $SYNTH_DIR"                 | tee -a "$LOG"
    (cd "$AI/ai8x-synthesis" && \
       python ai8xize.py \
         --test-dir "synthed_net_${V}" \
         --prefix "${V}" \
         --checkpoint-file "$Q_CKPT" \
         --config-file "$YAML" \
         --sample-input "$SAMPLE_NPY" \
         --softmax --device MAX78000 --compact-data \
         --mexpress --timer 0 --display-checkpoint --verbose --overwrite) \
       2>&1 | tee -a "$LOG"
  else
    echo "[4/4] up-to-date → $SYNTH_DIR"                 | tee -a "$LOG"
  fi
  deactivate

  echo "✓ $V → $SYNTH_DIR"                               | tee -a "$LOG"
}

# ---- run order --------------------------------------------------------

for v in $VARIANTS; do
  synth_one "$v"
done

# ---- summary ----------------------------------------------------------

echo ""                                                 | tee -a "$LOG"
echo "==== finished $(date) ===="                       | tee -a "$LOG"
echo ""
echo "================================================================"
echo "Synthesized projects:"
for v in $VARIANTS; do
  SYNTH_DIR="$AI/ai8x-synthesis/synthed_net_${v}/${v}"
  if [ -d "$SYNTH_DIR" ]; then
    echo "  ✓ $v   → $SYNTH_DIR"
  else
    echo "  ✗ $v   (failed — see $LOG)"
  fi
done
echo ""
echo "Next — flash + measure (per variant):"
echo ""
echo "  Per-layer profile (1 inference; reports CNN cycles + CPU cycles):"
echo "    ./bash_device_scripts/device_profile.sh <variant>      # build + flash"
echo "    ./bash_device_scripts/host_profile.sh   <variant>      # read UART block"
echo ""
echo "  Full test-set inference (10000 images via UART; reports device acc):"
echo "    ./bash_device_scripts/device_testset.sh <variant>      # build + flash"
echo "    ./bash_device_scripts/host_testset.sh   <variant>      # stream + collect"
echo ""
echo "  Metrics produced after device measurement:"
echo "    MAC_Ops/Cycle  = (model_MACs × 2) / cpu_cycles_reported"
echo "    Inference time = cnn_cycles / 100 MHz (or cnn_us directly)"
echo "    MAC_Ops/W      = (model_MACs × 2) / (inference_time × measured_power)"
echo "                     (needs INA219 / Joulescope for power)"
echo "================================================================"
