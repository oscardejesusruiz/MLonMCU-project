#!/usr/bin/env bash
# Train MAX78000-deployable variants — supports two modes:
#
#   fp32  (default)  Full 80-epoch run, NO QAT.  Stored as <v>_fp32.pth.tar.
#                    Reference accuracy and starting point for QAT fine-tuning.
#
#   qat              QAT fine-tune (40 epochs) starting from the existing
#                    fp32 checkpoint at half the fp32 LR.
#                    Stored as <v>_qat_train.pth.tar so the fp32 files are
#                    NEVER overwritten.
#                    QAT switch at epoch 5 (5 ep fp32 warmup → 35 ep QAT).
#                    Industry-standard QAT protocol: fine-tune a trained
#                    fp32 model to the int8 grid, NOT train QAT from scratch.
#                    Requires: <v>_fp32.pth.tar already exists.
#
# Usage:
#   ./train_max78000_models.sh <variant|all> [fp32|qat]
#
#   ./train_max78000_models.sh all              # fp32 training for all variants
#   ./train_max78000_models.sh improved         # fp32 for improved only
#   ./train_max78000_models.sh all qat          # QAT fine-tune all variants
#   ./train_max78000_models.sh mininet qat      # QAT fine-tune mininet only
#
# Variants:
#   baseline, improved, mininet, deeper, nascifarnet, ressimplenet
#
# After training, run synthesize_all.sh to produce the C deployment project.

set -uo pipefail

# ---------- args + paths --------------------------------------------------

VARIANTS_ALL="baseline improved mininet deeper nascifarnet ressimplenet"

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  echo "usage: $0 {all|baseline|improved|mininet|deeper|nascifarnet|ressimplenet} [fp32|qat]" >&2
  exit 1
fi

case "$1" in
  all) VARIANTS="$VARIANTS_ALL" ;;
  baseline|improved|mininet|deeper|ressimplenet|nascifarnet) VARIANTS="$1" ;;
  *) echo "unknown variant: $1" >&2; exit 1 ;;
esac

MODE="${2:-fp32}"
case "$MODE" in
  fp32|qat) ;;
  *) echo "unknown mode: $MODE  (choose fp32 or qat)" >&2; exit 1 ;;
esac

# TODO: adjust if your max78000 stack lives elsewhere.
AI="/Users/oscardejesusruiz/Desktop/project/max78000"
THIS_DIR="$(cd "$(dirname "$0")" && pwd)"
export MAXIM_PATH="$AI/ai8x-synthesis/sdk"

TRAINED_DIR="$THIS_DIR/trained_models"
LOG_DIR="$THIS_DIR/reports"
mkdir -p "$TRAINED_DIR" "$LOG_DIR"

if [ "$MODE" = "fp32" ]; then
  LOG="$LOG_DIR/_train_fp32.log"
  MODE_LABEL="fp32 (80 epochs, no QAT)"
else
  LOG="$LOG_DIR/_train_qat.log"
  MODE_LABEL="qat (40 epochs fine-tune, QAT from epoch 5)"
fi

echo "" >> "$LOG"
echo "==== started $(date) ====" >> "$LOG"
echo "================================================================"
echo "MAX78000 training — mode: $MODE_LABEL"
echo "  variants      : $VARIANTS"
echo "  AI            : $AI"
echo "  TRAINED_DIR   : $TRAINED_DIR"
echo "  LOG           : $LOG"
echo "================================================================"

# ---------- venv ----------------------------------------------------------

cd "$AI/ai8x-training"
source .venv/bin/activate

# ---------- fp32 training function ----------------------------------------

train_fp32() {
  local V=$1
  local ARCH                                                
  case "$V" in
    nascifarnet)   ARCH="ai85nascifarnet" ;;
    ressimplenet)  ARCH="ai85ressimplenetbn" ;;   # BN-augmented variant; BN folds at synth
    *)             ARCH="ai85net_cmsis_$V" ;;
  esac
  local OUT_CKPT="$TRAINED_DIR/${V}_fp32.pth.tar"

  echo ""                                                 | tee -a "$LOG"
  echo "######## $V ($ARCH) [fp32] $(date) ########"      | tee -a "$LOG"

  if [ -f "$OUT_CKPT" ]; then
    echo "✓ already trained: $OUT_CKPT"                   | tee -a "$LOG"
    return 0
  fi

  local LR BS EPOCHS SCHEDULE_YAML EXTRA_ARGS
  case "$V" in
    mininet)
      LR=0.005; BS=64; EPOCHS=80; OPTIM=adam; MOMENTUM=""
      SCHEDULE_YAML="policies/schedule_cosine_80.yaml"
      EXTRA_ARGS="--weight-decay 1e-4"
      ;;
    nascifarnet)
      LR=0.001; BS=100; EPOCHS=80; OPTIM=adam; MOMENTUM=""
      SCHEDULE_YAML="policies/schedule-cifar-nas.yaml"
      EXTRA_ARGS=""
      ;;
    ressimplenet)
      # Now uses BN-augmented variant (ai85ressimplenetbn) — train with the
      # same standard hyperparameters as the other dense-conv variants;
      # BN keeps activations well-conditioned across the 14-layer residual
      # stack, so the previous Adam-LR-collapse problem is gone.
      LR=0.001; BS=100; EPOCHS=80; OPTIM=adam; MOMENTUM=""
      SCHEDULE_YAML="policies/schedule.yaml"
      EXTRA_ARGS=""
      ;;
    *)
      LR=0.001; BS=100; EPOCHS=80; OPTIM=adam; MOMENTUM=""
      SCHEDULE_YAML="policies/schedule.yaml"
      EXTRA_ARGS=""
      ;;
  esac

  echo "→ lr=$LR  batch=$BS  epochs=$EPOCHS  schedule=$(basename $SCHEDULE_YAML)" | tee -a "$LOG"

  python train.py \
    --model "$ARCH" \
    --dataset CIFAR10 \
    --lr "$LR" --optimizer "$OPTIM" --epochs "$EPOCHS" --batch-size "$BS" $MOMENTUM \
    --deterministic \
    --compress "$SCHEDULE_YAML" \
    --qat-policy policies/no_qat_policy.yaml \
    --confusion --param-hist --pr-curves \
    --use-bias \
    $EXTRA_ARGS \
    --device MAX78000 \
    2>&1 | tee -a "$LOG"

  local LATEST=""
  for d in $(ls -td "$AI/ai8x-training/logs/"*/); do
      if [ -f "${d}best.pth.tar" ] && \
        grep -q "$ARCH" "$d"/*.log 2>/dev/null; then      
        LATEST="$d"; break
      fi
  done

  if [ -z "$LATEST" ]; then
    echo "✗ ERROR: no fp32 ckpt found for $V"             | tee -a "$LOG"
    return 1
  fi

  cp "${LATEST}best.pth.tar" "$OUT_CKPT"
  echo "✓ saved → $OUT_CKPT"                              | tee -a "$LOG"
}

# ---------- QAT fine-tuning function --------------------------------------
# Loads <v>_fp32.pth.tar and fine-tunes for 40 epochs with QAT starting at
# epoch 5 (5 ep fp32 warmup + 35 ep QAT) at half the fp32 LR.
#
# This is Maxim's recommended QAT flow and matches industry standard practice
# (NVIDIA TensorRT, Google TF-Lite QAT, Apple Core ML, etc.): QAT is applied
# as a short, low-LR fine-tune on top of a fully trained fp32 model. Training
# QAT from scratch deprives the network of the unconstrained feature-learning
# phase and consistently underperforms fine-tuning at int8 deployment.
#
# Saves the result as <v>_qat_train.pth.tar so the fp32 ckpt is preserved.

train_qat() {
  local V=$1
  local ARCH
  case "$V" in
    nascifarnet)   ARCH="ai85nascifarnet" ;;
    ressimplenet)  ARCH="ai85ressimplenetbn" ;;   # BN-augmented; BN folds at synth
    *)             ARCH="ai85net_cmsis_$V" ;;
  esac
  local FP32_CKPT="$TRAINED_DIR/${V}_fp32.pth.tar"
  local OUT_CKPT="$TRAINED_DIR/${V}_qat_train.pth.tar"

  echo ""                                                 | tee -a "$LOG"
  echo "######## $V ($ARCH) [qat fine-tune] $(date) ########" | tee -a "$LOG"

  if [ ! -f "$FP32_CKPT" ]; then
    echo "✗ skip $V: fp32 checkpoint not found at $FP32_CKPT" | tee -a "$LOG"
    echo "  run: ./train_max78000_models.sh $V fp32  first"        | tee -a "$LOG"
    return 1
  fi

  if [ -f "$OUT_CKPT" ]; then
    echo "✓ already trained: $OUT_CKPT"                   | tee -a "$LOG"
    return 0
  fi

  # QAT is a fine-tune: 40 epochs total, half the fp32 LR, switch at ep 5.
  local LR BS EPOCHS SCHEDULE_YAML EXTRA_ARGS
  case "$V" in
    mininet)
      # Mininet is the only variant that uses cosine LR + weight-decay
      # for fp32 (deep narrow VGG-style needs the regularization). At
      # fine-tune time, both of those become liabilities:
      #   - cosine restart from half-of-fp32 LR (0.0025) creates a HUGE
      #     LR shock vs the near-zero LR at the end of the cosine fp32
      #     run, destroying the carefully-trained weights at epoch 0
      #     of QAT before the QAT switch even kicks in;
      #   - weight-decay keeps moving the weights away from their fp32
      #     optimum throughout the fine-tune for no good reason.
      # Solution (mirrors pc-implementation/train_models.sh): drop LR by
      # 25x (0.0001) so the cosine restart is gentle, and disable
      # weight-decay. Keep cosine schedule so the LR still anneals.
      LR=0.0001; BS=64; EPOCHS=40
      SCHEDULE_YAML="policies/schedule_cosine_40.yaml"
      EXTRA_ARGS=""
      ;;
    nascifarnet)
      LR=0.0005; BS=100; EPOCHS=40                      # half of fp32 LR (0.001)
      SCHEDULE_YAML="policies/schedule_40.yaml"
      EXTRA_ARGS=""
      ;;
    ressimplenet)
      LR=0.0005; BS=100; EPOCHS=40                      # half of fp32 LR (0.001)
      SCHEDULE_YAML="policies/schedule_40.yaml"
      EXTRA_ARGS=""
      ;;
    *)
      LR=0.0005; BS=100; EPOCHS=40                      # half of fp32 LR (0.001)
      SCHEDULE_YAML="policies/schedule_40.yaml"
      EXTRA_ARGS=""
      ;;
  esac

  echo "→ QAT fine-tune from: $FP32_CKPT"                | tee -a "$LOG"
  echo "  lr=$LR  batch=$BS  epochs=$EPOCHS  qat_start=5  schedule=$(basename $SCHEDULE_YAML)" | tee -a "$LOG"

  python train.py \
    --model "$ARCH" \
    --dataset CIFAR10 \
    --lr "$LR" --optimizer adam --epochs "$EPOCHS" --batch-size "$BS" \
    --deterministic \
    --compress "$SCHEDULE_YAML" \
    --qat-policy policies/qat_policy_cifar10.yaml \
    --exp-load-weights-from "$FP32_CKPT" \
    --confusion --param-hist --pr-curves \
    --use-bias \
    $EXTRA_ARGS \
    --device MAX78000 \
    2>&1 | tee -a "$LOG"

  local LATEST=""
  for d in $(ls -td "$AI/ai8x-training/logs/"*/); do
    if [ -f "${d}qat_qat_best.pth.tar" ] && \
       grep -q "$ARCH" "$d"/*.log 2>/dev/null; then 
      LATEST="$d"; break
    fi
  done

  if [ -z "$LATEST" ]; then
    echo "✗ ERROR: no QAT ckpt found for $V"              | tee -a "$LOG"
    return 1
  fi

  cp "${LATEST}qat_qat_best.pth.tar" "$OUT_CKPT"
  echo "✓ saved → $OUT_CKPT (fp32 checkpoint preserved)"  | tee -a "$LOG"
}

# ---------- run -----------------------------------------------------------

for v in $VARIANTS; do
  if [ "$MODE" = "fp32" ]; then
    train_fp32 "$v"
  else
    train_qat "$v"
  fi
done

# ---------- summary -------------------------------------------------------

echo ""                                                    | tee -a "$LOG"
echo "==== finished $(date) ===="                          | tee -a "$LOG"
echo ""
echo "================================================================"
if [ "$MODE" = "fp32" ]; then
  echo "fp32 checkpoints:"
  ls "$TRAINED_DIR/"*_fp32.pth.tar 2>/dev/null | awk '{print "  "$0}' || echo "  (none)"
  echo ""
  echo "Next options:"
  echo "  QAT fine-tune : ./train_max78000_models.sh <variant|all> qat"
  echo "  Synthesize    : ./synthesize_all.sh [variant]     (uses fp32)"
else
  echo "QAT fine-tuned checkpoints:"
  ls "$TRAINED_DIR/"*_qat_train.pth.tar 2>/dev/null | awk '{print "  "$0}' || echo "  (none)"
  echo ""
  echo "Next: synthesize from QAT checkpoints:"
  echo "  ./synthesize_all.sh <variant> --from-qat"
  echo "  (or edit synthesize_all.sh FP32_CKPT → \${V}_qat_train.pth.tar)"
fi
echo "================================================================"
