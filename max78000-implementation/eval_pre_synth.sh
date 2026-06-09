#!/usr/bin/env bash
# Evaluate each MAX78000 model checkpoint on the CIFAR-10 test set BEFORE
# flashing the device.  Supports two modes:
#
#   fp32 (default)   Three stages:
#     1. fp32      — trained_models/<v>_fp32.pth.tar          (reference)
#     2. fused     — $AI/.../trained/<v>_fused.pth.tar        (BN folded)
#     3. int8 sim  — $AI/.../trained/<v>_q.pth.tar            (quantized, int8 sim)
#
#   qat              Three stages:
#     1. qat       — trained_models/<v>_qat_train.pth.tar     (QAT fine-tuned)
#     2. qat-fused — $AI/.../trained/<v>_qat_fused.pth.tar    (BN folded)
#     3. int8 sim  — $AI/.../trained/<v>_qat_q.pth.tar        (quantized, int8 sim)
#
# Usage:
#   ./eval_pre_synth.sh                       # fp32, all variants
#   ./eval_pre_synth.sh qat                   # qat, all variants
#   ./eval_pre_synth.sh fp32 improved mininet # fp32, subset
#   ./eval_pre_synth.sh qat improved mininet  # qat, subset
#
# Output: summary table + per-variant logs in reports/_eval_pre_synth/{fp32,qat}/

set -uo pipefail
cd "$(dirname "$0")"

THIS_DIR="$(pwd)"
AI="${AI:-$HOME/Desktop/project/max78000}"
TRAINED_PC="$THIS_DIR/trained_models"
TRAINED_AI="$AI/ai8x-synthesis/trained"

# ---- parse args: optional mode as first arg, rest are variant names --------

MODE="fp32"
VARIANT_ARGS=()
for arg in "$@"; do
  case "$arg" in
    fp32|qat) MODE="$arg" ;;
    *)        VARIANT_ARGS+=("$arg") ;;
  esac
done

# Checkpoint suffixes per mode
if [ "$MODE" = "qat" ]; then
  SRC_SUFFIX="_qat_train.pth.tar"
  FUSED_SUFFIX="_qat_fused.pth.tar"
  Q_SUFFIX="_qat_q.pth.tar"
  COL_A="qat acc"
  COL_B="qat-fused"
  MODE_LABEL="QAT fine-tuned"
else
  SRC_SUFFIX="_fp32.pth.tar"
  FUSED_SUFFIX="_fused.pth.tar"
  Q_SUFFIX="_q.pth.tar"
  COL_A="fp32 acc"
  COL_B="fused acc"
  MODE_LABEL="fp32 + PTQ"
fi

LOG_DIR="$THIS_DIR/reports/_eval_pre_synth/$MODE"
mkdir -p "$LOG_DIR"

# ---- pick variants ---------------------------------------------------------

if [ ${#VARIANT_ARGS[@]} -gt 0 ]; then
  VARIANTS=("${VARIANT_ARGS[@]}")
else
  VARIANTS=()
  for ckpt in "$TRAINED_PC"/*"${SRC_SUFFIX}"; do
    [ -f "$ckpt" ] || continue
    v=$(basename "$ckpt" "${SRC_SUFFIX}")
    case "$v" in *_OLD*) continue ;; esac
    VARIANTS+=("$v")
  done
fi

if [ ${#VARIANTS[@]} -eq 0 ]; then
  echo "No checkpoints found for mode=$MODE in $TRAINED_PC"
  echo "Run: ./train_max78000_models.sh all ${MODE}" >&2
  exit 1
fi

echo "================================================================"
echo "Pre-synth evaluation  [mode: $MODE_LABEL]"
echo "  variants : ${VARIANTS[*]}"
echo "  src      : $TRAINED_PC/<v>${SRC_SUFFIX}"
echo "  fused    : $TRAINED_AI/<v>${FUSED_SUFFIX}"
echo "  int8     : $TRAINED_AI/<v>${Q_SUFFIX}"
echo "  logs     : $LOG_DIR/"
echo "================================================================"
echo

# ---- 1+2. src + fused (via verify_fold.py) ---------------------------------

VERIFY_LOG="$LOG_DIR/verify_fold.log"
echo "[1/3] $COL_A + $COL_B via verify_fold.py  (log → $VERIFY_LOG)"
source "$AI/ai8x-training/.venv/bin/activate"
AI="$AI" python scripts/verify_fold.py --mode "$MODE" "${VARIANTS[@]}" 2>&1 | tee "$VERIFY_LOG"
deactivate
echo

# ---- 3. int8 simulated (train.py --evaluate -8) ----------------------------

echo "[2/3] int8 simulated via train.py --evaluate -8"
echo

# macOS bash 3.2 has no associative arrays → stash accs in per-variant files.
ACC_DIR="$LOG_DIR/_acc"
rm -rf "$ACC_DIR" && mkdir -p "$ACC_DIR"

# Parse verify_fold.py output: "<variant>  XX.XX% YY.YY% ZZ.ZZ%  W.WWWW"
for v in "${VARIANTS[@]}"; do
  line=$(grep -E "^${v}[[:space:]]" "$VERIFY_LOG" || true)
  col_a=$(echo "$line" | awk '{print $2}' | tr -d '%')
  col_b=$(echo "$line" | awk '{print $3}' | tr -d '%')
  echo "${col_a:--}" > "$ACC_DIR/col_a_${v}"
  echo "${col_b:--}" > "$ACC_DIR/col_b_${v}"
  echo "-"           > "$ACC_DIR/int8_${v}"
done

source "$AI/ai8x-training/.venv/bin/activate"
cd "$AI/ai8x-training"
for v in "${VARIANTS[@]}"; do
  Q_CKPT="$TRAINED_AI/${v}${Q_SUFFIX}"
  if [ ! -f "$Q_CKPT" ]; then
    echo "[$v] skip int8: $Q_CKPT not found"
    if [ "$MODE" = "qat" ]; then
      echo "       run: ./synthesize_all.sh $v --from-qat"
    else
      echo "       run: ./synthesize_all.sh $v"
    fi
    continue
  fi
  # Map variant → arch name. Most variants use ai85net_cmsis_<v>, but
  # nascifarnet and ressimplenet are Maxim/custom architectures with their
  # own arch names — keep in sync with train_max78000_models.sh / synthesize_all.sh.
  case "$v" in
    nascifarnet)   ARCH="ai85nascifarnet" ;;
    ressimplenet)  ARCH="ai85ressimplenetbn" ;;
    *)             ARCH="ai85net_cmsis_${v}" ;;
  esac
  ILOG="$LOG_DIR/int8_${v}.log"
  echo "[$v] int8 sim → $ILOG"
  python train.py \
      --model "$ARCH" \
      --dataset CIFAR10 \
      --confusion \
      --evaluate \
      --exp-load-weights-from "$Q_CKPT" \
      --use-bias \
      -8 \
      --device MAX78000 \
      --batch-size 100 \
      > "$ILOG" 2>&1
  top1=$(grep -oE 'Top1: [0-9.]+' "$ILOG" | tail -1 | awk '{print $2}')
  echo "${top1:-?}" > "$ACC_DIR/int8_${v}"
  echo "[$v] int8 acc: ${top1:--}%"
done
cd "$THIS_DIR"
deactivate

echo
echo "[3/3] summary  [mode: $MODE_LABEL]"
echo

# ---- final summary table ---------------------------------------------------

printf '%-14s  %10s  %10s  %10s\n' "variant" "$COL_A" "$COL_B" "int8 sim"
printf '%-14s  %10s  %10s  %10s\n' "-------" "----------" "----------" "--------"
for v in "${VARIANTS[@]}"; do
  a=$(cat "$ACC_DIR/col_a_${v}" 2>/dev/null || echo "-")
  b=$(cat "$ACC_DIR/col_b_${v}" 2>/dev/null || echo "-")
  c=$(cat "$ACC_DIR/int8_${v}"  2>/dev/null || echo "-")
  printf '%-14s  %9s%%  %9s%%  %9s%%\n' "$v" "$a" "$b" "$c"
done

echo
if [ "$MODE" = "fp32" ]; then
  echo "Interpretation:"
  echo "  fp32 ≈ fused   →  BN fold OK"
  echo "  fused ≫ int8  →  PTQ hurts; consider QAT: ./train_max78000_models.sh all qat"
  echo "  int8 ≈ fused   →  device acc should match int8 sim"
else
  echo "Interpretation:"
  echo "  qat ≈ qat-fused  →  BN fold on QAT ckpt OK"
  echo "  qat ≫ int8      →  quantize.py hurts even QAT model (rare)"
  echo "  int8 ≈ qat       →  device acc should match int8 sim (~QAT acc)"
fi
