#!/usr/bin/env bash
# Sequential supervisor — trains the 6 main architectures × 3 techniques:
#   1. fp32 + PTQ        (80 epochs from scratch)
#   2. QAT fine-tune     (40 epochs, loads <v>_fp32.pt then switches to QAT
#                          at epoch 5 — matches the MAX78000 recipe exactly:
#                            5 ep fp32 warmup → 35 ep QAT fine-tuning
#                          at half the original LR. This is the industry-
#                          standard QAT protocol (NVIDIA TensorRT, Google
#                          TF-Lite, Maxim ai8x): start from a fully trained
#                          fp32 model and adapt the weights to int8 with a
#                          short, low-LR fine-tune, rather than training QAT
#                          from scratch which deprives the network of the
#                          unconstrained feature-learning phase.)
#   3. Pruning 50%       (load fp32, prune, fine-tune 10 epochs)
#
# Variants: baseline_5x5, baseline, improved, deeper, mininet, ressimplenet,
#           nascifarnet
#
# Idempotent: skips a stage if its checkpoint already exists. Launch with:
#   nohup ./train_pc_models.sh > reports/_train_pc_models.log 2>&1 &

set -uo pipefail   # no -e: a single failure shouldn't kill the rest
cd "$(dirname "$0")"

LOG=reports/_train_main.log
mkdir -p reports/predictions trained_models
echo "==== started $(date) ====" >> "$LOG"

VARIANTS="baseline_5x5 baseline improved deeper mininet ressimplenet nascifarnet"

train_variant () {
  local V=$1
  echo ""                                  | tee -a "$LOG"
  echo "######## $V $(date) ########"      | tee -a "$LOG"

  # ARGS_FP32: 80-epoch from-scratch training (Phase 1).
  # ARGS_QAT:  40-epoch fine-tune starting from the fp32 checkpoint, at half
  #            the fp32 LR. QAT switch happens at epoch 5 (industry-standard
  #            warmup), so 35 epochs of QAT fine-tuning on already-optimized
  #            fp32 weights — exactly the MAX78000 protocol.
  if [ "$V" = "mininet" ]; then
    ARGS_FP32="--optimizer adam --lr 0.005 --batch-size 64 --epochs 80 \
          --weight-decay 1e-4 --scheduler cosine --input-size 32 --augment"
    # ↓ LR molt més baix per al QAT, sense weight-decay
    ARGS_QAT="--optimizer adam --lr 0.0001 --batch-size 64 --epochs 40 \
          --weight-decay 0 --scheduler cosine --input-size 32 --augment"
  else
    ARGS_FP32="--optimizer adam --lr 0.001 --batch-size 100 --epochs 80 \
          --weight-decay 0.0 --augment"
    ARGS_QAT="--optimizer adam --lr 0.0005 --batch-size 100 --epochs 40 \
          --weight-decay 0.0 --augment"
  fi

  # 1. fp32 + PTQ (80 epochs from scratch)
  if [ ! -f "trained_models/${V}_fp32.pt" ]; then
    echo "→ ${V}_fp32"                     | tee -a "$LOG"
    uv run python -m scripts.run_experiment "$V" $ARGS_FP32 --tag "${V}_fp32" \
      >> "$LOG" 2>&1
  else
    echo "✓ ${V}_fp32 already exists"      | tee -a "$LOG"
  fi

  # 2. QAT fine-tune: load <V>_fp32.pt and switch to QAT at epoch 5
  #    (MAX78000-style: 5 ep fp32 warmup + 35 ep QAT, total 40 ep, LR÷2)
  if [ ! -f "trained_models/${V}_qat.pt" ] && [ -f "trained_models/${V}_fp32.pt" ]; then
    echo "→ ${V}_qat  (loading ${V}_fp32.pt, QAT from epoch 5)" | tee -a "$LOG"
    uv run python -m scripts.run_experiment "$V" $ARGS_QAT \
        --load-fp32 "trained_models/${V}_fp32.pt" \
        --qat-start-epoch 5 \
        --tag "${V}_qat" \
      >> "$LOG" 2>&1
  else
    echo "✓ ${V}_qat already exists (or fp32 missing)" | tee -a "$LOG"
  fi

  # 3. Pruning 50% + fine-tune
  if [ ! -f "trained_models/prev/${V}_prune50.pt" ] && [ -f "trained_models/${V}_fp32.pt" ]; then
    echo "→ ${V}_prune50"                  | tee -a "$LOG"
    uv run python -m scripts.run_pruning \
        --model "$V" \
        --base-ckpt "trained_models/${V}_fp32.pt" \
        --sparsity 0.5 --finetune-epochs 10 \
        --input-size 32 \
        --tag "${V}_prune50" >> "$LOG" 2>&1
  else
    echo "✓ ${V}_prune50 already exists"   | tee -a "$LOG"
  fi
}

# ---------- run order ---------------------------------------------------

for v in $VARIANTS; do
  train_variant "$v"
done

# ---------- final reports -----------------------------------------------

echo "→ build_report + plot"                | tee -a "$LOG"
uv run python -m scripts.build_report       >> "$LOG" 2>&1 || true
uv run python -m scripts.plot_models_report all                         >> "$LOG" 2>&1 || true
uv run python -m scripts.plot_models_report all --variant int8           >> "$LOG" 2>&1 || true
uv run python -m scripts.eval_int8  >> "$LOG" 2>&1 || true

echo ""                                     | tee -a "$LOG"
echo "==== finished $(date) ===="           | tee -a "$LOG"
