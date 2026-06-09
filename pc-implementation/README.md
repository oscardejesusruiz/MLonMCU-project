# PC-side reference — CIFAR-10 training & int8 simulation

Pure-PyTorch training pipeline that produces the fp32 and QAT checkpoints
consumed by both deployment targets:

- [`../max78000-implementation`](../max78000-implementation) — MAX78000 CNN accelerator
- [`../imx500-implementation`](../imx500-implementation) — Sony IMX500 intelligent vision sensor

Also serves as the host-side **int8 PTQ simulator** for the MAX78000 path,
mirroring the on-device behaviour (BN folding → symmetric per-tensor int8)
so MAX78000 deployment-cost numbers are reproducible without a board. The
IMX500 path runs its own MCT-based PTQ on the same checkpoints, so the
same `*.pt` files feed two completely different quantization toolchains.

Runs on Apple Silicon (MPS), CUDA, or CPU — no microcontroller required.

---

## Pipeline overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   train_pc_models.sh   (fp32 + QAT + optional pruning)                  │
│   ──────────────                                                        │
│   for each variant:                                                     │
│     1. fp32 training  (80 ep from scratch)  →  <v>_fp32.pt              │
│     2. QAT fine-tune  (40 ep, load fp32)    →  <v>_qat.pt               │
│     3. pruning 50%    (optional)            →  <v>_prune50.pt           │
│                                                                         │
│           │                                                             │
│           ▼                                                             │
│                                                                         │
│   scripts/eval_int8.py        scripts/build_report.py                   │
│   ─────────────────────       ───────────────────────                   │
│   MAX78000-realistic int8     Aggregate metrics →                       │
│   PTQ + activation calib      reports/summary.md                        │
│   → reports/int8_eval/...     reports/figures/*.png                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Models evaluated

Seven architectures total. All seven deploy on the IMX500. Six of them
are mirrored 1:1 by the MAX78000 implementation (same channel widths,
same BN-fold-equivalent topology); `baseline_5x5` cannot deploy on the
MAX78000 because its 5×5 kernels are outside the accelerator's op-set,
but the IMX500 has no such restriction.

| Variant | Family | Paper / source | MAX78000 | IMX500 |
|---|---|---|---|---|
| `baseline` | Dense conv, shallow | Lai et al. CMSIS-NN (arXiv 2018), 3×3 adaptation | ✓ | ✓ |
| `baseline_5x5` | Dense conv with 5×5 kernels | Lai et al. CMSIS-NN — paper-faithful 5×5 version | **✗** | ✓ |
| `improved` | Dense conv + BN (ablation) | this work | ✓ | ✓ |
| `deeper` | Dense conv, deeper (ablation) | this work | ✓ | ✓ |
| `mininet` | Dense conv, deep VGG-style | Banbury et al. MicroNets (MLSys 2021) | ✓ | ✓ |
| `nascifarnet` | NAS-found, MCU-targeted | Maxim ai8x-training (2021) | ✓ | ✓ |
| `ressimplenet` | Residual SimpleNet | HasanPour et al. (arXiv 2016) | ✓ (BN-augmented variant) | ✓ |

The four CMSIS-NN-style models (`baseline`, `baseline_5x5`, `improved`,
`deeper`) use the same Conv-BN-ReLU-Pool template at different widths and
depths. `mininet` is narrower and deeper (VGG-Micro family). `nascifarnet`
and `ressimplenet` are external architectures from Maxim's reference repo,
ported to PC so the PC↔device comparison is apples-to-apples.

---

## Directory structure

```
pc-implementation/
├── pyproject.toml                ← uv-managed deps (torch, torchvision, matplotlib, …)
├── train_pc_models.sh            ← sequential trainer (run this first)
├── plot_network_diagrams.sh      ← render 3D layered-block diagram per variant
├── training/                     ← reusable library (models, data, engine, quantize, …)
│   ├── models.py                   # MODEL_REGISTRY (all seven architectures)
│   ├── data.py                     # CIFAR-10 loaders with optional augmentation
│   ├── engine.py                   # train/eval loop with optional QAT switch
│   ├── quantize.py                 # MAX78000-realistic int8 PTQ + QAT helpers
│   └── utils.py                    # MAC/param accounting, device pick
├── scripts/                      ← experiment drivers + analysis
│   ├── run_experiment.py           # one variant → fp32 (+ optional QAT switch)
│   ├── run_pruning.py              # magnitude pruning of an fp32 checkpoint
│   ├── run_qat_prune.py            # prune a QAT-trained checkpoint
│   ├── run_distillation.py         # knowledge-distillation training
│   ├── eval_int8.py                # MAX78000-realistic int8 PTQ on existing checkpoints
│   ├── build_report.py             # aggregate *_metrics.json → summary.md + Pareto plot
│   ├── plot_models_report.py       # per-experiment training curves + Pareto figure
│   └── plot_network_diagrams.py    # render <variant>_layered.png via visualtorch
├── trained_models/               ← *.pt checkpoints (one per variant × mode)
├── reports/                      ← metrics JSONs, predictions, plots
│   └── network_diagrams/           # output of plot_network_diagrams.sh
├── data/                         ← CIFAR-10 (auto-downloaded by torchvision)
└── checkpoints/                  ← TensorBoard event files, if enabled
```

### What each `training/` module does

| Module | Purpose |
|---|---|
| `models.py` | `MODEL_REGISTRY` — every architecture as a `nn.Module` class. The MAX78000-targeted variants include BatchNorm so PC and on-device training are stable; BN is folded into the preceding Conv at deployment, so the device sees a no-BN model with equivalent weights. |
| `data.py` | `get_loaders(...)` returns `(train, val, test)` CIFAR-10 loaders. Optional augmentation (random crop + horizontal flip + ColorJitter), per-channel normalization matching Lai 2018. |
| `engine.py` | `train(...)` with an optional `qat_start_epoch`: at that epoch the Conv/Linear layers are swapped for QAT-aware versions (`quantize.convert_to_qat`). `evaluate(...)` returns top-1 acc, loss, predictions, and probabilities. |
| `quantize.py` | Two entry points: `quantize_model_ptq` (BN-fold → weight quant → activation calibration → fake-quant hooks; mirrors what `bn_fuser_v2.py` + `quantize.py` do on the MAX78000 side); `convert_to_qat` (swap Conv/Linear for QAT-aware modules during training). |
| `utils.py` | `compute_stats(model)` returns total params, fp32 / int8 weight bytes, MACs/inference, and per-layer MAC breakdown. |

---

## Quick start

### 1. Install deps (one-time)

```bash
cd pc-implementation
uv sync                    # creates .venv with everything from pyproject.toml
```

### 2. Train everything

```bash
./train_pc_models.sh       # ~3-6 h on Apple Silicon MPS or CUDA GPU
```

For each variant the script runs three stages:
1. **fp32** — 80 epochs from scratch. Adam, LR=0.001 (mininet: cosine + LR=0.005 + WD=1e-4). Output: `trained_models/<v>_fp32.pt` + `<v>_fp32_ptq.pt`.
2. **QAT** — 40 epochs fine-tune starting from `<v>_fp32.pt`, switch to QAT modules at epoch 5. LR is halved (mininet uses LR=0.0001 with no WD to avoid cosine-restart shock). Output: `trained_models/<v>_qat.pt`.
3. **Pruning 50%** — magnitude prune, then fine-tune 10 epochs. Output: `trained_models/<v>_prune50.pt`.

The script is idempotent: re-running skips a stage whose output checkpoint already exists. Per-stage metrics (loss curves, fp32 + int8 accuracy, params, MACs) land in `reports/<tag>_metrics.json`.

### 3. Compute MAX78000-realistic int8 PTQ for any checkpoint

```bash
uv run python -m scripts.eval_int8                          # all variants × {fp32, qat}
uv run python -m scripts.eval_int8 --no-power-of-two        # tighter scales, more aggressive
```

Outputs to `reports/int8_eval/`:
- `summary.txt` — one row per (variant, mode) with params, MACs/inf, fp32 acc, int8 acc
- `acc_vs_macs.png` — scatter plot, accuracy vs MACs/inference (linear axis)

This script uses the **same `quantize_model_ptq` function** as `run_experiment.py`, so its int8 numbers are directly comparable to the `int8_test_acc` field of the per-model JSON. fp32-trained models show a real BN-fold-induced PTQ drop (typically 5-20 pp); QAT-trained models should recover most of it.

### 4. Build the aggregate report

```bash
uv run python -m scripts.build_report                # writes reports/summary.md + figures/pareto.png
uv run python -m scripts.plot_models_report all      # per-experiment curves
```

`build_report.py` reads every `<tag>_metrics.json` in `reports/`, produces:
- `reports/summary.md` — markdown table comparing all runs (acc / params / MACs / weight bytes)
- `reports/figures/pareto.png` — accuracy-vs-compute Pareto frontier
- `reports/figures/training_curves.png` — train/test accuracy over epochs

### 5. Render per-model architecture diagrams (optional)

```bash
./plot_network_diagrams.sh                                   # all variants
./plot_network_diagrams.sh --variants baseline mininet       # subset
```

Produces one `<variant>_layered.png` per architecture under
`reports/network_diagrams/` — a 3D stacked-block view (Conv2D /
MaxPool / Flatten / Dense rendered as labeled volumes) using
[`visualtorch`](https://pypi.org/project/visualtorch/). The wrapper
auto-installs `visualtorch` into the project venv on first run; no
external binary (Graphviz, etc.) needed.

Useful when you want a quick visual reference of each network's depth /
width profile for the paper, slides, or sanity-checking before training.

---

## Single-experiment workflow

If you want to re-run one variant (e.g. after tweaking hyperparams) without
the full supervisor:

```bash
# fp32 from scratch
uv run python -m scripts.run_experiment improved \
    --optimizer adam --lr 0.001 --batch-size 100 --epochs 80 \
    --weight-decay 0.0 --augment \
    --tag improved_fp32

# QAT fine-tune from existing fp32 checkpoint
uv run python -m scripts.run_experiment improved \
    --optimizer adam --lr 0.0005 --batch-size 100 --epochs 40 \
    --weight-decay 0.0 --augment \
    --load-fp32 trained_models/improved_fp32.pt \
    --qat-start-epoch 5 \
    --tag improved_qat
```

Useful flags:
- `--load-fp32 PATH` — start from a pre-trained fp32 checkpoint (industry-standard QAT)
- `--qat-start-epoch N` — switch Conv/Linear → QAT-aware modules at epoch N
- `--quant-power-of-two` / `--no-quant-power-of-two` — toggle CMSIS-NN q7 convention
- `--scheduler cosine|step|none` — LR schedule

---

## Cross-platform comparison

The PC results serve two roles:

1. **fp32 reference (upper bound)** — what's achievable with ideal float-precision hardware.
2. **int8 PTQ simulation (MAX78000-realistic)** — what the MAX78000 *would* achieve if its quantization were as gentle as the PC simulator (per-tensor symmetric + power-of-two + calibrated activations). The IMX500 has its own PTQ flow that runs against the same `*.pt` files — see [`../imx500-implementation`](../imx500-implementation).

The MAX78000 deployment typically loses a few extra percentage points relative to PC PTQ due to the silicon's additional output-shift + bias quantization + clamping per layer. The IMX500 deployment, in contrast, stays within ≤0.3 pp of fp32 thanks to per-channel scales + activation calibration in the MCT toolchain. The gap between the three (PC-sim, MAX78000-device, IMX500-device) is part of the paper's findings.

| Column | Source |
|---|---|
| fp32 accuracy | `reports/<v>_fp32_metrics.json` → `fp32_test_acc` (PC) |
| int8 PTQ accuracy (PC sim, MAX78000-realistic) | `reports/<v>_fp32_metrics.json` → `int8_test_acc` or `reports/int8_eval/summary.txt` |
| int8 PTQ accuracy (MAX78000 device) | `../max78000-implementation/reports/_eval_pre_synth/fp32/_acc/int8_<v>` |
| int8 QAT accuracy (MAX78000 device) | `../max78000-implementation/reports/_eval_pre_synth/qat/_acc/int8_<v>` |
| int8 PTQ accuracy (IMX500) | `../imx500-implementation/outputs/reports/<v>_fp32_metrics.json` → `int8_test_acc` |
| int8 QAT accuracy (IMX500) | `../imx500-implementation/outputs/reports/<v>_qat_metrics.json` → `int8_test_acc` |

The MAX78000 side's `scripts/plot_acc_comparison.py` combines the MAX78000 columns into the 3-bar-per-variant figure that's the headline of that target's deployment-cost story; the IMX500 side's `build_report.py` does the equivalent aggregation for the IMX500 columns.

---

## Hyperparameter recipe summary

All non-mininet variants share the same fp32 + QAT recipe; mininet has its
own because it's deeper and narrower.

| Hyperparam | Most variants | `mininet` (fp32) | `mininet` (QAT) |
|---|---|---|---|
| Optimizer | Adam | Adam | Adam |
| LR | 0.001 (fp32) / 0.0005 (QAT) | 0.005 | **0.0001** (very low to avoid cosine-restart shock) |
| Batch size | 100 | 64 | 64 |
| Epochs | 80 (fp32) / 40 (QAT) | 80 | 40 |
| Weight decay | 0 | 1e-4 | **0** (kept off during QAT) |
| Scheduler | none | cosine | cosine |
| Input size | 32 | 32 | 32 |
| Augmentation | random crop + flip | + ColorJitter | + ColorJitter |
| QAT switch epoch | 5 (5 ep warmup) | n/a | 5 |

These exact recipes are mirrored in `../max78000-implementation/train_max78000_models.sh`, so PC and device runs differ only in the hardware target — not in training methodology.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: torch` (or similar) | venv not activated / not synced | `uv sync` then either `uv run python …` or `source .venv/bin/activate`. |
| Training crashes on MPS with `Expected all tensors to be on …` | Stale `device` argument passed somewhere | The pipeline uses `training.utils.pick_device()` which auto-selects MPS/CUDA/CPU. |
| `int8_acc` ≈ `fp32_acc` with no drop on `*_fp32` models | `eval_int8.py` was run before `quantize.py` got BN folding | Pull the latest `training/quantize.py`; verify `fold_all_bn` runs by adding a `print` inside `quantize_model_ptq`. |
| PC `int8_acc` doesn't match MAX78000 device `int8 sim` | Expected: PC is the upper bound, device has extra per-layer rescaling | Report both; the gap is the paper's finding. |
| `mininet` fp32 trained but `mininet_qat` keeps degrading | Cosine LR restart from 0.0025 was destroying the fp32 weights | Already fixed: `mininet` uses `LR=0.0001` and `weight_decay=0` for QAT. |

---

## What's next

After running the PC pipeline:

1. Open `reports/summary.md` and `reports/figures/pareto.png` to see the trade-offs.
2. Cross to `../max78000-implementation/` for the MAX78000 deployment (mirror training → synthesize → flash → measure), and/or `../imx500-implementation/` for the IMX500 deployment (MCT PTQ → ONNX → `imxconv-pt` → `imx500-package` → run on the Pi AI Camera).
3. Once the deployment results are in, use `../max78000-implementation/scripts/plot_acc_comparison.py` for the MAX78000 headline figure (**fp32 vs int8 PTQ device vs int8 QAT device** per variant) and `../imx500-implementation/build_report.py` for the IMX500 equivalent.
