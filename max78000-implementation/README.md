# MAX78000 — CIFAR-10 deployment

End-to-end pipeline for training, quantizing, synthesizing and measuring
CIFAR-10 classifiers on the Maxim (Analog Devices) **MAX78000** CNN
accelerator (FTHR_RevA board).

Sister project: [`../pc-implementation`](../pc-implementation) — pure-PyTorch
reference training pipeline that produces the fp32 checkpoints this directory
consumes, and also simulates int8 PTQ for an "ideal-hardware" upper bound on
deployment accuracy.

---

## Pipeline overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                                                                        │
│   train_max78000_models.sh   (uses ai8x-training)                      │
│   ────────────────────────                                             │
│   1. fp32 training (80 epochs)        →  trained_models/<v>_fp32.pth.tar│
│   2. QAT fine-tune  (40 epochs)       →  trained_models/<v>_qat_train.*│
│                                                                        │
│           │                                                            │
│           ▼                                                            │
│                                                                        │
│   synthesize_all.sh   (uses ai8x-synthesis)                            │
│   ────────────────                                                     │
│   3. BN-fold + quantize + ai8xize.py  →  $AI/.../synthed_net_<v>/<v>/  │
│                                          (C project ready to flash)    │
│                                                                        │
│           │                                                            │
│           ▼                                                            │
│                                                                        │
│   bash_device_scripts/device_*.sh + host_*.sh                          │
│   ──────────────────────────────                                       │
│   4. flash firmware → measure on device                                │
│      • per-layer profile (cycles, MAC_Ops/Cycle)                       │
│      • full test-set accuracy (UART streamed)                          │
│      • energy/inference (GPIO sync + external power monitor)           │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Models evaluated

Six architectures, each trained from scratch in fp32 (80 ep) and then
fine-tuned with QAT (40 ep) starting from the fp32 checkpoint. All deploy
end-to-end on the MAX78000 with no manual YAML editing required.

| Variant | Family | Paper / source | Params | MACs/inf | Notes |
|---|---|---|---|---|---|
| `baseline` | Dense conv, shallow | Lai et al. CMSIS-NN (arXiv 2018) | 38 K | 4.4 M | Canonical MCU CIFAR-10 reference, adapted to 3×3 kernels (MAX78000 doesn't support 5×5). |
| `improved` | Dense conv + BN (ablation) | this work | 38 K | 4.4 M | `baseline` + BatchNorm + dropout. Same MACs, BN folded at synthesis. |
| `deeper` | Dense conv, deeper (ablation) | this work | 141 K | 8.0 M | `improved` with 2 extra conv layers. |
| `mininet` | Dense conv, deep VGG-style | Banbury et al. MicroNets (MLSys 2021) | 316 K | 23 M | Narrow-and-deep 6-layer VGG-Micro. The only variant that uses cosine LR + weight-decay at fp32 (and a much lower LR + no-WD recipe at QAT — see `train_max78000_models.sh`). |
| `nascifarnet` | NAS-found, MCU-targeted | Maxim ai8x-training (2021) | 303 K | 36 M | NAS-discovered specifically under the MAX78000 op-set constraints. Native ai8x arch (`ai85nascifarnet`), no rewriting needed. |
| `ressimplenet` | Residual SimpleNet | HasanPour et al. (arXiv 2016) | 374 K | 18 M | 14-layer Residual SimpleNet adapted to MAX78000. Uses the BN-augmented variant (`ai85ressimplenetbn`) — BN is fused into Conv at synthesis, so the deployed weights are equivalent to a no-BN model but training is dramatically more stable. |

Excluded from this directory:
- `baseline_5x5` (5×5 kernels not supported by the MAX78000 CNN accelerator)
- `separable` (depthwise separable; supported but extremely inefficient — Maxim reserves their depthwise hardware support for the MAX78002)
- `resnet8` (MLPerf Tiny ResNet-8; the `eltwise: add` SRAM-layout requirements are non-trivial to satisfy — `ressimplenet` is our residual representative instead)

---

## Directory structure

### Top-level orchestration (run these)

| File | Purpose |
|---|---|
| `train_max78000_models.sh` | Train all (or one) variant. `[fp32\|qat]` selects mode. |
| `synthesize_all.sh` | BN-fold → int8 quantize → generate C deployment project. |
| `eval_pre_synth.sh` | Evaluate `fp32` / `fused` / `int8 sim` accuracy on the host, before flashing. |

### `networks/` — synthesis configs

One `network_<variant>.yaml` per active model. These are the processor-map and offset configs consumed by `ai8xize.py`.

### `scripts/` — Python helpers

| File | Purpose | Called by |
|---|---|---|
| `bn_fuser_v2.py` | Correct BN folding for arbitrarily-nested ai8x state dicts. | `synthesize_all.sh` |
| `verify_fold.py` | Sanity-check that BN-fold preserves argmax on the test set. | `eval_pre_synth.sh` |
| `estimate_metrics.py` | Per-variant static MAC/param/memory estimates → `reports/models_estimation.json` (consumed by `plot_acc_comparison.py`). | manual |
| `plot_models_report.py` | Pareto plot + training curves (symlink to PC's). | `bash_device_scripts/host_testset.sh` |
| `plot_acc_comparison.py` | Two figures: (1) grouped bar plot fp32 vs int8 PTQ vs int8 QAT per variant; (2) scatter MACs/inf vs int8 acc with PTQ + QAT overlaid. | manual |
| `models_ai8x.py` | Reference copy of `ai85net_cmsis_*` classes (canonical lives in `$AI/ai8x-training/models/project_models.py`). | (reference) |
| `parse_ai8x_log.py` | Convert an `ai8x-training/logs/<run>/` into a PC-style metrics JSON. | manual utility |
| `eval_ai8x.py` | Evaluate a QAT/int8 ai8x checkpoint, dump predictions as `.npz`. | manual utility |

### `bash_device_scripts/` — board interaction

| File | Purpose |
|---|---|
| `device_testset.sh` | Build + flash `inference_test_set.c` (UART-driven full test-set inference). |
| `host_testset.sh` | Stream 10 K CIFAR-10 images over UART, collect device predictions. |
| `device_profile.sh` | Build + flash `profile_layers.c` (single-inference profile). |
| `host_profile.sh` | Read UART profile block + emit ST.AI-style per-layer table. |
| `_common.sh` | Shared shell helpers (port detection, `MAXIM_PATH`, `build_and_flash` with DAPLINK fallback). |

### `host/` — Python companions of the device firmware

| File | Purpose |
|---|---|
| `host_test_set.py` | Streams the CIFAR-10 test set to `inference_test_set.c` over UART, computes device accuracy. |
| `host_profile.py` | Parses the single-inference profile block from `profile_layers.c`. |
| `gui_classify.py` | Tkinter live demo: pick a class → board classifies → see results. |

### `c_harness/` — drop-in `main.c` files

| File | Purpose |
|---|---|
| `inference_test_set.c` | UART-driven inference loop (companion of `host_test_set.py`). |
| `profile_layers.c` | Single inference, reports CNN/CPU cycles via UART. |
| `measure_inference.c` | Times N inferences, toggles GPIO for external energy measurement. |

### Generated / runtime

| Path | Purpose |
|---|---|
| `trained_models/` | `<v>_fp32.pth.tar` and `<v>_qat_train.pth.tar` produced by `train_max78000_models.sh`. |
| `estimate.json` | Static MAC / param / memory budget per variant. Regenerated by `scripts/estimate_metrics.py`. |
| `reports/` | Logs, predictions, plots. Most subdirs auto-generated. |

### Docs

| File | Purpose |
|---|---|
| `SETUP_AI8X_MAC.md` | Apple Silicon install guide (pyenv, both ai8x venvs, MSDK, GCC, OpenOCD). |
| `FLASH_AND_RUN.md` | Detailed step-by-step from synthesized project to on-device measurements. |
| `c_harness/README.md` | Conventions for the firmware drop-ins. |

---

## End-to-end: from clean checkout to numbers on the board

### Prerequisites

1. **Train fp32 on PC first** (see [`../pc-implementation/README.md`](../pc-implementation/README.md)) — produces `pc-implementation/trained_models/<v>_fp32.pt` checkpoints that the MAX78000 side mirrors with its own training run.
2. **ai8x stack installed** — follow [`SETUP_AI8X_MAC.md`](SETUP_AI8X_MAC.md) for Apple Silicon, or Maxim's official docs for Linux.
3. **`models_ai8x.py` registered** with ai8x-training — copy `scripts/models_ai8x.py` into `$AI/ai8x-training/models/project_models.py` (one-time setup, also in `SETUP_AI8X_MAC.md`).
4. **DAPLINK + USB serial** wired to the FTHR board.

### Step 1 — Train all variants

```bash
./train_max78000_models.sh all fp32     # 80 ep from-scratch, all variants
./train_max78000_models.sh all qat      # 40 ep QAT fine-tune from fp32 ckpt
```

`fp32` is independent for each variant; `qat` requires the corresponding `<v>_fp32.pth.tar` to exist. Both are idempotent — re-running skips a variant whose output checkpoint already exists. Subset training:

```bash
./train_max78000_models.sh mininet fp32       # one variant
./train_max78000_models.sh nascifarnet qat
```

### Step 2 — Synthesize C deployment projects

```bash
./synthesize_all.sh                     # PTQ flow from <v>_fp32.pth.tar
./synthesize_all.sh --from-qat          # QAT flow from <v>_qat_train.pth.tar
./synthesize_all.sh mininet --from-qat  # single variant, QAT
```

Per variant, this performs: **BN-fold → int8 quantization → `ai8xize.py` synthesis → C project at `$AI/ai8x-synthesis/synthed_net_<v>/<v>/`**. The script is idempotent and skips up-to-date intermediates.

### Step 3 — Host-side accuracy check (no board needed)

```bash
./eval_pre_synth.sh                     # fp32 + fused + int8 sim for all variants
./eval_pre_synth.sh qat                 # same for QAT models
./eval_pre_synth.sh fp32 mininet        # subset
```

Produces `reports/_eval_pre_synth/<mode>/`:
- `verify_fold.log` — fp32 vs BN-folded argmax agreement (should be ≈100%)
- `int8_<v>.log` — int8 simulation acc per variant
- `_acc/` — per-variant accuracy text files (consumed by `plot_acc_comparison.py`)
- summary printed to stdout

Reading this output answers: *did BN folding preserve accuracy?* and *how much does PTQ vs QAT cost in int8 deployment?* — without touching the board.

### Step 4 — Flash and measure on the board

```bash
# Per-layer profile (cycles, latency)
./bash_device_scripts/device_profile.sh baseline           # build + flash
./bash_device_scripts/host_profile.sh   baseline           # read UART → ST.AI table

# Full test-set accuracy (10 000 images over UART)
./bash_device_scripts/device_testset.sh baseline           # build + flash
./bash_device_scripts/host_testset.sh   baseline           # stream + collect predictions
```

The `device_*.sh` scripts swap in the appropriate `c_harness/*.c`, build with `make`, and flash via DAPLINK mass-storage (OpenOCD path also supported but unreliable on macOS). The `host_*.sh` scripts then read the UART output. See [`FLASH_AND_RUN.md`](FLASH_AND_RUN.md) for the full step-by-step, including the energy-measurement workflow.

### Step 5 — Generate the comparison plots for the paper

```bash
# (deactivate conda first if you're in (base))
# one-time / whenever architectures change: regenerate the MAC catalogue
python3 scripts/estimate_metrics.py             # → reports/models_estimation.json

# the two figures
/usr/bin/python3 scripts/plot_acc_comparison.py
# → reports/fig_acc_comparison.png   (3 bars per variant: fp32 / PTQ / QAT)
# → reports/fig_acc_vs_macs.png      (scatter: MACs/inf vs int8 acc, PTQ+QAT overlaid)
```

The two figures together are the headline of the deployment-cost story:
- **`fig_acc_comparison.png`** shows the magnitude of PTQ drop and the QAT recovery, per variant.
- **`fig_acc_vs_macs.png`** places each variant on the Pareto plane (compute vs accuracy) and visualises the QAT lift with a dashed connector between PTQ and QAT points.

### Step 6 — Live demo (optional)

```bash
/usr/bin/python3 host/gui_classify.py
```

Tkinter GUI: click a CIFAR-10 class → script picks a random test image → sends it to the board over UART → board returns predictions + cycles → GUI displays the image, the prediction probabilities, and the round-trip latency. Requires `inference_test_set.c` to be flashed (i.e. Step 4 already done).

---

## Reading the results

After the full pipeline runs, the key outputs live at:

| Output | What's in it |
|---|---|
| `reports/_eval_pre_synth/fp32/_acc/` | Per-variant fp32 + int8 sim accuracy from `eval_pre_synth.sh` (PTQ path) |
| `reports/_eval_pre_synth/qat/_acc/` | Same but QAT path |
| `reports/fig_acc_comparison.png` | The 3-bar comparison plot |
| `reports/predictions/<v>_device.npz` | Per-image device predictions from `host_testset.sh` (compatible with PC report tooling) |
| `reports/profile_<v>.txt` | Per-layer ST.AI-style profile from `host_profile.sh` |
| `estimate.json` | Static budget (MACs, params, memory) per variant |

For the paper, the canonical claim is built from these three numbers per variant:

- **fp32 reference accuracy** (from PC; serves as upper bound) — `pc-implementation/reports/<v>_fp32_metrics.json`
- **int8 PTQ device accuracy** (naive deployment cost) — `reports/_eval_pre_synth/fp32/_acc/int8_<v>`
- **int8 QAT device accuracy** (QAT recovers PTQ drop) — `reports/_eval_pre_synth/qat/_acc/int8_<v>`

These three columns demonstrate the value proposition of QAT for fixed-point deployment.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `arm-none-eabi-gcc: command not found` | Toolchain not on PATH | See `bash_device_scripts/_common.sh` for auto-discovery; or `export PATH=/usr/local/arm-gnu-toolchain-12.3.rel1/bin:$PATH`. |
| `Network architecture of configuration file ... does not match checkpoint file` | YAML arch name ≠ checkpoint arch name | `nascifarnet` → `ai85nascifarnet`, `ressimplenet` → `ai85ressimplenetbn`. Check `synthesize_all.sh::synth_one`'s `case` statement. |
| Device accuracy ≈ 10 % (random) | BN fold bug or missing `memcpy32` to CNN SRAM | Ensure `scripts/bn_fuser_v2.py` is used (not upstream `batchnormfuser.py`). Verify `c_harness/inference_test_set.c` calls `memcpy32((uint32_t*)0x50400000, input_0, 1024)`. |
| OpenOCD flash fails with SIGSEGV on macOS | Known bug in the bundled OpenOCD 2021 binary | The `build_and_flash` helper auto-falls back to DAPLINK mass-storage copy via Finder. |
| `mininet_qat` accuracy lower than `mininet_fp32` | Cosine-LR restart at half-of-fp32 LR shocks the fine-tune | Already fixed: mininet QAT uses `LR=0.0001`, `weight-decay=0`, see comment in `train_max78000_models.sh::train_qat()`. |
| `int8 sim` drops 15-25 pp vs `fused` (PTQ path) | This is **expected** — BN folding + per-tensor symmetric quantization redistributes per-channel weight magnitudes, hurting per-tensor scale | QAT recovers it — see `int8 sim` in the `qat` mode of `eval_pre_synth.sh`. |

---

## How this compares to the PC reference

| Quantity | PC (`../pc-implementation`) | MAX78000 (this dir) |
|---|---|---|
| fp32 training | 80 ep, Adam | 80 ep, Adam (same hyperparams per variant) |
| QAT fine-tune | 40 ep from `<v>_fp32.pt`, switch at ep 5 | 40 ep from `<v>_fp32.pth.tar`, switch at ep 5 (matching `qat_policy_cifar10.yaml`) |
| int8 PTQ | `quantize_model_ptq` (BN fold + sym per-tensor + power-of-two scales) | `bn_fuser_v2.py` + `quantize.py` (ai8x-synthesis) — same algorithmic intent, with additional `output_shift` per layer at deployment |
| Where deployed | nowhere (host-side simulation only) | the actual FTHR board |
| Architectural fidelity | identical to MAX78000 deployment, BN included for training stability and folded at deployment | same architecture; ai8x layer wrappers used to satisfy the synthesis toolchain |

The PC int8 numbers are an "ideal-hardware" upper bound (no `output_shift`, no learned activation thresholds). The MAX78000 device numbers are what the silicon actually achieves. Both are valid; the gap is part of the paper's findings.
