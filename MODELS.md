# Model architectures — CIFAR-10 on MCU

This document is the **architecture specification** for every model trained
in this project. It uses the same layer-by-layer table format as Lai et al.
2018 (CMSIS-NN), Table 1, so comparison with the paper baseline is direct.

For each model we report:

| Column | Meaning |
|---|---|
| **Layer Type** | Conv / Pool / FC / Add |
| **Filter Shape** | `Kh × Kw × Cin × Cout` (and weight memory in KB) |
| **Output Shape** | `H × W × C` (and activation memory in KB) |
| **Ops** | Multiply-accumulate × 2 (paper convention; `Ops = 2 × MACs`) |
| **Notes** | Padding, stride, BN-fold, etc. |

Weight memory assumes int8 deployment. Per-layer MAC counts include the
bias add (`Cout` extra adds per spatial position).

---

## Variant summary

| # | Variant | Backbone | MAX78000 | IMX500 | Family / source |
|---|---|---|---|---|---|
| 1 | `baseline_5x5`   | 3-conv CNN, **5×5 kernels** | ✗ (no 5×5 on MAX78000) | ✓ | Lai et al. CMSIS-NN, paper-faithful (arXiv 2018) |
| 2 | `baseline`       | 3-conv CNN, 3×3 kernels | ✓ | ✓ | Lai et al. CMSIS-NN, 3×3 adaptation |
| 3 | `improved`       | baseline + **BN + Dropout** | ✓ | ✓ | this work (recipe ablation on #2) |
| 4 | `deeper`         | improved + 2 extra stages | ✓ | ✓ | this work (depth ablation on #3) |
| 5 | `mininet`        | 7-conv VGG-Micro | ✓ | ✓ | Banbury et al. MicroNets (MLSys 2021) |
| 6 | `nascifarnet`    | NAS-found 10-conv | ✓ | ✓ | Maxim ai8x-training (2021) |
| 7 | `ressimplenet`   | 14-conv with 3 residuals | ✓ | ✓ | HasanPour et al. SimpleNet (arXiv 2016) |

**MAX78000 active variants (6):** `baseline`, `improved`, `deeper`, `mininet`, `nascifarnet`, `ressimplenet`.
**IMX500 active variants (7):** all of the above, plus `baseline_5x5` (the IMX500 has no kernel-size restriction).

The four CMSIS-NN-style models (#1-#4) all use the same Conv-BN-ReLU-Pool
template at different widths and depths. `mininet` (#5) is narrower and
deeper (VGG-Micro family). `nascifarnet` (#6) is NAS-found specifically
under the MAX78000 op-set. `ressimplenet` (#7) adds residual connections
in the SimpleNet style.

---

## Compression techniques applied to each variant

Each variant produces two deployable models. Tags:

| Technique | Tag suffix | Trained from | Script |
|---|---|---|---|
| **fp32 + PTQ** | `_fp32` (+ `_fp32_ptq` snapshot on PC) | from scratch, 80 epochs | `run_experiment.py` (PC) / `train_max78000_models.sh ... fp32` (MAX78000) |
| **QAT** (Quantization-Aware Training) | `_qat` (PC) / `_qat_train` (MAX78000) | load `_fp32` checkpoint, 40 ep fine-tune, switch at epoch 5 | `run_experiment.py --load-fp32 ... --qat-start-epoch 5` (PC) / `train_max78000_models.sh ... qat` (MAX78000) |

Each `*.pt` checkpoint produced by the PC pipeline is then quantized by
**two completely different toolchains**, one per deployment target:

| Target | PTQ flow | Granularity |
|---|---|---|
| MAX78000 | `bn_fuser_v2.py` + `quantize.py` (ai8x-synthesis) → C project | per-tensor symmetric, power-of-two scales, per-layer `output_shift` + clamping |
| IMX500 | `post_training_compress.py` (MCT, IMX500 TPC v1) → ONNX → `imxconv-pt` → `.rpk` | per-channel scales with calibrated activation ranges |

The PC-side simulator (`scripts/eval_int8.py`) uses **CMSIS-NN q7 semantics**
(per-tensor symmetric, power-of-two) and is therefore the ideal-hardware
upper bound for the **MAX78000 path**. Typical deltas:

- PC PTQ sim ↔ MAX78000 device: 5–20 pp drop (per-layer `output_shift` + clamping). QAT recovers most of it.
- IMX500 device vs PC fp32: **≤0.3 pp drop across all 7 variants** — the
  per-channel + activation-calibrated PTQ is gentle enough that QAT brings
  only marginal additional gains.

The size of this gap as a function of the silicon's quantization
granularity is one of the headline findings of the project.

---

## 1. `baseline_5x5` — Paper-faithful reference (PC + IMX500)

The exact architecture from **Lai et al. 2018, Table 1**. Excluded from
the MAX78000 deployment because the accelerator only supports 1×1 and
3×3 standard convolutions, but deploys natively on the IMX500 (no
kernel-size restriction).

### Layer table

|         | Layer Type      | Filter Shape          | Output Shape          | Ops      | Notes |
|---------|-----------------|------------------------|------------------------|----------|---|
| Layer 1 | Convolution     | 5×5×3×32 (2.3 KB)      | 32×32×32 (32 KB)       | 4.9 M    | pad=2, ReLU |
| Layer 2 | Max Pooling     | N.A.                   | 16×16×32 (8 KB)        | 73.7 K   | 2×2, stride 2 |
| Layer 3 | Convolution     | 5×5×32×32 (25 KB)      | 16×16×32 (8 KB)        | 13.1 M   | pad=2, ReLU |
| Layer 4 | Max Pooling     | N.A.                   | 8×8×32 (2 KB)          | 18.4 K   | 2×2, stride 2 |
| Layer 5 | Convolution     | 5×5×32×64 (50 KB)      | 8×8×64 (4 KB)          | 6.6 M    | pad=2, ReLU |
| Layer 6 | Max Pooling     | N.A.                   | 4×4×64 (1 KB)          | 9.2 K    | 2×2, stride 2 |
| Layer 7 | Fully-connected | 4×4×64×10 (10 KB)      | 10                     | 20 K     | softmax |
| **Total** |               | **87.5 KB weights**    | **55 KB activations**  | **24.7 M** | reproduces 79.9% paper int8 |

PyTorch class: `CMSISNNBaseline5x5` (`pc-implementation/training/models.py`).

---

## 2. `baseline` — MAX78000-portable adaptation

Same topology as #1 with every 5×5 → 3×3. Receptive field shrinks but
the spatial flow (32→16→8→4) is identical.

### Layer table

|         | Layer Type      | Filter Shape          | Output Shape         | Ops      | Notes |
|---------|-----------------|------------------------|----------------------|----------|---|
| Layer 1 | Convolution     | 3×3×3×32 (0.85 KB)     | 32×32×32 (32 KB)     | 1.77 M   | pad=1, ReLU |
| Layer 2 | Max Pooling     | N.A.                   | 16×16×32 (8 KB)      | 73.7 K   | 2×2 / 2 |
| Layer 3 | Convolution     | 3×3×32×32 (9 KB)       | 16×16×32 (8 KB)      | 4.72 M   | pad=1, ReLU |
| Layer 4 | Max Pooling     | N.A.                   | 8×8×32 (2 KB)        | 18.4 K   | 2×2 / 2 |
| Layer 5 | Convolution     | 3×3×32×64 (18 KB)      | 8×8×64 (4 KB)        | 2.36 M   | pad=1, ReLU |
| Layer 6 | Max Pooling     | N.A.                   | 4×4×64 (1 KB)        | 9.2 K    | 2×2 / 2 |
| Layer 7 | Fully-connected | 4×4×64×10 (10 KB)      | 10                   | 20 K     | softmax |
| **Total** |               | **38.0 KB weights**    | **55 KB activations**| **8.87 M**| ~36% of #1 ops, ~43% of #1 weights |

PyTorch class: `CMSISNNBaseline` (`pc-implementation/training/models.py`).
ai8x class: `ai85net_cmsis_baseline` (`ai8x-training/models/project_models.py`).

---

## 3. `improved` — Recipe ablation on `baseline`

Same architecture as `baseline`. **BatchNorm + Dropout added**; BN folds
into the preceding conv at deployment so the int8 weight memory is
unchanged. Training adds augmentation.

### Layer table

|         | Layer Type        | Filter Shape          | Output Shape         | Ops     | Notes |
|---------|-------------------|------------------------|----------------------|---------|---|
| Layer 1 | Conv + BN + ReLU  | 3×3×3×32 (0.85 KB)     | 32×32×32 (32 KB)     | 1.77 M  | BN folded at deploy |
| Layer 2 | Max Pooling       | N.A.                   | 16×16×32 (8 KB)      | 73.7 K  | |
| Layer 3 | Conv + BN + ReLU  | 3×3×32×32 (9 KB)       | 16×16×32 (8 KB)      | 4.72 M  | |
| Layer 4 | Max Pooling       | N.A.                   | 8×8×32 (2 KB)        | 18.4 K  | |
| Layer 5 | Conv + BN + ReLU  | 3×3×32×64 (18 KB)      | 8×8×64 (4 KB)        | 2.36 M  | |
| Layer 6 | Max Pooling       | N.A.                   | 4×4×64 (1 KB)        | 9.2 K   | |
| Layer 7 | Dropout(0.1) + FC | 4×4×64×10 (10 KB)      | 10                   | 20 K    | Dropout train-time only |
| **Total** |                 | **38.1 KB weights**    | **55 KB activations**| **8.87 M** | |

PyTorch class: `CMSISNNImproved`.
ai8x class: `ai85net_cmsis_improved`.

---

## 4. `deeper` — Depth ablation on `improved`

`improved` + 2 extra conv stages. Tests if pure depth (without
depthwise) closes the gap to `mininet`.

### Layer table

|         | Layer Type        | Filter Shape          | Output Shape          | Ops      |
|---------|-------------------|------------------------|------------------------|----------|
| Layer 1 | Conv + BN + ReLU  | 3×3×3×32 (0.85 KB)     | 32×32×32 (32 KB)       | 1.77 M   |
| Layer 2 | Max Pool          | N.A.                   | 16×16×32 (8 KB)        | 73.7 K   |
| Layer 3 | Conv + BN + ReLU  | 3×3×32×32 (9 KB)       | 16×16×32 (8 KB)        | 4.72 M   |
| Layer 4 | Max Pool          | N.A.                   | 8×8×32 (2 KB)          | 18.4 K   |
| Layer 5 | Conv + BN + ReLU  | 3×3×32×64 (18 KB)      | 8×8×64 (4 KB)          | 2.36 M   |
| Layer 6 | Conv + BN + ReLU  | 3×3×64×64 (36 KB)      | 8×8×64 (4 KB)          | 4.72 M   |
| Layer 7 | Max Pool          | N.A.                   | 4×4×64 (1 KB)          | 9.2 K    |
| Layer 8 | Conv + BN + ReLU  | 3×3×64×128 (72 KB)     | 4×4×128 (2 KB)         | 2.36 M   |
| GAP     | Avg Pool 4        | N.A.                   | 1×1×128 (0.13 KB)      | 2.05 K   |
| FC      | Linear            | 128×10 (1.3 KB)        | 10                     | 2.6 K    |
| **Total**|                  | **137.7 KB weights**   | **~63 KB activations** | **16.0 M**| |

PyTorch class: `CMSISNNDeeper`.
ai8x class: `ai85net_cmsis_deeper`.

---

## 5. `mininet` — VGG-Micro (MicroNets-inspired)

Purely sequential VGG-Micro with standard 3×3 convs throughout.
Stride-2 convs for downsampling instead of pooling. Tight against
MAX78000's 442 KiB weight memory budget.

### Layer table (32×32 input)

|          | Layer Type        | Filter Shape          | Output Shape       | Ops     | Notes |
|----------|-------------------|------------------------|--------------------|---------|---|
| Block 1a | Conv + BN + ReLU6 | 3×3×3×32 (0.85 KB)     | 32×32×32 (32 KB)   | 1.77 M  | pad=1 |
| Block 1b | Conv + BN + ReLU6 | 3×3×32×48 (13.5 KB)    | 16×16×48 (12 KB)   | 7.08 M  | stride 2 |
| Block 2a | Conv + BN + ReLU6 | 3×3×48×64 (27.0 KB)    | 16×16×64 (16 KB)   | 14.16 M | pad=1 |
| Block 2b | Conv + BN + ReLU6 | 3×3×64×80 (45.0 KB)    | 8×8×80 (5 KB)      | 5.90 M  | stride 2 |
| Block 3a | Conv + BN + ReLU6 | 3×3×80×96 (67.5 KB)    | 8×8×96 (6 KB)      | 8.85 M  | pad=1 |
| Block 3b | Conv + BN + ReLU6 | 3×3×96×128 (108.0 KB)  | 4×4×128 (2 KB)     | 3.54 M  | stride 2 |
| Block 4  | Conv + BN + ReLU6 | 3×3×128×128 (144.0 KB) | 4×4×128 (2 KB)     | 4.72 M  | pad=1 |
| GAP      | Avg Pool 4        | N.A.                   | 1×1×128 (0.13 KB)  | 2.05 K  | |
| Dropout  | Dropout(0.4)      | N.A.                   | 128                | —       | train-only |
| FC       | Linear            | 128×10 (1.3 KB)        | 10                 | 2.6 K   | |
| **Total**|                   | **408 KB weights**     | **~75 KB activations** | **46.0 M** | 92% of MAX78000's 442 KiB budget |

PyTorch class: `MiniMobileNet`.
ai8x class: `ai85net_cmsis_mininet`.

**Training note (mininet only):** uses **cosine LR + weight-decay** for
fp32. For QAT, drops LR to **0.0001** and disables weight-decay to avoid
the cosine-restart shock (the cosine schedule restarts from half-of-fp32 LR
which would otherwise destroy the trained weights at epoch 0). See
`pc-implementation/train_pc_models.sh` and
`max78000-implementation/train_max78000_models.sh` for the exact recipe.

---

## 6. `nascifarnet` — NAS-found, MCU-targeted

Architecture discovered via Neural Architecture Search by Maxim,
specifically under the MAX78000 op-set constraints. Alternates 3×3 and
1×1 dense convolutions across 5 stages, terminating in a 512-input fully
connected classifier. As a NAS-optimized network, it tends to occupy a
favourable point on the Pareto frontier for fixed-op-set deployment.

### Layer table (32×32 input)

|         | Layer Type         | Filter Shape           | Output Shape       | Ops      | Notes |
|---------|--------------------|-------------------------|--------------------|----------|---|
| conv1_1 | Conv + BN(NoAffine)+ReLU | 3×3×3×64 (1.69 KB)  | 32×32×64 (64 KB)   | 3.54 M   | pad=1 |
| conv1_2 | Conv + BN + ReLU   | 1×1×64×32 (2.0 KB)      | 32×32×32 (32 KB)   | 4.19 M   | pad=0 |
| conv1_3 | Conv + BN + ReLU   | 3×3×32×64 (18.0 KB)     | 32×32×64 (64 KB)   | 37.75 M  | pad=1 |
| conv2_1 | MaxPool + Conv + BN + ReLU | 3×3×64×32 (18.0 KB) | 16×16×32 (8 KB) | 4.72 M  | pool 2/2, pad=1 |
| conv2_2 | Conv + BN + ReLU   | 1×1×32×64 (2.0 KB)      | 16×16×64 (16 KB)   | 1.05 M   | pad=0 |
| conv3_1 | MaxPool + Conv + BN + ReLU | 3×3×64×128 (72 KB) | 8×8×128 (8 KB)    | 4.72 M  | pool 2/2, pad=1 |
| conv3_2 | Conv + BN + ReLU   | 1×1×128×128 (16 KB)     | 8×8×128 (8 KB)     | 1.05 M   | pad=0 |
| conv4_1 | MaxPool + Conv + BN + ReLU | 3×3×128×64 (72 KB) | 4×4×64 (1 KB)    | 1.18 M  | pool 2/2, pad=1 |
| conv4_2 | Conv + BN + ReLU   | 3×3×64×128 (72 KB)      | 4×4×128 (2 KB)     | 1.18 M   | pad=1 |
| conv5_1 | MaxPool + Conv + BN + ReLU | 1×1×128×128 (16 KB) | 2×2×128 (0.5 KB)| 65.5 K  | pool 2/2, pad=0 |
| FC      | Linear             | 512×10 (5 KB)           | 10                 | 5.12 K   | |
| **Total**|                   | **~300 KB weights**     | **~204 KB activations** | **~60 M** | NAS-optimized for MAX78000 |

PyTorch class: `NASCifarNet` (PC twin).
ai8x class: `ai85nascifarnet` (Maxim's original at `ai8x-training/models/ai85net-nas-cifar.py`).

BN uses `affine=False` ("NoAffine") to match Maxim's reference. Folded
into Conv at synthesis just like any other BN.

---

## 7. `ressimplenet` — Residual SimpleNet

14-layer Residual SimpleNet adapted to MAX78000. Uses the **BN-augmented
variant** (`ai85ressimplenetbn`): every conv is `FusedConv2dBNReLU` for
training stability across the 14 layers. BN is folded into Conv at
synthesis so the deployed weights match a no-BN model.

### Layer table

|          | Layer Type            | Filter Shape          | Output Shape       | Ops      | Notes |
|----------|-----------------------|------------------------|--------------------|----------|---|
| conv1    | Conv + BN + ReLU      | 3×3×3×16 (0.42 KB)     | 32×32×16 (16 KB)   | 0.89 M   | pad=1 |
| conv2    | Conv + BN + ReLU      | 3×3×16×20 (2.81 KB)    | 32×32×20 (20 KB)   | 5.90 M   | x_res branch |
| conv3    | Conv + BN + ReLU      | 3×3×20×20 (3.52 KB)    | 32×32×20 (20 KB)   | 7.37 M   | |
| **resid1** | **eltwise add** (conv3 + conv2) | — | 32×32×20 (20 KB) | — | |
| conv4    | Conv + BN + ReLU      | 3×3×20×20 (3.52 KB)    | 32×32×20 (20 KB)   | 7.37 M   | |
| conv5    | MaxPool + Conv + BN + ReLU | 3×3×20×20 (3.52 KB) | 16×16×20 (5 KB)  | 1.84 M  | x_res branch, pool 2/2 |
| conv6    | Conv + BN + ReLU      | 3×3×20×20 (3.52 KB)    | 16×16×20 (5 KB)    | 1.84 M   | |
| **resid2** | **eltwise add** (conv6 + conv5) | — | 16×16×20 (5 KB) | — | |
| conv7    | Conv + BN + ReLU      | 3×3×20×44 (7.73 KB)    | 16×16×44 (11 KB)   | 4.05 M   | |
| conv8    | MaxPool + Conv + BN + ReLU | 3×3×44×48 (18.56 KB) | 8×8×48 (3 KB)   | 1.22 M  | x_res branch, pool 2/2 |
| conv9    | Conv + BN + ReLU      | 3×3×48×48 (20.25 KB)   | 8×8×48 (3 KB)      | 1.33 M   | |
| **resid3** | **eltwise add** (conv9 + conv8) | — | 8×8×48 (3 KB)  | — | |
| conv10   | MaxPool + Conv + BN + ReLU | 3×3×48×96 (40.5 KB) | 4×4×96 (1.5 KB)  | 0.66 M  | pool 2/2 |
| conv11   | MaxPool + Conv + BN + ReLU | 1×1×96×512 (48.0 KB) | 2×2×512 (4 KB)  | 0.20 M  | pool 2/2 |
| conv12   | Conv + BN + ReLU      | 1×1×512×128 (64.0 KB)  | 2×2×128 (1 KB)     | 0.26 M   | pad=0 |
| conv13   | MaxPool + Conv + BN + ReLU | 3×3×128×128 (144 KB) | 1×1×128 (0.13 KB) | 0.15 M | pool 2/2 |
| conv14   | Conv (wide, no act)   | 1×1×128×10 (1.25 KB)   | 1×1×10             | 1.28 K   | int32 output |
| **Total**|                      | **~370 KB weights**    | **~110 KB activations** | **~33 M** | 3 residual sums |

PyTorch class: `ResSimpleNet`.
ai8x class: `ai85ressimplenetbn` (`ai8x-training/models/ai85ressimplenetbn.py`).
Reference: HasanPour et al., *Lets keep it simple, using simple
architectures to outperform deeper and more complex architectures*,
arXiv:1608.06037 (2016).

---

## 8. Side-by-side comparison

CIFAR-10 at 32×32 input. fp32 is the trained float32 accuracy.

**PC + MAX78000 column block.** int8 PTQ is the PC-side simulation
(`scripts/eval_int8.py`, BN-folded, MAX78000-realistic per-tensor
symmetric quantization). int8 QAT is from the QAT-fine-tuned checkpoint
under the same simulator. These numbers stand in for the MAX78000
deployment as an "ideal-hardware" upper bound (real device numbers in
`max78000-implementation/reports/`).

| Variant          | Params  | Wt KiB | MACs (M) | fp32 acc | int8 PTQ (MAX78000 sim) | int8 QAT (MAX78000 sim) | Notes |
|------------------|--------:|-------:|---------:|---------:|------------------------:|------------------------:|---|
| `baseline_5x5`   | 89,578  | 87.5   | 12.30    | 79.79    | 79.76                   | 79.04                   | not deployed on MAX78000 — IMX500 only |
| `baseline`       | 38,890  | 38.0   | 4.43     | 80.98    | 75.50                   | 80.95                   | |
| `improved`       | 39,018  | 38.1   | 4.43     | 81.70    | 62.38                   | 82.86                   | QAT recovers all of the PTQ drop |
| `deeper`         | 141,034 | 137.7  | 7.96     | 83.17    | 65.71                   | 84.74                   | |
| `mininet`        | 316,000 | 408.2  | 23.00    | 88.22    | 69.21                   | 84.61                   | best fp32 in the zoo |
| `nascifarnet`    | 303,000 | ~300   | 36.00    | 87.02    | 80.50*                  | 89.25                   | NAS-optimized |
| `ressimplenet`   | 374,000 | ~370   | 18.00    | 86.39    | 78.00*                  | 83.07                   | residual |

\* `int8 PTQ` for `nascifarnet` and `ressimplenet` may show "n/a" if
`eval_pre_synth.sh` ran before the synthesis-arch-mapping fix; re-run
`./synthesize_all.sh <v>` + `./eval_pre_synth.sh fp32 <v>` to fill in.

**IMX500 column block.** Real device numbers from
`imx500-implementation/reports/summary.md`. PTQ is MCT
`pytorch_post_training_quantization` against the IMX500 TPC v1; QAT
column is the same MCT PTQ but applied to the QAT-trained PC checkpoint.

| Variant          | fp32 acc | int8 PTQ (IMX500) | int8 QAT (IMX500) | HW inf. (ms) | Δ (fp32 → QAT) |
|------------------|---------:|------------------:|------------------:|-------------:|---------------:|
| `baseline_5x5`   | 79.79    | 79.83             | 81.04             | 1.52         | +1.25 pp |
| `baseline`       | 80.98    | 80.73             | 80.94             | 1.53         | -0.04 pp |
| `improved`       | 81.40    | 81.48             | 81.75             | 1.53         | +0.35 pp |
| `deeper`         | 85.14    | 85.00             | 85.62             | 1.53         | +0.48 pp |
| `mininet`        | 88.63    | 88.48             | 88.52             | 1.53         | -0.11 pp |
| `nascifarnet`    | 87.61    | 87.64             | 88.95             | 1.53         | +1.34 pp |
| `ressimplenet`   | 86.90    | 86.97             | 88.32             | 1.53         | +1.42 pp |

### Headline reading

- **MAX78000 path:** PTQ hurts substantially on BN-bearing variants (15-20 pp drop on improved/deeper/mininet) due to BN folding redistributing per-channel weight magnitudes — the cost of per-tensor symmetric quantization at fixed-op-set hardware. **QAT recovers almost all of the PTQ drop**; for improved and deeper, int8 QAT even slightly *exceeds* fp32 (the fake-quant noise acts as regularization). **`baseline` is barely affected** because it has no BN to fold — the "best case" for naive per-tensor PTQ, and exactly why the other variants need QAT.
- **IMX500 path:** PTQ alone hits ≤0.3 pp drop across the board — the per-channel scales + activation calibration in the MCT IMX500 TPC v1 absorb the BN-fold disruption that destroys the per-tensor MAX78000 path. QAT brings only marginal additional gains (typically +0-1.5 pp).
- **HW inference time is essentially constant (~1.53 ms) across all 7 IMX500 models** — Sony dimensions the on-sensor accelerator to absorb a CIFAR-class network at the camera's native frame rate, so the bottleneck is the I/O loop (~90 ms frame-to-frame), not the inference itself.

### Reference: SOTA from the literature

| Source                            | Architecture        | Params  | Ops      | Acc    |
|-----------------------------------|---------------------|---------|----------|--------|
| Lai et al. 2018 (CMSIS-NN)        | 5×5 baseline (#1)   | 89,578  | 24.7 M   | 79.9%  |
| Banbury et al. 2021 (MicroNets)   | MicroNet-CIFAR      | ~70 K   | ~15 M    | ~88%   |
| Banbury et al. 2021 (MLPerf Tiny) | ResNet-8            | ~78 K   | ~25 M    | ~85%   |
| HasanPour et al. 2016 (SimpleNet) | SimpleNet           | 5.4 M   | —        | 94.5%  |
| Maxim 2021 (ai8x reference)       | ai85nascifarnet     | ~300 K  | ~36 M    | (this work, ~89%) |

---

## 9. Implementation status

| Variant         | PC class                  | MAX78000 arch              | YAML                          | MAX78000 synthesis | IMX500 deploy |
|-----------------|---------------------------|-----------------------------|--------------------------------|---------------------|----------------|
| `baseline_5x5`  | `CMSISNNBaseline5x5`      | n/a (5×5 unsupported)      | —                              | ✗                   | ✓ (3 flavours) |
| `baseline`      | `CMSISNNBaseline`         | `ai85net_cmsis_baseline`   | `networks/network_baseline.yaml` | ✓                   | ✓ (3 flavours) |
| `improved`      | `CMSISNNImproved`         | `ai85net_cmsis_improved`   | `networks/network_improved.yaml` | ✓                   | ✓ (3 flavours) |
| `deeper`        | `CMSISNNDeeper`           | `ai85net_cmsis_deeper`     | `networks/network_deeper.yaml` | ✓                   | ✓ (3 flavours) |
| `mininet`       | `MiniMobileNet`           | `ai85net_cmsis_mininet`    | `networks/network_mininet.yaml` | ✓                   | ✓ (3 flavours) |
| `nascifarnet`   | `NASCifarNet`             | `ai85nascifarnet`          | `networks/network_nascifarnet.yaml` | ✓                   | ✓ (3 flavours) |
| `ressimplenet`  | `ResSimpleNet`            | `ai85ressimplenetbn`       | `networks/network_ressimplenet.yaml` | ✓                   | ✓ (3 flavours) |

"3 flavours" on the IMX500 side = `_fp32`, `_fp32_ptq`, `_qat` PC
checkpoints, each separately quantized by MCT and packaged into a
`network.rpk`. See [`imx500-implementation/README.md`](imx500-implementation/README.md).

### MAX78000 notes per variant

- **`mininet`**: 408 KiB int8 weights = 92% of MAX78000's 442 KiB budget.
  Tight but verified to synthesize.
- **`nascifarnet`**: BN uses `affine=False` (Maxim's "NoAffine") — folded
  identically to affine BN at synthesis.
- **`ressimplenet`**: deploys the BN-augmented variant
  `ai85ressimplenetbn` (BN is folded at synthesis, so the device weights
  are equivalent to a no-BN model with proper initialization — BN exists
  only as a training-time scaffold to keep the 14-layer stack
  well-conditioned). The deployed YAML uses `eltwise: add` for the 3
  residual connections; the canonical Maxim reference network handled all
  the SRAM-layout & processor-mapping details for free.

---

## 10. Per-model report artifacts

Each variant × technique combination produces a standard file set:

```
reports/
  <V>_<technique>_metrics.json          # accuracy, params, MACs, training history
  predictions/
    <V>_<technique>.npz                 # y_true, fp32_y_pred/probs, int8_y_pred/probs
  figures/<V>_<technique>/
    fp32_confusion.png    int8_confusion.png
    fp32_roc.png          int8_roc.png
    fp32_pr.png           int8_pr.png
    fp32_confidence.png   int8_confidence.png
    loss_curve.png        acc_curve.png
```

PC + MAX78000 use the **same plotting code** (`plot_models_report.py`
in `pc-implementation/scripts/`, symlinked from `max78000-implementation/scripts/`).
Figures are format-equivalent across platforms.

The headline figure for the paper is the three-bar comparison
(fp32 / int8 PTQ / int8 QAT) per variant, produced by
`max78000-implementation/scripts/plot_acc_comparison.py` →
`reports/fig_acc_comparison.png`.

---

## 11. How to add a new variant

1. Define `nn.Module` in `pc-implementation/training/models.py` and register in `MODEL_REGISTRY`.
2. Add the name to `choices` in `scripts/run_experiment.py` (and the prune/QAT-prune scripts if you'll use those compression modes).
3. **For MAX78000 deployment**: mirror with ai8x layers either in `max78000-implementation/scripts/models_ai8x.py` (then copy into `$AI/ai8x-training/models/project_models.py`) or as a separate module under `$AI/ai8x-training/models/`. Register in the `models = [...]` list at the bottom of the file.
4. **For MAX78000 synthesis**: create `max78000-implementation/networks/network_<variant>.yaml` describing processor mapping.
5. Add the variant name to:
   - `pc-implementation/train_pc_models.sh::VARIANTS`
   - `max78000-implementation/train_max78000_models.sh::VARIANTS_ALL` (and the case statement)
   - `max78000-implementation/synthesize_all.sh::synth_one` (arch mapping case)
   - `max78000-implementation/eval_pre_synth.sh` (arch mapping case)
   - `max78000-implementation/scripts/verify_fold.py::_SPECIAL_ARCH` and `scripts/eval_full.py::_SPECIAL_ARCH` if the arch name doesn't follow the `ai85net_cmsis_<v>` convention
6. **For IMX500 deployment**: usually nothing to do — `imx500-implementation/post_training_compress.py` auto-discovers `<v>_*.pt` under `pc-implementation/trained_models/` and runs MCT PTQ + ONNX export with no per-variant configuration. The only gotcha is variant-name resolution: if the checkpoint stem doesn't match a key in `MODEL_REGISTRY`, add an entry to `MODEL_NAME_MAP` in `post_training_compress.py` (e.g. `improved_distill → improved`). Then re-run `convert_all_onnx.sh` (host) and `package_all_rpk.sh` (Pi).
7. Train, generate reports, drop numbers into §8.

---

## References

- Lai, Suda, Chandra. *CMSIS-NN: Efficient Neural Network Kernels for Arm
  Cortex-M CPUs.* arXiv:[1801.06601](https://arxiv.org/abs/1801.06601), 2018.
- He, Zhang, Ren, Sun. *Deep Residual Learning for Image Recognition.*
  CVPR 2016.
- HasanPour, Rouhani, Fayyaz, Sabokrou. *Lets keep it simple, using simple
  architectures to outperform deeper and more complex architectures.*
  arXiv:[1608.06037](https://arxiv.org/abs/1608.06037), 2016.
- Banbury et al. *MicroNets: Neural Network Architectures for Deploying
  TinyML Applications on Commodity Microcontrollers.* MLSys 2021.
- Banbury et al. *MLPerf Tiny Benchmark.* NeurIPS Datasets and Benchmarks
  Track, 2021. arXiv:[2106.07597](https://arxiv.org/abs/2106.07597).
- Maxim Integrated (Analog Devices). *ai8x-training reference networks
  (ai85nascifarnet, ai85ressimplenet).*
  https://github.com/MaximIntegratedAI/ai8x-training, 2021.
- Sony Semiconductor. *Model Compression Toolkit (MCT) — IMX500 target
  platform capabilities (TPC v1).*
  https://github.com/sony/model_optimization, 2023.
