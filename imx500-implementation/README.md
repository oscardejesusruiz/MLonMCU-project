# IMX500 — CIFAR-10 deployment

End-to-end pipeline for quantizing, converting, packaging and measuring
CIFAR-10 classifiers on the Sony **IMX500** intelligent vision sensor
(Raspberry Pi AI Camera). Sister project to
[`../max78000-implementation`](../max78000-implementation): same model
zoo, same fp32/QAT checkpoints, but a completely different deployment
toolchain (MCT PTQ → ONNX → `imxconv-pt` → `imx500-package` → `.rpk`).

The PC training side ([`../pc-implementation`](../pc-implementation))
produces the float checkpoints this directory consumes.

---

## Pipeline overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│   ../pc-implementation/trained_models/<v>_{fp32,fp32_ptq,qat}.pt        │
│                                                                         │
│           │                                                             │
│           ▼                                                             │
│                                                                         │
│   post_training_compress.py     (Model Compression Toolkit, IMX500 TPC v1)
│   ─────────────────────────                                             │
│   • wraps each .pt in IMX500PrepWrapper (bakes 0-255 → CIFAR norm)      │
│   • runs MCT PyTorch PTQ with 10 calibration batches                    │
│   • exports quantized ONNX → outputs/onnx/<tag>_imx500_ptq.onnx         │
│   • writes outputs/reports/<tag>_metrics.json + outputs/summary.json    │
│                                                                         │
│           │                                                             │
│           ▼                                                             │
│                                                                         │
│   convert_all_onnx.sh           (imxconv-pt — host side, x86 or Pi)     │
│   ───────────────────                                                   │
│   • per-ONNX: produce packerOut.zip + dnnParams.xml + MemoryReport.json │
│   • outputs/imx500_converted/<tag>/                                     │
│                                                                         │
│           │                                                             │
│           ▼                                                             │
│                                                                         │
│   package_all_rpk.sh            (imx500-package — Raspberry Pi)         │
│   ──────────────────                                                    │
│   • per packerOut.zip: produce network.rpk                              │
│   • outputs/rpk/<tag>/network.rpk                                       │
│                                                                         │
│           │                                                             │
│           ▼                                                             │
│                                                                         │
│   camera_imx500_live.py / camera_imx500_view.py   (Raspberry Pi + AI Camera)
│   ────────────────────────────────────────────                          │
│   • IMX500 firmware loads the .rpk, runs inference on-sensor            │
│   • Picamera2 callbacks deliver logits + per-frame KPI (HW time)        │
│   • logs/<tag>.jsonl + logs/<tag>_summary.json                          │
│                                                                         │
│           │                                                             │
│           ▼                                                             │
│                                                                         │
│   build_report.py                                                       │
│   ───────────────                                                       │
│   • merges per-model metrics + memory reports + live device logs        │
│   • reports/summary.md + reports/figures/pareto.png                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Models evaluated

All seven architectures from the PC zoo deploy on the IMX500 — the
sensor has no 3×3-kernel restriction, so `baseline_5x5` is included
here (unlike the MAX78000 side which excludes it).

For each architecture three checkpoint flavours are quantized and
benchmarked:

| Suffix | Source checkpoint | Meaning |
|---|---|---|
| `_fp32` | `<v>_fp32.pt` (PC fp32 training) | float baseline → IMX500 PTQ |
| `_fp32_ptq` | `<v>_fp32_ptq.pt` (PC int8 PTQ already applied) | PC-PTQ'd weights → IMX500 PTQ (double-quant) |
| `_qat` | `<v>_qat.pt` (PC QAT fine-tune) | QAT-trained weights → IMX500 PTQ |

| Variant | Family | Source | Params | MACs/inf |
|---|---|---|---|---|
| `baseline_5x5` | Dense conv, 5×5 kernels | Lai et al. CMSIS-NN (arXiv 2018) | 89 K | 12.3 M |
| `baseline` | Dense conv, 3×3 | Lai et al. CMSIS-NN (3×3 adaptation) | 39 K | 4.4 M |
| `improved` | `baseline` + BN | this work (ablation) | 39 K | 4.4 M |
| `deeper` | `improved` + 2 extra conv layers | this work (ablation) | 141 K | 8.0 M |
| `mininet` | VGG-Micro (deep narrow stack) | Banbury et al. MicroNets (MLSys 2021) | 316 K | 24.5 M |
| `nascifarnet` | NAS-found, MCU-targeted | Maxim ai8x-training (2021) | 302 K | 36.2 M |
| `ressimplenet` | Residual SimpleNet (14 conv + 3 skips) | HasanPour et al. (arXiv 2016) | 373 K | 18.5 M |

Headline result from `reports/summary.md`: MCT PTQ on the IMX500 TPC
costs **≤0.3 pp** vs the PC fp32 reference across the board — much
gentler than the MAX78000's per-tensor symmetric quantization, because
the IMX500 toolchain uses per-channel scales with calibrated activation
ranges, and the `_fp32_ptq` rows that already collapsed on the MAX78000
side recover well here (e.g. `mininet_fp32_ptq`: 13% PC int8 → 12.8%
IMX500 int8 — the model was already destroyed; but `mininet_qat`:
88.6% fp32 → 88.5% int8, no drop).

---

## Directory structure

### Top-level orchestration

| File | Purpose |
|---|---|
| `post_training_compress.py` | MCT PyTorch PTQ for every `.pt` in `pc-implementation/trained_models/`. Wraps each model in `IMX500PrepWrapper` so the exported ONNX expects raw 0-255 sensor pixels. |
| `run_quantization.sh` | Convenience wrapper around `post_training_compress.py` (writes to `outputs_2/`, batch=64, skips ONNX export — for accuracy-only sweeps). |
| `convert_all_onnx.sh` | Batches every ONNX through `imxconv-pt` → `outputs/imx500_converted/<tag>/packerOut.zip`. |
| `package_all_rpk.sh` | Batches every `packerOut.zip` through `imx500-package` → `outputs/rpk/<tag>/network.rpk`. **Runs on the Pi.** |
| `imxconv_commands.sh` | Reference: equivalent of `convert_all_onnx.sh` but with `--model-insight` so each converted folder also contains `<tag>_Insights.json`. |
| `raspberry-terminal.sh` | One-liner Pi helper: iterates every `outputs/rpk/<tag>/network.rpk` and runs `camera_imx500_live.py --frames 500` per model. |
| `build_report.py` | Aggregates `outputs/summary.json`, converter `*_MemoryReport.json`, and live `logs/*_summary.json` into `reports/summary.md` + `reports/figures/pareto.png`. |
| `camera_imx500_live.py` | **Pi-side.** Loads a `.rpk`, runs Picamera2, prints per-frame predictions + HW inference KPI, writes JSONL log. 1-second rolling-mean smoother. |
| `camera_imx500_view.py` | **Pi-side.** Same as above plus a matplotlib window showing the camera frame + class bar chart. |

### `host/` — PC-side companions (not strictly needed for the IMX500 flow)

These mirror the MAX78000 host scripts so the same analysis tools work
across both deployment targets. They are kept here for symmetry with
the MAX78000 implementation, but the primary IMX500 data-collection path
runs on the Pi via `camera_imx500_live.py`.

### `outputs/` — generated artefacts

| Path | Purpose |
|---|---|
| `outputs/onnx/<tag>_imx500_ptq.onnx` | Quantized ONNX (one per checkpoint × mode = 21 files for 7 variants × 3 flavours). |
| `outputs/imx500_converted/<tag>/` | `packerOut.zip`, `dnnParams.xml`, `<tag>_MemoryReport.json` (footprint vs 8 MB on-chip budget). |
| `outputs/rpk/<tag>/network.rpk` | Packaged firmware blob loaded by Picamera2. |
| `outputs/reports/<tag>_metrics.json` | Per-model: params, MACs, fp32 vs int8 acc and mAP, TPC info. |
| `outputs/summary.json` | All 21 runs as a single JSON array (consumed by `build_report.py`). |
| `outputs/quant_info/<tag>/` | Verbose MCT `UserInformation` dump per checkpoint. |

### `logs/` — Pi-side live capture

| File | Purpose |
|---|---|
| `<tag>.jsonl` | Per-frame: `predicted_class`, `confidence`, `hw_inference` (ms), `frame_to_frame_ms`. Appended live by `camera_imx500_live.py`. |
| `<tag>_summary.json` | Sample count, frame limit, file paths. |

### `reports/` — paper-ready outputs

| File | Purpose |
|---|---|
| `summary.md` | Compact + full comparison tables, per-experiment notes, converter memory checks, live-Pi latency block. |
| `figures/pareto.png` | int8 accuracy vs MACs/inference scatter across all 21 runs. |

---

## End-to-end: from clean checkout to numbers on the AI Camera

### Prerequisites

1. **PC training done** — see [`../pc-implementation/README.md`](../pc-implementation/README.md). You need `<v>_{fp32,fp32_ptq,qat}.pt` under `pc-implementation/trained_models/`.
2. **MCT installed** in a Python 3.10+ env: `pip install model-compression-toolkit torch torchvision`.
3. **IMX500 converter (`imxconv-pt`)** installed and on PATH (`pip install imx500-converter[pt]` or the Sony SDK installer).
4. **Raspberry Pi 5 + AI Camera** with the IMX500 software stack: `sudo apt install imx500-tools imx500-all python3-picamera2`.
5. (Optional) `conda` env `mcu-pt` if you prefer the conda recipe — `post_training_compress.py` is environment-agnostic, only needs MCT + torch.

### Step 1 — Quantize every checkpoint and export ONNX (host)

```bash
conda run -n mcu-pt python imx500-implementation/post_training_compress.py \
    --source-dir pc-implementation/trained_models \
    --output-dir imx500-implementation/outputs \
    --include-non-fp32
```

The `--include-non-fp32` flag picks up `*_fp32_ptq.pt` and `*_qat.pt` in addition to the fp32 checkpoints. Drop it for an fp32-only sweep.

Per checkpoint the script:
1. Loads the float weights, wraps them in `IMX500PrepWrapper` so the ONNX accepts raw 0-255 sensor pixels.
2. Runs MCT PTQ with the IMX500 TPC v1 (`mct.ptq.pytorch_post_training_quantization`).
3. Evaluates float + quantized accuracy on the CIFAR-10 test split.
4. Exports the quantized model to ONNX.
5. Writes `outputs/reports/<tag>_metrics.json`.

After the sweep, `outputs/summary.json` collects every run.

### Step 2 — Convert ONNX → `packerOut.zip` (host)

```bash
bash imx500-implementation/convert_all_onnx.sh
```

Iterates `outputs/onnx/*.onnx` through `imxconv-pt`. Each model lands at `outputs/imx500_converted/<tag>/` containing `packerOut.zip` and `<tag>_MemoryReport.json`. The memory report tells you whether the model fits in the 8 MB on-chip budget (it does, with comfortable margin — the largest model is ~370 KB).

If you want per-layer insights too, use `imxconv_commands.sh` (same calls plus `--model-insight`).

### Step 3 — Package `packerOut.zip` → `network.rpk` (Pi)

```bash
# on the Raspberry Pi
bash imx500-implementation/package_all_rpk.sh
```

`imx500-package` is only shipped via `imx500-tools` on Raspberry Pi OS, so this step has to run on the Pi. Output: `outputs/rpk/<tag>/network.rpk` per model.

### Step 4 — Live inference on the AI Camera (Pi)

```bash
# single model
python imx500-implementation/camera_imx500_live.py \
    --model imx500-implementation/outputs/rpk/baseline_qat_imx500_ptq/network.rpk \
    --frames 500

# every model in outputs/rpk/, 500 frames each (writes per-model JSONL + summary)
bash imx500-implementation/raspberry-terminal.sh
```

`camera_imx500_live.py` loads the `.rpk` into the IMX500 firmware, starts Picamera2, and prints per-frame predictions with HW inference time (from `imx500.get_kpi_info`) and frame-to-frame wall time. A 1-second rolling-mean smooths the softmax distribution. Use `camera_imx500_view.py` for a matplotlib viewer with a class bar chart instead of text output.

### Step 5 — Build the comparison report (host)

```bash
uv run python imx500-implementation/build_report.py \
    --source imx500-implementation/outputs/summary.json \
    --out-dir imx500-implementation/reports \
    --logs-dir imx500-implementation/logs
```

Merges:
- accuracy + footprint from `outputs/summary.json`,
- converter memory checks from `outputs/imx500_converted/*/*_MemoryReport.json`,
- device timings from `logs/*_summary.json` (mean / p50 / p95 of HW inference + frame-to-frame).

Writes `reports/summary.md` (compact + full sweep tables, per-experiment notes) and `reports/figures/pareto.png`.

---

## Reading the results

`reports/summary.md` is the canonical artefact. The three numbers per
variant that tell the deployment story:

- **fp32 reference accuracy** — from `pc-implementation/reports/<v>_fp32_metrics.json`
- **IMX500 int8 accuracy (PTQ from fp32 ckpt)** — `<v>_fp32_imx500_ptq` row of `summary.md`
- **IMX500 int8 accuracy (PTQ from QAT ckpt)** — `<v>_qat_imx500_ptq` row of `summary.md`

The IMX500 toolchain's per-channel scales + calibration absorb most of
the PTQ drop on its own — QAT brings small additional gains (~0.5-1 pp),
unlike the MAX78000 where QAT recovery is 15-25 pp because the silicon's
per-tensor symmetric quantization is much harsher.

The `_fp32_ptq` rows are an instructive failure mode: those checkpoints
already had their weights crushed by the PC-side MAX78000-realistic PTQ,
so re-quantizing them on the IMX500 just preserves the existing damage
(`deeper_fp32_ptq`: 21% acc, `mininet_fp32_ptq`: 13% acc). These rows
are kept in the sweep as a control — they prove the IMX500 PTQ itself
is gentle.

### Headline numbers (from `reports/summary.md`)

| Variant (QAT) | fp32 acc | IMX500 int8 acc | Δ | HW inf. (ms) |
|---|---|---|---|---|
| `baseline` | 81.15% | 80.94% | -0.21 pp | 1.53 |
| `improved` | 81.91% | 81.75% | -0.16 pp | 1.53 |
| `deeper` | 85.69% | 85.62% | -0.07 pp | 1.53 |
| `mininet` | 88.59% | 88.52% | -0.07 pp | 1.53 |
| `nascifarnet` | 88.95% | 88.95% | +0.00 pp | 1.53 |
| `ressimplenet` | 88.20% | 88.32% | +0.12 pp | 1.53 |

HW inference time is essentially constant (~1.53 ms) across all models
— a deliberate choice by Sony: the on-sensor accelerator is dimensioned
to absorb a CIFAR-class network at the camera's native frame rate, so
the bottleneck is the I/O loop (~90 ms frame-to-frame), not the
inference itself.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `imxconv-pt: command not found` | IMX500 converter not installed / not on PATH | `pip install imx500-converter[pt]`, or set `IMX500_CONVERTER_BIN=/path/to/imxconv-pt`. |
| `imx500-package: command not found` (on Pi) | `imx500-tools` not installed | `sudo apt install imx500-tools`. Must be on the Pi — there is no x86 build. |
| MCT errors with `unsupported op` | A layer in the PC model isn't in the IMX500 TPC v1 op-set | `post_training_compress.py` wraps every model in `IMX500PrepWrapper` (only `/` and `-` extra). If the inner model uses an unsupported op, it must be removed at the PC training side. |
| Live predictions stuck on one class | `IMX500PrepWrapper` not applied at export → ONNX expects normalized input but firmware feeds 0-255 | Re-run `post_training_compress.py`; verify the exported ONNX graph has the `Div(255)` + `Sub(mean)/Div(std)` head. |
| `_fp32_ptq` accuracy collapses to ~10-40% | **Expected** — those checkpoints come from `<v>_fp32_ptq.pt`, which already had MAX78000-realistic PTQ applied on the PC side. Re-quantization preserves the existing damage. | Use the `_fp32` or `_qat` flavours for deployment. The `_fp32_ptq` rows are kept as a control. |
| Frame rate caps at ~11 fps even though `hw_inference` ≈ 1.5 ms | Picamera2 + display loop overhead, not the IMX500 | This is the camera I/O ceiling on Pi 5. Use `--no-preview` and run headless if you need to push frame rate. |

---

## How this compares to the MAX78000 deployment

| Quantity | MAX78000 ([`../max78000-implementation`](../max78000-implementation)) | IMX500 (this dir) |
|---|---|---|
| Quantization | per-tensor symmetric int8, power-of-two scales, per-layer `output_shift` | per-channel int8 with calibrated activation ranges (MCT IMX500 TPC v1) |
| QAT necessity | mandatory — naive PTQ loses 15-25 pp | optional — PTQ alone hits ≤0.3 pp drop |
| Architectures deployed | 6 (no `baseline_5x5` — 5×5 kernels not supported) | 7 (all PC models) |
| Toolchain | `ai8x-synthesis` + `ai8xize.py` → C project + `make` + DAPLINK flash | MCT PTQ → ONNX → `imxconv-pt` → `imx500-package` → `.rpk` loaded by Picamera2 |
| Where it runs | MAX78000 FTHR_RevA board (Arm Cortex-M4 + CNN accelerator) | Sony IMX500 sensor inside the Raspberry Pi AI Camera |
| Inference latency | Per-network — varies from ~hundreds of µs to ~ms depending on MACs | ~1.5 ms across all 7 models (sensor-dimensioned) |
| Energy story | GPIO-toggled, externally measured (Joulescope / INA219) | Not characterised here — sensor draws a fixed budget independent of model |
