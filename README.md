# ML on Microcontrollers Project

End-to-end study of CIFAR-10 image classification on resource-constrained
edge accelerators, comparing a pure-PyTorch reference against two
on-sensor / on-MCU deployment targets:

- **MAX78000** CNN accelerator (Maxim / Analog Devices) on the FTHR_RevA board
- **IMX500** intelligent vision sensor (Sony) on the Raspberry Pi AI Camera

For each network architecture we train an fp32 reference on PC, fine-tune
a QAT variant for int8 deployment, push it through each deployment
toolchain, and measure on-device accuracy / latency. The result is a
head-to-head trade-off table per architecture, per platform, and per
quantization regime.

---

## Repository layout

```
ml-on-microcontrollers/
├── README.md                    ← this file
├── MODELS.md                    ← per-architecture spec sheets (layer tables,
│                                  params, MACs, design rationale)
│
├── pc-implementation/           ← pure-PyTorch training + int8 simulation
│   ├── training/                  # model classes, data loaders, QAT helpers
│   ├── scripts/                   # train / eval / plotting drivers
│   ├── train_pc_models.sh         # supervisor — runs everything per variant
│   ├── plot_network_diagrams.sh   # visualize each architecture (3D blocks)
│   ├── trained_models/            # *.pt checkpoints (consumed by both targets)
│   └── reports/                   # metrics JSONs, figures, predictions
│
├── max78000-implementation/     ← MAX78000 deployment pipeline
│   ├── train_max78000_models.sh   # supervisor — fp32 + QAT, all variants
│   ├── synthesize_all.sh          # BN-fold → int8 quant → ai8xize.py → C project
│   ├── eval_pre_synth.sh          # host-side fp32 / fused / int8 sim accuracy check
│   ├── networks/                  # synthesis YAMLs (processor maps, offsets)
│   ├── scripts/                   # bn_fuser_v2, verify_fold, estimate_metrics, …
│   ├── bash_device_scripts/       # device_*.sh / host_*.sh (flash + measure)
│   ├── host/                      # Python-side UART companions + live GUI demo
│   ├── c_harness/                 # firmware drop-ins (inference / profile / measure)
│   ├── trained_models/            # *.pth.tar checkpoints
│   └── reports/                   # logs, predictions, comparison plots
│
└── imx500-implementation/       ← IMX500 deployment pipeline
    ├── post_training_compress.py  # MCT PTQ for every PC checkpoint → ONNX
    ├── convert_all_onnx.sh        # batch imxconv-pt → packerOut.zip
    ├── package_all_rpk.sh         # batch imx500-package → network.rpk (Pi)
    ├── camera_imx500_live.py      # Pi-side live inference (Picamera2 + IMX500)
    ├── camera_imx500_view.py      # Pi-side matplotlib viewer
    ├── build_report.py            # aggregate metrics + Pi logs → summary.md
    ├── outputs/                   # ONNX, converted, RPK, per-model metrics
    ├── logs/                      # per-model Pi-side JSONL + summary JSON
    └── reports/                   # summary.md + figures/pareto.png
```

See each subdirectory's own `README.md` for full pipeline details.

---

## Models evaluated

Seven architectures spanning four TinyML design families. All seven
deploy on the IMX500; six deploy on the MAX78000 (5×5 kernels not
supported by that accelerator).

| Variant | Family | Source | MAX78000 | IMX500 |
|---|---|---|---|---|
| `baseline_5x5`   | Dense shallow, 5×5 kernels | Lai et al. CMSIS-NN (2018) | ✗ | ✓ |
| `baseline`       | Dense shallow, 3×3 (MAX78000-portable) | Lai et al. CMSIS-NN (2018) | ✓ | ✓ |
| `improved`       | `baseline` + BatchNorm | this work (ablation) | ✓ | ✓ |
| `deeper`         | `improved` + 2 extra layers | this work (ablation) | ✓ | ✓ |
| `mininet`        | VGG-Micro (deep narrow stack) | Banbury et al. MicroNets, MLSys 2021 | ✓ | ✓ |
| `nascifarnet`    | NAS-found, MCU-targeted | Maxim ai8x-training (2021) | ✓ | ✓ |
| `ressimplenet`   | Residual SimpleNet (14 conv + 3 skips) | HasanPour et al. (arXiv 2016) | ✓ | ✓ |

Full layer-by-layer specs in [`MODELS.md`](MODELS.md). The architecture
selection rationale (and which SOTA families were considered + dropped,
and why) is documented there as well.

---

## Pipeline at a glance

```
┌──────────────────────┐    ┌────────────────────────────┐    ┌─────────────────────┐
│  pc-implementation   │ →  │  max78000-implementation   │ →  │  FTHR_RevA board    │
│                      │    │                            │    │                     │
│  • fp32 training     │    │  • mirror training         │    │  • flash via DAPLINK│
│  • QAT fine-tune     │    │  • BN-fold + quantize      │    │  • UART → host_*.sh │
│  • int8 PTQ sim      │    │  • ai8xize.py synthesis    │    │  • per-layer profile│
│  • Pareto plots      │    │  • host accuracy check     │    │  • full test-set    │
│                      │    │                            │    │  • energy (GPIO)    │
│   *.pt checkpoints   │    └────────────────────────────┘    └─────────────────────┘
│        │             │
│        │             │    ┌────────────────────────────┐    ┌─────────────────────┐
│        └───────────→ │ →  │  imx500-implementation     │ →  │  Pi AI Camera       │
│                      │    │                            │    │                     │
│                      │    │  • MCT PTQ (IMX500 TPC v1) │    │  • load .rpk        │
│                      │    │  • ONNX export             │    │  • Picamera2 capture│
│                      │    │  • imxconv-pt → packerOut  │    │  • per-frame HW KPI │
│                      │    │  • imx500-package → .rpk   │    │  • JSONL logs       │
│                      │    │                            │    │                     │
└──────────────────────┘    └────────────────────────────┘    └─────────────────────┘
```

Both MCU implementations consume the **same `*.pt` checkpoints** from
`pc-implementation/trained_models/`. The PC training recipe (optimizer,
LR, batch size, epoch budget per variant) is the single source of truth
— the only thing that varies between deployment targets is the
quantization toolchain and the silicon.

---

## Quick start

```bash
# 1. Train all variants on PC (fp32 + QAT, idempotent)
cd pc-implementation
uv sync
./train_pc_models.sh

# 2a. MAX78000 path — mirror training, synthesize, flash, measure
cd ../max78000-implementation
./train_max78000_models.sh all fp32
./train_max78000_models.sh all qat
./synthesize_all.sh
./eval_pre_synth.sh
./bash_device_scripts/device_testset.sh baseline
./bash_device_scripts/host_testset.sh   baseline

# 2b. IMX500 path — MCT PTQ, convert, package, run on Pi
cd ../imx500-implementation
python post_training_compress.py \
    --source-dir ../pc-implementation/trained_models \
    --output-dir outputs --include-non-fp32
bash convert_all_onnx.sh                    # host (needs imxconv-pt)
# ssh to Pi:
bash package_all_rpk.sh                     # needs imx500-tools
bash raspberry-terminal.sh                  # live inference, 500 frames per model
# back on host:
python build_report.py

# 3. Generate the headline comparison figures
cd ../pc-implementation
uv run python -m scripts.build_report
cd ../max78000-implementation
python3 scripts/plot_acc_comparison.py
```

Detailed instructions in each subdirectory's `README.md`.

---

## What you get

After the full pipeline runs, the headline outputs are:

- **`pc-implementation/reports/figures/pareto.png`** — accuracy-vs-compute Pareto frontier across all variants
- **`pc-implementation/reports/network_diagrams/*.png`** — per-variant 3D architecture diagrams (generated by `plot_network_diagrams.sh`)
- **`max78000-implementation/reports/_eval_pre_synth/{fp32,qat}/`** — fp32 vs fused vs int8-sim accuracy per variant
- **`max78000-implementation/reports/fig_acc_comparison.png`** — 3 bars per variant (fp32 / int8 PTQ / int8 QAT) on the MAX78000
- **`max78000-implementation/reports/fig_acc_vs_macs.png`** — scatter showing where PTQ collapses and QAT recovers on the MAX78000
- **`max78000-implementation/reports/profile_<v>.txt`** — ST.AI-style per-layer profile from the FTHR board
- **`imx500-implementation/reports/summary.md`** — IMX500 comparison: fp32 vs int8 acc + per-model HW inference time, with converter memory checks and live-Pi latency block
- **`imx500-implementation/reports/figures/pareto.png`** — IMX500 int8 acc vs MACs across all 21 runs (7 variants × 3 checkpoint flavours)

The science across the two targets:

- **MAX78000:** naive PTQ degrades int8 accuracy by 15-25 pp at deployment (BN folding redistributes per-channel weight magnitudes and the accelerator's per-tensor symmetric quantization can't accommodate the resulting outliers); **QAT recovers almost all of it** at the cost of ~40 extra training epochs.
- **IMX500:** the toolchain's per-channel scales + activation calibration absorb the PTQ drop on their own — fp32→int8 cost is **≤0.3 pp** across the board, QAT brings only marginal additional gains.

The contrast between the two is the central finding of the paper: **the
quantization-accuracy story is dominated by the silicon's
quantization granularity, not by the model architecture or training
recipe**.

---

## Hardware

- **MAX78000 FTHR_RevA** development board (DAPLINK + USB serial)
- **Raspberry Pi 5 + AI Camera** (IMX500 sensor) — for the IMX500 path
- Apple Silicon Mac (MPS) or any CUDA GPU for PC training
- Optional for energy measurement: Joulescope or INA219 + logic analyzer

---

## References

Detailed citations per architecture are in [`MODELS.md`](MODELS.md). The
core references:

- Lai, Suda, Chandra. *CMSIS-NN: Efficient Neural Network Kernels for Arm
  Cortex-M CPUs.* arXiv:[1801.06601](https://arxiv.org/abs/1801.06601), 2018.
- Banbury et al. *MicroNets: Neural Network Architectures for Deploying
  TinyML Applications on Commodity Microcontrollers.* MLSys 2021.
- HasanPour et al. *Lets keep it simple, using simple architectures to
  outperform deeper and more complex architectures.*
  arXiv:[1608.06037](https://arxiv.org/abs/1608.06037), 2016.
- Maxim Integrated (Analog Devices). *ai8x-training & ai8x-synthesis.*
  https://github.com/MaximIntegratedAI
- Sony Semiconductor. *IMX500 intelligent vision sensor — Model Compression Toolkit (MCT).*
  https://github.com/sony/model_optimization
