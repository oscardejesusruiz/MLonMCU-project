# ML on Microcontrollers Project

End-to-end study of CIFAR-10 image classification on resource-constrained
edge accelerators, comparing a pure-PyTorch reference against the
**MAX78000** CNN accelerator (Maxim / Analog Devices) and the **IMX500**
intelligent vision sensor (Sony).

For each network architecture we train an fp32 reference on PC, fine-tune
a QAT variant for int8 deployment, synthesize the C deployment project,
flash it to the FTHR_RevA board, and measure on-device accuracy /
latency / energy. The result is a head-to-head trade-off table per
architecture, per platform, and per quantization regime.

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
│   ├── trained_models/            # *.pt checkpoints
│   └── reports/                   # metrics JSONs, figures, predictions
│
└── max78000-implementation/     ← MAX78000 deployment pipeline
    ├── train_max78000_models.sh   # supervisor — fp32 + QAT, all variants
    ├── synthesize_all.sh          # BN-fold → int8 quant → ai8xize.py → C project
    ├── eval_pre_synth.sh          # host-side fp32 / fused / int8 sim accuracy check
    ├── networks/                  # synthesis YAMLs (processor maps, offsets)
    ├── scripts/                   # bn_fuser_v2, verify_fold, estimate_metrics, …
    ├── bash_device_scripts/       # device_*.sh / host_*.sh (flash + measure)
    ├── host/                      # Python-side UART companions + live GUI demo
    ├── c_harness/                 # firmware drop-ins (inference / profile / measure)
    ├── trained_models/            # *.pth.tar checkpoints
    └── reports/                   # logs, predictions, comparison plots
```

See each subdirectory's own `README.md` for full pipeline details.

---

## Models evaluated

Seven architectures spanning four TinyML design families. Six deploy
end-to-end on the MAX78000; one is PC-only (5×5 kernels not supported
on-device).

| Variant | Family | Source | MAX78000 |
|---|---|---|---|
| `baseline_5x5`   | Dense shallow, 5×5 kernels | Lai et al. CMSIS-NN (2018) | ✗ |
| `baseline`       | Dense shallow, 3×3 (MAX78000-portable) | Lai et al. CMSIS-NN (2018) | ✓ |
| `improved`       | `baseline` + BatchNorm | this work (ablation) | ✓ |
| `deeper`         | `improved` + 2 extra layers | this work (ablation) | ✓ |
| `mininet`        | VGG-Micro (deep narrow stack) | Banbury et al. MicroNets, MLSys 2021 | ✓ |
| `nascifarnet`    | NAS-found, MCU-targeted | Maxim ai8x-training (2021) | ✓ |
| `ressimplenet`   | Residual SimpleNet (14 conv + 3 skips) | HasanPour et al. (arXiv 2016) | ✓ |

Full layer-by-layer specs in [`MODELS.md`](MODELS.md). The architecture
selection rationale (and which SOTA families were considered + dropped,
and why) is documented there as well.

---

## Pipeline at a glance

```
┌──────────────────────┐    ┌─────────────────────────┐    ┌─────────────────────┐
│  pc-implementation   │ →  │  max78000-implementation│ →  │  FTHR_RevA board    │
│                      │    │                         │    │                     │
│  • fp32 training     │    │  • mirror training      │    │  • flash via DAPLINK│
│  • QAT fine-tune     │    │  • BN-fold + quantize   │    │  • UART → host_*.sh │
│  • int8 PTQ sim      │    │  • ai8xize.py synthesis │    │  • per-layer profile│
│  • Pareto plots      │    │  • host accuracy check  │    │  • full test-set    │
│                      │    │                         │    │  • energy (GPIO)    │
└──────────────────────┘    └─────────────────────────┘    └─────────────────────┘
```

Both implementations use **identical training recipes** (same optimizer,
LR, batch size, epoch budget per variant) — the only difference is the
target hardware. This keeps the PC ↔ device comparison apples-to-apples.

---

## Quick start

```bash
# 1. Train all variants on PC (fp32 + QAT, idempotent)
cd pc-implementation
uv sync
./train_pc_models.sh

# 2. Mirror the training on the MAX78000 side (uses ai8x-training)
cd ../max78000-implementation
./train_max78000_models.sh all fp32
./train_max78000_models.sh all qat

# 3. Synthesize C deployment projects (per variant)
./synthesize_all.sh

# 4. Host-side accuracy verification (no board needed)
./eval_pre_synth.sh
./eval_pre_synth.sh qat

# 5. Flash + measure on device (one variant at a time)
./bash_device_scripts/device_testset.sh baseline
./bash_device_scripts/host_testset.sh   baseline

# 6. Generate the headline comparison figures
python3 scripts/plot_acc_comparison.py
```

Detailed instructions in each subdirectory's `README.md`.

---

## What you get

After the full pipeline runs, the headline outputs are:

- **`pc-implementation/reports/figures/pareto.png`** — accuracy-vs-compute Pareto frontier across all variants
- **`pc-implementation/reports/network_diagrams/*.png`** — per-variant 3D architecture diagrams (generated by `plot_network_diagrams.sh`)
- **`max78000-implementation/reports/_eval_pre_synth/{fp32,qat}/`** — fp32 vs fused vs int8-sim accuracy per variant
- **`max78000-implementation/reports/fig_acc_comparison.png`** — 3 bars per variant (fp32 / int8 PTQ / int8 QAT), the central story of the paper
- **`max78000-implementation/reports/fig_acc_vs_macs.png`** — scatter showing where PTQ collapses and QAT recovers, plotted against compute
- **`max78000-implementation/reports/profile_<v>.txt`** — ST.AI-style per-layer profile from device

The science: **naive PTQ degrades int8 accuracy by 5-20 pp** at deployment
(due to BN folding redistributing per-channel weight magnitudes and the
accelerator's per-layer output rescaling), while **QAT fine-tuning recovers
almost all of it** at the cost of ~40 extra training epochs.

---

## Hardware

- **MAX78000 FTHR_RevA** development board (DAPLINK + USB serial)
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
