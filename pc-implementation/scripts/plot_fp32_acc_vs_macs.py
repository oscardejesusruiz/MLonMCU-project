"""Plot fp32 accuracy vs MACs (millions) for all PC-trained CIFAR-10 variants.

Each model is shown as a labelled point. The display names follow the
renaming convention adopted in the paper write-up:

    mininet       → VGG-Micro-7
    nascifarnet   → NAS-EdgeNet-10
    ressimplenet  → ResSimpleNet-14

baseline_5x5, baseline, improved, and deeper keep their original names.

Output:
    pc-implementation/reports/figures/fp32_acc_vs_macs.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

# (display_name, macs_millions, fp32_accuracy_percent, int8_weight_KiB)
MODELS = [
    #("baseline_5x5",     12.30, 79.79,  87.5),
    ("baseline",          4.43, 80.98,  38.0),
    ("improved",          4.43, 81.40,  38.1),
    ("deeper",            7.96, 85.14, 137.7),
    ("VGG-Micro-7",      24.48, 88.63, 309.0),  # mininet
    ("NAS-EdgeNet-10",   36.18, 87.61, 294.7),  # nascifarnet
    ("ResSimpleNet-14",  18.45, 86.90, 363.8),  # ressimplenet
]

# Reference point from Lai et al. 2018 (CMSIS-NN baseline on CIFAR-10).
# Their figure uses Ops (≈ 2 × MACs); we plot it as MACs for a fair
# comparison vs our column, so 24.7 MOps -> ~12.35 MMACs.
# Weight memory: 87 KB int8 (their paper, table 2).
REFERENCE = ("Lai et al. 2018 (ref.)", 12.35, 79.9, 87.0)

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
OUT_SCATTER  = FIG_DIR / "fp32_acc_vs_macs.png"
OUT_BUBBLE   = FIG_DIR / "fp32_acc_vs_macs_bubble.png"
OUT_LATENCY  = FIG_DIR / "max78000_acc_vs_latency.png"
OUT_TABLE    = FIG_DIR / "max78000_results_table.png"
OUT_OVERVIEW = FIG_DIR / "pc_models_overview_table.png"
OUT_TRAINING = FIG_DIR / "pc_models_training_table.png"

# Platform input power for MAC-Ops/W in the results table.
#
# The FTHR_RevA's MAX20303 PMIC reports a USB Vbus draw of 51 mA during
# camera+inference operation (from the SDK boot print "USB charge current
# is set to 51 mA"). At Vbus = 5 V that is 51 mA × 5 V = 255 mW of total
# *system* input power — MAX78000 core, CNN array, LEDs, camera, UART
# and the DAPLINK debug interface all included.
#
# This is therefore a platform-level efficiency figure (board-as-a-whole),
# not the bare CNN accelerator number. The CNN block alone is ~5–30 mW
# of that envelope per datasheet figures, so the true CNN-only Ops/W is
# ~8-50× higher than what this column reports.
ASSUMED_POWER_MW = 51.0 * 5.0   # USB charge current × Vbus = 255 mW

# MAX78000 CNN clock = 100 MHz. Used to convert cycles → seconds where the
# profile file doesn't report ms directly.
CNN_CLOCK_HZ = 100_000_000

# MAX78000 on-device inference time (ms) and int8 top-1 accuracy (%).
# Latencies are NOW the live-measured `cnn_us` averages from the
# end-to-end profiling activity (host/host_e2e_profile.py over UART,
# rolling window of 10 inferences). These supersede the
# profile_*.txt static-sim numbers and are tightly in agreement with
# them (~0.5% delta). Accuracies are the int8 (QAT-deployed) column
# from the metrics table.
#   (display_name, latency_ms, int8_acc_percent, int8_weight_KiB)
MAX78000_RUNS = [
    ("baseline",         1.004, 80.85,  38.0),
    ("improved",         1.004, 81.51,  38.1),
    ("deeper",           1.129, 84.76, 137.7),
    ("VGG-Micro-7",      1.761, 88.57, 309.0),  # mininet
    ("NAS-EdgeNet-10",   4.573, 87.44, 294.7),  # nascifarnet
    ("ResSimpleNet-14",  2.883, 86.71, 363.8),  # ressimplenet
]

# Full per-variant result table.
#   (display_name, params, kib_int8, macs_M, fp32_acc, ptq_acc, qat_acc,
#    t_cnn_ms, mac_per_cnn_cycle, tops_w_meas)
#
# Where each column comes from:
#   • fp32 / ptq from max78000-implementation/reports/_eval_pre_synth/fp32/_acc/int8_*
#     (PTQ collapses on the deeper / NAS / residual variants)
#   • qat  from max78000-implementation/reports/_eval_pre_synth/qat/_acc/int8_*
#   • t_cnn_ms, mac_per_cnn_cycle, tops_w_meas are the live-measured
#     averages from host/host_e2e_profile.py (rolling window of 10
#     inferences, c_harness/profile_camera.c firmware). MAC/cycle is at
#     50 MHz CNN clock, peak 576. TOPS/W is computed against the paper
#     Table I power figure (28 mW).
TABLE_ROWS = [
    # name             params    kib   macs   fp32   ptq    qat    t_cnn  t_e2e  MAC/cyc  TOPS/W
    ("baseline",         38_890,  38.0,  4.43, 80.98, 75.50, 80.95,  1.004, 92.30,  88.3,  0.315),
    ("improved",         39_018,  38.1,  4.43, 81.40, 62.38, 82.86,  1.004, 92.30,  88.3,  0.315),
    ("deeper",          141_034, 137.7,  7.96, 85.14, 65.71, 84.74,  1.129, 92.30, 141.1,  0.504),
    ("VGG-Micro-7",     316_458, 309.0, 24.48, 88.63, 69.21, 84.61,  1.761, 92.30, 278.0,  0.993),  # mininet
    ("NAS-EdgeNet-10",  301_770, 294.7, 36.18, 87.61, 22.10, 89.25,  4.573, 92.30, 158.2,  0.565),  # nascifarnet
    ("ResSimpleNet-14", 372_512, 363.8, 18.45, 86.90, 13.40, 83.07,  2.883, 92.30, 128.0,  0.457),  # ressimplenet
]

# Power used for Energy / inference in the results table. This is the
# *paper Table I* number for MAX78000 active power (Capogrosso et al.
# 2026) and matches exactly what the live profiling dashboard
# (host_e2e_profile.py) displays in its "Energy / inference" row.
# We deliberately use this — not the FTHR USB-Vbus 255 mW figure —
# so the table aligns with the on-device measurements.
MEASURED_POWER_MW = 28.0
PEAK_TOPS_W = 2.0      # paper Table I, MAX78000 peak efficiency

# ---------------------------------------------------------------------------
# Reference paper (cited by every figure footnote / table that uses its
# MAX78000 row from Table I). Centralized here so a single edit updates
# every figure.
# ---------------------------------------------------------------------------
PAPER_AUTHORS = "L. Capogrosso, P. Bonazzi, M. Magno"
PAPER_TITLE   = ("Performance Analysis of Edge and In-Sensor AI "
                 "Processors: A Comparative Review")
PAPER_YEAR    = 2026
PAPER_ARXIV   = "arXiv:2603.08725"
PAPER_URL     = "https://arxiv.org/pdf/2603.08725"

# Short footnote form (fits inline in figure captions).
PAPER_REF_SHORT = (
    f"{PAPER_AUTHORS.split(',')[0].split()[-1]} et al. "
    f"({PAPER_YEAR}), {PAPER_ARXIV}"
)
# Full reference (bottom-of-figure citation block).
PAPER_REF_FULL  = (
    f"Ref: {PAPER_AUTHORS} ({PAPER_YEAR}). "
    f"\"{PAPER_TITLE}\". {PAPER_ARXIV}. {PAPER_URL}"
)


# Per-point label offsets — used by both figures.
LABEL_OFFSETS = {
    #"baseline_5x5":     (8,  -14),
    "baseline":         (8,    -16),
    "improved":         (8,    8),
    "deeper":           (8,    8),
    "VGG-Micro-7":      (8,    8),
    "NAS-EdgeNet-10":   (8,    8),
    "ResSimpleNet-14":  (8,    8),
}


def _setup_axes(ax, xs, ys) -> None:
    ax.set_xlabel("MACs per inference (millions)", fontsize=12)
    ax.set_ylabel("fp32 top-1 accuracy on CIFAR-10 (%)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(max(xs), REFERENCE[1]) * 1.12)
    ax.set_ylim(min(min(ys), REFERENCE[2]) - 2.0,
                max(max(ys), REFERENCE[2]) + 2.0)


def plot_scatter(xs, ys) -> None:
    """Original scatter — fixed marker size, no encoding for int8 size."""
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.scatter(xs, ys, s=110, color="#1f77b4", edgecolor="black",
               linewidth=0.7, zorder=3, label="PC fp32 (this work)")
    ax.scatter([REFERENCE[1]], [REFERENCE[2]], marker="X", s=160,
               color="#d62728", edgecolor="black", linewidth=0.7,
               zorder=3, label="Lai et al. 2018 (reference)")

    for name, x, y, _kib in MODELS:
        dx, dy = LABEL_OFFSETS.get(name, (8, 8))
        ax.annotate(name, (x, y), xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=10, fontweight="medium")
    ax.annotate(REFERENCE[0], (REFERENCE[1], REFERENCE[2]),
                xytext=(8, -16), textcoords="offset points",
                fontsize=10, fontstyle="italic", color="#a02020")

    ax.set_title("PC-trained CIFAR-10 models — fp32 accuracy vs. compute",
                 fontsize=13, pad=12)
    _setup_axes(ax, xs, ys)

    fig.tight_layout()
    fig.savefig(OUT_SCATTER, dpi=180)
    print(f"wrote {OUT_SCATTER}")


def plot_bubble(xs, ys) -> None:
    """Same axes as plot_scatter, but each bubble's *area* is proportional
    to that model's int8 weight memory (KiB). The reference point uses the
    same area scaling so it can be compared visually like-for-like.

    Bubble area = AREA_PER_KIB * size_kib  →  marker radius ∝ √(size_kib).
    """
    fig, ax = plt.subplots(figsize=(10, 6.5))

    AREA_PER_KIB = 6.0   # tuned so the largest model (~364 KiB) is readable
                         # but doesn't swallow the figure.

    kibs   = [m[3] for m in MODELS]
    sizes  = [AREA_PER_KIB * k for k in kibs]
    ref_sz = AREA_PER_KIB * REFERENCE[3]

    ax.scatter(xs, ys, s=sizes, color="#1f77b4", alpha=0.55,
               edgecolor="black", linewidth=0.8, zorder=3,
               label="PC fp32 (this work)")
    ax.scatter([REFERENCE[1]], [REFERENCE[2]], s=ref_sz,
               color="#d62728", alpha=0.55, edgecolor="black",
               linewidth=0.8, zorder=3,
               label="Lai et al. 2018 (reference, 87 KB int8)")

    for name, x, y, kib in MODELS:
        dx, dy = LABEL_OFFSETS.get(name, (8, 8))
        ax.annotate(f"{name}\n({kib:.0f} KiB)", (x, y),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=9.5, fontweight="medium")
    ax.annotate(f"{REFERENCE[0]}\n({REFERENCE[3]:.0f} KB)",
                (REFERENCE[1], REFERENCE[2]),
                xytext=(8, -28), textcoords="offset points",
                fontsize=9.5, fontstyle="italic", color="#a02020")

    ax.set_title("PC-trained CIFAR-10 models — fp32 accuracy vs. compute\n"
                 "(bubble area ∝ int8 weight memory)",
                 fontsize=13, pad=12)
    _setup_axes(ax, xs, ys)

    # Size legend: three reference bubbles in axes-fraction coords so the
    # reader can map bubble area → KiB without guessing.
    legend_kibs = [40, 150, 350]
    handles = [
        plt.scatter([], [], s=AREA_PER_KIB * k, color="#1f77b4",
                    alpha=0.55, edgecolor="black", linewidth=0.8,
                    label=f"{k} KiB int8")
        for k in legend_kibs
    ]
    size_legend = ax.legend(handles=handles, title="int8 weight memory",
                            loc="lower right", labelspacing=1.6,
                            borderpad=1.1, framealpha=0.95,
                            handletextpad=1.4)
    ax.add_artist(size_legend)

    fig.tight_layout()
    fig.savefig(OUT_BUBBLE, dpi=180)
    print(f"wrote {OUT_BUBBLE}")


def plot_max78000_latency() -> None:
    """MAX78000 on-device inference latency (ms) vs int8 top-1 accuracy (%).

    Plain scatter — fixed marker size, same visual style as plot_scatter.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    xs = [m[1] for m in MAX78000_RUNS]
    ys = [m[2] for m in MAX78000_RUNS]

    ax.scatter(xs, ys, s=110, color="#2ca02c", edgecolor="black",
               linewidth=0.7, zorder=3, label="MAX78000 int8 (this work)")

    lat_offsets = {
        "baseline":         (8,   -16),
        "improved":         (8,     8),
        "deeper":           (8,     8),
        "VGG-Micro-7":      (8,     8),
        "NAS-EdgeNet-10":   (8,     8),
        "ResSimpleNet-14":  (8,     8),
    }
    for name, x, y, _kib in MAX78000_RUNS:
        dx, dy = lat_offsets.get(name, (8, 8))
        ax.annotate(name, (x, y), xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=10, fontweight="medium")

    ax.set_xlabel("MAX78000 on-device inference latency (ms / image)",
                  fontsize=12)
    ax.set_ylabel("int8 top-1 accuracy on CIFAR-10 (%)", fontsize=12)
    ax.set_title("MAX78000 deployment — int8 accuracy vs. on-device latency",
                 fontsize=13, pad=12)
    ax.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(xs) * 1.15)
    ax.set_ylim(min(ys) - 2.0, max(ys) + 2.0)

    # Paper reference (the chip's power/peak-TOPS figures come from this
    # paper — even when this specific plot doesn't show them, our project
    # framing builds on it). Bottom-of-figure citation block.
    fig.text(
        0.5, 0.004,
        PAPER_REF_FULL,
        ha="center", va="bottom", fontsize=7.5,
        fontstyle="italic", color="#666",
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUT_LATENCY, dpi=180)
    print(f"wrote {OUT_LATENCY}")


def plot_results_table() -> None:
    """Render the per-variant MAX78000 result table as a standalone PNG.

    Columns (9, in order):
        Model | PTQ acc | QAT acc | Δ acc (QAT−PTQ)
              | Inference latency (ms) | End-to-end latency (ms)
              | MAC / CNN-cyc (@50 MHz) | Energy / inference (µJ)
              | Measured efficiency (TOPS/W + % of peak)

    Every numeric column is either a live measurement from
    host/host_e2e_profile.py or a single derivation from one. Power for
    the Energy column is MEASURED_POWER_MW (28 mW, paper Table I) so
    the values exactly match the on-device profiling dashboard.
    """
    headers = [
        "Model",
        "PTQ acc", "QAT acc", "Δ acc\n(QAT−PTQ)",
        "Inference\nlatency (ms)",
        "End-to-end\nlatency (ms)",
        "MAC / CNN-cyc\n(@50 MHz)",
        f"Energy / inf.\n(µJ, @ {MEASURED_POWER_MW:.0f} mW)",
        "Measured efficiency\n(TOPS/W, % of peak)",
    ]

    rows: list[list[str]] = []
    for (name, _params, _kib, _macs_M, _fp32, ptq, qat,
         t_cnn_ms, t_e2e_ms, mac_per_cyc, tops_w_meas) in TABLE_ROWS:
        delta = qat - ptq
        # Energy per inference: E = P · t — units cancel: mW · ms = µJ.
        energy_uj = MEASURED_POWER_MW * t_cnn_ms
        pct_of_peak = tops_w_meas / PEAK_TOPS_W * 100.0
        rows.append([
            name,
            f"{ptq:.2f}%",
            f"{qat:.2f}%",
            f"{delta:+.2f} pp",
            f"{t_cnn_ms:.3f}",
            f"{t_e2e_ms:.2f}",
            f"{mac_per_cyc:.1f}",
            f"{energy_uj:.2f} µJ",
            f"{tops_w_meas:.3f}  ({pct_of_peak:4.1f}%)",
        ])

    n_cols = len(headers)
    n_rows = len(rows)

    fig_w = 14.0
    fig_h = 0.55 + 0.50 * (n_rows + 1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    # Column widths — give the name column and the multi-value
    # efficiency column extra room. Sum is ~1.0.
    col_widths = [0.13,
                  0.075, 0.075, 0.095,
                  0.10,  0.10,
                  0.10,
                  0.115,
                  0.16]

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        colWidths=col_widths,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.55)

    # Header styling.
    for c in range(n_cols):
        cell = table[0, c]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", weight="bold")
        cell.set_height(cell.get_height() * 1.25)

    # Zebra row striping + colour the Δ-accuracy column by sign +
    # colour the efficiency and MAC/cyc columns by absolute magnitude
    # (yellow → green as utilization approaches the architectural peak).
    delta_col    = 3   # Δ acc column index (9-col layout)
    macc_col     = 6   # MAC / CNN-cyc column index
    eff_col      = 8   # Measured efficiency column index
    PEAK_MAC_CYC = 576.0

    for r in range(1, n_rows + 1):
        for c in range(n_cols):
            cell = table[r, c]
            cell.set_facecolor("#f5f7fa" if r % 2 == 0 else "white")

        # Δ column tint — green proportional to recovery magnitude.
        delta_val = float(rows[r - 1][delta_col].split()[0])
        if delta_val > 0:
            intensity = min(delta_val / 70.0, 1.0)
            green = 0.55 + 0.35 * intensity
            table[r, delta_col].set_facecolor((0.55, green, 0.55))
            table[r, delta_col].set_text_props(weight="bold")

        # Efficiency column tint — combined "TOPS/W (% of peak)" cell.
        # Parse the leading TOPS/W value out of the string.
        eff_val = float(rows[r - 1][eff_col].split()[0])
        frac = min(eff_val / PEAK_TOPS_W, 1.0)
        # Interpolate cream (low efficiency) → green (near peak).
        r_c = 0.99 - 0.44 * frac
        g_c = 0.95 - 0.10 * frac
        b_c = 0.65 - 0.10 * frac
        table[r, eff_col].set_facecolor((r_c, g_c, b_c))
        table[r, eff_col].set_text_props(weight="bold")

        # MAC/cycle column tint — same gradient, normalized to peak 576.
        mac_val = float(rows[r - 1][macc_col])
        frac_m = min(mac_val / PEAK_MAC_CYC, 1.0)
        table[r, macc_col].set_facecolor(
            (0.99 - 0.44 * frac_m, 0.95 - 0.10 * frac_m, 0.65 - 0.10 * frac_m)
        )

    # Left-align the model name for readability.
    for r in range(1, n_rows + 1):
        table[r, 0].set_text_props(ha="left")
        table[r, 0].PAD = 0.04

    plt.suptitle(
        "MAX78000 — per-variant CIFAR-10 results "
        "(int8 deployment, fp32 reference, QAT recovery)",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.028,
        f"Δ acc = QAT − PTQ (pp).   "
        f"Inference latency, end-to-end latency, MAC/cyc, TOPS/W: "
        f"live-measured by host_e2e_profile.py (rolling window of 10 "
        f"inferences, profile_camera.c firmware).   "
        f"End-to-end is camera-pinned at OV7692 ~10.8 FPS, "
        f"CAMERA_FREQ = 10 MHz.   "
        f"MAC / CNN-cyc normalised to the 50 MHz CNN clock "
        f"(peak 576 MAC/cyc).   "
        f"Energy / inf. = P · t_inference with P = {MEASURED_POWER_MW:.0f} mW "
        f"[Table I, {PAPER_REF_SHORT}].   "
        f"% of peak = TOPS/W ÷ {PEAK_TOPS_W:.2f} TOPS/W "
        f"[Table I, {PAPER_REF_SHORT}].",
        ha="center", va="bottom", fontsize=8.5,
        fontstyle="italic", color="#444",
    )
    # Full reference, bottom of the figure.
    fig.text(
        0.5, 0.004,
        PAPER_REF_FULL,
        ha="center", va="bottom", fontsize=7.5,
        fontstyle="italic", color="#666",
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(OUT_TABLE, dpi=180, bbox_inches="tight")
    print(f"wrote {OUT_TABLE}")


def plot_models_overview_table() -> None:
    """Architecture-overview table — one row per variant with the static
    spec sheet (params, int8 weight memory, MACs, paper-conv Ops) plus
    the PC-side fp32 top-1 accuracy from the training metrics. Pure
    "what is this model?" reference; no on-device measurements.

    Columns (6):
        Model | Params | Wt. KiB (int8) | MACs (M) | Ops paper conv. (M)
              | fp32 acc (PC)
    """
    headers = [
        "Model", "Params", "Wt. KiB\n(int8)", "MACs\n(M)",
        "Ops paper\nconv. (M)", "fp32 acc\n(PC)",
    ]

    rows: list[list[str]] = []
    for (name, params, kib, macs_M, fp32, *_rest) in TABLE_ROWS:
        rows.append([
            name,
            f"{params:,}",
            f"{kib:.1f}",
            f"{macs_M:.2f}",
            f"{macs_M * 2:.2f}",
            f"{fp32:.2f}%",
        ])

    n_cols = len(headers)
    n_rows = len(rows)
    fig_w = 11.0
    fig_h = 0.55 + 0.50 * (n_rows + 1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    # 6 columns, sum ≈ 1.0 — wider for the name and accuracy.
    col_widths = [0.22, 0.13, 0.13, 0.12, 0.16, 0.13]

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        colWidths=col_widths,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.65)

    # Header styling — same slate-blue used by max78000_results_table.png.
    for c in range(n_cols):
        cell = table[0, c]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", weight="bold")
        cell.set_height(cell.get_height() * 1.25)

    # Zebra striping + colour the fp32-accuracy column by magnitude.
    fp32_col = 5
    fp32_min = min(float(r[fp32_col].rstrip("%")) for r in rows)
    fp32_max = max(float(r[fp32_col].rstrip("%")) for r in rows)
    span = max(fp32_max - fp32_min, 1.0)

    for r in range(1, n_rows + 1):
        for c in range(n_cols):
            cell = table[r, c]
            cell.set_facecolor("#f5f7fa" if r % 2 == 0 else "white")
        # fp32 acc tint — cream → green as we approach the top accuracy.
        val = float(rows[r - 1][fp32_col].rstrip("%"))
        frac = (val - fp32_min) / span
        table[r, fp32_col].set_facecolor(
            (0.99 - 0.44 * frac, 0.95 - 0.10 * frac, 0.65 - 0.10 * frac)
        )
        table[r, fp32_col].set_text_props(weight="bold")
        # Left-align the model-name column for readability.
        table[r, 0].set_text_props(ha="left")
        table[r, 0].PAD = 0.04

    plt.suptitle(
        "Architecture overview — CIFAR-10 models (PC-trained fp32 reference)",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.025,
        "Params, Wt. KiB (int8), MACs: static model specs.   "
        "Ops paper conv. = 2 × MACs (1 MAC = 2 OPS, paper convention).   "
        "fp32 acc = PC-trained top-1 on CIFAR-10 test set.",
        ha="center", va="bottom", fontsize=8.5,
        fontstyle="italic", color="#444",
    )

    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(OUT_OVERVIEW, dpi=180, bbox_inches="tight")
    print(f"wrote {OUT_OVERVIEW}")


def plot_training_specs_table() -> None:
    """Per-variant training-hyperparameter table.

    Columns: Variant | Optimizer | LR (fp32) | LR (QAT) | LR (prune)
             | Scheduler | Batch | Epochs (fp32 / QAT)

    Numbers extracted from `pc-implementation/train_pc_models.sh`. The
    `baseline_5x5` variant is intentionally omitted (5×5 kernels are
    not MAX78000-portable so it lives outside the deployment story).
    """
    headers = [
        "Variant", "Optimizer", "LR (fp32)", "LR (QAT)", "LR (prune)",
        "Scheduler", "Batch", "Epochs\n(fp32 / QAT)",
    ]

    # All non-mininet variants share the same recipe (constant LR, batch 100,
    # 80/40 epochs). mininet uses its own (cosine, batch 64, higher LR).
    rows = [
        ["baseline",         "Adam", "1e-3", "5e-4", "1e-4",
         "constant", "100", "80 / 40"],
        ["improved",         "Adam", "1e-3", "5e-4", "1e-4",
         "constant", "100", "80 / 40"],
        ["deeper",           "Adam", "1e-3", "5e-4", "1e-4",
         "constant", "100", "80 / 40"],
        ["VGG-Micro-7",      "Adam", "5e-3", "1e-4", "1e-4",
         "cosine",   "64",  "80 / 40"],
        ["NAS-EdgeNet-10",   "Adam", "1e-3", "5e-4", "1e-4",
         "constant", "100", "80 / 40"],
        ["ResSimpleNet-14",  "Adam", "1e-3", "5e-4", "1e-4",
         "constant", "100", "80 / 40"],
    ]

    n_cols = len(headers)
    n_rows = len(rows)
    fig_w = 13.0
    fig_h = 0.55 + 0.55 * (n_rows + 1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    col_widths = [0.18, 0.09, 0.10, 0.10, 0.10, 0.11, 0.08, 0.14]

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        colWidths=col_widths,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.7)

    # Header — same slate-blue as the other tables for visual parity.
    for c in range(n_cols):
        cell = table[0, c]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", weight="bold")
        cell.set_height(cell.get_height() * 1.25)

    # Zebra stripes + highlight the row that breaks the pattern.
    # mininet/VGG-Micro-7 has its own recipe — tint the whole row pale
    # amber so it visually pops as "this one's different".
    accent_row_name = "VGG-Micro-7"
    for r in range(1, n_rows + 1):
        row_name = rows[r - 1][0]
        for c in range(n_cols):
            cell = table[r, c]
            if row_name == accent_row_name:
                cell.set_facecolor("#fff4d6")   # pale amber accent
            elif r % 2 == 0:
                cell.set_facecolor("#f5f7fa")
            else:
                cell.set_facecolor("white")
        # Left-align variant name; bold the accent row.
        table[r, 0].set_text_props(
            ha="left",
            weight="bold" if row_name == accent_row_name else "normal",
        )
        table[r, 0].PAD = 0.04

    plt.suptitle(
        "PC training hyperparameters — per-variant recipe",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.020,
        "All variants use Adam + CIFAR-10 augmentation + CrossEntropy loss.   "
        "QAT runs 40 epochs total: 5 epochs fp32 warmup + 35 epochs fake-quant "
        "fine-tune at half the fp32 LR.   "
        "Prune phase: 50% global L1-unstructured + 10 epochs fine-tune "
        "(cosine, Adam, lr = 1e-4).   "
        "VGG-Micro-7 (highlighted) uses its own recipe (cosine + 5× LR + "
        "batch 64 + weight decay 1e-4) because its deeper stack benefits from "
        "regularization the shallower nets don't need.",
        ha="center", va="bottom", fontsize=8.5,
        fontstyle="italic", color="#444", wrap=True,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    fig.savefig(OUT_TRAINING, dpi=180, bbox_inches="tight")
    print(f"wrote {OUT_TRAINING}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    xs = [m[1] for m in MODELS]
    ys = [m[2] for m in MODELS]
    plot_scatter(xs, ys)
    plot_bubble(xs, ys)
    plot_max78000_latency()
    plot_results_table()
    plot_models_overview_table()
    plot_training_specs_table()


if __name__ == "__main__":
    main()
