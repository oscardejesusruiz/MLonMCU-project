"""Grouped bar plot: fp32 vs int8-PTQ vs int8-QAT accuracy per variant.

Three bars per architecture, all from the MAX78000-realistic int8 simulation
pipeline (`eval_pre_synth.sh`):

  1. fp32 accuracy
        source: reports/_eval_pre_synth/fp32/_acc/col_a_<variant>
        meaning: the fp32-trained model evaluated in fp32 → upper bound

  2. int8 PTQ accuracy
        source: reports/_eval_pre_synth/fp32/_acc/int8_<variant>
        meaning: same fp32 model → BN-folded → quantize.py → ai8x sim eval
        i.e. what naive post-training-quantization gives at MAX78000 deployment

  3. int8 QAT accuracy
        source: reports/_eval_pre_synth/qat/_acc/int8_<variant>
        meaning: QAT-fine-tuned model → BN-folded → quantize.py → ai8x sim eval
        i.e. what QAT-aware deployment gives at MAX78000

To produce the input data, first run:
    ./eval_pre_synth.sh          # fills reports/_eval_pre_synth/fp32/_acc/
    ./eval_pre_synth.sh qat      # fills reports/_eval_pre_synth/qat/_acc/

Then:
    uv run python plot_acc_comparison.py        # or python plot_acc_comparison.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# Display names used in the final report (paper-style architecture names
# in place of the internal tag). Internal tags stay everywhere on the data
# side (file paths, JSON keys) — only axis labels and annotations get the
# pretty rename.
DISPLAY_NAME = {
    "baseline":     "baseline",
    "improved":     "improved",
    "deeper":       "deeper",
    "mininet":      "VGG-Micro-7",
    "nascifarnet":  "NAS-EdgeNet-10",
    "ressimplenet": "ResSimpleNet-14",
}


def _display(variant: str) -> str:
    """Return the pretty display name for a variant tag, falling back to
    the tag itself if no rename is registered."""
    return DISPLAY_NAME.get(variant, variant)


def _read_acc(path: Path) -> float | None:
    """Read a one-line accuracy file. Returns None if missing or '-'/'?'."""
    if not path.exists():
        return None
    txt = path.read_text().strip()
    if txt in ("", "-", "?"):
        return None
    try:
        return float(txt)
    except ValueError:
        return None


def _load_macs(estimate_path: Path) -> dict[str, int]:
    """Return {variant: macs_per_inference} loaded from estimate.json.

    estimate.json is the per-variant static-metrics catalogue produced by
    scripts/estimate_metrics.py — see ../README.md §estimate.json. Returns
    an empty dict if the file is missing so the caller can fall back to
    skipping the MACs plot rather than crashing.
    """
    if not estimate_path.exists():
        return {}
    try:
        data = json.loads(estimate_path.read_text())
    except json.JSONDecodeError:
        return {}
    return {
        name: int(entry["macs"])
        for name, entry in data.items()
        if isinstance(entry, dict) and "macs" in entry
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--variants", nargs="+",
        default=["baseline", "improved", "deeper", "mininet",
                 "nascifarnet", "ressimplenet"],
        help="Variants to plot, left-to-right order.",
    )
    ap.add_argument(
        "--in-dir", type=Path,
        default=REPO_ROOT / "reports" / "_eval_pre_synth",
        help="Root dir holding fp32/_acc and qat/_acc subfolders.",
    )
    ap.add_argument(
        "--out", type=Path,
        default=REPO_ROOT / "reports" / "fig_acc_comparison.png",
        help="Output PNG path for the 3-bar (fp32 / PTQ / QAT) figure.",
    )
    ap.add_argument(
        "--out-macs", type=Path,
        default=REPO_ROOT / "reports" / "fig_acc_vs_macs.png",
        help="Output PNG path for the MACs vs int8-accuracy figure (one "
             "PTQ point and one QAT point per variant, joined by a thin "
             "vertical line showing the QAT recovery).",
    )
    ap.add_argument(
        "--estimate", type=Path,
        default=REPO_ROOT / "reports" / "models_estimation.json",
        help="JSON with per-variant MACs/inference. Generate it via "
             "`python scripts/estimate_metrics.py`. If the file is "
             "missing, the MACs vs accuracy plot is skipped silently.",
    )
    ap.add_argument(
        "--title", default=None,
        help="Plot title (default: auto).",
    )
    args = ap.parse_args()

    fp32_dir = args.in_dir / "fp32" / "_acc"
    qat_dir = args.in_dir / "qat" / "_acc"

    fp32_vals, ptq_vals, qat_vals = [], [], []
    for v in args.variants:
        fp32_vals.append(_read_acc(fp32_dir / f"col_a_{v}"))   # fp32 acc
        ptq_vals.append(_read_acc(fp32_dir / f"int8_{v}"))     # fp32→PTQ→int8
        qat_vals.append(_read_acc(qat_dir / f"int8_{v}"))      # QAT→int8

    # ---------- bar plot ----------
    n = len(args.variants)
    x = np.arange(n)
    width = 0.27

    fig, ax = plt.subplots(figsize=(max(10, n * 1.6), 6.0))

    def _safe(values):
        return [v if v is not None else 0.0 for v in values]

    bars_fp32 = ax.bar(
        x - width, _safe(fp32_vals), width,
        label="fp32 reference",
        color="#2ca02c", edgecolor="black", linewidth=0.5, zorder=3,
    )
    bars_ptq = ax.bar(
        x, _safe(ptq_vals), width,
        label="int8 PTQ",# (fp32 ckpt → BN-fold → quantize)",
        color="#d62728", edgecolor="black", linewidth=0.5, zorder=3,
    )
    bars_qat = ax.bar(
        x + width, _safe(qat_vals), width,
        label="int8 QAT",# (QAT ckpt → BN-fold → quantize)",
        color="#1f77b4", edgecolor="black", linewidth=0.5, zorder=3,
    )

    # Annotate each bar; show "n/a" where data is missing
    for bars, values in [
        (bars_fp32, fp32_vals),
        (bars_ptq, ptq_vals),
        (bars_qat, qat_vals),
    ]:
        for bar, val in zip(bars, values):
            cx = bar.get_x() + bar.get_width() / 2
            if val is None:
                ax.text(cx, 2, "n/a",
                        ha="center", va="bottom", fontsize=8, color="grey")
            else:
                ax.text(cx, val + 0.6, f"{val:.1f}",
                        ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([_display(v) for v in args.variants],
                       fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("Top-1 accuracy on CIFAR-10  [%]", fontsize=11)
    ax.set_title(args.title or
                 "CIFAR-10 — fp32 reference vs int8 PTQ vs int8 QAT"
                 "",
                 fontsize=12)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3, linestyle="--", zorder=0)
    ax.legend(loc="lower left", framealpha=0.95, fontsize=10)
    ax.tick_params(axis="x", which="both", length=0)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")

    # ====================================================================
    # Second plot: int8 accuracy vs MACs/inference (PTQ and QAT overlaid)
    # ====================================================================
    # Both points (PTQ and QAT) for the same variant sit at the SAME x
    # (same architecture → same MACs), only the y differs. A thin vertical
    # line between them visualises the QAT recovery.
    macs_table = _load_macs(args.estimate)
    missing_macs = [v for v in args.variants if v not in macs_table]
    if missing_macs:
        print(f"  ! skipping MACs plot for: {missing_macs} "
              f"(no entry in {args.estimate.name}; re-run "
              f"scripts/estimate_metrics.py)")

    plot_rows = [
        (v, macs_table[v] / 1e6, ptq, qat)
        for v, ptq, qat in zip(args.variants, ptq_vals, qat_vals)
        if v in macs_table
    ]

    if not plot_rows:
        print("  no variants have both MACs and accuracy data — skipping MACs plot")
    else:
        fig2, ax2 = plt.subplots(figsize=(10, 6.5))

        # Pull the same per-variant colour palette as the bar plot so a
        # quick eye-flick between the two figures reads identical variants.
        palette = plt.cm.tab10.colors
        color_of = {v: palette[i % 10] for i, v in enumerate(args.variants)}

        for v, macs_m, ptq, qat in plot_rows:
            colour = color_of[v]

            # Thin connector showing QAT vs PTQ delta
            if ptq is not None and qat is not None:
                ax2.plot(
                    [macs_m, macs_m], [ptq, qat],
                    color=colour, linestyle="--", linewidth=1, alpha=0.5,
                    zorder=2,
                )

            # PTQ point (hollow circle)
            if ptq is not None:
                ax2.scatter(
                    macs_m, ptq,
                    facecolor="white", edgecolor=colour, linewidths=2,
                    marker="o", s=140, zorder=3,
                )
            # QAT point (filled triangle)
            if qat is not None:
                ax2.scatter(
                    macs_m, qat,
                    color=colour, edgecolor="black", linewidths=0.6,
                    marker="^", s=180, zorder=4,
                )

            # Annotation: pretty display name to the right of the QAT point
            label_y = qat if qat is not None else ptq
            if label_y is not None:
                ax2.annotate(
                    _display(v), (macs_m, label_y),
                    xytext=(8, 0), textcoords="offset points",
                    fontsize=9, va="center", color=colour,
                )

        # Two-part legend: marker semantic + variant colour
        legend_handles = [
            plt.Line2D([0], [0], marker="o", color="w", markersize=10,
                       markerfacecolor="white", markeredgecolor="grey",
                       markeredgewidth=2,
                       label="int8 PTQ (fp32 ckpt → BN-fold → quantize)"),
            plt.Line2D([0], [0], marker="^", color="w", markersize=11,
                       markerfacecolor="grey", markeredgecolor="black",
                       label="int8 QAT (QAT ckpt → BN-fold → quantize)"),
            plt.Line2D([], [], color="grey", linestyle="--", linewidth=1,
                       alpha=0.6, label="QAT recovery (same architecture)"),
        ]
        ax2.legend(handles=legend_handles, loc="lower right",
                   framealpha=0.95, fontsize=10)

        ax2.set_xlabel("MACs / Inference  [M]", fontsize=11)
        ax2.set_ylabel("Top-1 accuracy on CIFAR-10  [%]   (int8 inference)",
                       fontsize=11)
        ax2.set_title(
            "Int8 deployment accuracy vs compute — PTQ vs QAT per variant",
            fontsize=12,
        )
        ax2.set_ylim(0, 100)
        ax2.set_xlim(left=0)
        ax2.grid(True, alpha=0.3, linestyle="--", zorder=0)

        fig2.tight_layout()
        args.out_macs.parent.mkdir(parents=True, exist_ok=True)
        fig2.savefig(args.out_macs, dpi=150)
        print(f"wrote {args.out_macs}")

    # ---------- text summary to stdout ----------
    print()
    print(f"{'variant':<14} {'fp32':>8} {'int8 PTQ':>10} {'int8 QAT':>10} "
          f"{'PTQ drop':>10} {'QAT vs PTQ':>12}")
    print("-" * 75)
    for v, f, p, q in zip(args.variants, fp32_vals, ptq_vals, qat_vals):
        f_s = f"{f:6.2f}%" if f is not None else "    n/a"
        p_s = f"{p:6.2f}%" if p is not None else "    n/a"
        q_s = f"{q:6.2f}%" if q is not None else "    n/a"
        if f is not None and p is not None:
            ptq_drop = f"{f - p:+7.2f}pp"
        else:
            ptq_drop = "      —"
        if q is not None and p is not None:
            qat_rec = f"{q - p:+7.2f}pp"
        else:
            qat_rec = "      —"
        print(f"{v:<14} {f_s:>8} {p_s:>10} {q_s:>10} "
              f"{ptq_drop:>10} {qat_rec:>12}")


if __name__ == "__main__":
    main()
