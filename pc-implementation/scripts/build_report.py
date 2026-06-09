"""Aggregate per-experiment metric JSONs into a markdown summary + Pareto plot.

Reads every `<tag>_metrics.json` in `reports/` and produces:
  reports/summary.md       — comparison table + per-experiment notes
  reports/figures/pareto.png — accuracy vs MACs (size = weight bytes)
  reports/figures/training_curves.png — train/test acc per experiment
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)


def load_runs() -> list[dict]:
    runs = []
    for path in sorted(REPORT_DIR.glob("*_metrics.json")):
        runs.append(json.loads(path.read_text()))
    return runs


def write_summary(runs: list[dict]) -> None:
    lines = [
        "# CIFAR-10 on MCU — Phase 1 + 2 results",
        "",
        "| Tag | Model | Params | Wt. KiB (int8) | MACs (M) | Ops paper conv. (M) | fp32 acc | int8 acc | Δ acc | Train time (s) |",
        "|-----|-------|--------|----------------|----------|----------------------|----------|----------|-------|----------------|",
    ]
    for r in runs:
        delta = r["int8_test_acc"] - r["fp32_test_acc"]
        lines.append(
            f"| {r['tag']} | {r['model']} | {r['params']:,} "
            f"| {r['weight_kib_int8']:.1f} | {r['macs']/1e6:.2f} "
            f"| {r['ops_paper_convention']/1e6:.2f} "
            f"| {r['fp32_test_acc']*100:.2f}% | {r['int8_test_acc']*100:.2f}% "
            f"| {delta*100:+.2f}pp | {r['train_time_seconds']:.0f} |"
        )
    lines.append("")
    lines.append("**Reference (Lai et al. 2018):** 24.7 MOps/inference, 87 KB int8 weights, 79.9% int8 accuracy on CIFAR-10.")
    lines.append("")
    lines.append("## Per-experiment notes")
    for r in runs:
        lines.append(f"### {r['tag']}")
        a = r["args"]
        is_qat = r.get("int8_mode") == "qat"
        if is_qat:
            tech = r.get("technique", "qat")
            epochs = a.get("epochs") or a.get("finetune_epochs", "?")
            sparsity_str = ""
            if "actual_sparsity" in r:
                sparsity_str = f", sparsity={r['actual_sparsity']*100:.0f}%"
            lines.append(
                f"- [{tech}] "
                f"lr={a.get('lr','?')}, epochs={epochs}, "
                f"augment={a.get('augment', False)}"
                f"{', QAT switch @ epoch ' + str(a.get('qat_start_epoch')) if a.get('qat_start_epoch') else ''}"
                f"{sparsity_str}"
            )
            lines.append(
                f"- fp32 test acc: **{r['fp32_test_acc']*100:.2f}%**, "
                f"QAT int8 test acc: **{r['int8_test_acc']*100:.2f}%**"
            )
        else:
            tech = r.get("technique", "fp32+ptq")
            epochs = a.get("epochs") or a.get("finetune_epochs", "?")
            lines.append(
                f"- Model: `{r['model']}` [{tech}], "
                f"optimizer: `{a.get('optimizer','-')}`, "
                f"lr={a.get('lr','?')}, wd={a.get('weight_decay','?')}, "
                f"scheduler={a.get('scheduler','none')}, "
                f"augment={a.get('augment', False)}, epochs={epochs}"
            )
            extra = ""
            if tech == "pruning":
                extra = f", sparsity={r.get('actual_sparsity', 0)*100:.0f}%"
            lines.append(
                f"- fp32 test acc: **{r['fp32_test_acc']*100:.2f}%**, "
                f"int8 test acc: **{r['int8_test_acc']*100:.2f}%**{extra}"
            )
        lines.append("")

    (REPORT_DIR / "summary.md").write_text("\n".join(lines))
    print(f"wrote {REPORT_DIR / 'summary.md'}")


def plot_pareto(runs: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in runs:
        macs_m = r["macs"] / 1e6
        ax.scatter(macs_m, r["int8_test_acc"] * 100, s=80, label=r["tag"])
        ax.annotate(
            r["tag"],
            (macs_m, r["int8_test_acc"] * 100),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )
    # Reference baseline marker
    ax.axhline(79.9, color="gray", linestyle="--", linewidth=1, label="Lai et al. (79.9%)")
    ax.set_xlabel("MACs / inference (M)")
    ax.set_ylabel("int8 accuracy (%)")
    ax.set_title("Pareto: accuracy vs compute")
    ax.grid(True, alpha=0.3)
    #ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = FIG_DIR / "pareto.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def plot_training_curves(runs: list[dict]) -> None:
    if not runs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for r in runs:
        h = r["history"]
        epochs = range(1, len(h["val_acc"]) + 1)
        axes[0].plot(epochs, h["train_loss"], label=r["tag"])
        axes[1].plot(epochs, [a * 100 for a in h["val_acc"]], label=r["tag"])
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("train loss"); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("test acc (%)"); axes[1].grid(alpha=0.3)
    axes[1].axhline(79.9, color="gray", linestyle="--", linewidth=1, label="Lai et al.")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "training_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    runs = load_runs()
    if not runs:
        print("no metrics found in reports/")
        return
    write_summary(runs)
    plot_pareto(runs)
    plot_training_curves(runs)


if __name__ == "__main__":
    main()
