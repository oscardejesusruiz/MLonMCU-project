"""Per-model report: confusion matrix, ROC curves, loss/acc curves.

Usage:
    uv run python -m scripts.plot_model_report baseline_3x3
    uv run python -m scripts.plot_model_report baseline_3x3 --variant int8
    uv run python -m scripts.plot_model_report all          # every tag with predictions
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

# Resolve repo root from cwd so this script works from both pc-implementation/
# and max78000-implementation/ (the latter symlinks to this file).
# Override with --root if you want to invoke from elsewhere.
_DEFAULT_ROOT = Path.cwd()
REPO_ROOT = _DEFAULT_ROOT
REPORT_DIR = REPO_ROOT / "reports"
PRED_DIR = REPORT_DIR / "predictions"
FIG_DIR = REPORT_DIR / "figures"

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def plot_confusion(y_true, y_pred, out_path, title):
    cm = confusion_matrix(y_true, y_pred, labels=range(10), normalize="true")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(10), CIFAR10_CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(10), CIFAR10_CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(10):
        for j in range(10):
            v = cm[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if v > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_roc(y_true, y_probs, out_path, title):
    y_true_oh = np.eye(10)[y_true]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    aucs = []
    for c in range(10):
        fpr, tpr, _ = roc_curve(y_true_oh[:, c], y_probs[:, c])
        a = auc(fpr, tpr); aucs.append(a)
        ax.plot(fpr, tpr, lw=1, label=f"{CIFAR10_CLASSES[c]} (AUC={a:.3f})")
    # macro-avg
    fpr_grid = np.linspace(0, 1, 200)
    mean_tpr = np.zeros_like(fpr_grid)
    for c in range(10):
        fpr, tpr, _ = roc_curve(y_true_oh[:, c], y_probs[:, c])
        mean_tpr += np.interp(fpr_grid, fpr, tpr)
    mean_tpr /= 10
    ax.plot(fpr_grid, mean_tpr, "k--", lw=2,
            label=f"macro avg (AUC={auc(fpr_grid, mean_tpr):.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=0.5)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title(title); ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return float(np.mean(aucs))


def plot_confidence(y_true, y_pred, y_probs, out_path, title):
    """Histogram of the max softmax probability per prediction, split by
    correct vs incorrect. A well-calibrated model has incorrect predictions
    skewed toward low confidence and correct ones toward high confidence."""
    conf = y_probs.max(axis=1)
    correct = (y_pred == y_true)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 1, 41)
    ax.hist(conf[correct],  bins=bins, alpha=0.6, label=f"correct (n={correct.sum()})", color="#2E7D32")
    ax.hist(conf[~correct], bins=bins, alpha=0.6, label=f"incorrect (n={(~correct).sum()})", color="#C62828")
    ax.set_xlabel("max softmax probability (confidence)")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pr_curves(y_true, y_probs, out_path, title):
    """Per-class precision-recall curves (one-vs-rest) + average precision."""
    y_true_oh = np.eye(10)[y_true]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    aps = []
    for c in range(10):
        precision, recall, _ = precision_recall_curve(y_true_oh[:, c], y_probs[:, c])
        ap = average_precision_score(y_true_oh[:, c], y_probs[:, c])
        aps.append(ap)
        ax.plot(recall, precision, lw=1, label=f"{CIFAR10_CLASSES[c]} (AP={ap:.3f})")
    macro_ap = float(np.mean(aps))
    ax.set_xlabel("recall"); ax.set_ylabel("precision")
    ax.set_title(f"{title}   (macro mAP = {macro_ap:.4f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return macro_ap


def plot_curves(history, out_loss, out_acc, tag):
    epochs = range(1, len(history["train_loss"]) + 1)
    # loss
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, history["train_loss"], label="train")
    if "val_loss" in history and history["val_loss"]:
        ax.plot(epochs, history["val_loss"], label="val")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title(f"{tag} — loss"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out_loss, dpi=150); plt.close(fig)
    # acc
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [a * 100 for a in history["train_acc"]], label="train")
    ax.plot(epochs, [a * 100 for a in history["val_acc"]], label="val")
    ax.set_xlabel("epoch"); ax.set_ylabel("accuracy (%)")
    ax.set_title(f"{tag} — accuracy"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out_acc, dpi=150); plt.close(fig)


def run_one(tag: str, variant: str) -> None:
    metrics_path = REPORT_DIR / f"{tag}_metrics.json"
    pred_path = PRED_DIR / f"{tag}.npz"
    if not metrics_path.exists():
        print(f"skip {tag}: no metrics file"); return
    if not pred_path.exists():
        print(f"skip {tag}: no predictions ({pred_path})"); return

    metrics = json.loads(metrics_path.read_text())
    data = np.load(pred_path)
    y_true = data["y_true"]
    y_pred = data[f"{variant}_y_pred"]
    y_probs = data[f"{variant}_y_probs"]

    out_dir = FIG_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion(y_true, y_pred, out_dir / f"{variant}_confusion.png",
                   f"{tag} — {variant} confusion (normalized)")
    auroc = plot_roc(y_true, y_probs, out_dir / f"{variant}_roc.png",
                     f"{tag} — {variant} ROC (one-vs-rest)")
    plot_pr_curves(y_true, y_probs, out_dir / f"{variant}_pr.png",
                   f"{tag} — {variant} Precision-Recall")
    plot_confidence(y_true, y_pred, y_probs,
                    out_dir / f"{variant}_confidence.png",
                    f"{tag} — {variant} confidence distribution")
    plot_curves(metrics["history"],
                out_dir / "loss_curve.png", out_dir / "acc_curve.png", tag)

    mAP = float(average_precision_score(np.eye(10)[y_true], y_probs, average="macro"))
    acc = float((y_pred == y_true).mean())
    print(
        f"{tag} [{variant}] — "
        f"params={metrics['params']:,}  MACs={metrics['macs']/1e6:.2f}M  "
        f"acc={acc*100:.2f}%  mAP={mAP:.4f}  macro AUROC={auroc:.4f}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("tag", help="run tag, or 'all'")
    p.add_argument("--variant", choices=["fp32", "int8"], default="fp32")
    p.add_argument("--root", type=Path, default=None,
                   help="override repo root (defaults to cwd). The script reads "
                        "<root>/reports/{predictions,figures} and <root>/reports/<tag>_metrics.json.")
    args = p.parse_args()

    if args.root is not None:
        global REPO_ROOT, REPORT_DIR, PRED_DIR, FIG_DIR
        REPO_ROOT = args.root.resolve()
        REPORT_DIR = REPO_ROOT / "reports"
        PRED_DIR = REPORT_DIR / "predictions"
        FIG_DIR = REPORT_DIR / "figures"

    FIG_DIR.mkdir(exist_ok=True, parents=True)
    if args.tag == "all":
        tags = sorted({p.stem for p in PRED_DIR.glob("*.npz")})
        if not tags:
            print("no predictions found"); sys.exit(1)
        for t in tags:
            run_one(t, args.variant)
    else:
        run_one(args.tag, args.variant)


if __name__ == "__main__":
    main()