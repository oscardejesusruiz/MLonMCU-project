"""Evaluate PC checkpoints in float and int8 modes — MAX78000-realistic PTQ.

For each (variant × training_mode) where training_mode ∈ {fp32, qat}:
  1. Load the .pt checkpoint from trained_models/
  2. Evaluate on CIFAR-10 test set with FLOAT weights → top1_float
  3. Apply post-training int8 quantization that mirrors the MAX78000 deploy
     pipeline (see `training/quantize.py::quantize_model_ptq`):
       • Fold every Conv2d→BatchNorm2d pair into the preceding conv
         (this is what `bn_fuser_v2.py` does at device synthesis)
       • Quantize the post-fold Conv2d/Linear weights to symmetric per-tensor
         int8 (power-of-two scales, matching CMSIS-NN q7 convention)
       • Calibrate activation scales from a handful of train batches
       • Install permanent fake-quant pre-hooks on every Conv/Linear input
  4. Evaluate on CIFAR-10 test set with the quantized model → top1_int8
  5. Record MACs, params, weight bytes

Important: this is the SAME quantization function the main training script
(`run_experiment.py`) uses for fp32 + PTQ metrics, so the int8 numbers
reported here are directly comparable to the `int8_test_acc` field in the
per-model JSON. fp32-trained models should now show a real drop comparable
to what the MAX78000 reports at deployment (5-25 pp depending on the
architecture's sensitivity to BN-fold-induced per-channel outliers).
QAT-trained models should still recover most of that drop.

Output (under `reports/int8_eval/`):
  summary.txt         — per-model metrics, plain text table
  acc_vs_gops.png     — Top-1 accuracy vs G-Ops plot

Run with:
    uv run python -m scripts.eval_int8
or:
    uv run python scripts/eval_int8.py --device cpu --batch-size 256
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Make the local `training` package importable when the script is run
# both as `python scripts/eval_int8.py` and `python -m scripts.eval_int8`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.data import get_loaders
from training.models import build_model
from training.quantize import QuantConfig, quantize_model_ptq
from training.utils import compute_stats


# Variants and training-mode suffixes to evaluate. Only fp32 and qat — we
# deliberately skip *_ptq (already quantized; we re-derive from fp32 here
# with a consistent config) and *_prune50 (orthogonal axis).
VARIANTS = ["baseline", "deeper", "mininet", "improved",
            "nascifarnet", "ressimplenet"]
MODES = ["fp32", "qat"]


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    """Top-1 accuracy [%] over `loader`."""
    model = model.to(device).eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / max(total, 1)


def load_checkpoint(model: nn.Module, ckpt_path: Path) -> nn.Module:
    """Load a `.pt` into `model` in-place. Handles dict-wrapped and bare
    state_dict layouts, and strips DDP `module.` prefixes."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        sd = state["state_dict"]
    elif isinstance(state, dict) and "model" in state:
        sd = state["model"]
    elif isinstance(state, dict):
        sd = state if all(torch.is_tensor(v) for v in state.values()) else state
    else:
        sd = state
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"    ! missing keys: {len(missing)} (first: {missing[:3]})",
              file=sys.stderr)
    if unexpected:
        print(f"    ! unexpected keys: {len(unexpected)} "
              f"(first: {unexpected[:3]})", file=sys.stderr)
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cpu",
                    help="Inference device (cpu/cuda/mps). cpu is fine — "
                         "CIFAR-10 test set is small.")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-calib-batches", type=int, default=20,
                    help="Train batches used by quantize_model_ptq to "
                         "calibrate activation scales. Matches the default "
                         "in QuantConfig.")
    ap.add_argument("--no-power-of-two", dest="power_of_two",
                    action="store_false",
                    help="Disable power-of-two scale rounding (use exact "
                         "max-abs scales). Default: power-of-two ON, "
                         "matches CMSIS-NN q7 convention.")
    ap.set_defaults(power_of_two=True)
    ap.add_argument("--out-dir", type=Path,
                    default=REPO_ROOT / "reports" / "int8_eval")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"device         : {args.device}")
    print(f"batch size     : {args.batch_size}")
    print(f"calib batches  : {args.n_calib_batches} "
          f"(~{args.n_calib_batches * args.batch_size} imgs)")
    print(f"power-of-two   : {args.power_of_two}")
    print(f"out dir        : {args.out_dir}")
    print()

    train_loader, _, test_loader = get_loaders(
        batch_size=args.batch_size, num_workers=2, augment=False)

    qcfg = QuantConfig(
        power_of_two=args.power_of_two,
        n_calib_batches=args.n_calib_batches,
    )

    results: list[dict] = []
    for variant in VARIANTS:
        for mode in MODES:
            tag = f"{variant}_{mode}"
            ckpt = REPO_ROOT / "trained_models" / f"{tag}.pt"
            if not ckpt.exists():
                print(f"[{tag}] skip — no checkpoint at {ckpt.name}")
                continue
            print(f"[{tag}]")

            model = build_model(variant, num_classes=10)
            load_checkpoint(model, ckpt)

            stats = compute_stats(model)

            # 1) Float-weight accuracy (with BN runtime as trained)
            float_acc = evaluate(model, test_loader, args.device)
            print(f"    float acc : {float_acc:6.2f}%")

            # 2) MAX78000-realistic int8 PTQ:
            #    fold BN → quantize weights → calibrate activations → fake-quant
            #    quantize_model_ptq deepcopies the model internally so the
            #    original stays intact for the next iteration.
            model_q, _ = quantize_model_ptq(
                model, calib_loader=train_loader, config=qcfg,
            )
            int8_acc = evaluate(model_q, test_loader, args.device)
            drop = int8_acc - float_acc
            print(f"    int8  acc : {int8_acc:6.2f}%   "
                  f"(Δ = {drop:+.2f} pp)")
            print(f"    params    : {stats.params:,}  "
                  f"size int8 = {stats.weight_bytes_int8 / 1024:.1f} KiB  "
                  f"MACs = {stats.macs / 1e6:.2f} M")

            results.append({
                "tag": tag,
                "variant": variant,
                "mode": mode,
                "params": stats.params,
                "weight_bytes_fp32": stats.weight_bytes_fp32,
                "weight_bytes_int8": stats.weight_bytes_int8,
                "macs": stats.macs,
                "gops": 2.0 * stats.macs / 1e9,
                "float_acc": float_acc,
                "int8_acc": int8_acc,
            })

    if not results:
        print("\nNo checkpoints found — nothing to plot or report.")
        return

    # ---------- TXT report ----------
    txt_path = args.out_dir / "summary.txt"
    with open(txt_path, "w") as f:
        f.write("PC MODEL EVALUATION — float vs MAX78000-realistic int8 PTQ\n")
        f.write("=" * 105 + "\n")
        f.write("Quantization pipeline (training/quantize.py::quantize_model_ptq):\n")
        f.write("  0. BN fold     : every Conv2d→BatchNorm2d pair fused into\n")
        f.write("                   the conv weight/bias (same step the MAX78000\n")
        f.write("                   does via bn_fuser_v2.py at device synthesis)\n")
        f.write("  1. weight quant: symmetric per-tensor int8, "
                f"power-of-two={args.power_of_two}\n")
        f.write("                   scale = max(|w_folded|) / 127\n")
        f.write(f"  2. activations : calibrated over {args.n_calib_batches} train batches "
                f"(~{args.n_calib_batches * args.batch_size} imgs)\n")
        f.write("  3. fake-quant  : pre-forward hooks round each Conv/Linear input\n")
        f.write("                   to the int8 grid before the multiply\n")
        f.write("\n")
        f.write("This matches what the MAX78000 actually deploys — fp32 models\n")
        f.write("will show a real PTQ drop, QAT models should recover most of it.\n")
        f.write("\n")
        f.write(f"{'model':<22} {'params':>10} {'size_fp32':>11} "
                f"{'size_int8':>11} {'MACs/inf':>14} {'MACs/inf [M]':>14} "
                f"{'fp32_acc':>10} {'int8_acc':>10}\n")
        f.write("-" * 110 + "\n")
        for r in results:
            f.write(
                f"{r['tag']:<22} "
                f"{r['params']:>10,} "
                f"{r['weight_bytes_fp32'] / 1024:>10.1f}K "
                f"{r['weight_bytes_int8'] / 1024:>10.1f}K "
                f"{r['macs']:>14,} "
                f"{r['macs'] / 1e6:>14.3f} "
                f"{r['float_acc']:>9.2f}% "
                f"{r['int8_acc']:>9.2f}%\n"
            )
        f.write("\n")
        f.write("Column legend\n")
        f.write("-------------\n")
        f.write("  params         : total learnable parameters\n")
        f.write("  size_fp32      : weight bytes assuming fp32 storage (4 B/param)\n")
        f.write("  size_int8      : weight bytes assuming int8 storage (1 B/param)\n")
        f.write("  MACs/inf       : multiply-accumulate operations per inference\n")
        f.write("                   (raw count; 1 MAC = 1 multiply + 1 add)\n")
        f.write("  MACs/inf [M]   : same number expressed in millions for readability\n")
        f.write("  fp32_acc       : top-1 CIFAR-10 test accuracy with float weights\n")
        f.write("                   AND BN preserved as runtime layer\n")
        f.write("  int8_acc       : top-1 CIFAR-10 test accuracy after BN folding,\n")
        f.write("                   weight quantization, and activation fake-quant\n")
    print(f"\n  → wrote {txt_path}")

    # ---------- Plot: Top-1 acc vs MACs / Inference ----------
    fig, ax = plt.subplots(figsize=(10, 6.5))
    palette = plt.cm.tab10.colors
    color_of = {v: palette[i % 10] for i, v in enumerate(VARIANTS)}
    marker_of = {"fp32": "o", "qat": "s"}   # ○ float-trained, ▪ QAT-trained

    # X-axis: MACs per inference, expressed in millions (M-MACs). This is the
    # canonical TinyML cost unit — one MAC = one multiply-accumulate op. Each
    # inference forwards N MACs through the accelerator; smaller is cheaper.
    for r in results:
        macs_m = r["macs"] / 1e6
        ax.scatter(
            macs_m, r["int8_acc"],
            color=color_of[r["variant"]],
            marker=marker_of[r["mode"]],
            s=150, edgecolors="black", linewidths=0.8, zorder=3,
        )
        ax.annotate(
            r["tag"], (macs_m, r["int8_acc"]),
            xytext=(7, 4), textcoords="offset points",
            fontsize=8, color=color_of[r["variant"]],
        )

    legend_handles = []
    for v in VARIANTS:
        if any(r["variant"] == v for r in results):
            legend_handles.append(
                plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=color_of[v], markersize=9,
                           markeredgecolor="black", label=v)
            )
    legend_handles.append(plt.Line2D([], [], color="none", label=""))
    legend_handles.append(
        plt.Line2D([0], [0], marker="o", color="w", markersize=9,
                   markerfacecolor="lightgray", markeredgecolor="black",
                   label="fp32-trained → BN-fold + PTQ")
    )
    legend_handles.append(
        plt.Line2D([0], [0], marker="s", color="w", markersize=9,
                   markerfacecolor="lightgray", markeredgecolor="black",
                   label="QAT-trained → int8")
    )
    ax.legend(handles=legend_handles, loc="lower right", framealpha=0.95)

    ax.set_xlabel("MACs / Inference  [M]")
    ax.set_ylabel("Top-1 accuracy on CIFAR-10  [%]   (int8 inference)")
    ax.set_title("Accuracy vs Compute — CIFAR-10, MAX78000-realistic int8 PTQ")
    ax.grid(True, alpha=0.3)
    # Linear x-axis starts at 0 so the visual distance reflects the actual
    # MAC ratio between models (e.g. nascifarnet is ~8x more MACs than
    # baseline — that 8x is now visible at a glance, unlike on log-scale).
    ax.set_xlim(left=0)
    fig.tight_layout()

    plot_path = args.out_dir / "acc_vs_macs.png"
    fig.savefig(plot_path, dpi=150)
    print(f"  → wrote {plot_path}")


if __name__ == "__main__":
    main()
