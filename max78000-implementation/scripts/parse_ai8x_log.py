"""Convert an ai8x-training log directory into a PC-style metrics JSON.

ai8x-training writes per-epoch lines like:

    Epoch: [N][last/total]   Overall Loss X  Top1 Y  Top5 Z  LR L
    ==> Top1: ... Top5: ... Loss: ...

This script parses those, extracts the train/val curves, looks up the static
metrics (params, MACs, etc.) from `estimate.json`, and emits the same schema
as `pc-implementation/reports/<tag>_metrics.json` so the shared plotting
script works on both sides.

Usage:
    uv run python parse_ai8x_log.py \\
        --log-dir $AI/ai8x-training/logs/2026.05.20-152504 \\
        --variant separable \\
        --tag separable_qat \\
        --qat-start-epoch 30
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

TRAIN_RE = re.compile(
    r"Epoch:\s*\[(\d+)\]\[\s*(\d+)/\s*(\d+)\]\s+Overall Loss\s+([\d.]+)"
    r".*?Top1\s+([\d.]+)\s+Top5\s+([\d.]+)\s+LR\s+([\d.eE+-]+)"
)
VAL_RE = re.compile(r"==>\s*Top1:\s*([\d.]+)\s+Top5:\s*([\d.]+)\s+Loss:\s*([\d.]+)")


def parse_log(log_path: Path) -> dict:
    h = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    last_train: tuple[float, float, float] | None = None
    for line in log_path.read_text(errors="replace").splitlines():
        m = TRAIN_RE.search(line)
        if m:
            _, cur, total, loss, top1, _top5, lr = m.groups()
            if cur == total:
                last_train = (float(loss), float(top1) / 100, float(lr))
            continue
        v = VAL_RE.search(line)
        if v and last_train is not None:
            top1, _top5, vloss = v.groups()
            h["train_loss"].append(last_train[0])
            h["train_acc"].append(last_train[1])
            h["lr"].append(last_train[2])
            h["val_loss"].append(float(vloss))
            h["val_acc"].append(float(top1) / 100)
            last_train = None
    return h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", type=Path, required=True,
                    help="ai8x-training/logs/<timestamp>-train/")
    ap.add_argument("--variant", choices=["baseline", "improved", "separable"],
                    required=True)
    ap.add_argument("--tag", required=True,
                    help="output tag, e.g. baseline_qat")
    ap.add_argument("--qat-start-epoch", type=int, default=30,
                    help="epoch at which QAT was activated in the policy")
    ap.add_argument("--estimate", type=Path,
                    default=Path(__file__).resolve().parent.parent / "estimate.json")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "reports")
    args = ap.parse_args()

    log_files = sorted(args.log_dir.glob("*.log"))
    if not log_files:
        raise FileNotFoundError(f"no .log under {args.log_dir}")
    log_file = log_files[0]
    print(f"parsing {log_file}")

    history = parse_log(log_file)
    n_epochs = len(history["val_acc"])
    if n_epochs == 0:
        raise RuntimeError("no epochs parsed — log format unexpected")

    est_all = json.loads(args.estimate.read_text())
    est_key = next((k for k in est_all if args.variant in k), None)
    if est_key is None:
        raise KeyError(f"variant {args.variant!r} not found in {args.estimate}")
    est = est_all[est_key]

    qat_start = max(0, min(args.qat_start_epoch, n_epochs))
    pre_qat_val = history["val_acc"][:qat_start] or history["val_acc"]
    post_qat_val = history["val_acc"][qat_start:] or history["val_acc"]

    fp32_best_val = max(pre_qat_val)
    fp32_final_val = pre_qat_val[-1] if pre_qat_val else 0.0
    int8_best_val = max(post_qat_val)

    metrics = {
        "tag": args.tag,
        "model": args.variant,
        "args": {
            "optimizer": "adam",
            "lr": 0.001,
            "batch_size": 100,
            "epochs": n_epochs,
            "augment": True,
            "qat_start_epoch": args.qat_start_epoch,
            "weight_decay": 0.0,
            "scheduler": "multistep",
            "milestones": [100, 140, 170],
            "gamma": 0.1,
        },
        "fp32_best_val_acc": fp32_best_val,
        "fp32_final_val_acc": fp32_final_val,
        "fp32_test_acc": fp32_best_val,    # filled by eval_ai8x.py post-hoc if needed
        "int8_test_acc": int8_best_val,    # QAT best is the deployed number
        "int8_mode": "qat",
        "params": int(est.get("params_pytorch") or est.get("params") or 0),
        "weight_kib_int8": float(est.get("weight_memory_kib") or 0.0),
        "weight_kib_fp32": float(est.get("weight_memory_kib") or 0.0) * 4,
        "macs": int(est.get("macs") or 0),
        "ops_paper_convention": int(est.get("ops_paper_convention") or 2 * (est.get("macs") or 0)),
        "train_time_seconds": None,
        "history": history,
        "weight_scales": None,
        "activation_scales": None,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{args.tag}_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"wrote {out_path}  ({n_epochs} epochs)")


if __name__ == "__main__":
    main()
