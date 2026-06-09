"""Magnitude-based pruning + fine-tune.

Loads a fp32 checkpoint, applies L1-unstructured global pruning at a target
sparsity, then fine-tunes the remaining weights for a few epochs.

Usage:
    uv run python -m scripts.run_pruning \\
        --model separable --base-ckpt trained_models/separable_3x3.pt \\
        --sparsity 0.5 --finetune-epochs 10 --tag separable_prune50
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from training.data import get_loaders
from training.engine import evaluate as eval_full
from training.models import build_model
from training.utils import compute_stats, pick_device


REPO_ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = REPO_ROOT / "trained_models"
REPORT_DIR = REPO_ROOT / "reports"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   choices=["baseline", "baseline_5x5", "improved",
                            "mininet", "deeper", "ressimplenet", "nascifarnet"])
    p.add_argument("--base-ckpt", type=Path, required=True,
                   help="fp32 starting checkpoint (.pt)")
    p.add_argument("--sparsity", type=float, default=0.5,
                   help="target sparsity (0..1). 0.5 = prune half of weights")
    p.add_argument("--finetune-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--input-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", required=True)
    return p.parse_args()


def collect_prune_params(model: nn.Module) -> list[tuple[nn.Module, str]]:
    """Collect (module, 'weight') tuples for every Conv2d + Linear."""
    out = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            out.append((m, "weight"))
    return out


def sparsity_report(prune_targets) -> float:
    total = nonzero = 0
    for mod, name in prune_targets:
        w = getattr(mod, name)
        total += w.numel()
        nonzero += (w != 0).sum().item()
    return 1.0 - (nonzero / total)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device()
    print(f"device: {device}")
    print(f"loading base model {args.model} from {args.base_ckpt}")

    train_loader, val_loader, test_loader = get_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
        augment=True, input_size=args.input_size,
    )

    model = build_model(args.model)
    model.load_state_dict(torch.load(args.base_ckpt, map_location="cpu",
                                     weights_only=True))
    model = model.to(device)

    stats = compute_stats(build_model(args.model))
    base_acc = eval_full(model, test_loader, device)
    print(f"base test acc (before prune): {base_acc*100:.2f}%")

    # ---------- 1. Global magnitude pruning ---------------------------------
    targets = collect_prune_params(model)
    prune.global_unstructured(
        targets, pruning_method=prune.L1Unstructured, amount=args.sparsity,
    )
    actual_sparsity = sparsity_report(targets)
    pre_ft_acc = eval_full(model, test_loader, device)
    print(f"sparsity after prune: {actual_sparsity*100:.2f}%")
    print(f"test acc immediately after prune (no fine-tune): {pre_ft_acc*100:.2f}%")

    # ---------- 2. Fine-tune ------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.finetune_epochs)
    crit = nn.CrossEntropyLoss()
    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": [], "lr": [], "epoch_time": []}
    best_acc = 0.0
    t0 = time.time()

    for epoch in range(1, args.finetune_epochs + 1):
        model.train()
        loss_sum = correct = total = 0
        epoch_t = time.time()
        bar = tqdm(train_loader, desc=f"ft {epoch}/{args.finetune_epochs}", leave=False)
        for x, y in bar:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = crit(logits, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        scheduler.step()

        val_out = eval_full(model, val_loader, device, return_loss=True)
        history["train_loss"].append(loss_sum/total)
        history["train_acc"].append(correct/total)
        history["val_loss"].append(val_out["loss"])
        history["val_acc"].append(val_out["acc"])
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["epoch_time"].append(time.time() - epoch_t)
        best_acc = max(best_acc, val_out["acc"])
        print(f"[{epoch:3d}/{args.finetune_epochs}] "
              f"loss={loss_sum/total:.4f} train_acc={correct/total:.4f} "
              f"val_acc={val_out['acc']:.4f}  (sparsity {actual_sparsity*100:.1f}%)")

    train_time = time.time() - t0

    # ---------- 3. Make pruning permanent + save ---------------------------
    for mod, name in targets:
        prune.remove(mod, name)   # bakes the mask into the weights

    test_out = eval_full(model.cpu(), test_loader, torch.device("cpu"),
                         return_loss=True, return_predictions=True)
    test_acc = test_out["acc"]
    mAP = float(average_precision_score(
        np.eye(10)[test_out["y_true"]], test_out["y_probs"], average="macro"))

    CKPT_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "predictions").mkdir(exist_ok=True)

    ckpt = CKPT_DIR / f"{args.tag}.pt"
    torch.save(model.state_dict(), ckpt)

    np.savez(REPORT_DIR / "predictions" / f"{args.tag}.npz",
             y_true=test_out["y_true"],
             fp32_y_pred=test_out["y_pred"], fp32_y_probs=test_out["y_probs"],
             int8_y_pred=test_out["y_pred"], int8_y_probs=test_out["y_probs"])

    # effective weight count after pruning: total_weights * (1 - sparsity)
    eff_kib = stats.weight_bytes_int8 / 1024 * (1 - actual_sparsity)

    metrics = {
        "tag": args.tag, "model": args.model,
        "technique": "pruning",
        "args": vars(args) | {"base_ckpt": str(args.base_ckpt)},
        "target_sparsity": args.sparsity,
        "actual_sparsity": actual_sparsity,
        "fp32_best_val_acc": best_acc,
        "fp32_final_val_acc": history["val_acc"][-1],
        "fp32_test_acc": test_acc,
        "int8_test_acc": test_acc,
        "int8_mode": "none",
        "fp32_mAP": mAP, "int8_mAP": mAP,
        "params": stats.params,
        "weight_kib_int8": stats.weight_bytes_int8 / 1024,
        "weight_kib_int8_effective": eff_kib,
        "weight_kib_fp32": stats.weight_bytes_fp32 / 1024,
        "macs": stats.macs,
        "macs_effective": int(stats.macs * (1 - actual_sparsity)),
        "ops_paper_convention": stats.macs * 2,
        "train_time_seconds": train_time,
        "history": history,
        "base_test_acc": base_acc,
        "pre_finetune_test_acc": pre_ft_acc,
        "weight_scales": None, "activation_scales": None,
    }
    (REPORT_DIR / f"{args.tag}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\npruned model — sparsity={actual_sparsity*100:.1f}%, "
          f"final test acc: {test_acc*100:.2f}% (was {base_acc*100:.2f}% before)")
    print(f"effective int8 weight memory: {eff_kib:.1f} KiB "
          f"(was {stats.weight_bytes_int8/1024:.1f} KiB)")
    print(f"wrote {ckpt}")


if __name__ == "__main__":
    main()
