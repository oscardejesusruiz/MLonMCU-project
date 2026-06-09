"""Combine QAT and pruning: load a QAT checkpoint, prune the QAT modules,
then fine-tune with QAT still active.

Workflow:
    1. Build fp32 architecture.
    2. Convert to QAT (calibrate with train_loader → installs QATConv2d/Linear).
    3. Load the saved QAT state_dict on top.
    4. Magnitude-prune the QAT modules at the requested sparsity.
    5. Fine-tune for N epochs (QAT remains active throughout).
    6. Bake the mask in (prune.remove) and save.

Usage:
    uv run python -m scripts.run_qat_prune \\
        --model improved \\
        --qat-ckpt trained_models/improved_qat.pt \\
        --sparsity 0.5 --finetune-epochs 10 \\
        --tag improved_qat_prune50
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
from training.quantize import QuantConfig, convert_to_qat
from training.utils import compute_stats, pick_device


REPO_ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = REPO_ROOT / "trained_models"
REPORT_DIR = REPO_ROOT / "reports"

_MODEL_CHOICES = ["baseline", "baseline_5x5", "improved",
                  "mininet", "deeper", "ressimplenet", "nascifarnet"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=_MODEL_CHOICES)
    p.add_argument("--qat-ckpt", type=Path, required=True,
                   help="path to a QAT-trained .pt checkpoint")
    p.add_argument("--sparsity", type=float, default=0.5,
                   help="target sparsity (0..1). 0.5 = half of weights zeroed")
    p.add_argument("--finetune-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--input-size", type=int, default=32)
    p.add_argument("--quant-power-of-two", action="store_true")
    p.add_argument("--no-quant-power-of-two", dest="quant_power_of_two",
                   action="store_false")
    p.set_defaults(quant_power_of_two=True)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", required=True)
    return p.parse_args()


def collect_prune_params(model: nn.Module) -> list[tuple[nn.Module, str]]:
    """Collect (module, 'weight') for every module exposing a 'weight'
    nn.Parameter — covers nn.Conv2d, nn.Linear, and our QATConv2d/QATLinear."""
    out = []
    for m in model.modules():
        if hasattr(m, "weight") and isinstance(m.weight, nn.Parameter):
            # skip BatchNorm — it has 'weight' but we don't want to prune gammas
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                              nn.GroupNorm, nn.LayerNorm)):
                continue
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
    print(f"loading QAT model {args.model} from {args.qat_ckpt}")

    train_loader, val_loader, test_loader = get_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
        augment=True, input_size=args.input_size,
    )

    # ---- 1. build fp32 architecture + convert to QAT structure --------------
    fp32 = build_model(args.model)
    qmodel, _ = convert_to_qat(
        fp32, calib_loader=train_loader,
        config=QuantConfig(power_of_two=args.quant_power_of_two),
    )
    # ---- 2. load the trained QAT weights ------------------------------------
    state = torch.load(args.qat_ckpt, map_location="cpu", weights_only=True)
    missing, unexpected = qmodel.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[warn] missing={len(missing)} unexpected={len(unexpected)}")
    qmodel = qmodel.to(device)
    stats = compute_stats(build_model(args.model))
    base_acc = eval_full(qmodel, test_loader, device)
    print(f"base QAT test acc (before prune): {base_acc*100:.2f}%")

    # ---- 3. magnitude pruning on the QAT modules ----------------------------
    targets = collect_prune_params(qmodel)
    prune.global_unstructured(
        targets, pruning_method=prune.L1Unstructured, amount=args.sparsity,
    )
    actual_sparsity = sparsity_report(targets)
    pre_ft_acc = eval_full(qmodel, test_loader, device)
    print(f"sparsity after prune: {actual_sparsity*100:.2f}%")
    print(f"test acc immediately after prune: {pre_ft_acc*100:.2f}%")

    # ---- 4. fine-tune (QAT modules remain active) ---------------------------
    optimizer = torch.optim.Adam(qmodel.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.finetune_epochs)
    crit = nn.CrossEntropyLoss()
    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": [], "lr": [], "epoch_time": []}
    best_acc = 0.0
    t0 = time.time()

    for epoch in range(1, args.finetune_epochs + 1):
        qmodel.train()
        loss_sum = correct = total = 0
        epoch_t = time.time()
        bar = tqdm(train_loader, desc=f"qat-ft {epoch}/{args.finetune_epochs}",
                   leave=False)
        for x, y in bar:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            logits = qmodel(x)
            loss = crit(logits, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        scheduler.step()
        val_out = eval_full(qmodel, val_loader, device, return_loss=True)
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

    # ---- 5. bake mask, save, report -----------------------------------------
    for mod, name in targets:
        prune.remove(mod, name)

    test_out = eval_full(qmodel.cpu(), test_loader, torch.device("cpu"),
                         return_loss=True, return_predictions=True)
    test_acc = test_out["acc"]
    mAP = float(average_precision_score(
        np.eye(10)[test_out["y_true"]], test_out["y_probs"], average="macro"))

    CKPT_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "predictions").mkdir(exist_ok=True)

    ckpt = CKPT_DIR / f"{args.tag}.pt"
    torch.save(qmodel.state_dict(), ckpt)

    np.savez(REPORT_DIR / "predictions" / f"{args.tag}.npz",
             y_true=test_out["y_true"],
             fp32_y_pred=test_out["y_pred"], fp32_y_probs=test_out["y_probs"],
             int8_y_pred=test_out["y_pred"], int8_y_probs=test_out["y_probs"])

    eff_kib = stats.weight_bytes_int8 / 1024 * (1 - actual_sparsity)

    metrics = {
        "tag": args.tag, "model": args.model,
        "technique": "qat_pruning",
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "target_sparsity": args.sparsity,
        "actual_sparsity": actual_sparsity,
        "fp32_best_val_acc": best_acc,
        "fp32_final_val_acc": history["val_acc"][-1],
        "fp32_test_acc": test_acc,
        "int8_test_acc": test_acc,
        "int8_mode": "qat",
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
    print(f"\nQAT+pruned model — sparsity={actual_sparsity*100:.1f}%, "
          f"final test acc: {test_acc*100:.2f}% (was {base_acc*100:.2f}% before)")
    print(f"effective int8 weight memory: {eff_kib:.1f} KiB")
    print(f"wrote {ckpt}")


if __name__ == "__main__":
    main()
