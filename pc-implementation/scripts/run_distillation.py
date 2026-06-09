"""Knowledge distillation: train a small `student` model using soft labels
from a larger pre-trained `teacher`.

Loss = alpha * KL(student_logits/T, teacher_logits/T) * T^2
     + (1 - alpha) * CE(student_logits, hard_labels)

Default T=4, alpha=0.7 (Hinton et al. 2015 conventions).

Optional QAT switch mid-training (mirrors run_experiment.py): if
`--qat-start-epoch N` is set, the student is converted to QAT modules at
epoch N and training continues with the distillation loss.

Optional PTQ at end (default: enabled for fp32-only runs, skipped for QAT
runs since QAT model is already int8 numerics).

Usage:
    # fp32 distillation + PTQ at end
    uv run python -m scripts.run_distillation \\
        --teacher mininet --teacher-ckpt trained_models/mininet_fp32.pt \\
        --student improved --tag improved_distill_fp32 --epochs 80

    # distillation with inline QAT
    uv run python -m scripts.run_distillation \\
        --teacher mininet --teacher-ckpt trained_models/mininet_fp32.pt \\
        --student improved --tag improved_distill_qat --epochs 80 \\
        --qat-start-epoch 30
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from training.data import get_loaders
from training.engine import evaluate as eval_full
from training.models import build_model
from training.quantize import QuantConfig, convert_to_qat, quantize_model_ptq
from training.utils import compute_stats, pick_device


REPO_ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = REPO_ROOT / "trained_models"
REPORT_DIR = REPO_ROOT / "reports"

_MODEL_CHOICES = ["baseline", "baseline_5x5", "improved",
                  "mininet", "deeper", "ressimplenet", "nascifarnet"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True, choices=_MODEL_CHOICES)
    p.add_argument("--teacher-ckpt", type=Path, required=True,
                   help="path to fp32 teacher .pt")
    p.add_argument("--student", required=True, choices=_MODEL_CHOICES)
    p.add_argument("--tag", required=True, help="output tag")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--input-size", type=int, default=32)
    p.add_argument("--temperature", "-T", type=float, default=4.0)
    p.add_argument("--alpha", type=float, default=0.7,
                   help="weight on soft-label loss (1-alpha on hard CE)")
    p.add_argument("--qat-start-epoch", type=int, default=None,
                   help="switch student to QAT modules at this epoch")
    p.add_argument("--quant-power-of-two", action="store_true")
    p.add_argument("--no-quant-power-of-two", dest="quant_power_of_two",
                   action="store_false")
    p.set_defaults(quant_power_of_two=True)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def distill_loss(student_logits, teacher_logits, hard_labels, T, alpha):
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction="batchmean",
    ) * (T * T)
    hard_loss = F.cross_entropy(student_logits, hard_labels)
    return alpha * soft_loss + (1 - alpha) * hard_loss


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = pick_device()
    print(f"device: {device}")
    print(f"teacher: {args.teacher}  ←  {args.teacher_ckpt}")
    print(f"student: {args.student}")
    print(f"T={args.temperature}  alpha={args.alpha}")
    if args.qat_start_epoch is not None:
        print(f"QAT switch at epoch {args.qat_start_epoch}")

    train_loader, val_loader, test_loader = get_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
        augment=True, input_size=args.input_size,
    )

    teacher = build_model(args.teacher)
    teacher.load_state_dict(torch.load(args.teacher_ckpt, map_location="cpu",
                                       weights_only=True))
    teacher = teacher.to(device).eval()
    for p_ in teacher.parameters():
        p_.requires_grad = False

    student = build_model(args.student).to(device)
    stats = compute_stats(build_model(args.student))   # before any QAT swap
    print(f"student stats: {stats.summary()}")

    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=args.epochs)

    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": [], "lr": [], "epoch_time": []}
    best_acc = 0.0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        # ---- mid-training QAT switch ----
        if args.qat_start_epoch is not None and epoch == args.qat_start_epoch:
            print(f"[epoch {epoch}] switching student to QAT (8-bit weights)")
            student, _ = convert_to_qat(
                student.cpu(),
                calib_loader=train_loader,
                config=QuantConfig(power_of_two=args.quant_power_of_two),
            )
            student = student.to(device)
            optimizer = torch.optim.Adam(
                student.parameters(),
                lr=optimizer.param_groups[0]["lr"],
                weight_decay=args.weight_decay,
            )
            # rebuild scheduler so it points at the new optimizer
            remaining = args.epochs - epoch + 1
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=remaining)

        student.train()
        loss_sum = correct = total = 0
        epoch_t = time.time()
        bar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for x, y in bar:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            with torch.no_grad():
                t_logits = teacher(x)
            s_logits = student(x)
            loss = distill_loss(s_logits, t_logits, y, args.temperature, args.alpha)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            loss_sum += loss.item() * y.size(0)
            correct += (s_logits.argmax(1) == y).sum().item()
            total += y.size(0)

        scheduler.step()
        val_out = eval_full(student, val_loader, device, return_loss=True)
        history["train_loss"].append(loss_sum / total)
        history["train_acc"].append(correct / total)
        history["val_loss"].append(val_out["loss"])
        history["val_acc"].append(val_out["acc"])
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["epoch_time"].append(time.time() - epoch_t)
        best_acc = max(best_acc, val_out["acc"])
        print(f"[{epoch:3d}/{args.epochs}] "
              f"loss={loss_sum/total:.4f} train_acc={correct/total:.4f} "
              f"val_acc={val_out['acc']:.4f}")

    train_time = time.time() - t0

    # ---- final fp32 (or QAT) test pass + optional PTQ -----------------------
    fp_out = eval_full(student.cpu(), test_loader, torch.device("cpu"),
                       return_loss=True, return_predictions=True)
    fp32_test_acc = fp_out["acc"]
    fp32_mAP = float(average_precision_score(
        np.eye(10)[fp_out["y_true"]], fp_out["y_probs"], average="macro"))

    if args.qat_start_epoch is None:
        # apply PTQ on top of the fp32 distilled student
        print("calibrating + applying PTQ...")
        qmodel, qstats = quantize_model_ptq(
            student,
            calib_loader=train_loader,
            config=QuantConfig(power_of_two=args.quant_power_of_two),
        )
        q_out = eval_full(qmodel, test_loader, torch.device("cpu"),
                          return_loss=True, return_predictions=True)
        int8_mode = "ptq"
        weight_scales = qstats.weight_scales
        activation_scales = qstats.activation_scales
        # also save PTQ checkpoint
        ptq_path = CKPT_DIR / f"{args.tag}_ptq.pt"
        torch.save(qmodel.state_dict(), ptq_path)
        print(f"saved PTQ checkpoint to {ptq_path}")
    else:
        # QAT distillation: the student already represents int8 numerics
        print("QAT-distill student — skipping PTQ")
        q_out = fp_out
        int8_mode = "qat"
        weight_scales = None
        activation_scales = None

    quant_acc = q_out["acc"]
    quant_mAP = float(average_precision_score(
        np.eye(10)[q_out["y_true"]], q_out["y_probs"], average="macro"))
    print(f"int8 ({int8_mode}) accuracy: {quant_acc:.4f}  mAP: {quant_mAP:.4f}")

    CKPT_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "predictions").mkdir(exist_ok=True)

    ckpt = CKPT_DIR / f"{args.tag}.pt"
    torch.save(student.state_dict(), ckpt)

    np.savez(REPORT_DIR / "predictions" / f"{args.tag}.npz",
             y_true=fp_out["y_true"],
             fp32_y_pred=fp_out["y_pred"], fp32_y_probs=fp_out["y_probs"],
             int8_y_pred=q_out["y_pred"],  int8_y_probs=q_out["y_probs"])

    metrics = {
        "tag": args.tag, "model": args.student,
        "technique": "distillation" + ("_qat" if args.qat_start_epoch else ""),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "teacher": args.teacher,
        "fp32_best_val_acc": best_acc,
        "fp32_final_val_acc": history["val_acc"][-1],
        "fp32_test_acc": fp32_test_acc,
        "int8_test_acc": quant_acc,
        "int8_mode": int8_mode,
        "fp32_mAP": fp32_mAP, "int8_mAP": quant_mAP,
        "params": stats.params,
        "weight_kib_int8": stats.weight_bytes_int8 / 1024,
        "weight_kib_fp32": stats.weight_bytes_fp32 / 1024,
        "macs": stats.macs,
        "ops_paper_convention": stats.macs * 2,
        "train_time_seconds": train_time,
        "history": history,
        "weight_scales": weight_scales,
        "activation_scales": activation_scales,
    }
    (REPORT_DIR / f"{args.tag}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\ndistilled student fp32={fp32_test_acc*100:.2f}%  int8={quant_acc*100:.2f}%  mAP={quant_mAP:.4f}")
    print(f"wrote {ckpt} + {REPORT_DIR / f'{args.tag}_metrics.json'}")


if __name__ == "__main__":
    main()
