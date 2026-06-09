"""End-to-end driver: train one model variant, PTQ-quantize, write metrics.

Phase 1 (paper-faithful):
    uv run python -m scripts.run_experiment baseline \
        --epochs 50 --optimizer sgd --lr 0.001 --weight-decay 0.004

Phase 2 (improvements):
    uv run python -m scripts.run_experiment improved \
        --epochs 80 --augment --optimizer sgd --lr 0.05 \
        --scheduler cosine --weight-decay 5e-4
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
from training.engine import evaluate as eval_full
import torch

from training.data import get_loaders
from training.engine import evaluate, train
from training.models import build_model
from training.quantize import QuantConfig, quantize_model_ptq
from training.utils import compute_stats, pick_device


REPO_ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = REPO_ROOT / "trained_models"
REPORT_DIR = REPO_ROOT / "reports"
CKPT_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("model", choices=["baseline", "baseline_5x5", "improved",
                                       "mininet", "deeper", "ressimplenet", "nascifarnet"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.004)
    p.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument(
        "--scheduler",
        choices=["none", "cosine", "step"],
        default="none",
    )
    p.add_argument("--augment", action="store_true")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--input-size", type=int, default=32,
                   help="Spatial input resolution. CIFAR-10 default 32. "
                        "MiniMobileNet uses 64 (resized from 32).")
    p.add_argument("--quant-power-of-two", action="store_true",
                   help="Constrain int8 scales to powers of two (CMSIS-NN q7 convention).")
    p.add_argument("--no-quant-power-of-two", dest="quant_power_of_two", action="store_false")
    p.set_defaults(quant_power_of_two=True)
    p.add_argument("--tag", default=None,
                   help="Suffix for checkpoint + report file names. Default = model name.")
    p.add_argument("--qat-start-epoch", type=int, default=None,
                   help="Switch to QAT modules at this epoch (ai8x-style). "
                        "None = no inline QAT (use run_qat.py instead).")
    p.add_argument("--load-fp32", type=Path, default=None,
                   help="Path to an fp32 checkpoint to load BEFORE training "
                        "starts. Mirrors the MAX78000 QAT flow "
                        "(--exp-load-weights-from): load fp32 weights, then "
                        "fine-tune with QAT for a smaller number of epochs. "
                        "Pair with --qat-start-epoch to enable QAT.")
    p.add_argument("--milestones", type=int, nargs="*", default=None,
                   help="MultiStepLR milestones (in epochs).")
    p.add_argument("--gamma", type=float, default=0.1,
                   help="MultiStepLR decay factor.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tag = args.tag or args.model
    device = pick_device()
    print(f"using device: {device}")

    train_loader, val_loader, test_loader = get_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=args.augment,
        input_size=args.input_size,
    )

    model = build_model(args.model)

    # MAX78000-style QAT flow: optionally start from a previously-trained
    # fp32 checkpoint and fine-tune (typically with --qat-start-epoch set).
    if args.load_fp32 is not None:
        if not args.load_fp32.exists():
            raise FileNotFoundError(
                f"--load-fp32 checkpoint not found: {args.load_fp32}")
        state = torch.load(args.load_fp32, map_location="cpu", weights_only=False)
        sd = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"loaded fp32 weights from {args.load_fp32}")
        if missing:
            print(f"  ! missing keys: {len(missing)} (first 3: {missing[:3]})")
        if unexpected:
            print(f"  ! unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")

    stats_pre = compute_stats(model)
    print("--- model stats ---")
    print(stats_pre.summary())

    def make_scheduler(opt):
        if args.milestones:
            return torch.optim.lr_scheduler.MultiStepLR(
                opt, milestones=args.milestones, gamma=args.gamma,
            )
        if args.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        if args.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)
        return None

    scheduler_factory = make_scheduler if (args.scheduler != "none" or args.milestones) else None

    t0 = time.time()
    result = train(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        device=device, scheduler_factory=scheduler_factory,
        optimizer_name=args.optimizer, momentum=args.momentum,
        qat_start_epoch=args.qat_start_epoch,
        qat_calib_loader=train_loader,
    )
    train_time = time.time() - t0

    ckpt_path = CKPT_DIR / f"{tag}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"saved checkpoint to {ckpt_path}")
    
    # --- final fp32 predictions on the test set ---
    fp_out = eval_full(
        model.cpu(), test_loader, torch.device("cpu"),
        return_loss=True, return_predictions=True,
    )
    fp32_test_acc = fp_out["acc"]
    fp32_test_loss = fp_out["loss"]
    fp32_mAP = float(average_precision_score(
        np.eye(10)[fp_out["y_true"]], fp_out["y_probs"], average="macro"
    ))

    # --- PTQ ---
    if args.qat_start_epoch is None:
        # fp32-only run: apply PTQ for the deployed int8 numbers
        print("calibrating + applying PTQ...")
        qmodel, qstats = quantize_model_ptq(
            model.cpu(),
            calib_loader=train_loader,
            config=QuantConfig(power_of_two=args.quant_power_of_two),
        )
        q_out = eval_full(
            qmodel, test_loader, torch.device("cpu"),
            return_loss=True, return_predictions=True,
        )
        int8_mode = "ptq"
        weight_scales = qstats.weight_scales
        activation_scales = qstats.activation_scales
        ptq_path = CKPT_DIR / f"{tag}_ptq.pt"
        torch.save(qmodel.state_dict(), ptq_path)
        print(f"saved PTQ checkpoint to {ptq_path}")
    else:
        # QAT run: the model already is int8 numerics; no PTQ step
        print("QAT model — skipping PTQ")
        q_out = fp_out
        int8_mode = "qat"
        weight_scales = None
        activation_scales = None

    quant_acc = q_out["acc"]
    quant_mAP = float(average_precision_score(
        np.eye(10)[q_out["y_true"]], q_out["y_probs"], average="macro"
    ))
    print(f"int8 ({int8_mode}) accuracy: {quant_acc:.4f}  mAP: {quant_mAP:.4f}")

    # save raw predictions for plotting
    preds_dir = REPORT_DIR / "predictions"
    preds_dir.mkdir(exist_ok=True)
    np.savez(
        preds_dir / f"{tag}.npz",
        y_true=fp_out["y_true"],
        fp32_y_pred=fp_out["y_pred"], fp32_y_probs=fp_out["y_probs"],
        int8_y_pred=q_out["y_pred"],  int8_y_probs=q_out["y_probs"],
    )

    metrics = {
        "tag": tag,
        "model": args.model,
        "args": {k: v for k, v in vars(args).items() if k not in {"tag"}},
        "fp32_best_val_acc":  result["best_acc"],     # best across epochs, on val
        "fp32_final_val_acc": result["final_acc"],    # last epoch, on val
        "fp32_test_acc":      fp32_test_acc,          # final fp32 pass on test (already in dict below)
        "int8_test_acc":      quant_acc,              # PTQ or QAT model on test
        "params": stats_pre.params,
        "weight_kib_int8": stats_pre.weight_bytes_int8 / 1024,
        "weight_kib_fp32": stats_pre.weight_bytes_fp32 / 1024,
        "macs": stats_pre.macs,
        "ops_paper_convention": stats_pre.macs * 2,  # mul + add
        "train_time_seconds": train_time,
        "history": result["history"],
        "fp32_test_loss": fp32_test_loss,
        "fp32_mAP": fp32_mAP,
        "int8_mAP": quant_mAP,
        "int8_mode": int8_mode,           # "ptq" or "qat"
        "weight_scales": weight_scales,
        "activation_scales": activation_scales,
    }
    metrics_path = REPORT_DIR / f"{tag}_metrics.json"
    # `default=str` so PosixPath (from --load-fp32) and any other
    # non-JSON-native types are stringified instead of crashing.
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
    print(f"wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
