"""Training/eval loops shared across phases."""
from __future__ import annotations

import time
from collections.abc import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from training.quantize import QuantConfig, convert_to_qat


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    return_loss: bool = False,
    return_predictions: bool = False,
):
    """Eval accuracy. Optionally returns mean loss and/or (y_true, y_pred, y_probs)."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    correct = total = 0
    loss_sum = 0.0
    all_true, all_pred, all_prob = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        if return_loss:
            loss_sum += criterion(logits, y).item()
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        if return_predictions:
            all_true.append(y.cpu())
            all_pred.append(pred.cpu())
            all_prob.append(torch.softmax(logits, dim=1).cpu())

    acc = correct / total
    out = {"acc": acc}
    if return_loss:
        out["loss"] = loss_sum / total
    if return_predictions:
        out["y_true"] = torch.cat(all_true).numpy()
        out["y_pred"] = torch.cat(all_pred).numpy()
        out["y_probs"] = torch.cat(all_prob).numpy()
    if not return_loss and not return_predictions:
        return acc       # back-compat: old callers still get a float
    return out


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    scheduler_factory=None,
    optimizer_name: str = "sgd",
    momentum: float = 0.9,
    log_every: int = 50,
    qat_start_epoch: int | None = None,
    qat_config: QuantConfig | None = None,
    qat_calib_loader: DataLoader | None = None,
) -> dict:
    """Train loop returning history + best test accuracy.

    Defaults match Caffe cifar10_quick: SGD + momentum + weight decay 0.004.
    Phase 2 can pass a scheduler factory (e.g. cosine annealing).
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"unknown optimizer {optimizer_name!r}")

    scheduler = scheduler_factory(optimizer) if scheduler_factory else None

    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": [],
               "lr": [], "epoch_time": []}
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        if qat_start_epoch is not None and epoch == qat_start_epoch:
            print(f"[epoch {epoch}] switching to QAT (8-bit weights)")
            model, _ = convert_to_qat(
                model.cpu(),
                calib_loader=qat_calib_loader or train_loader,
                config=qat_config or QuantConfig(power_of_two=True),
            )
            model = model.to(device)
            # rebuild optimizer to point at the new module's params
            if optimizer_name == "sgd":
                optimizer = torch.optim.SGD(
                    model.parameters(), lr=optimizer.param_groups[0]["lr"],
                    momentum=momentum, weight_decay=weight_decay,
                )
            else:
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=optimizer.param_groups[0]["lr"],
                    weight_decay=weight_decay,
                )
            # scheduler is rebuilt only if user supplied a factory
            if scheduler_factory is not None:
                scheduler = scheduler_factory(optimizer)
        model.train()
        running_loss = correct = total = 0
        t0 = time.time()
        bar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for step, (x, y) in enumerate(bar, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
            if step % log_every == 0:
                bar.set_postfix(loss=running_loss / total, acc=correct / total)

        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss / total
        train_acc = correct / total
        eval_out = evaluate(model, val_loader, device, return_loss=True)
        val_acc, val_loss = eval_out["acc"], eval_out["loss"]
        epoch_time = time.time() - t0
        cur_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(cur_lr)
        history["epoch_time"].append(epoch_time)

        best_acc = max(best_acc, val_acc)
        print(
            f"[{epoch:3d}/{epochs}] "
            f"loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_acc={val_acc:.4f} lr={cur_lr:.4g} "
            f"time={epoch_time:.1f}s"
        )

    return {"history": history, "best_acc": best_acc, "final_acc": history["val_acc"][-1]}
