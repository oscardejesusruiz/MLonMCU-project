"""Run a QAT (or int8) ai8x checkpoint on the CIFAR-10 test set and save
predictions in the same .npz format as `pc-implementation/`.

Output:
    reports/predictions/<tag>.npz   keys: y_true, fp32_y_pred, fp32_y_probs,
                                          int8_y_pred, int8_y_probs

For ai8x QAT checkpoints `fp32_*` and `int8_*` are identical (the QAT model
*is* the deployed numerics).

Usage:
    uv run python eval_ai8x.py \\
        --checkpoint $AI/ai8x-synthesis/trained/separable_qat.pth.tar \\
        --model-name ai85net_cmsis_separable \\
        --tag separable_qat

Must run inside the ai8x-training venv (it needs `ai8x` and the registered
model factories on sys.path).
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def _add_ai8x_to_path(ai_root: Path) -> None:
    ai_train = ai_root / "ai8x-training"
    if not ai_train.is_dir():
        raise FileNotFoundError(f"ai8x-training not found under {ai_root}")
    sys.path.insert(0, str(ai_train))


def _load_model(model_name: str, checkpoint: Path) -> torch.nn.Module:
    import ai8x  # noqa: F401 — populates the device registry on import

    ai8x.set_device(85, simulate=False, round_avg=False)

    # ai8x-training models live as `models.<name>` after sys.path injection
    try:
        mod = importlib.import_module(f"models.{model_name}")
    except ModuleNotFoundError:
        # some ai8x models register under different module file names — fall back
        # to a wildcard search in ai8x-training/models/
        mod = None
        for f in (Path(sys.path[0]) / "models").glob("*.py"):
            try:
                cand = importlib.import_module(f"models.{f.stem}")
            except Exception:
                continue
            if hasattr(cand, model_name):
                mod = cand
                break
        if mod is None:
            raise

    factory = getattr(mod, model_name)
    # bias=True matches the train.py `--use-bias` flag: ai8x requires it for
    # FusedConv2dBNReLU layers (improved + separable variants) so BN can be
    # folded into the preceding conv at synthesis.
    model = factory(num_classes=10, num_channels=3, dimensions=(32, 32), bias=True)

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    # ai8x checkpoints often have module. prefix
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)} (first: {missing[:3]})", file=sys.stderr)
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)} (first: {unexpected[:3]})", file=sys.stderr)
    model.eval()
    return model


def _get_test_loader(ai_root: Path, batch_size: int) -> DataLoader:
    from datasets.cifar import cifar10_get_datasets

    class _Args:
        truncate_testset = False
        act_mode_8bit = False
        dataset = "CIFAR10"
        device = "MAX78000"

    data_dir = ai_root / "ai8x-training" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    _, test_set = cifar10_get_datasets((str(data_dir), _Args()),
                                       load_train=False, load_test=True)
    # num_workers=0 to avoid pickling _Args across worker processes.
    return DataLoader(test_set, batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="QAT or quantized .pth.tar from ai8x-synthesis/trained/")
    ap.add_argument("--model-name", required=True,
                    help="ai85net_cmsis_{baseline,improved,separable}")
    ap.add_argument("--tag", required=True,
                    help="run tag, e.g. baseline_qat")
    ap.add_argument("--ai-root", type=Path,
                    default=Path(os.environ.get("AI",
                        Path.home() / "Desktop/project/max78000")))
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="defaults to <repo>/reports/predictions/")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    _add_ai8x_to_path(args.ai_root)

    model = _load_model(args.model_name, args.checkpoint)
    loader = _get_test_loader(args.ai_root, args.batch_size)

    out_dir = args.out_dir or (Path(__file__).resolve().parent.parent
                               / "reports" / "predictions")
    out_dir.mkdir(parents=True, exist_ok=True)

    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []
    qs: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            ys.append(y.numpy())
            ps.append(logits.argmax(1).numpy())
            qs.append(torch.softmax(logits, dim=1).numpy())

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    y_probs = np.concatenate(qs)
    acc = float((y_pred == y_true).mean())

    out_path = out_dir / f"{args.tag}.npz"
    np.savez(out_path,
             y_true=y_true,
             fp32_y_pred=y_pred, fp32_y_probs=y_probs,
             int8_y_pred=y_pred, int8_y_probs=y_probs)
    print(f"test acc = {acc*100:.2f}%")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
