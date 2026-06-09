"""Verify that BN folding preserves model semantics.

For each variant, loads two checkpoints and compares their predictions:

  fp32 mode (default):
    A = trained_models/<v>_fp32.pth.tar          (fp32, BN as separate layer)
    B = $AI/ai8x-synthesis/trained/<v>_fused.pth.tar  (BN folded by bn_fuser_v2)

  qat mode (--mode qat):
    A = trained_models/<v>_qat_train.pth.tar     (QAT fine-tuned, BN fused by ai8x)
    B = $AI/ai8x-synthesis/trained/<v>_qat_fused.pth.tar  (BN fold by bn_fuser_v2)

Prints per-variant: acc_A, acc_B, argmax-agreement %, max |Δlogit|.
Agreement should be ~100% for a correct fold.

Usage:
    AI=$HOME/Desktop/project/max78000 python verify_fold.py [--mode fp32|qat] [variant ...]
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

_DEFAULT_VARIANTS = ["baseline", "improved", "mininet", "deeper", "wide_improved"]


def _add_ai8x_to_path(ai_root: Path) -> None:
    sys.path.insert(0, str(ai_root / "ai8x-training"))


def _build(variant: str) -> torch.nn.Module:
    import ai8x
    ai8x.set_device(85, simulate=False, round_avg=False)

    # Most variants live in project_models.py with the ai85net_cmsis_<v> name;
    # nascifarnet and ressimplenet are Maxim/custom modules with their own
    # arch names — keep this mapping in sync with train_max78000_models.sh and
    # synthesize_all.sh.
    _SPECIAL_ARCH = {
        "nascifarnet":  ("ai85net-nas-cifar",   "ai85nascifarnet"),
        "ressimplenet": ("ai85ressimplenetbn",  "ai85ressimplenetbn"),
    }
    if variant in _SPECIAL_ARCH:
        module_name, arch = _SPECIAL_ARCH[variant]
        mod = importlib.import_module(f"models.{module_name}")
    else:
        mod = importlib.import_module("models.project_models")
        arch = f"ai85net_cmsis_{variant}"

    return getattr(mod, arch)(num_classes=10, num_channels=3,
                              dimensions=(32, 32), bias=True)


def _load_into(model: torch.nn.Module, ckpt_path: Path) -> None:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)


def _testloader(ai_root: Path) -> DataLoader:
    from datasets.cifar import cifar10_get_datasets

    class _Args:
        truncate_testset = False
        act_mode_8bit = False
        dataset = "CIFAR10"
        device = "MAX78000"

    data_dir = ai_root / "ai8x-training" / "data"
    _, test_set = cifar10_get_datasets((str(data_dir), _Args()),
                                       load_train=False, load_test=True)
    return DataLoader(test_set, batch_size=512, shuffle=False, num_workers=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fp32", "qat"], default="fp32",
                    help="fp32: compare fp32 vs fused.  qat: compare qat_train vs qat_fused.")
    ap.add_argument("variants", nargs="*", default=_DEFAULT_VARIANTS)
    args = ap.parse_args()

    ai_root = Path(os.environ.get("AI", Path.home() / "Desktop/project/max78000"))
    _add_ai8x_to_path(ai_root)

    impl_dir  = Path(__file__).resolve().parent.parent
    trained_pc = impl_dir / "trained_models"
    fused_dir  = ai_root / "ai8x-synthesis" / "trained"

    if args.mode == "fp32":
        src_suffix   = "_fp32.pth.tar"
        fused_suffix = "_fused.pth.tar"
        col_a, col_b = "fp32 acc", "fused acc"
    else:
        src_suffix   = "_qat_train.pth.tar"
        fused_suffix = "_qat_fused.pth.tar"
        col_a, col_b = "qat acc", "qat-fused"

    print(f"{'variant':<14} {col_a:>9} {col_b:>10} {'agree %':>8} {'max|Δlogit|':>11}")
    print("-" * 60)

    loader = None   # lazily initialised after ai8x path is ready

    for v in args.variants:
        src_ckpt   = trained_pc / f"{v}{src_suffix}"
        fused_ckpt = fused_dir  / f"{v}{fused_suffix}"

        if not src_ckpt.exists():
            print(f"{v:<14} (missing {src_ckpt.name} — skipped)")
            continue
        if not fused_ckpt.exists():
            print(f"{v:<14} (missing {fused_ckpt.name} — run synthesize_all.sh {'--from-qat ' if args.mode == 'qat' else ''}first)")
            continue

        m_a = _build(v); _load_into(m_a, src_ckpt);   m_a.eval()
        m_b = _build(v); _load_into(m_b, fused_ckpt); m_b.eval()

        if loader is None:
            loader = _testloader(ai_root)

        ys, pa, pb, dmax = [], [], [], 0.0
        with torch.no_grad():
            for x, y in loader:
                la = m_a(x); lb = m_b(x)
                dmax = max(dmax, (la - lb).abs().max().item())
                ys.append(y.numpy())
                pa.append(la.argmax(1).numpy())
                pb.append(lb.argmax(1).numpy())

        y_true = np.concatenate(ys)
        p_a = np.concatenate(pa); p_b = np.concatenate(pb)
        acc_a  = (p_a == y_true).mean()
        acc_b  = (p_b == y_true).mean()
        agree  = (p_a == p_b).mean()
        print(f"{v:<14} {acc_a*100:8.2f}% {acc_b*100:9.2f}% {agree*100:7.2f}% {dmax:11.4f}")


if __name__ == "__main__":
    main()
