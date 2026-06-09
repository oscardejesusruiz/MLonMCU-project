"""CIFAR-10 dataset loaders.

Paper-faithful baseline: per-channel mean subtraction only (matches Caffe
cifar10_quick / Lai et al. 2018). Phase 2 swaps in heavier augmentation.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


def _baseline_transforms(input_size: int = 32) -> tuple[transforms.Compose, transforms.Compose]:
    """Paper-faithful: ToTensor + normalize. No augmentation.

    If input_size != 32, resize as the first step (matches the TF MiniMobileNet
    reference, which resizes CIFAR-10 32x32 → 64x64 before training).
    """
    norm = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    pre = [transforms.Resize(input_size)] if input_size != 32 else []
    train_tf = transforms.Compose([*pre, transforms.ToTensor(), norm])
    eval_tf = transforms.Compose([*pre, transforms.ToTensor(), norm])
    return train_tf, eval_tf


def _augmented_transforms(input_size: int = 32) -> tuple[transforms.Compose, transforms.Compose]:
    """Phase 2: random crop + horizontal flip + normalize.

    For input_size != 32 we resize first, then random-crop at the larger
    resolution with proportional padding (mimics the TF aug pipeline:
    resize-pad-crop chain).
    """
    norm = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    if input_size == 32:
        train_tf = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                norm,
            ]
        )
        eval_tf = transforms.Compose([transforms.ToTensor(), norm])
    else:
        pad = max(1, input_size // 8)   # ≈ 8 px for 64x64, matches TF (4 of 32)
        train_tf = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.RandomCrop(input_size, padding=pad),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                norm,
            ]
        )
        eval_tf = transforms.Compose([
            transforms.Resize(input_size), transforms.ToTensor(), norm,
        ])
    return train_tf, eval_tf


def get_loaders(
    batch_size: int = 128,
    num_workers: int = 2,
    augment: bool = False,
    data_root: Path = DATA_ROOT,
    val_fraction: float = 0.1,
    split_seed: int = 42,
    input_size: int = 32,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_tf, eval_tf = (
        _augmented_transforms(input_size) if augment else _baseline_transforms(input_size)
    )

    full_train = datasets.CIFAR10(
        root=str(data_root), train=True, download=True, transform=train_tf
    )
    # val gets the eval transform (no augmentation) — build a parallel dataset
    # with the eval transform and index into it via the same val indices.
    full_train_eval = datasets.CIFAR10(
        root=str(data_root), train=True, download=True, transform=eval_tf
    )
    test_set = datasets.CIFAR10(
        root=str(data_root), train=False, download=True, transform=eval_tf
    )

    n_total = len(full_train)
    n_val = int(round(n_total * val_fraction))
    g = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(n_total, generator=g).tolist()
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_set = torch.utils.data.Subset(full_train, train_idx)
    val_set = torch.utils.data.Subset(full_train_eval, val_idx)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin,
                              persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False,
                            num_workers=num_workers, pin_memory=pin,
                            persistent_workers=num_workers > 0)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                             num_workers=num_workers, pin_memory=pin,
                             persistent_workers=num_workers > 0)
    return train_loader, val_loader, test_loader
