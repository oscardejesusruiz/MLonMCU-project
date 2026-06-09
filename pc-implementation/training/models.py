"""Model definitions.

All three variants use **3x3 convolutions** so the same architectures
deploy across CMSIS-NN (Cortex-M), MAX78000, and IMX500 without
hardware-forced rewrites. MAX78000 only supports 1x1 and 3x3 kernels;
IMX500 supports any kernel size, so 3x3 is the lowest common denominator
that lets the comparison be apples-to-apples on the same architecture.

History: an earlier version of `CMSISNNBaseline` used 5x5 kernels to
mirror Lai et al. 2018 Table 1 exactly (~87 KB int8 weights, 24.7
MOps/inference). That gave a paper-faithful reproduction on CMSIS-NN but
forced a 5x5→3x3 substitution before MAX78000 deployment, breaking
architecture parity between the two MCU paths. We now use 3x3 throughout
and report the deviation from the paper explicitly.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CMSISNNBaseline(nn.Module):
    """Paper-faithful topology with **3x3** kernels for MCU-portability.

    Conv 3x3 (pad=1) — ReLU — MaxPool 2x2/s=2, three times, then FC.
    Spatial flow 32→32→16→16→8→8→4 unchanged from the 5x5 version.

    Parameter / MAC delta vs. the original 5x5 Lai et al. architecture:
      params:   89,578 (5x5)  →  38,762 (3x3)        ~43% smaller
      MACs:     12.30 M       →   4.43 M             ~36% of original
      receptive field: 5x5 effective area per conv reduces to 3x3.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),                # 32x32x32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                      # 16x16x32
            nn.Conv2d(32, 32, kernel_size=3, padding=1),               # 16x16x32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                      # 8x8x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),               # 8x8x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                      # 4x4x64
        )
        self.classifier = nn.Linear(4 * 4 * 64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class CMSISNNImproved(nn.Module):
    """Phase-2: Conv 3x3-BN-ReLU-Pool x3 + FC head. Same MACs as baseline."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(4 * 4 * 64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.dw = nn.Conv2d(
            in_ch, in_ch, kernel_size, padding=pad, groups=in_ch, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x


class CMSISNNSeparable(nn.Module):
    """Phase-2: depthwise-separable variant. Far fewer MACs than baseline."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.1) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.block1 = DepthwiseSeparableBlock(32, 64)
        self.pool1 = nn.MaxPool2d(2, 2)            # 16x16x64
        self.block2 = DepthwiseSeparableBlock(64, 64)
        self.pool2 = nn.MaxPool2d(2, 2)            # 8x8x64
        self.block3 = DepthwiseSeparableBlock(64, 128)
        self.pool3 = nn.AdaptiveAvgPool2d(1)        # 1x1x128
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.block3(x)
        x = self.pool3(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class _VGGBlock(nn.Module):
    """One Conv 3x3 + BN + ReLU unit. Used by MiniMobileNet (VGG-Micro)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class MiniMobileNet(nn.Module):
    """VGG-Micro — architecture aligned with AI85NetCmsisMininet (MAX78000).

    Downsampling uses MaxPool 2×2 + stride-1 conv (same as the MAX78000 port),
    NOT stride-2 conv. This makes the two sides architecturally identical:
    same channel widths, same spatial flow, same ReLU, same fixed AvgPool.

    Topology (32×32×3 input):
        L1: Conv 3×3,  3 → 32                       32×32×32
        L2: MaxPool 2 + Conv 3×3, 32 → 64           16×16×64
        L3: Conv 3×3, 64 → 64                        16×16×64
        L4: MaxPool 2 + Conv 3×3, 64 → 64             8×8×64
        L5: Conv 3×3, 64 → 128                         8×8×128
        L6: MaxPool 2 + Conv 3×3, 128 → 128            4×4×128
        AvgPool(4) → Dropout(0.4) → Linear(128 → 10)

    MACs are unchanged vs. the stride-2 version (output spatial size is the
    same; the pool has negligible cost). Weight count is identical.
    """

    def __init__(
        self,
        num_classes: int = 10,
        dropout: float = 0.4,
        input_size: int = 32,
    ) -> None:
        super().__init__()
        self.input_size = input_size

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.l1 = _VGGBlock(3,   32)   # 32×32×32
        self.l2 = _VGGBlock(32,  64)   # (after pool) 16×16×64
        self.l3 = _VGGBlock(64,  64)   # 16×16×64
        self.l4 = _VGGBlock(64,  64)   # (after pool)  8×8×64
        self.l5 = _VGGBlock(64,  128)  # 8×8×128
        self.l6 = _VGGBlock(128, 128)  # (after pool)  4×4×128

        self.avgpool = nn.AvgPool2d(kernel_size=4)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        x = self.l2(self.pool2(x))
        x = self.l3(x)
        x = self.l4(self.pool2(x))
        x = self.l5(x)
        x = self.l6(self.pool2(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)


class CMSISNNBaseline5x5(nn.Module):
    """Lai et al. 2018 exact — three 5x5 Conv-ReLU-Pool blocks + FC.

    PC-only (MAX78000 doesn't support 5x5). 87 KB weights, 24.7 M ops,
    79.9% paper int8 accuracy.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,  32, kernel_size=5, padding=2),   # 32x32x32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                             # 16x16x32
            nn.Conv2d(32, 32, kernel_size=5, padding=2),   # 16x16x32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                             # 8x8x32
            nn.Conv2d(32, 64, kernel_size=5, padding=2),   # 8x8x64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                             # 4x4x64
        )
        self.classifier = nn.Linear(4 * 4 * 64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class CMSISNNDeeper(nn.Module):
    """`improved_3x3` + an extra 64-channel conv + 128-channel head.

    Tests whether depth alone (without depthwise-separable decomposition)
    can match `mininet`. ~140 KB int8 weights, ~16 M ops.

    Spatial flow:  32x32x3 → 32x32x32 → 16x16x32 → 16x16x32 → 8x8x32 →
                   8x8x64 → 8x8x64 → 4x4x64 → 4x4x128 → 1x1x128 → 10
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,  32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                                  # 16x16x32

            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                                  # 8x8x32

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),         # extra layer
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                                  # 4x4x64

            nn.Conv2d(64, 128, 3, padding=1, bias=False),        # extra layer
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                             # 1x1x128
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class _ResNetBasicBlock(nn.Module):
    """ResNet BasicBlock with MAX78000-compatible downsampling.

    The MAX78000 accelerator doesn't support stride-2 conv, so it does
    `MaxPool(2) → Conv(stride=1)` for any spatial reduction. We mirror
    that here: a single pool at the block entry, then both the main
    branch and the shortcut work on the already-pooled tensor.

    Why one pool (not two like the MAX78000 wiring): on MAX78000 the
    pool is fused into BOTH `FusedMaxPoolConv2d` ops (main + shortcut),
    so it's "pooled twice" in the layer list — but max-pool is
    deterministic so the result is identical to pooling once and
    splitting. Doing it once here keeps the PyTorch code clean.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        if stride == 2:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool = nn.Identity()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        # Shortcut runs on the SAME pooled input as conv1, so no extra pool.
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()
        self.act = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)                              # single pool at block entry
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.act(out)

class ResNet8(nn.Module):
    """Tiny CIFAR ResNet, MicroNets-inspired.

    Stem Conv 3x3 (3→16) + 4 BasicBlocks (16, 32, 64, 64) with stride-2
    downsampling at blocks 2 and 3. ~150K params (incl BN), ~25 M MACs.
    Residuals are essentially free — no extra weights or MACs beyond the
    1x1 shortcut convs at channel-change boundaries.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False),
        )
        self.b1 = _ResNetBasicBlock(16, 16, stride=1)   # 32x32x16
        self.b2 = _ResNetBasicBlock(16, 32, stride=2)   # 16x16x32
        self.b3 = _ResNetBasicBlock(32, 64, stride=2)   # 8x8x64
        self.b4 = _ResNetBasicBlock(64, 64, stride=1)   # 8x8x64
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.b1(x); x = self.b2(x); x = self.b3(x); x = self.b4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class WideImproved(nn.Module):
    """`improved_3x3` with wider channels: 48/48/96 instead of 32/32/64.

    Same depth (3 conv stages), wider features. Tests width-scaling vs
    depth-scaling at a similar MAC budget.
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                                # 16x16x48
            nn.Conv2d(48, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                                # 8x8x48
            nn.Conv2d(48, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                                # 4x4x96
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(4 * 4 * 96, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class NASCifarNet(nn.Module):
    """NAS-found architecture for MAX78000 (Maxim 2021).
    PC twin of ai85nascifarnet. Same channel widths, same conv pattern,
    same activations. BatchNorm uses affine=False to match ai8x's NoAffine
    semantics (no learned scale/shift, only running statistics).
    """
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1_1 = self._cbr(3,   64, k=3)
        self.conv1_2 = self._cbr(64,  32, k=1)
        self.conv1_3 = self._cbr(32,  64, k=3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2_1 = self._cbr(64,  32, k=3)
        self.conv2_2 = self._cbr(32,  64, k=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3_1 = self._cbr(64, 128, k=3)
        self.conv3_2 = self._cbr(128,128, k=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4_1 = self._cbr(128, 64, k=3)
        self.conv4_2 = self._cbr(64, 128, k=3)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5_1 = self._cbr(128,128, k=1)
        self.fc = nn.Linear(512, num_classes)

    @staticmethod
    def _cbr(in_ch: int, out_ch: int, k: int) -> nn.Sequential:
        pad = k // 2
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch, affine=False),   # ai8x uses NoAffine
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        x = self.conv1_1(x); x = self.conv1_2(x); x = self.conv1_3(x)
        x = self.pool1(x)
        x = self.conv2_1(x); x = self.conv2_2(x)
        x = self.pool2(x)
        x = self.conv3_1(x); x = self.conv3_2(x)
        x = self.pool3(x)
        x = self.conv4_1(x); x = self.conv4_2(x)
        x = self.pool4(x)
        x = self.conv5_1(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class ResSimpleNet(nn.Module):
    """SimpleNet v1 + residuals — PC twin of ai85ressimplenetbn (Maxim).

    Architecture from HasanPour et al. (2016) with three residual additions
    inserted by Maxim at strategic points. 14 conv layers + GAP-style 1x1
    classifier head, ~370K parameters.

    PC <-> MAX78000 parity:
      Every conv block here is Conv2d → BatchNorm2d → ReLU.
      On the MAX78000 side, the matching `ai85ressimplenetbn` model uses
      `FusedConv2dBNReLU` / `FusedMaxPoolConv2dBNReLU`. At synthesis time,
      `ai8x-synthesis` folds each BN into the preceding convolution, so the
      DEPLOYED architecture on-device is bit-equivalent to the no-BN variant
      `ai85ressimplenet` originally published by Maxim. BN exists only for
      training stability — without it, the 14-layer residual stack collapses
      to chance accuracy under standard hyperparameters.

    Reference: HasanPour, Rouhani, Fayyaz, Sabokrou.
    "Lets keep it simple, using simple architectures to outperform deeper
     and more complex architectures." arXiv:1608.06037, 2016.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1  = self._cbr(3,   16, k=3)
        self.conv2  = self._cbr(16,  20, k=3)
        self.conv3  = self._cbr(20,  20, k=3)
        self.conv4  = self._cbr(20,  20, k=3)
        # resid1: x = conv3(x) + conv2(x)
        self.pool1  = nn.MaxPool2d(2, 2)
        self.conv5  = self._cbr(20,  20, k=3)   # x_res branch
        self.conv6  = self._cbr(20,  20, k=3)
        # resid2: x = conv6(x) + conv5(x)
        self.conv7  = self._cbr(20,  44, k=3)
        self.pool2  = nn.MaxPool2d(2, 2)
        self.conv8  = self._cbr(44,  48, k=3)   # x_res branch
        self.conv9  = self._cbr(48,  48, k=3)
        # resid3: x = conv9(x) + conv8(x)
        self.pool3  = nn.MaxPool2d(2, 2)
        self.conv10 = self._cbr(48,  96, k=3)
        self.pool4  = nn.MaxPool2d(2, 2)
        self.conv11 = self._cbr(96, 512, k=1)
        self.conv12 = self._cbr(512, 128, k=1)
        self.pool5  = nn.MaxPool2d(2, 2)
        self.conv13 = self._cbr(128, 128, k=3)
        # conv14 is the wide classifier — NO BN, NO ReLU (final logits)
        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1,
                                    stride=1, padding=0, bias=False)

    @staticmethod
    def _cbr(in_ch: int, out_ch: int, k: int) -> nn.Sequential:
        """Conv → BN → ReLU block matching ai8x's FusedConv2dBNReLU.

        BN gets folded into Conv at synthesis on MAX78000 so the deployed
        weights match a no-BN model with proper initialization — the BN is
        purely a training-time scaffold to keep activations well-conditioned
        across the 14-layer stack.
        """
        pad = k // 2
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, stride=1, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x_res = self.conv2(x)
        x = self.conv3(x_res)
        x = x + x_res                       # resid1
        x = self.conv4(x)
        x = self.pool1(x)
        x_res = self.conv5(x)
        x = self.conv6(x_res)
        x = x + x_res                       # resid2
        x = self.conv7(x)
        x = self.pool2(x)
        x_res = self.conv8(x)
        x = self.conv9(x_res)
        x = x + x_res                       # resid3
        x = self.pool3(x)
        x = self.conv10(x)
        x = self.pool4(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.pool5(x)
        x = self.conv13(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1)

MODEL_REGISTRY = {
    "baseline":      CMSISNNBaseline,
    "baseline_5x5":  CMSISNNBaseline5x5,
    "improved":      CMSISNNImproved,
    "separable":     CMSISNNSeparable,
    "mininet":       MiniMobileNet,
    "deeper":        CMSISNNDeeper,
    "resnet8":       ResNet8,
    "wide_improved": WideImproved,
    "nascifarnet": NASCifarNet,
    "ressimplenet":  ResSimpleNet,  
}


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {name!r}. Options: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](num_classes=num_classes)
