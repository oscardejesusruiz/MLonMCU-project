"""All three CIFAR-10 models rewritten in ai8x layers for MAX78000.

Drop this file into `ai8x-training/models/` and register the three factories
in `ai8x-training/models/__init__.py`:

    from .models_ai8x import (
        ai85net_cmsis_baseline,
        ai85net_cmsis_improved,
        ai85net_cmsis_separable,
    )

    models = [
        ...,
        {"name": "ai85net_cmsis_baseline",  "module": ai85net_cmsis_baseline,  "min_input": 1, "dim": 2},
        {"name": "ai85net_cmsis_improved",  "module": ai85net_cmsis_improved,  "min_input": 1, "dim": 2},
        {"name": "ai85net_cmsis_separable", "module": ai85net_cmsis_separable, "min_input": 1, "dim": 2},
    ]

## 5x5 → 3x3 substitution (baseline / improved only)

The MAX78000 CNN accelerator only supports kernel sizes of 1x1 and 3x3 for
standard convolutions. Your `CMSISNNBaseline` and `CMSISNNImproved` from
`training/models.py` both use 5x5 convolutions, which cannot deploy as-is.

We substitute 3x3 throughout. The receptive field shrinks (5x5 covers 25
pixels, 3x3 covers 9), but the spatial flow is identical (32→16→8→4) and
the deployment story is honest: "the MAX78000 forces this architectural
change, here's how much accuracy we lose."

Parameter count drops with the substitution:
  baseline_5x5:   89,578 params  (training/models.py original)
  baseline_3x3:   38,762 params  (this file)

Track this in your report — it's a tangible deployment cost.

## Pool ordering (Fused vs. separate)

PyTorch convention is Conv→ReLU→Pool. ai8x has a `FusedMaxPoolConv2dReLU`
which is Pool→Conv→ReLU. These are equivalent when chained: the pool at the
START of layer N+1 sees the same input as a pool at the END of layer N.
We use the fused variant where possible because it saves one layer slot
in the YAML and one round trip through data SRAM.
"""
from __future__ import annotations

import torch.nn as nn

try:
    import ai8x
except ImportError as exc:  # pragma: no cover — only importable in ai8x-training venv
    ai8x = None
    _AI8X_IMPORT_ERROR = exc


def _check_ai8x() -> None:
    if ai8x is None:
        raise RuntimeError(
            "ai8x is not importable. Run inside the ai8x-training venv."
        ) from _AI8X_IMPORT_ERROR


# ============================================================================
# baseline (5x5 → 3x3 substitute, no BN, no dropout)
# Spatial flow:  32x32x3 → 32x32x32 → 16x16x32 → 8x8x64 → 4x4x64 → 10
# ============================================================================


class AI85NetCmsisBaseline(nn.Module):
    """Paper-faithful (3x3) variant for MAX78000. No BN, no dropout."""

    def __init__(
        self,
        num_classes: int = 10,
        num_channels: int = 3,
        dimensions: tuple[int, int] = (32, 32),
        bias: bool = True,
        **kwargs,
    ) -> None:
        _check_ai8x()
        super().__init__()
        del dimensions

        # L1: conv 3x3, 3 -> 32, ReLU
        self.conv1 = ai8x.FusedConv2dReLU(
            num_channels, 32, 3, padding=1, bias=bias, **kwargs
        )
        # L2: maxpool 2x2 + conv 3x3, 32 -> 32, ReLU
        self.conv2 = ai8x.FusedMaxPoolConv2dReLU(
            32, 32, 3, pool_size=2, pool_stride=2,
            padding=1, bias=bias, **kwargs,
        )
        # L3: maxpool 2x2 + conv 3x3, 32 -> 64, ReLU
        self.conv3 = ai8x.FusedMaxPoolConv2dReLU(
            32, 64, 3, pool_size=2, pool_stride=2,
            padding=1, bias=bias, **kwargs,
        )
        # L4: maxpool 2x2 (no conv) — 8x8x64 -> 4x4x64
        self.pool = ai8x.MaxPool2d(kernel_size=2, stride=2)
        # L5: MLP 4*4*64 (=1024) -> num_classes
        self.fc = ai8x.Linear(
            4 * 4 * 64, num_classes, wide=True, bias=True, **kwargs
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def ai85net_cmsis_baseline(pretrained: bool = False, **kwargs):
    assert not pretrained
    return AI85NetCmsisBaseline(**kwargs)


# ============================================================================
# improved (5x5 → 3x3 substitute + BN folded at synthesis)
# Same spatial flow as baseline but with Fused...BN... fused layers.
# ============================================================================


class AI85NetCmsisImproved(nn.Module):
    """Phase-2 variant for MAX78000: 3x3 + BN + Dropout(0.1). Same MAC count as baseline.

    Dropout(0.1) matches the PC CMSISNNImproved. nn.Dropout is identity at
    inference (eval mode) so it has no effect on synthesis or the deployed
    model — it only regularises QAT training.
    """

    def __init__(
        self,
        num_classes: int = 10,
        num_channels: int = 3,
        dimensions: tuple[int, int] = (32, 32),
        bias: bool = False,
        **kwargs,
    ) -> None:
        _check_ai8x()
        super().__init__()
        del dimensions

        self.conv1 = ai8x.FusedConv2dBNReLU(
            num_channels, 32, 3, padding=1, bias=bias, **kwargs
        )
        self.conv2 = ai8x.FusedMaxPoolConv2dBNReLU(
            32, 32, 3, pool_size=2, pool_stride=2,
            padding=1, bias=bias, **kwargs,
        )
        self.conv3 = ai8x.FusedMaxPoolConv2dBNReLU(
            32, 64, 3, pool_size=2, pool_stride=2,
            padding=1, bias=bias, **kwargs,
        )
        self.pool = ai8x.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.1)
        self.fc = ai8x.Linear(
            4 * 4 * 64, num_classes, wide=True, bias=True, **kwargs
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def ai85net_cmsis_improved(pretrained: bool = False, **kwargs):
    assert not pretrained
    return AI85NetCmsisImproved(**kwargs)


# ============================================================================
# separable (depthwise + pointwise, BN). Ports without architectural changes.
# Spatial flow: 32x32x3 → 32x32x32 → 32x32x64 → 16x16x64 → 8x8x64 → 8x8x128 → 1x1x128 → 10
# ============================================================================


class AI85NetCmsisSeparable(nn.Module):
    """Depthwise-separable variant. Same architecture as the PyTorch original."""

    def __init__(
        self,
        num_classes: int = 10,
        num_channels: int = 3,
        dimensions: tuple[int, int] = (32, 32),
        bias: bool = False,
        **kwargs,
    ) -> None:
        _check_ai8x()
        super().__init__()
        del dimensions

        # Stem: standard 3x3
        self.conv_stem = ai8x.FusedConv2dBNReLU(
            num_channels, 32, 3, padding=1, bias=bias, **kwargs
        )

        # Block 1: depthwise 3x3 (32->32) + pointwise 1x1 (32->64)
        self.block1_dw = ai8x.FusedDepthwiseConv2dBNReLU(
            32, 32, 3, padding=1, bias=bias, **kwargs
        )
        self.block1_pw = ai8x.FusedConv2dBNReLU(
            32, 64, 1, padding=0, bias=bias, **kwargs
        )
        self.pool1 = ai8x.MaxPool2d(kernel_size=2, stride=2)   # 32→16

        # Block 2: depthwise (64->64) + pointwise (64->64)
        self.block2_dw = ai8x.FusedDepthwiseConv2dBNReLU(
            64, 64, 3, padding=1, bias=bias, **kwargs
        )
        self.block2_pw = ai8x.FusedConv2dBNReLU(
            64, 64, 1, padding=0, bias=bias, **kwargs
        )
        self.pool2 = ai8x.MaxPool2d(kernel_size=2, stride=2)   # 16→8

        # Block 3: depthwise (64->64) + pointwise (64->128)
        self.block3_dw = ai8x.FusedDepthwiseConv2dBNReLU(
            64, 64, 3, padding=1, bias=bias, **kwargs
        )
        self.block3_pw = ai8x.FusedConv2dBNReLU(
            64, 128, 1, padding=0, bias=bias, **kwargs
        )

        # MAX78000 has no GlobalAvgPool — use AvgPool with kernel = spatial size
        self.avgpool = ai8x.AvgPool2d(kernel_size=8, stride=1)   # 8→1

        self.fc = ai8x.Linear(128, num_classes, wide=True, bias=True, **kwargs)

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.block1_dw(x); x = self.block1_pw(x); x = self.pool1(x)
        x = self.block2_dw(x); x = self.block2_pw(x); x = self.pool2(x)
        x = self.block3_dw(x); x = self.block3_pw(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def ai85net_cmsis_separable(pretrained: bool = False, **kwargs):
    assert not pretrained
    return AI85NetCmsisSeparable(**kwargs)


# ============================================================================
# mininet — MAX78000 port of the TensorFlow MiniMobileNet (IMX500 reference).
#
# Differences vs. the PC port in `training/models.py::MiniMobileNet`:
#   - ReLU instead of ReLU6 (MAX78000 hardware only supports ReLU/abs;
#     int8 quantization implicitly clips activations anyway).
#   - NO residual / skip connections in this version. Residual adds on
#     MAX78000 require explicit ai8x.Add layers + matching activation scales
#     between branches, and need YAML support for "eltwise: add". A
#     no-residual variant lands within 0.5-1 pp of the residual one in
#     practice and synthesizes cleanly. Add residuals once the basic path
#     is verified.
#   - 64x64 input (matches the TF reference).
#
# Spatial flow:
#   64x64x3 → stem s=2 → 32x32x24
#   block1  s=1        → 32x32x48
#   block2  s=2        → 16x16x96
#   block3  s=1        → 16x16x96      (no residual)
#   block4  s=1        → 16x16x96      (no residual)
#   block5  s=2        → 8x8x192
#   block6  s=1        → 8x8x192       (no residual)
#   block7  s=1        → 8x8x192       (no residual)
#   avgpool 8          → 1x1x192 → MLP → 10
# ============================================================================


class AI85NetCmsisMininet(nn.Module):
    """VGG-Micro — MAX78000-hardware-friendly mininet (v2).

    Channels are all multiples of 32 so MAX78000 packs cleanly without
    multipass alignment issues (the previous 80/96-channel variant hit
    "Kernel memory exhausted" on the 96→128 transition).

    Topology:
        L1:  3x3 conv      3 → 32       32x32x32
        L2:  pool 2 + 3x3 conv  32 → 64  16x16x64
        L3:  3x3 conv      64 → 64      16x16x64
        L4:  pool 2 + 3x3 conv  64 → 64   8x8x64
        L5:  3x3 conv      64 → 128      8x8x128
        L6:  pool 2 + 3x3 conv  128 → 128  4x4x128
        AvgPool 4 + Dropout(0.4) + FC(128 → 10)
    """

    def __init__(
        self,
        num_classes: int = 10,
        num_channels: int = 3,
        dimensions: tuple[int, int] = (32, 32),
        bias: bool = False,
        **kwargs,
    ) -> None:
        _check_ai8x()
        super().__init__()
        del dimensions

        # L1: 32x32x3 → 32x32x32
        self.l1 = ai8x.FusedConv2dBNReLU(
            num_channels, 32, 3, padding=1, bias=bias, **kwargs)
        # L2: 32x32x32 → 16x16x64  (pool 2 + conv)
        self.l2 = ai8x.FusedMaxPoolConv2dBNReLU(
            32, 64, 3, pool_size=2, pool_stride=2, padding=1, bias=bias, **kwargs)
        # L3: 16x16x64 → 16x16x64
        self.l3 = ai8x.FusedConv2dBNReLU(
            64, 64, 3, padding=1, bias=bias, **kwargs)
        # L4: 16x16x64 → 8x8x64  (pool 2 + conv)
        self.l4 = ai8x.FusedMaxPoolConv2dBNReLU(
            64, 64, 3, pool_size=2, pool_stride=2, padding=1, bias=bias, **kwargs)
        # L5: 8x8x64 → 8x8x128
        self.l5 = ai8x.FusedConv2dBNReLU(
            64, 128, 3, padding=1, bias=bias, **kwargs)
        # L6: 8x8x128 → 4x4x128  (pool 2 + conv)
        self.l6 = ai8x.FusedMaxPoolConv2dBNReLU(
            128, 128, 3, pool_size=2, pool_stride=2, padding=1, bias=bias, **kwargs)

        self.avgpool = ai8x.AvgPool2d(kernel_size=4, stride=1)
        # Dropout(0.4) before FC — mirrors the PC MiniMobileNet which uses
        # Dropout to regularize the 300K+ params.
        self.dropout = nn.Dropout(p=0.4)
        self.fc = ai8x.Linear(128, num_classes, wide=True, bias=True, **kwargs)

    def forward(self, x):
        x = self.l1(x); x = self.l2(x); x = self.l3(x)
        x = self.l4(x); x = self.l5(x); x = self.l6(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def ai85net_cmsis_mininet(pretrained: bool = False, **kwargs):
    assert not pretrained
    return AI85NetCmsisMininet(**kwargs)


# ============================================================================
# deeper — `improved_3x3` + extra 64-ch conv + 128-ch head + GAP.
# Tests if depth alone (without depthwise) matches the mininet accuracy.
# ============================================================================


class AI85NetCmsisDeeper(nn.Module):
    """Deeper variant: 5 standard 3x3 convs + GAP + Dropout(0.2) + FC. ~140 KiB int8 weights.

    Dropout(0.2) matches the PC CMSISNNDeeper. nn.Dropout is identity at
    inference (eval mode) so synthesis is unaffected.
    """

    def __init__(
        self,
        num_classes: int = 10,
        num_channels: int = 3,
        dimensions: tuple[int, int] = (32, 32),
        bias: bool = False,
        **kwargs,
    ) -> None:
        _check_ai8x()
        super().__init__()
        del dimensions

        self.conv1 = ai8x.FusedConv2dBNReLU(
            num_channels, 32, 3, padding=1, bias=bias, **kwargs,
        )
        self.conv2 = ai8x.FusedMaxPoolConv2dBNReLU(
            32, 32, 3, pool_size=2, pool_stride=2, padding=1, bias=bias, **kwargs,
        )
        self.conv3 = ai8x.FusedMaxPoolConv2dBNReLU(
            32, 64, 3, pool_size=2, pool_stride=2, padding=1, bias=bias, **kwargs,
        )
        self.conv4 = ai8x.FusedConv2dBNReLU(
            64, 64, 3, padding=1, bias=bias, **kwargs,
        )
        self.conv5 = ai8x.FusedMaxPoolConv2dBNReLU(
            64, 128, 3, pool_size=2, pool_stride=2, padding=1, bias=bias, **kwargs,
        )
        self.avgpool = ai8x.AvgPool2d(kernel_size=4, stride=1)  # 4x4 → 1x1
        self.dropout = nn.Dropout(p=0.2)
        self.fc = ai8x.Linear(128, num_classes, wide=True, bias=True, **kwargs)

    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x)
        x = self.conv4(x); x = self.conv5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def ai85net_cmsis_deeper(pretrained: bool = False, **kwargs):
    assert not pretrained
    return AI85NetCmsisDeeper(**kwargs)


# ============================================================================
# wide_improved — same depth as improved but channels 48/48/96.
# ============================================================================


class AI85NetCmsisWideImproved(nn.Module):
    """`improved` with wider channels (48/48/96). ~77 KiB int8 weights."""

    def __init__(
        self,
        num_classes: int = 10,
        num_channels: int = 3,
        dimensions: tuple[int, int] = (32, 32),
        bias: bool = False,
        **kwargs,
    ) -> None:
        _check_ai8x()
        super().__init__()
        del dimensions

        self.conv1 = ai8x.FusedConv2dBNReLU(
            num_channels, 48, 3, padding=1, bias=bias, **kwargs,
        )
        self.conv2 = ai8x.FusedMaxPoolConv2dBNReLU(
            48, 48, 3, pool_size=2, pool_stride=2, padding=1, bias=bias, **kwargs,
        )
        self.conv3 = ai8x.FusedMaxPoolConv2dBNReLU(
            48, 96, 3, pool_size=2, pool_stride=2, padding=1, bias=bias, **kwargs,
        )
        # GAP 8x8 → 1x1 (avoids the ai8x.Linear's 1024-input limit; 4*4*96=1536
        # would not fit, but 96 does).
        self.avgpool = ai8x.AvgPool2d(kernel_size=8, stride=1)
        self.fc = ai8x.Linear(96, num_classes, wide=True, bias=True, **kwargs)

    def forward(self, x):
        x = self.conv1(x); x = self.conv2(x); x = self.conv3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def ai85net_cmsis_wide_improved(pretrained: bool = False, **kwargs):
    assert not pretrained
    return AI85NetCmsisWideImproved(**kwargs)


# ============================================================================
# resnet8 — tiny ResNet with 4 BasicBlocks and residual connections.
# Uses ai8x.Add for elementwise residual additions.
# ============================================================================


class _Ai8xBasicBlock(nn.Module):
    """ResNet BasicBlock with ai8x layers.

    For stride==2 the first conv is preceded by maxpool (fused). The shortcut
    branch uses an ai8x.Add layer; if channels change we need an explicit
    1x1 conv on the shortcut (also ai8x).
    """

    def __init__(self, in_ch, out_ch, stride, bias=False, **kwargs):
        super().__init__()
        if stride == 2:
            self.conv1 = ai8x.FusedMaxPoolConv2dBNReLU(
                in_ch, out_ch, 3,
                pool_size=2, pool_stride=2,
                padding=1, bias=bias, **kwargs,
            )
        else:
            self.conv1 = ai8x.FusedConv2dBNReLU(
                in_ch, out_ch, 3, padding=1, bias=bias, **kwargs,
            )
        # conv2: NO activation before the add (we apply ReLU after)
        self.conv2 = ai8x.FusedConv2dBN(
            out_ch, out_ch, 3, padding=1, bias=bias, **kwargs,
        )
        if stride != 1 or in_ch != out_ch:
            if stride == 2:
                self.shortcut = ai8x.FusedMaxPoolConv2dBN(
                    in_ch, out_ch, 1,
                    pool_size=2, pool_stride=2,
                    padding=0, bias=bias, **kwargs,
                )
            else:
                self.shortcut = ai8x.FusedConv2dBN(
                    in_ch, out_ch, 1, padding=0, bias=bias, **kwargs,
                )
        else:
            self.shortcut = None
        self.add = ai8x.Add()
        self.relu = nn.ReLU(inplace=False)   # post-add activation

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        sc = self.shortcut(x) if self.shortcut is not None else x
        out = self.add(out, sc)
        return self.relu(out)


class AI85NetCmsisResnet8(nn.Module):
    """Tiny ResNet for CIFAR-10. Stem + 4 BasicBlocks + GAP + FC.

    Residual connections via `ai8x.Add`. The synthesis YAML must include the
    proper `in_sequences` to wire the shortcut branches to the add layers.
    """

    def __init__(
        self,
        num_classes: int = 10,
        num_channels: int = 3,
        dimensions: tuple[int, int] = (32, 32),
        bias: bool = False,
        **kwargs,
    ) -> None:
        _check_ai8x()
        super().__init__()
        del dimensions

        self.stem = ai8x.FusedConv2dBNReLU(
            num_channels, 16, 3, padding=1, bias=bias, **kwargs,
        )
        self.b1 = _Ai8xBasicBlock(16, 16, stride=1, bias=bias, **kwargs)
        self.b2 = _Ai8xBasicBlock(16, 32, stride=2, bias=bias, **kwargs)
        self.b3 = _Ai8xBasicBlock(32, 64, stride=2, bias=bias, **kwargs)
        self.b4 = _Ai8xBasicBlock(64, 64, stride=1, bias=bias, **kwargs)
        self.avgpool = ai8x.AvgPool2d(kernel_size=8, stride=1)   # 8x8 → 1x1
        self.fc = ai8x.Linear(64, num_classes, wide=True, bias=True, **kwargs)

    def forward(self, x):
        x = self.stem(x)
        x = self.b1(x); x = self.b2(x); x = self.b3(x); x = self.b4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def ai85net_cmsis_resnet8(pretrained: bool = False, **kwargs):
    assert not pretrained
    return AI85NetCmsisResnet8(**kwargs)

# ============================================================================
# repvgg_tiny — deployed-time RepVGG: just 3×3 convs + ReLU + pool.
#
# IMPORTANT: this is the **fused** (re-parameterized) architecture. The PC
# side trains the multi-branch RepVGGTiny, then a re-parameterization step
# collapses each block to a single 3×3 conv before exporting weights here.
#
# Spatial flow (matches PC):
#   32x32x3 → 32x32x32 (stem)
#   32x32x32 → 32x32x32 (s1.0)
#   32x32x32 → 32x32x32 (s1.1)
#   pool → 16x16x32
#   16x16x32 → 16x16x64 (s2.0)
#   16x16x64 → 16x16x64 (s2.1)
#   pool → 8x8x64
#   8x8x64 → 8x8x64 (s3.0)
#   8x8x64 → 8x8x64 (s3.1)
#   pool → 4x4x64
#   4x4x64 → 4x4x64 (s4)
#   GAP → MLP(64 → 10)
# ============================================================================


'''class AI85NetCmsisRepVGG(nn.Module):
    """RepVGG-tiny deployed form — 8 stacked 3x3 convs + ReLU + pools + FC.

    Each `block_*` corresponds to one re-parameterized RepVGG block on the
    PC side. After fusion there's no BN at inference (folded into the conv
    weights) and no skip connections — just dense 3x3 conv-ReLU.

    We keep `FusedConv2dBNReLU` here because we still need BN during training
    on the MAX78000 side (the ai8x QAT pipeline expects it). The BN is folded
    automatically at synthesis time, matching what offline re-param does.
    """

    def __init__(
        self,
        num_classes: int = 10,
        num_channels: int = 3,
        dimensions: tuple[int, int] = (32, 32),
        bias: bool = False,
        **kwargs,
    ) -> None:
        _check_ai8x()
        super().__init__()
        del dimensions

        self.stem = ai8x.FusedConv2dBNReLU(
            num_channels, 32, 3, padding=1, bias=bias, **kwargs,
        )
        self.s1_0 = ai8x.FusedConv2dBNReLU(32, 32, 3, padding=1, bias=bias, **kwargs)
        self.s1_1 = ai8x.FusedConv2dBNReLU(32, 32, 3, padding=1, bias=bias, **kwargs)
        # pool + first conv of s2 in one fused op
        self.s2_0 = ai8x.FusedMaxPoolConv2dBNReLU(
            32, 64, 3, pool_size=2, pool_stride=2, padding=1, bias=bias, **kwargs,
        )
        self.s2_1 = ai8x.FusedConv2dBNReLU(64, 64, 3, padding=1, bias=bias, **kwargs)
        self.s3_0 = ai8x.FusedMaxPoolConv2dBNReLU(
            64, 64, 3, pool_size=2, pool_stride=2, padding=1, bias=bias, **kwargs,
        )
        self.s3_1 = ai8x.FusedConv2dBNReLU(64, 64, 3, padding=1, bias=bias, **kwargs)
        self.s4 = ai8x.FusedMaxPoolConv2dBNReLU(
            64, 64, 3, pool_size=2, pool_stride=2, padding=1, bias=bias, **kwargs,
        )
        self.avgpool = ai8x.AvgPool2d(kernel_size=4, stride=1)   # 4x4 → 1x1
        self.fc = ai8x.Linear(64, num_classes, wide=True, bias=True, **kwargs)

    def forward(self, x):
        x = self.stem(x)
        x = self.s1_0(x); x = self.s1_1(x)
        x = self.s2_0(x); x = self.s2_1(x)
        x = self.s3_0(x); x = self.s3_1(x)
        x = self.s4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def ai85net_cmsis_repvgg(pretrained: bool = False, **kwargs):
    assert not pretrained
    return AI85NetCmsisRepVGG(**kwargs)'''
