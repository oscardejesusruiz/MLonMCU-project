"""Int8 quantization that mirrors CMSIS-NN q7 semantics.

Symmetric, per-tensor, signed 8-bit. Optionally constrains scales to powers of
two — that's what Lai et al. 2018 use so that re-quantization between layers
collapses to a shift, no multiplier needed.

Two flavors are supported:

* `quantize_model_ptq` — post-training: weights are replaced in-place with their
  fake-quantized values, activations are fake-quanted via pre-forward hooks
  using calibrated scales.
* `convert_to_qat` — quantization-aware: each Conv2d/Linear is swapped for a
  QAT version that fake-quants weight+activation in `forward` with a straight-
  through estimator on the backward pass, so the network can learn to absorb
  the quantization noise.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

INT8_QMIN, INT8_QMAX = -128, 127


def _power_of_two_scale(scale: float) -> float:
    """Round scale UP to the next power of two so quantized values still fit."""
    if scale <= 0:
        return 1.0
    return 2 ** math.ceil(math.log2(scale))


def fake_quant_symmetric(x: torch.Tensor, scale: float) -> torch.Tensor:
    if scale <= 0:
        return x
    q = torch.round(x / scale).clamp(INT8_QMIN, INT8_QMAX)
    return q * scale


@dataclass
class QuantConfig:
    power_of_two: bool = True   # CMSIS-NN q7 convention
    n_calib_batches: int = 20


@dataclass
class QuantStats:
    weight_scales: dict[str, float]
    activation_scales: dict[str, float]


def _compute_weight_scale(weight: torch.Tensor, power_of_two: bool) -> float:
    max_abs = weight.detach().abs().max().item()
    if max_abs == 0:
        return 1.0
    scale = max_abs / INT8_QMAX
    if power_of_two:
        scale = _power_of_two_scale(scale)
    return scale


def _fuse_bn_into_conv(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> None:
    """Fold one BN's running stats + affine params into the preceding Conv.

    Math (BN in eval mode follows Conv → activation):
        z = Conv(x) = W·x + b
        y = γ·(z − μ)/√(σ²+ε) + β
          = (γ/√(σ²+ε))·W · x   +   (γ/√(σ²+ε))·(b − μ) + β
            └─── new W' ────┘       └────── new b' ──────┘

    Handles `affine=False` (γ=1, β=0) and conv without bias (creates one).
    The model is mutated in place; call this BEFORE quantization so the
    redistributed weight magnitudes (often with per-channel outliers when
    a BN has small running_var) drive the quantization scale.
    """
    mean = bn.running_mean.detach().clone()
    var = bn.running_var.detach().clone()
    eps = bn.eps
    std = torch.sqrt(var + eps)

    if bn.affine:
        gamma = bn.weight.detach().clone()
        beta = bn.bias.detach().clone()
    else:
        gamma = torch.ones_like(mean)
        beta = torch.zeros_like(mean)

    scale_per_channel = gamma / std            # (out_channels,)
    # Broadcast over (in_ch, kH, kW) for the conv weight tensor shape
    conv.weight.data = conv.weight.data * scale_per_channel.reshape(-1, 1, 1, 1)

    if conv.bias is None:
        # Create a bias so the fused offset has somewhere to live
        conv.bias = nn.Parameter(torch.zeros(conv.out_channels))
    conv.bias.data = (conv.bias.data - mean) * scale_per_channel + beta


def fold_all_bn(model: nn.Module) -> int:
    """Walk the model and fold every detectable (Conv2d → BatchNorm2d) pair.

    Replaces each folded BN with `nn.Identity()` so subsequent forwards
    return numerically identical results in fp32. Detects two patterns:

      (A) `nn.Sequential(..., Conv2d, BatchNorm2d, ...)` — consecutive children
          (used by ResSimpleNet's `_cbr`, NASCifarNet's `_cbr`,
           CMSISNNBaseline/Improved/Deeper/WideImproved, MiniMobileNet's
           `_VGGBlock` when wrapped in Sequential, etc.)

      (B) Sibling attributes following the `convN`/`bnN` naming convention
          (used by ResNet8's `_ResNetBasicBlock`: `self.conv1` + `self.bn1`,
           and MiniMobileNet's `_VGGBlock` direct attributes).

    Returns the number of (conv, bn) pairs folded.

    Model must be in eval() so BN uses running_mean/running_var (not the
    batch statistics from whatever happened to be in the buffer).
    """
    assert not model.training, "fold_all_bn requires model.eval() to be set first"
    folded = 0

    # --- Pattern A: nn.Sequential children ---
    for parent in model.modules():
        if not isinstance(parent, nn.Sequential):
            continue
        children = list(parent.children())
        for i in range(len(children) - 1):
            cur, nxt = children[i], children[i + 1]
            if isinstance(cur, nn.Conv2d) and isinstance(nxt, nn.BatchNorm2d):
                _fuse_bn_into_conv(cur, nxt)
                parent[i + 1] = nn.Identity()   # BN now a no-op
                folded += 1

    # --- Pattern B: sibling attributes convN/bnN ---
    for parent in model.modules():
        if isinstance(parent, nn.Sequential):
            continue   # already handled by pattern A
        # Find every Conv2d attribute of this parent
        for attr_name in list(vars(parent).get("_modules", {})):
            conv = getattr(parent, attr_name, None)
            if not isinstance(conv, nn.Conv2d):
                continue
            # Try to find a matching BN attribute: convN → bnN, conv → bn
            if attr_name.startswith("conv"):
                bn_name = "bn" + attr_name[len("conv"):]
            else:
                continue
            bn = getattr(parent, bn_name, None)
            if isinstance(bn, nn.BatchNorm2d):
                _fuse_bn_into_conv(conv, bn)
                setattr(parent, bn_name, nn.Identity())
                folded += 1

    return folded


def quantize_model_ptq(
    model: nn.Module,
    calib_loader: DataLoader,
    config: QuantConfig | None = None,
) -> tuple[nn.Module, QuantStats]:
    """Return a fake-quantized copy of `model` along with the chosen scales.

    Mirrors the MAX78000 deployment pipeline:

    0. **BN fold** every detected (Conv2d → BatchNorm2d) pair into the
       Conv's weight & bias. This is the step every fixed-point accelerator
       has to do (the MAX78000 does it during synthesis via `bn_fuser_v2`)
       and is the single largest source of accuracy loss from PTQ — channels
       with small BN running_var produce outlier folded weights that
       collapse per-tensor symmetric scales.
    1. Replace each Conv2d/Linear weight tensor with its int8 fake-quant
       (symmetric per-tensor, power-of-two scales by default).
    2. Calibrate activation scales (per Conv/Linear input) with a pre-forward
       hook that records running max-abs over `n_calib_batches` batches.
    3. Install permanent fake-quant pre-hooks on those layers.

    After this function returns, the model's BN layers are `Identity`,
    its Conv weights are on the int8 grid, and every Conv/Linear input is
    rounded to int8 at runtime — matching what the MAX78000 actually
    deploys, not the optimistic "BN-preserved" simulation.
    """
    cfg = config or QuantConfig()
    qmodel = copy.deepcopy(model).cpu().eval()

    # 0) fold every Conv2d → BatchNorm2d pair into the conv
    n_folded = fold_all_bn(qmodel)

    # 1) static weight quantization (now over post-BN-fold weights)
    weight_scales: dict[str, float] = {}
    for name, mod in qmodel.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            scale = _compute_weight_scale(mod.weight.data, cfg.power_of_two)
            weight_scales[name] = scale
            with torch.no_grad():
                mod.weight.data.copy_(fake_quant_symmetric(mod.weight.data, scale))

    # 2) activation calibration (pre-forward hooks recording running max)
    act_max: dict[str, float] = {}
    calib_handles = []

    def _make_calib_hook(layer_name: str):
        def hook(_mod: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            x = inputs[0]
            cur = x.detach().abs().max().item()
            act_max[layer_name] = max(act_max.get(layer_name, 0.0), cur)
        return hook

    for name, mod in qmodel.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            calib_handles.append(mod.register_forward_pre_hook(_make_calib_hook(name)))

    with torch.no_grad():
        for i, (x, _) in enumerate(calib_loader):
            if i >= cfg.n_calib_batches:
                break
            qmodel(x)

    for h in calib_handles:
        h.remove()

    activation_scales: dict[str, float] = {}
    for name, max_abs in act_max.items():
        if max_abs == 0:
            activation_scales[name] = 1.0
            continue
        scale = max_abs / INT8_QMAX
        if cfg.power_of_two:
            scale = _power_of_two_scale(scale)
        activation_scales[name] = scale

    # 3) permanent fake-quant pre-hooks on inputs of every Conv/Linear
    def _make_quant_hook(scale: float):
        def hook(_mod: nn.Module, inputs: tuple[torch.Tensor, ...]):
            x = inputs[0]
            return (fake_quant_symmetric(x, scale),) + inputs[1:]
        return hook

    for name, mod in qmodel.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            mod.register_forward_pre_hook(_make_quant_hook(activation_scales[name]))

    return qmodel, QuantStats(weight_scales=weight_scales, activation_scales=activation_scales)


# ---- QAT ---------------------------------------------------------------------


class _FakeQuantSTE(torch.autograd.Function):
    """Fake-quant in forward, straight-through gradient in backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        if scale <= 0:
            return x
        q = torch.round(x / scale).clamp(INT8_QMIN, INT8_QMAX)
        return q * scale

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


def _scale_from_max_abs(max_abs: float, power_of_two: bool) -> float:
    if max_abs <= 0:
        return 1.0
    s = max_abs / INT8_QMAX
    return _power_of_two_scale(s) if power_of_two else s


class QATConv2d(nn.Module):
    def __init__(self, original: nn.Conv2d, in_act_scale: float, power_of_two: bool) -> None:
        super().__init__()
        self.weight = original.weight  # nn.Parameter — gradient flows
        self.bias = original.bias
        self.stride = original.stride
        self.padding = original.padding
        self.dilation = original.dilation
        self.groups = original.groups
        self.power_of_two = power_of_two
        self.register_buffer("in_act_scale", torch.tensor(float(in_act_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _FakeQuantSTE.apply(x, self.in_act_scale.item())
        w_scale = _scale_from_max_abs(
            self.weight.detach().abs().max().item(), self.power_of_two
        )
        w = _FakeQuantSTE.apply(self.weight, w_scale)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


class QATLinear(nn.Module):
    def __init__(self, original: nn.Linear, in_act_scale: float, power_of_two: bool) -> None:
        super().__init__()
        self.weight = original.weight
        self.bias = original.bias
        self.power_of_two = power_of_two
        self.register_buffer("in_act_scale", torch.tensor(float(in_act_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _FakeQuantSTE.apply(x, self.in_act_scale.item())
        w_scale = _scale_from_max_abs(
            self.weight.detach().abs().max().item(), self.power_of_two
        )
        w = _FakeQuantSTE.apply(self.weight, w_scale)
        return F.linear(x, w, self.bias)


def _replace_module(parent: nn.Module, dotted_name: str, new_mod: nn.Module) -> None:
    parts = dotted_name.split(".")
    obj = parent
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], new_mod)


def convert_to_qat(
    model: nn.Module,
    calib_loader: DataLoader,
    config: QuantConfig | None = None,
) -> tuple[nn.Module, dict[str, float]]:
    """Calibrate activation scales then swap Conv2d/Linear for QAT modules.

    The returned model has the same parameter tensors as the input — gradients
    flow into the original weights — but every conv/linear forward applies
    weight + activation fake-quant.
    """
    cfg = config or QuantConfig()
    qmodel = copy.deepcopy(model).cpu().eval()

    # 1) calibrate activation scales (no weight quant yet — the calibration is
    #    on the un-quantized network so scales reflect the real dynamic range)
    act_max: dict[str, float] = {}
    handles = []

    def _make_calib_hook(layer_name: str):
        def hook(_mod: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            x = inputs[0]
            cur = x.detach().abs().max().item()
            act_max[layer_name] = max(act_max.get(layer_name, 0.0), cur)
        return hook

    for name, mod in qmodel.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            handles.append(mod.register_forward_pre_hook(_make_calib_hook(name)))

    with torch.no_grad():
        for i, (x, _) in enumerate(calib_loader):
            if i >= cfg.n_calib_batches:
                break
            qmodel(x)

    for h in handles:
        h.remove()

    activation_scales = {
        name: _scale_from_max_abs(m, cfg.power_of_two) for name, m in act_max.items()
    }

    # 2) swap modules
    qat_targets: list[tuple[str, nn.Module]] = [
        (name, mod)
        for name, mod in qmodel.named_modules()
        if isinstance(mod, (nn.Conv2d, nn.Linear))
    ]
    for name, mod in qat_targets:
        scale = activation_scales[name]
        if isinstance(mod, nn.Conv2d):
            _replace_module(qmodel, name, QATConv2d(mod, scale, cfg.power_of_two))
        else:
            _replace_module(qmodel, name, QATLinear(mod, scale, cfg.power_of_two))

    return qmodel, activation_scales
