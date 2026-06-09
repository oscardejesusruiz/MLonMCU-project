"""Model accounting helpers: MAC count, parameter/byte sizes."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ModelStats:
    params: int
    weight_bytes_fp32: int
    weight_bytes_int8: int
    macs: int
    layer_breakdown: list[tuple[str, int]]  # (layer_name, macs)

    def summary(self) -> str:
        lines = [
            f"Parameters       : {self.params:>12,}",
            f"Weights (fp32)   : {self.weight_bytes_fp32 / 1024:>9.1f} KiB",
            f"Weights (int8)   : {self.weight_bytes_int8 / 1024:>9.1f} KiB",
            f"MACs / inference : {self.macs / 1e6:>9.2f} M",
            "",
            "Per-layer MACs:",
        ]
        for name, m in self.layer_breakdown:
            lines.append(f"  {name:<30} {m / 1e6:>8.3f} M")
        return "\n".join(lines)


def _count_module_macs(module: nn.Module, x: torch.Tensor) -> int:
    """MACs for the most common compute layers given input shape."""
    if isinstance(module, nn.Conv2d):
        out_h = (
            x.shape[2] + 2 * module.padding[0] - module.kernel_size[0]
        ) // module.stride[0] + 1
        out_w = (
            x.shape[3] + 2 * module.padding[1] - module.kernel_size[1]
        ) // module.stride[1] + 1
        kh, kw = module.kernel_size
        in_per_group = module.in_channels // module.groups
        return out_h * out_w * module.out_channels * in_per_group * kh * kw
    if isinstance(module, nn.Linear):
        return module.in_features * module.out_features
    return 0


def compute_stats(model: nn.Module, input_shape: tuple[int, ...] = (1, 3, 32, 32)) -> ModelStats:
    """Forward a dummy tensor and tally MACs + parameter bytes.

    Run on CPU so we don't push tensors to MPS just for accounting.
    """
    model = model.cpu().eval()
    layer_macs: list[tuple[str, int]] = []
    handles = []

    def make_hook(name: str):
        def hook(mod: nn.Module, inputs, output):
            x = inputs[0]
            m = _count_module_macs(mod, x)
            if m:
                layer_macs.append((name, m))
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Linear)):
            handles.append(mod.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        model(torch.zeros(input_shape))

    for h in handles:
        h.remove()

    params = sum(p.numel() for p in model.parameters())
    return ModelStats(
        params=params,
        weight_bytes_fp32=params * 4,
        weight_bytes_int8=params,
        macs=sum(m for _, m in layer_macs),
        layer_breakdown=layer_macs,
    )


def pick_device(prefer_mps: bool = True) -> torch.device:
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
