"""Fold BatchNorm into the preceding Conv for arbitrarily-nested ai8x state
dicts (the upstream `batchnormfuser.py` breaks on >1 level of nesting like
`b1.conv1.bn.*`).

For each `<layer>.bn.running_mean` key, fold the BN params into the conv
weight at `<layer>.op.weight` (or `<layer>.conv2d.weight` as fallback).
The same beta/gamma/r_mean/r_std math the upstream uses (with the 0.25
ai8x-specific rescale on beta and gamma).

Usage:
    python bn_fuser_v2.py -i input.pth.tar -o output_fused.pth.tar -oa <arch>
"""
from __future__ import annotations

import argparse

import torch


def bn_fuser(state_dict: dict) -> dict:
    # Discover every layer that has a BN sibling, by looking for
    # `*.bn.running_mean` keys and stripping the trailing `.bn.running_mean`.
    suffix = ".bn.running_mean"
    layers = sorted({k[:-len(suffix)] for k in state_dict.keys() if k.endswith(suffix)})

    for layer in layers:
        # Find the conv-weight key for this layer
        for cand in (f"{layer}.op.weight", f"{layer}.conv2d.weight"):
            if cand in state_dict:
                w_key = cand
                conv_key = cand[:-len(".weight")]
                break
        else:
            print(f"[skip] no conv weight found for layer '{layer}'")
            continue

        b_key       = f"{conv_key}.bias"
        bn_key      = f"{layer}.bn"
        r_mean_key  = f"{bn_key}.running_mean"
        r_var_key   = f"{bn_key}.running_var"
        beta_key    = f"{bn_key}.weight"    # in PyTorch nn.BatchNorm2d, weight = gamma
        gamma_key   = f"{bn_key}.bias"      # bias = beta (note: ai8x uses swapped naming)
        batches_key = f"{bn_key}.num_batches_tracked"

        w = state_dict[w_key]
        device = w.device

        b      = state_dict.get(b_key, torch.zeros(w.shape[0], device=device))
        r_mean = state_dict[r_mean_key]
        r_var  = state_dict[r_var_key]
        r_std  = torch.sqrt(r_var + 1e-20)
        beta   = state_dict.get(beta_key,  torch.ones (w.shape[0], device=device))
        gamma  = state_dict.get(gamma_key, torch.zeros(w.shape[0], device=device))

        # NOTE: do NOT apply the 0.25 rescale that upstream's
        # `batchnormfuser.py` does. That factor is only correct for the legacy
        # QAT-saved-weight scale convention; for fp32→PTQ (our pipeline) it
        # divides activations by 4 per layer and the int8-quantized result
        # collapses to chance. The official in-process fold inside
        # `ai8x.fuse_bn_layers` (ai8x.py line 2060) does NOT apply 0.25 — we
        # mirror that here. Verified with verify_fold.py.
        # beta  = 0.25 * beta   # ← removed
        # gamma = 0.25 * gamma  # ← removed

        w_new = w * (beta / r_std).reshape((w.shape[0],) + (1,) * (len(w.shape) - 1))
        b_new = (b - r_mean) / r_std * beta + gamma

        state_dict[w_key] = w_new
        state_dict[b_key] = b_new

        for k in (r_mean_key, r_var_key, beta_key, gamma_key, batches_key):
            state_dict.pop(k, None)

        print(f"[fold] {layer:<30s}  → {w_key}  (Cout={w.shape[0]})")

    return state_dict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--inp_path", required=True, help="Input checkpoint")
    ap.add_argument("-o", "--out_path", required=True, help="Output (BN-fused) checkpoint")
    ap.add_argument("-oa", "--out_arch", required=True, help="Architecture name to embed")
    args = ap.parse_args()

    ckpt = torch.load(args.inp_path, map_location="cpu", weights_only=False)
    ckpt["state_dict"] = bn_fuser(ckpt["state_dict"])
    ckpt["arch"] = args.out_arch
    torch.save(ckpt, args.out_path)
    print(f"New checkpoint saved to: {args.out_path}")


if __name__ == "__main__":
    main()
