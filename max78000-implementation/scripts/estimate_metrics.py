"""Offline metrics for all five active CIFAR-10 models targeting MAX78000.

Reports the four numbers the project rubric asks for, minus energy
(which needs the board). Run from the repo root:

    .venv/bin/python max78000/estimate_metrics.py

Or with uv: `uv run python max78000/estimate_metrics.py`

The output values for "Weight memory" and "MACs" match what `ai8xize.py`
will print at synthesis time — both are computed from the architecture
without needing the hardware.

Use: python estimate_metrics.py > reports/models_estimation.txt

"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Reuse the PC-implementation's model + stats code (single source of truth).
# Layout: ml-on-microcontrollers/{pc-implementation, max78000-implementation}
_PC_DIR = Path(__file__).resolve().parents[2] / "pc-implementation"
if not (_PC_DIR / "training").is_dir():
    raise RuntimeError(
        f"expected pc-implementation/training/ next to this script, "
        f"looked under {_PC_DIR}"
    )
sys.path.insert(0, str(_PC_DIR))

from training.models import build_model  # noqa: E402
from training.utils import compute_stats  # noqa: E402

MAX78000_WEIGHT_MEM_BYTES = 442_368
MAX78000_CNN_CLOCK_HZ = 100_000_000
PEAK_MACS_PER_SEC = 64 * MAX78000_CNN_CLOCK_HZ   # 64 procs × 1 MAC/cycle


def _bn_param_count(model_name: str) -> int:
    """Sum of BN gamma+beta channels (folded into preceding conv at synthesis)."""
    model = build_model(model_name)
    return sum(
        2 * m.num_features
        for m in model.modules()
        if type(m).__name__ == "BatchNorm2d"
    )


def estimate_for(model_name: str) -> dict:
    model = build_model(model_name)
    stats = compute_stats(model)
    bn_params = _bn_param_count(model_name)
    pre_fold = stats.params
    post_fold = pre_fold - bn_params
    macs = stats.macs
    t_lb = macs / PEAK_MACS_PER_SEC

    return {
        "model": model_name,
        "params_pytorch": pre_fold,
        "params_on_device_after_bn_fold": post_fold,
        "weight_memory_bytes_int8": post_fold,
        "weight_memory_kib": post_fold / 1024,
        "weight_memory_cap_bytes": MAX78000_WEIGHT_MEM_BYTES,
        "weight_memory_utilization_pct": 100 * post_fold / MAX78000_WEIGHT_MEM_BYTES,
        "macs": macs,
        "ops_paper_convention": 2 * macs,
        "latency_lower_bound_us": t_lb * 1e6,
        "fps_upper_bound": 1 / t_lb,
        "peak_gops": 2 * macs / t_lb / 1e9,
        "per_layer_macs": [{"name": n, "macs": m} for n, m in stats.layer_breakdown],
    }


def _row(d: dict) -> str:
    return (
        f"  {d['model']:<10s}  "
        f"params={d['params_pytorch']:>7,}  "
        f"post_fold={d['params_on_device_after_bn_fold']:>7,}  "
        f"wt={d['weight_memory_kib']:>6.2f} KiB  "
        f"util={d['weight_memory_utilization_pct']:>5.2f}%  "
        f"MACs={d['macs']/1e6:>5.2f}M  "
        f"latency_lb={d['latency_lower_bound_us']:>6.1f}us  "
        f"fps_ub={d['fps_upper_bound']:>5.0f}"
    )


def main() -> None:
    print("=" * 96)
    print("MAX78000 deployment estimate — five active CIFAR-10 variants")
    print("=" * 96)
    print()

    out = {}
    for name in ("baseline", "improved", "mininet", "deeper", "nascifarnet", "ressimplenet"):
        try:
            d = estimate_for(name)
        except Exception as e:
            print(f"  [skip {name}: {e}]")
            continue
        out[name] = d
        print(_row(d))

    print()
    print(f"Hardware cap: {MAX78000_WEIGHT_MEM_BYTES:,} bytes "
          f"({MAX78000_WEIGHT_MEM_BYTES/1024:.0f} KiB)")
    print()
    print("Notes:")
    print("  * post_fold = on-device int8 weight count after BN is folded into the preceding conv at synthesis.")
    print("  * latency_lb = best-case theoretical (all 64 procs busy at 1 MAC/cycle @ 100 MHz).")
    print("  * Real on-device latency is typically 5-10x the lower bound for models this small,")
    print("    because depthwise layers can't fill all 64 procs and pool/ReLU overhead dominates.")
    print("  * Cross-check 'wt' and 'MACs' against the table that ai8xize.py prints at synthesis time.")

    # parent.parent because this script lives in max78000-implementation/scripts/
    out_path = Path(__file__).resolve().parent.parent / "reports/models_estimation.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
