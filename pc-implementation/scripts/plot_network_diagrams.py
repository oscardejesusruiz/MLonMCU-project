"""Generate per-model 3D layered-block network diagrams.

For every variant in `training.models.MODEL_REGISTRY`, produce one PNG
diagram in `reports/network_diagrams/`:

  <variant>_layered.png   3D stacked-block view (visualtorch's
                          `layered_view`) — Conv2D / MaxPool / Flatten /
                          Dense blocks rendered as 3-D volumes whose
                          dimensions reflect channels × spatial.

Requires `visualtorch` only — no Graphviz, no external binaries.

Run:
    uv run python -m scripts.plot_network_diagrams
    uv run python -m scripts.plot_network_diagrams --variants baseline mininet
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from training.models import MODEL_REGISTRY, build_model  # noqa: E402

DEFAULT_VARIANTS = list(MODEL_REGISTRY)


def _make_layered_view(model: torch.nn.Module, out_path: Path,
                       input_shape: tuple[int, ...]) -> bool:
    """3D stacked-block view (visualtorch). Returns True on success."""
    try:
        import visualtorch
    except ImportError:
        print("    skip: pip install visualtorch")
        return False
    try:
        img = visualtorch.layered_view(
            model,
            input_shape=input_shape,
            legend=True,
            draw_volume=True,
            spacing=40,
            scale_xy=2.0,
            scale_z=1.0,
            max_z=200,
        )
        img.save(str(out_path))
        return True
    except Exception as e:  # noqa: BLE001
        print(f"    layered_view failed: {type(e).__name__}: {e}")
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--variants", nargs="+", default=DEFAULT_VARIANTS,
        help="Subset of MODEL_REGISTRY to render. Default: all.",
    )
    ap.add_argument(
        "--out-dir", type=Path,
        default=REPO_ROOT / "reports" / "network_diagrams",
    )
    ap.add_argument(
        "--input-size", type=int, default=32,
        help="Spatial input size (CIFAR-10 default 32).",
    )
    ap.add_argument(
        "--num-classes", type=int, default=10,
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"output dir : {args.out_dir}")
    print(f"variants   : {args.variants}")
    print(f"input shape: (1, 3, {args.input_size}, {args.input_size})")
    print()

    input_shape = (1, 3, args.input_size, args.input_size)

    summary = []
    for v in args.variants:
        if v not in MODEL_REGISTRY:
            print(f"[{v}] not in MODEL_REGISTRY; skipping")
            continue

        print(f"[{v}]")
        try:
            model = build_model(v, num_classes=args.num_classes).eval()
        except Exception as e:  # noqa: BLE001
            print(f"    build_model failed: {type(e).__name__}: {e}")
            summary.append((v, False))
            continue

        out_path = args.out_dir / f"{v}_layered.png"
        ok = _make_layered_view(model, out_path, input_shape)
        if ok:
            print(f"    → {out_path.name}")
        summary.append((v, ok))

    # ------------------------------------------------------ summary
    print()
    print(f"{'variant':<16} {'layered':>9}")
    print("-" * 28)
    for v, ok in summary:
        print(f"{v:<16} {'✓' if ok else '–':>9}")


if __name__ == "__main__":
    # Suppress visualtorch warnings unrelated to diagram quality
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
