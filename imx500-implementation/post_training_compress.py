"""Post-training quantization for the shared PyTorch CIFAR-10 models.

This is the PyTorch counterpart to the TensorFlow script the user already has.

What it does:

- scans ``pc-implementation/trained_models_2`` for the float checkpoints;
- loads each model with the shared ``training.models`` factory;
- runs MCT PyTorch PTQ with the IMX500 TPC v1;
- evaluates float vs quantized accuracy on CIFAR-10;
- exports the quantized model as ONNX for the IMX500 converter;
- writes one metrics file per checkpoint plus a compact run summary.

The script focuses on the float checkpoints (``*_fp32.pt``) because those are
the true post-training quantization targets. If you want, it can also be told to
include the already-compressed checkpoints for comparison only.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import model_compression_toolkit as mct
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

# Short explicit mapping for checkpoint variant names -> registry keys
# Use this to handle historical or renamed checkpoints (quick pragmatic fix).
MODEL_NAME_MAP = {
    "improved_distill": "improved",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _pc_impl_root() -> Path:
    return _repo_root() / "pc-implementation"


def _add_pc_impl_to_path() -> None:
    pc_dir = _pc_impl_root()
    if not (pc_dir / "training").is_dir():
        raise FileNotFoundError(f"expected pc-implementation/training under {pc_dir}")
    sys.path.insert(0, str(pc_dir))


def _load_training_modules():
    _add_pc_impl_to_path()
    models = importlib.import_module("training.models")
    utils = importlib.import_module("training.utils")
    engine = importlib.import_module("training.engine")
    return models, utils, engine


def _default_image_size(variant: str) -> int:
    # All checkpoints in trained_models_2 are CIFAR-10 models trained for 32x32 inputs.
    return 32


def _checkpoint_variant(path: Path) -> str:
    stem = path.stem
    for suffix in ("_fp32_ptq", "_fp32", "_qat_prune50", "_qat", "_prune50"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _checkpoint_mode(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_fp32.pt"):
        return "fp32"
    if stem.endswith("_fp32_ptq.pt"):
        return "ptq"
    if stem.endswith("_qat_prune50.pt"):
        return "qat_prune50"
    if stem.endswith("_qat.pt"):
        return "qat"
    if stem.endswith("_prune50.pt"):
        return "prune50"
    return "unknown"


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    sd = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    return {k.replace("module.", ""): v for k, v in sd.items()}


# ==========================================
# NEW: HARDWARE COMPATIBILITY HELPERS
# ==========================================
def scale_to_raw(x: torch.Tensor) -> torch.Tensor:
    """Scales data back to 0-255 to mimic raw camera sensor output."""
    return x * 255.0


class IMX500PrepWrapper(torch.nn.Module):
    """Bakes ToTensor and Normalization math directly into the ONNX graph.

    This protects your original model layout while ensuring the exported ONNX
    file natively processes raw 0-255 images coming from the camera sensor.
    """
    def __init__(self, original_model: torch.nn.Module):
        super().__init__()
        self.model = original_model
        # Register buffers so they are properly embedded as graph constants during ONNX export
        self.register_buffer("mean", torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CIFAR_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Reverse ToTensor scale -> gets data to 0.0 - 1.0 range
        x = x / 255.0
        # 2. Re-apply the normalizations your training process used
        x = (x - self.mean) / self.std
        # 3. Pass perfectly normalized floats down to your core model weights
        return self.model(x)
# ==========================================


def _make_loaders(data_root: Path, image_size: int, batch_size: int, num_workers: int) -> tuple[DataLoader, DataLoader]:
    # CHANGED: Replaced transforms.Normalize with transforms.Lambda(scale_to_raw)
    # This ensures your dataset outputs 0-255 values, matching the physical camera sensor.
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(scale_to_raw),
        ]
    )
    train_set = datasets.CIFAR10(root=str(data_root), train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root=str(data_root), train=False, download=True, transform=transform)

    generator = torch.Generator().manual_seed(42)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def _representative_dataset_gen(loader: DataLoader, n_iter: int):
    for batch_index, (images, _labels) in enumerate(loader):
        if batch_index >= n_iter:
            break
        yield [images.cpu().numpy()]


def _target_platform_capabilities():
    return mct.get_target_platform_capabilities("pytorch", "imx500", "v1")


def _mAP(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    return float(average_precision_score(np.eye(10)[y_true], y_probs, average="macro"))


def _checkpoint_state(model, checkpoint: Path):
    missing, unexpected = model.load_state_dict(_load_state_dict(checkpoint), strict=False)
    if missing:
        print(f"[warn] {checkpoint.name}: missing keys {len(missing)} (first: {missing[:3]})", file=sys.stderr)
    if unexpected:
        print(f"[warn] {checkpoint.name}: unexpected keys {len(unexpected)} (first: {unexpected[:3]})", file=sys.stderr)
    return model.eval()


def _build_model_with_fallback(models_module, variant: str, checkpoint: Path):
    """Try to build a model by variant name; when that fails, attempt simple fallbacks.

    Returns a built model instance. Raises ValueError if no candidate found.
    """
    try:
        return models_module.build_model(variant)
    except Exception as exc:  # pragma: no cover - best-effort fallback
        # Try to inspect a declared registry if available
        registry = getattr(models_module, "MODEL_REGISTRY", None)
        candidates = []
        if registry:
            try:
                candidates = list(registry)
            except Exception:
                candidates = list(registry)

        # Heuristics: prefer registry entries that appear in the variant name
        for name in candidates:
            if name in variant or variant in name:
                try:
                    print(f"[info] mapping variant '{variant}' -> model '{name}'")
                    return models_module.build_model(name)
                except Exception:
                    continue

        # Try simple prefix heuristic (e.g., improved_distill -> improved)
        base = variant.split("_")[0]
        if base != variant:
            try:
                print(f"[info] trying base variant '{base}' for '{variant}'")
                return models_module.build_model(base)
            except Exception:
                pass

        # No fallback found; re-raise a clearer error
        raise ValueError(f"Unknown model '{variant}' for checkpoint {checkpoint.name}; no fallback found")


def _discover_checkpoints(source_dir: Path, include_non_fp32: bool) -> list[Path]:
    checkpoints = sorted(source_dir.glob("*.pt"))
    if include_non_fp32:
        return checkpoints
    return [path for path in checkpoints if path.stem.endswith("_fp32")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-training quantization for the shared PyTorch CIFAR-10 models.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=_pc_impl_root() / "trained_models_2",
        help="directory containing PyTorch checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_repo_root() / "imx500-implementation" / "outputs",
        help="directory to write quantized exports and metrics",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=_repo_root() / "imx500-implementation" / "data",
        help="CIFAR-10 download/cache directory",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="batch size for evaluation and calibration")
    parser.add_argument("--n-iter", type=int, default=10, help="number of representative batches used for PTQ")
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="override the input size used for all checkpoints",
    )
    parser.add_argument(
        "--include-non-fp32",
        action="store_true",
        help="also process already-compressed checkpoints instead of only *_fp32.pt",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="only run PTQ and evaluation; do not export the quantized ONNX model",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="skip accuracy evaluation and only perform quantization/export",
    )
    return parser.parse_args()


def _save_quantization_artifacts(model_dir: Path, tag: str, quantization_info: object) -> None:
    report_path = model_dir / "quantization_info.txt"
    report_path.write_text(repr(quantization_info) + "\n", encoding="utf-8")

    try:
        import pickle
        import pprint

        pkl_path = model_dir / "quantization_info.pkl"
        with pkl_path.open("wb") as handle:
            pickle.dump(quantization_info, handle)

        summary = {}
        for attr in dir(quantization_info):
            if attr.startswith("_"):
                continue
            try:
                value = getattr(quantization_info, attr)
                summary[attr] = type(value).__name__
            except Exception as exc:  # pragma: no cover - defensive
                summary[attr] = f"ERROR: {exc}"

        summary_path = model_dir / "quantization_info_summary.txt"
        summary_path.write_text(pprint.pformat(summary), encoding="utf-8")
    except Exception:
        print(f"[warn] could not save verbose quantization info for {tag}", file=sys.stderr)


def _export_quantized_model(quantized_model: torch.nn.Module, export_path: Path, repr_dataset) -> None:
    export_path.parent.mkdir(parents=True, exist_ok=True)
    mct.exporter.pytorch_export_model(
        model=quantized_model,
        save_model_path=str(export_path),
        repr_dataset=repr_dataset
    )


def run_quantization(
    model_path: Path,
    output_dir: Path,
    data_root: Path,
    batch_size: int,
    n_iter: int,
    image_size: int | None,
    evaluate: bool,
    export: bool,
) -> dict:
    models, utils, engine = _load_training_modules()

    variant = _checkpoint_variant(model_path)
    # Apply explicit name mapping if available (keeps backwards compatibility)
    variant = MODEL_NAME_MAP.get(variant, variant)
    resolved_image_size = image_size or _default_image_size(variant)

    # output layout under <output_dir>:
    # - onnx/                     <- all exported ONNX files
    # - reports/                  <- per-checkpoint metrics JSONs and summary
    # - quant_info/<checkpoint>/  <- verbose quantization info per checkpoint
    # - models/<checkpoint>/      <- any model-specific artifacts
    base_out = output_dir
    onnx_dir = base_out / "onnx"
    reports_dir = base_out / "reports"
    quant_dir = base_out / "quant_info"
    models_dir = base_out / "models"

    model_out_dir = models_dir / model_path.stem
    model_out_dir.mkdir(parents=True, exist_ok=True)
    onnx_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (quant_dir / model_path.stem).mkdir(parents=True, exist_ok=True)

    print(f"Loading float checkpoint: {model_path}")
    try:
        # CHANGED: Wrapped the core model inside IMX500PrepWrapper.
        # This injects the normalization steps dynamically right into the model layout.
        core_model = _checkpoint_state(_build_model_with_fallback(models, variant, model_path), model_path)
        float_model = IMX500PrepWrapper(core_model)
    except Exception as exc:
        print(f"[warn] skipping {model_path.name}: {exc}", file=sys.stderr)
        return {
            "tag": model_path.stem,
            "model": variant,
            "checkpoint": str(model_path),
            "error": str(exc),
        }
    stats = utils.compute_stats(models.build_model(variant), input_shape=(1, 3, resolved_image_size, resolved_image_size))
    train_loader, test_loader = _make_loaders(data_root, resolved_image_size, batch_size, num_workers=2)

    def representative_dataset_gen():
        yield from _representative_dataset_gen(train_loader, n_iter)

    print("Creating IMX500 target platform capabilities (TPC v1)...")
    target_platform_cap = _target_platform_capabilities()

    print("Running MCT post-training quantization...")
    quantized_model, quantization_info = mct.ptq.pytorch_post_training_quantization(
        in_module=float_model,
        representative_data_gen=representative_dataset_gen,
        target_platform_capabilities=target_platform_cap,
    )

    fp_out = None
    q_out = None
    if evaluate:
        print("Evaluating float and quantized models on CIFAR-10 test split...")
        fp_out = engine.evaluate(
            float_model.cpu(),
            test_loader,
            torch.device("cpu"),
            return_loss=True,
            return_predictions=True,
        )
        q_out = engine.evaluate(
            quantized_model,
            test_loader,
            torch.device("cpu"),
            return_loss=True,
            return_predictions=True,
        )
        print(f"Float model   - loss: {fp_out['loss']:.4f}, acc: {fp_out['acc']:.4f}, mAP: {_mAP(fp_out['y_true'], fp_out['y_probs']):.4f}")
        print(f"Quant model   - loss: {q_out['loss']:.4f}, acc: {q_out['acc']:.4f}, mAP: {_mAP(q_out['y_true'], q_out['y_probs']):.4f}")

    export_path = onnx_dir / f"{model_path.stem}_imx500_ptq.onnx"
    if export:
        print(f"Exporting quantized PyTorch model to: {export_path}")
        _export_quantized_model(
            quantized_model,
            export_path,
            repr_dataset=representative_dataset_gen,
        )

    # Save verbose quantization info under quant_info/<checkpoint>/
    _save_quantization_artifacts(quant_dir / model_path.stem, model_path.stem, quantization_info)

    metrics = {
        "tag": model_path.stem,
        "model": variant,
        "checkpoint": str(model_path),
        "output_dir": str(model_out_dir),
        "export_path": str(export_path) if export else None,
        "params": stats.params,
        "weight_kib_fp32": stats.weight_bytes_fp32 / 1024,
        "weight_kib_int8": stats.weight_bytes_int8 / 1024,
        "macs": stats.macs,
        "ops_paper_convention": stats.macs * 2,
        "tpc_fw_name": "pytorch",
        "tpc_target_platform": "imx500",
        "tpc_version": "v1",
        "quantization_info": repr(quantization_info),
    }
    if fp_out is not None and q_out is not None:
        metrics.update(
            {
                "fp32_test_acc": fp_out["acc"],
                "fp32_mAP": _mAP(fp_out["y_true"], fp_out["y_probs"]),
                "int8_test_acc": q_out["acc"],
                "int8_mAP": _mAP(q_out["y_true"], q_out["y_probs"]),
            }
        )

    metrics_path = reports_dir / f"{model_path.stem}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"wrote {metrics_path}")
    return metrics


def main() -> None:
    args = parse_args()
    checkpoints = _discover_checkpoints(args.source_dir, args.include_non_fp32)
    if not checkpoints:
        raise SystemExit(f"no checkpoints found in {args.source_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for checkpoint in checkpoints:
        summary.append(
            run_quantization(
                model_path=checkpoint,
                output_dir=args.output_dir,
                data_root=args.data_root,
                batch_size=args.batch_size,
                n_iter=args.n_iter,
                image_size=args.image_size,
                evaluate=not args.skip_eval,
                export=not args.skip_export,
            )
        )

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()