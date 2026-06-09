"""Build a comparison report for IMX500 experiments.

This mirrors the MAX78000 reporting shape:

- reports/summary.md
- reports/figures/pareto.png
- reports/figures/training_curves.png

The script is intentionally generic: it reads every <tag>_metrics.json in a
source directory and writes the rendered report to a separate output tree.
That lets you point it at existing PC metrics today, and at IMX500 metrics
once the device-side pipeline is added.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


def _format_percent(value: object, precision: int = 2) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value) * 100:.{precision}f}%"
    except (TypeError, ValueError):
        return "—"


def _format_number(value: object, precision: int = 1) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return "—"


def _format_mops(value: object) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value) / 1e6:.2f}"
    except (TypeError, ValueError):
        return "—"


def _format_int(value: object) -> str:
    if value is None:
        return "—"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "—"


def _is_primary_model(tag: str) -> bool:
    return tag in {
        "baseline_5x5_fp32",
        "baseline_fp32",
        "deeper_fp32",
        "improved_fp32",
        "mininet_fp32",
        "wide_improved_fp32",
    }


def load_runs(source_dir: Path) -> list[dict]:
    if source_dir.is_file():
        data = json.loads(source_dir.read_text())
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        return []
    runs: list[dict] = []
    for path in sorted(source_dir.glob("*_metrics.json")):
        runs.append(json.loads(path.read_text()))
    return runs


def _parse_latency_ms(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        first = value.split(",", 1)[0].strip().split(" ", 1)[0]
        try:
            return float(first)
        except ValueError:
            return None
    return None


def load_converter_runs(converted_dir: Path) -> list[dict]:
    runs: list[dict] = []
    for path in sorted(converted_dir.glob("*/**/*_MemoryReport.json")):
        try:
            report = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        memory_report = report.get("Memory Report", {})
        runs.append(
            {
                "tag": memory_report.get("Name", path.parent.name),
                "fit_in_chip": memory_report.get("Fit In Chip"),
                "runtime_kb": memory_report.get("Runtime Memory Physical Size"),
                "model_kb": memory_report.get("Model Memory Physical Size"),
                "usage": memory_report.get("Memory Usage"),
                "utilization": memory_report.get("Memory Utilization"),
                "path": str(path),
            }
        )
    return runs


def load_log_runs(log_dir: Path) -> list[dict]:
    runs: list[dict] = []
    for path in sorted(log_dir.glob("*_summary.json")):
        try:
            summary = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        jsonl_path = path.with_name(path.name.replace("_summary.json", ".jsonl"))
        inference_latencies: list[float] = []
        latencies: list[float] = []
        if jsonl_path.exists():
            for line in jsonl_path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                inference_latency = _parse_latency_ms(record.get("hw_inference"))
                if inference_latency is not None:
                    inference_latencies.append(inference_latency)
                frame_latency = _parse_latency_ms(record.get("frame_to_frame_ms"))
                if frame_latency is not None:
                    latencies.append(frame_latency)
        runs.append(
            {
                "tag": path.stem.replace("_summary", ""),
                "samples": summary.get("samples"),
                "frame_limit": summary.get("frame_limit"),
                "log_file": summary.get("log_file"),
                "summary_file": summary.get("summary_file"),
                "device_inference_ms_mean": float(statistics.fmean(inference_latencies)) if inference_latencies else None,
                "device_inference_ms_p50": float(statistics.median(inference_latencies)) if inference_latencies else None,
                "device_inference_ms_p95": float(__import__("numpy").percentile(inference_latencies, 95)) if inference_latencies else None,
                "device_inference_ms_min": min(inference_latencies) if inference_latencies else None,
                "device_inference_ms_max": max(inference_latencies) if inference_latencies else None,
                "inference_mean": float(statistics.fmean(inference_latencies)) if inference_latencies else None,
                "inference_p50": float(statistics.median(inference_latencies)) if inference_latencies else None,
                "inference_p95": float(__import__("numpy").percentile(inference_latencies, 95)) if inference_latencies else None,
                "inference_min": min(inference_latencies) if inference_latencies else None,
                "inference_max": max(inference_latencies) if inference_latencies else None,
                "latency_mean": float(statistics.fmean(latencies)) if latencies else None,
                "latency_p50": float(statistics.median(latencies)) if latencies else None,
                "latency_p95": float(__import__("numpy").percentile(latencies, 95)) if latencies else None,
                "latency_min": min(latencies) if latencies else None,
                "latency_max": max(latencies) if latencies else None,
            }
        )
    return runs


def write_summary(runs: list[dict], out_dir: Path, title: str) -> None:
    compact_runs = [run for run in runs if _is_primary_model(str(run.get("tag", "")))]

    lines = [
        f"# {title}",
        "",
        "## Compact comparison",
        "",
        "| Tag | Model | Params | Wt. KiB (int8) | MACs (M) | Ops paper conv. (M) | fp32 acc | int8 acc | Δ acc | HW inference (ms) | Frame loop (ms) |",
        "|-----|-------|--------|----------------|----------|----------------------|----------|----------|-------|-------------------|-----------------|"]

    for run in compact_runs:
        delta = None
        if run.get("fp32_test_acc") is not None and run.get("int8_test_acc") is not None:
            delta = float(run["int8_test_acc"]) - float(run["fp32_test_acc"])
        hw_inference = run.get("device_inference_ms_mean")
        frame_loop = run.get("device_frame_latency_ms_mean") or run.get("latency_mean")
        lines.append(
            f"| {run.get('tag', '—')} | {run.get('model', '—')} | {_format_int(run.get('params'))} "
            f"| {_format_number(run.get('weight_kib_int8'))} | {_format_mops(run.get('macs'))} | {_format_mops(run.get('ops_paper_convention'))} "
            f"| {_format_percent(run.get('fp32_test_acc'))} | {_format_percent(run.get('int8_test_acc'))} "
            f"| {f'{delta * 100:+.2f}pp' if delta is not None else '—'} | "
            f"{_format_number(hw_inference, precision=2)} | {_format_number(frame_loop, precision=2)} |"
        )

    lines.extend([
        "",
        "## Full sweep",
        "",
        "| Tag | Model | Params | Wt. KiB (int8) | MACs (M) | Ops paper conv. (M) | fp32 acc | int8 acc | Δ acc | HW inference (ms) | Frame loop (ms) |",
        "|-----|-------|--------|----------------|----------|----------------------|----------|----------|-------|-------------------|-----------------|"])

    for run in runs:
        delta = None
        if run.get("fp32_test_acc") is not None and run.get("int8_test_acc") is not None:
            delta = float(run["int8_test_acc"]) - float(run["fp32_test_acc"])
        hw_inference = run.get("device_inference_ms_mean")
        frame_loop = run.get("device_frame_latency_ms_mean") or run.get("latency_mean")
        lines.append(
            f"| {run.get('tag', '—')} | {run.get('model', '—')} | {_format_int(run.get('params'))} "
            f"| {_format_number(run.get('weight_kib_int8'))} | {_format_mops(run.get('macs'))} | {_format_mops(run.get('ops_paper_convention'))} "
            f"| {_format_percent(run.get('fp32_test_acc'))} | {_format_percent(run.get('int8_test_acc'))} "
            f"| {f'{delta * 100:+.2f}pp' if delta is not None else '—'} | "
            f"{_format_number(hw_inference, precision=2)} | {_format_number(frame_loop, precision=2)} |"
        )

    lines.append("")
    lines.append("**Reference (Lai et al. 2018):** 24.7 MOps/inference, 87 KB int8 weights, 79.9% int8 accuracy on CIFAR-10.")
    lines.append("")
    lines.append("## Per-experiment notes")
    for run in runs:
        lines.append(f"### {run.get('tag', '—')}")
        args = run.get("args", {})
        lines.append(
            f"- Model: `{run.get('model', '—')}`, "
            f"optimizer: `{args.get('optimizer', '-')}`, "
            f"lr={args.get('lr', '?')}, "
            f"batch_size={args.get('batch_size', args.get('batch-size', '?'))}, "
            f"epochs={args.get('epochs', args.get('finetune_epochs', '?'))}, "
            f"augment={args.get('augment', False)}"
        )
        if run.get("device_test_acc") is not None:
            lines.append(f"- Device test acc: **{float(run['device_test_acc']) * 100:.2f}%**")
        if run.get("device_inference_ms_mean") is not None:
            lines.append(
                "- HW inference time: **"
                f"{float(run['device_inference_ms_mean']):.2f} ms mean, "
                f"{float(run.get('device_inference_ms_p50') or run['device_inference_ms_mean']):.2f} ms p50, "
                f"{float(run.get('device_inference_ms_p95') or run['device_inference_ms_mean']):.2f} ms p95**"
            )
        if run.get("device_frame_latency_ms_mean") is not None:
            lines.append(
                "- Frame loop time: **"
                f"{float(run['device_frame_latency_ms_mean']):.2f} ms mean, "
                f"{float(run.get('device_frame_latency_ms_p50') or run['device_frame_latency_ms_mean']):.2f} ms p50, "
                f"{float(run.get('device_frame_latency_ms_p95') or run['device_frame_latency_ms_mean']):.2f} ms p95**"
            )
        lines.append(
            f"- fp32 test acc: **{_format_percent(run.get('fp32_test_acc'))}**, "
            f"int8 test acc: **{_format_percent(run.get('int8_test_acc'))}**"
        )
        if run.get("device_profile"):
            lines.append(f"- Device profile: `{run['device_profile']}`")
        lines.append("")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text("\n".join(lines))
    print(f"wrote {out_dir / 'summary.md'}")


def write_converter_section(runs: list[dict], lines: list[str]) -> None:
    if not runs:
        return
    lines.append("## Converter memory checks")
    lines.append("")
    lines.append("| Tag | Fit in chip | Runtime KB | Model KB | Usage | Utilization |")
    lines.append("|-----|-------------|------------|----------|-------|-------------|")
    for run in runs:
        lines.append(
            f"| {run.get('tag', '—')} | {run.get('fit_in_chip', '—')} | {run.get('runtime_kb', '—')} | "
            f"{run.get('model_kb', '—')} | {run.get('usage', '—')} | {run.get('utilization', '—')} |"
        )
    lines.append("")


def write_logs_section(runs: list[dict], lines: list[str]) -> None:
    if not runs:
        return
    lines.append("## Live Pi logs")
    lines.append("")
    lines.append("| Tag | Samples | Frame limit | HW mean ms | HW p50 ms | HW p95 ms | Frame mean ms | Frame p50 ms | Frame p95 ms |")
    lines.append("|-----|---------|-------------|------------|-----------|-----------|--------------|-------------|-------------|")
    for run in runs:
        lines.append(
            f"| {run.get('tag', '—')} | {run.get('samples', '—')} | {run.get('frame_limit', '—')} | "
            f"{_format_number(run.get('inference_mean'), precision=2)} | {_format_number(run.get('inference_p50'), precision=2)} | "
            f"{_format_number(run.get('inference_p95'), precision=2)} | {_format_number(run.get('latency_mean'), precision=2)} | "
            f"{_format_number(run.get('latency_p50'), precision=2)} | {_format_number(run.get('latency_p95'), precision=2)} |"
        )
    lines.append("")


def plot_pareto(runs: list[dict], fig_dir: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for run in runs:
        macs = run.get("macs")
        acc = run.get("device_test_acc") if run.get("device_test_acc") is not None else run.get("int8_test_acc")
        if macs is None or acc is None:
            continue
        macs_m = float(macs) / 1e6
        ax.scatter(macs_m, float(acc) * 100, s=80, label=run.get("tag", "—"))
        ax.annotate(
            run.get("tag", "—"),
            (macs_m, float(acc) * 100),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )
    ax.set_xlabel("MACs / inference (M)")
    ax.set_ylabel("int8 accuracy (%)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "pareto.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def plot_latency(runs: list[dict], fig_dir: Path) -> None:
    latency_runs = [r for r in runs if r.get("device_frame_latency_ms_mean") is not None or r.get("device_kpi_latency_ms_mean") is not None]
    if not latency_runs:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = []
    values = []
    for run in latency_runs:
        labels.append(run.get("tag", "—"))
        values.append(float(run.get("device_frame_latency_ms_mean") or run.get("device_kpi_latency_ms_mean")))
    ax.bar(labels, values)
    ax.set_ylabel("latency (ms)")
    ax.set_title("IMX500 device latency")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    out = fig_dir / "latency.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def plot_training_curves(runs: list[dict], fig_dir: Path) -> None:
    if not runs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    plotted = False
    for run in runs:
        history = run.get("history") or {}
        val_acc = history.get("val_acc") or []
        train_loss = history.get("train_loss") or []
        if not val_acc or not train_loss:
            continue
        epochs = range(1, min(len(val_acc), len(train_loss)) + 1)
        axes[0].plot(epochs, train_loss[: len(val_acc)], label=run.get("tag", "—"))
        axes[1].plot(epochs, [float(a) * 100 for a in val_acc[: len(train_loss)]], label=run.get("tag", "—"))
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("train loss")
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("test acc (%)")
    axes[1].grid(alpha=0.3)
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    out = fig_dir / "training_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "summary.json",
        help="summary.json file or directory containing <tag>_metrics.json files",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "reports",
        help="directory to write summary.md and figures/",
    )
    ap.add_argument(
        "--converted-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "imx500_converted",
        help="directory containing converter MemoryReport.json artifacts",
    )
    ap.add_argument(
        "--logs-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "logs",
        help="directory containing live JSONL logs and summary JSON files",
    )
    ap.add_argument(
        "--title",
        default="CIFAR-10 on MCU — IMX500 results",
        help="report title",
    )
    args = ap.parse_args()

    runs = load_runs(args.source)
    if not runs:
        raise SystemExit(f"no metrics found in {args.source}")

    converter_runs = load_converter_runs(args.converted_dir) if args.converted_dir.exists() else []
    log_runs = load_log_runs(args.logs_dir) if args.logs_dir.exists() else []
    def _normalize_tag(tag: object) -> str:
        text = str(tag or "")
        return text[:-11] if text.endswith("_imx500_ptq") else text

    log_by_tag = {_normalize_tag(run.get("tag")): run for run in log_runs}
    merged_runs = []
    for run in runs:
        merged = dict(run)
        log_run = log_by_tag.get(_normalize_tag(run.get("tag")))
        if log_run:
            for key, value in log_run.items():
                if key != "tag":
                    merged.setdefault(key, value)
        merged_runs.append(merged)

    fig_dir = args.out_dir / "figures"
    write_summary(merged_runs, args.out_dir, args.title)
    plot_pareto(merged_runs, fig_dir, "Pareto: accuracy vs compute")
    plot_latency(merged_runs, fig_dir)
    plot_training_curves(merged_runs, fig_dir)

    summary_path = args.out_dir / "summary.md"
    lines = summary_path.read_text().splitlines()
    if converter_runs:
        write_converter_section(converter_runs, lines)
    if log_runs:
        write_logs_section(log_runs, lines)
    summary_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()