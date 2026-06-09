"""Single-inference profiling report for the MAX78000, ST.AI-style.

Combines:
  - device-measured numbers from `c_harness/profile_layers.c` over UART
    (total CNN cycles, total CPU cycles, the 10 logits, the clock frequency)
  - static per-layer breakdown from `estimate.json` (params, MACs per layer)

…and emits a table that mirrors the layout from STM32CubeMX.AI profiler:

    nb samples       : 1
    duration         : XX.XXX ms
    macc             : Y
    cycles/MACC      : Z
    CPU cycles       : W
    used stack/heap  : not monitored

    Inference time per node
     c_id  m_id  type                  dur (ms)    %   cumul   CPU cycles  name
     0     0     Conv2d                  ...

    Statistic per tensor
     I.0   10  i8[1,32,32,3]:3072  ...
     O.0   10  i8[1,1,10]:10      ...

Usage (after flashing profile_layers.c):
    uv run python host/host_profile.py \\
        --port /dev/cu.usbmodemXXXX \\
        --variant baseline \\
        --tag baseline_qat
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import serial


PROFILE_BEGIN = "# profile-begin"
PROFILE_END   = "# profile-end"


def _wait_for_profile(ser: serial.Serial, timeout_s: float) -> dict:
    """Block reading lines from device until we see the profile block."""
    import time
    deadline = time.time() + timeout_s
    raw_lines: list[str] = []
    in_block = False
    parsed: dict[str, str] = {}
    while time.time() < deadline:
        line = ser.readline().decode("ascii", errors="replace").strip()
        if not line:
            continue
        raw_lines.append(line)
        if line == PROFILE_BEGIN:
            in_block = True
            continue
        if line == PROFILE_END:
            return parsed
        if in_block and "=" in line:
            k, _, v = line.partition("=")
            parsed[k.strip()] = v.strip()
    raise TimeoutError(f"no profile block within {timeout_s}s — last lines:\n  "
                       + "\n  ".join(raw_lines[-10:]))


def _load_per_layer(est: dict) -> list[dict]:
    """Pull the per-layer breakdown from estimate.json entry."""
    layers = est.get("per_layer_macs") or []
    return [{"name": L["name"], "macs": int(L["macs"])} for L in layers]


def _fmt_table(rows: list[list[str]]) -> str:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    out: list[str] = []
    for r in rows:
        out.append("  ".join(c.ljust(widths[i]) for i, c in enumerate(r)))
    return "\n".join(out)


def render_report(dev: dict, layers: list[dict], variant: str,
                  tag: str) -> str:
    cnn_cycles = int(dev["cnn_cycles"])
    cpu_cycles = int(dev["cpu_cycles"])
    clock_hz   = int(dev.get("cnn_clock_hz", 100_000_000))
    duration_ms = cnn_cycles / clock_hz * 1e3

    total_macs = sum(L["macs"] for L in layers) or 1
    cycles_per_mac = cnn_cycles / total_macs

    out: list[str] = []
    out.append(f"MAX78000 Profiling results v1.0 — \"{tag}\"")
    out.append("")
    out.append(f"  nb sample(s)     : 1")
    out.append(f"  duration         : {duration_ms:.3f} ms")
    out.append(f"  macc             : {total_macs:,}")
    out.append(f"  cycles/MACC      : {cycles_per_mac:.2f}")
    out.append(f"  CPU cycles       : {cpu_cycles:,}")
    out.append(f"  used stack/heap  : not monitored / 0 bytes")
    out.append("")
    out.append(f"Inference time per node ({variant})")
    out.append("")

    rows = [["c_id", "type", "macs", "%", "cumul %", "est. cycles", "est. dur (ms)", "name"]]
    cumul = 0.0
    for i, L in enumerate(layers):
        pct = 100.0 * L["macs"] / total_macs
        cumul += pct
        est_cycles = int(cnn_cycles * L["macs"] / total_macs)
        est_ms = est_cycles / clock_hz * 1e3
        rows.append([
            str(i),
            "Conv2d/Pool",   # we don't have type per layer in estimate.json
            f"{L['macs']:,}",
            f"{pct:.1f}",
            f"{cumul:.1f}",
            f"{est_cycles:,}",
            f"{est_ms:.3f}",
            L["name"],
        ])
    rows.append(["total", "", f"{total_macs:,}", "100.0", "100.0",
                 f"{cnn_cycles:,}", f"{duration_ms:.3f}", ""])
    out.append(_fmt_table(rows))
    out.append("")
    out.append("Note: per-layer cycles are distributed by MAC count (synthesizer-")
    out.append("estimated). MAX78000 does not expose layer-by-layer wall-clock from")
    out.append("a single cnn_start() — for true per-layer timing, re-synthesize with")
    out.append("`ai8xize.py --unload` (slower, dilates total inference time).")
    out.append("")
    out.append("Statistic per tensor")
    out.append("  I.0   1   i8[1,3,32,32]:3072       input")
    out.append("  O.0   1   i32[1,10]:40             logits")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200,
                    help="profile_layers.c uses 115200; test_set firmware uses 921600")
    ap.add_argument("--variant",
                    choices=["baseline", "improved", "mininet", "deeper", "nascifarnet", "ressimplenet"],
                    required=True)
    ap.add_argument("--tag", default=None,
                    help="report title; defaults to <variant>_qat")
    ap.add_argument("--timeout-s", type=float, default=10.0)
    ap.add_argument("--estimate", type=Path,
                    default=Path(__file__).resolve().parents[1] / "reports/models_estimation.json")
    ap.add_argument("--out", type=Path, default=None,
                    help="defaults to reports/profile_<variant>.txt")
    args = ap.parse_args()

    tag = args.tag or f"{args.variant}_qat"

    print(f"opening {args.port} @ {args.baud}, waiting for device profile block...")
    ser = serial.Serial(args.port, args.baud, timeout=1.0)
    try:
        dev = _wait_for_profile(ser, args.timeout_s)
    finally:
        ser.close()

    est_all = json.loads(args.estimate.read_text())
    est_key = next((k for k in est_all if args.variant in k), None)
    if est_key is None:
        print(f"variant {args.variant!r} not in {args.estimate}", file=sys.stderr)
        sys.exit(1)
    layers = _load_per_layer(est_all[est_key])

    report = render_report(dev, layers, args.variant, tag)
    print(report)

    out_path = args.out or (Path(__file__).resolve().parents[1]
                            / "reports" / f"profile_{args.variant}.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
