"""Live end-to-end profiling monitor for the camera-driven inference firmware.

Reads per-inference packets from the MAX78000 (sent by
`c_harness/profile_camera.c`), maintains a rolling N-inference window,
and continuously displays:

    • End-to-end latency (DWT-measured full pipeline)        — ms
    • CNN latency        (MSDK timer in cnn_time)            — ms
    • CNN cycles         (cnn_us × 50 MHz)                   — cycles
    • MAC / CNN-cycle    (variant MACs / CNN cycles)         — utilization
    • End-to-end energy  (P × t_e2e)                         — µJ
    • Inference energy   (P × t_cnn)                         — µJ
    • Power normalized   (P / f_CPU and P / f_CNN)           — µW/MHz
    • Measured efficiency (2 × MACs / (P × t_cnn))           — TOPS/W
    • Peak efficiency    (paper Table I)                     — TOPS/W

All metrics that need a per-variant constant (MACs, peak TOPS/W) are
keyed off `--variant`. Power assumption is `--power-mw` (default 28 mW,
the value Capogrosso et al. 2026 quote in Table I for the MAX78000).

Wire protocol — one packet per inference, little-endian, 50 bytes:

    dev -> host:  uint8   sync = 0xCB
                  uint8   inference_id   (mod 256, drop-detection only)
                  uint32  e2e_us
                  uint32  cnn_us
                  int32   logits[10]

Run via the bash wrapper (recommended):

    ./bash_device_scripts/host_profile.sh <variant>

Direct invocation:

    uv run --project ../pc-implementation \\
        python host/host_e2e_profile.py --port /dev/cu.usbmodemXXXX \\
                                        --variant baseline
"""
from __future__ import annotations

import argparse
import glob
import struct
import sys
import time
from collections import deque

import numpy as np
import serial

SYNC_BYTE = 0xCB
PACKET_BYTES = 1 + 4 + 4 + 4 * 10           # after sync byte: 49 bytes
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# ---------------------------------------------------------------------------
# Per-variant constants
# ---------------------------------------------------------------------------
# MACs come from pc-implementation/reports/*_metrics.json. Peak TOPS/W is
# from Capogrosso et al. 2026 Table I ("Max 78000 / 0.056 TOPS / 0.028 W /
# 2.00 TOPS/W") and is shared across variants — chip property, not model
# property.
MAC_PER_VARIANT = {
    "baseline":      4_433_920,
    "improved":      4_433_920,
    "deeper":        7_963_904,
    "mininet":      24_478_976,
    "ressimplenet": 18_449_664,
    "nascifarnet":  36_180_992,
}
PEAK_TOPS_W = 2.00       # MAX78000 peak (paper Table I)

# Clocks
F_CPU_MHZ = 100.0        # Cortex-M4 / HCLK
F_CNN_MHZ = 50.0         # PCLK / DIV1 = HCLK / 2 = 50 MHz
PEAK_MAC_PER_CNN_CYCLE = 576  # 64 procs × 9-MAC kernel, architectural peak


# ---------- serial helpers -------------------------------------------------

def autodetect_port() -> str | None:
    for pat in ("/dev/cu.usbmodem*", "/dev/tty.usbmodem*", "/dev/ttyACM*"):
        found = sorted(glob.glob(pat))
        if found:
            return found[0]
    return None


def read_packet(ser: serial.Serial):
    """Sync on SYNC_BYTE, then read the 49-byte body. Returns
    (inference_id, e2e_us, cnn_us, logits)."""
    while True:
        b = ser.read(1)
        if not b:
            raise IOError("serial timeout — is the firmware running?")
        if b[0] == SYNC_BYTE:
            break
    body = ser.read(PACKET_BYTES)
    if len(body) != PACKET_BYTES:
        raise IOError(f"short read: got {len(body)}/{PACKET_BYTES} bytes "
                      f"after sync byte")
    inference_id = body[0]
    e2e_us, cnn_us = struct.unpack("<II", body[1:9])
    logits = np.frombuffer(body[9:], dtype="<i4")
    return inference_id, e2e_us, cnn_us, logits


# ---------- math / display -------------------------------------------------

def softmax(logits: np.ndarray) -> np.ndarray:
    """Auto-temperature softmax matching host_camera.py — the device's
    int32 logits often span thousands, so we rescale before exp()."""
    x = logits.astype(np.float64)
    rng = float(x.max() - x.min())
    T = max(rng / 10.0, 1.0)
    x = (x - x.max()) / T
    e = np.exp(x)
    return (e / e.sum()).astype(np.float32)


def clear_screen() -> None:
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


# ---------- main -----------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default=None,
                    help="serial port (auto-detected if omitted)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--timeout-s", type=float, default=10.0)
    ap.add_argument("--variant", required=True,
                    choices=sorted(MAC_PER_VARIANT.keys()),
                    help="model variant (selects MAC count for the metrics)")
    ap.add_argument("--power-mw", type=float, default=28.0,
                    help="chip active power assumption "
                         "(paper Table I = 28 mW)")
    ap.add_argument("--window", type=int, default=10,
                    help="rolling-average window length (default 10)")
    ap.add_argument("--log", type=str, default=None,
                    help="append CSV: ts,id,e2e_us,cnn_us,top_class,top_prob")
    ap.add_argument("--no-clear", action="store_true",
                    help="print each frame on a new line (logging-friendly)")
    args = ap.parse_args()

    macs = MAC_PER_VARIANT[args.variant]
    power_w = args.power_mw / 1000.0

    port = args.port or autodetect_port()
    if not port:
        print("ERROR: no /dev/cu.usbmodem* found; pass --port", file=sys.stderr)
        sys.exit(1)

    print(f"opening {port} @ {args.baud}")
    ser = serial.Serial(port, args.baud, timeout=args.timeout_s)

    # Boot-message snapshot (same pattern as host_camera.py — take ONE
    # peek at whatever bytes accumulated during boot, then reset the
    # input buffer so packet parsing starts on a clean boundary).
    time.sleep(3.0)
    n_waiting = ser.in_waiting
    drained = ser.read(n_waiting) if n_waiting > 0 else b""
    if drained:
        text_bytes = bytes(c for c in drained
                           if 0x20 <= c < 0x7F or c in (0x09, 0x0A, 0x0D))
        msg = text_bytes.decode("ascii", errors="replace").strip()
        if msg:
            print("[boot]")
            for line in msg.splitlines():
                if line.strip():
                    print(f"  {line.strip()}")
        print(f"[boot] drained {len(drained)} bytes "
              f"({len(text_bytes)} printable)")
    else:
        print("[boot] (no output — check port + reset button)")
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    log_fp = open(args.log, "a") if args.log else None
    if log_fp:
        log_fp.write("ts,inference_id,e2e_us,cnn_us,top_class,top_prob\n")

    # Rolling window of last N inferences.
    e2e_hist: deque[int] = deque(maxlen=args.window)
    cnn_hist: deque[int] = deque(maxlen=args.window)

    prev_id: int | None = None
    n_drops = 0

    print(f"\n=== Profiling {args.variant} "
          f"({macs / 1e6:.2f} MMACs, P = {args.power_mw:.1f} mW) ===\n")

    try:
        while True:
            try:
                inference_id, e2e_us, cnn_us, logits = read_packet(ser)
            except IOError as e:
                print(f"\n[stream error] {e}", file=sys.stderr)
                ser.reset_input_buffer()
                time.sleep(0.1)
                continue

            # Drop detection — inference_id wraps at 256.
            if prev_id is not None:
                expected = (prev_id + 1) & 0xFF
                if inference_id != expected:
                    gap = (inference_id - expected) & 0xFF
                    n_drops += gap
            prev_id = inference_id

            e2e_hist.append(e2e_us)
            cnn_hist.append(cnn_us)

            # Wait until we have a full window before printing the metrics.
            if len(e2e_hist) < args.window:
                if not args.no_clear:
                    clear_screen()
                print(f"warming up — {len(e2e_hist)}/{args.window} inferences "
                      f"collected (latest id {inference_id})")
                continue

            # ----- aggregate over the rolling window ----------------------
            avg_e2e_us = sum(e2e_hist) / args.window
            avg_cnn_us = sum(cnn_hist) / args.window
            min_e2e_us, max_e2e_us = min(e2e_hist), max(e2e_hist)
            min_cnn_us, max_cnn_us = min(cnn_hist), max(cnn_hist)

            avg_e2e_ms = avg_e2e_us / 1000.0
            avg_cnn_ms = avg_cnn_us / 1000.0

            # CNN cycles = µs × MHz (µs × 10⁶ Hz = cycles per inverted second).
            cnn_cycles_avg = avg_cnn_us * F_CNN_MHZ
            mac_per_cycle = (macs / cnn_cycles_avg) if cnn_cycles_avg > 0 else 0.0

            # Energy: P [mW] × t [ms] = E [µJ]. Units cancel exactly.
            e_e2e_uj = args.power_mw * avg_e2e_ms
            e_cnn_uj = args.power_mw * avg_cnn_ms

            # Power-per-MHz in both clock domains (same physical P).
            p_per_mhz_cpu = args.power_mw * 1000.0 / F_CPU_MHZ
            p_per_mhz_cnn = args.power_mw * 1000.0 / F_CNN_MHZ

            # Measured TOPS/W on this network:
            #   OPS = 2 × MACs (paper convention)
            #   throughput = OPS / t_cnn[s]
            #   efficiency = throughput / P  ÷ 1e12
            tops_w_meas = (
                (2.0 * macs / (avg_cnn_us * 1e-6)) / power_w / 1e12
            )

            # Sanity: top class from softmax of the last frame's logits.
            probs = softmax(logits)
            top = int(np.argmax(probs))

            if not args.no_clear:
                clear_screen()

            print("┌──────────────────────────────────────────────────────────────┐")
            print(f"│  MAX78000 profile — variant: {args.variant:<31} │")
            print(f"│  rolling window of {args.window:>2} inferences   "
                  f"latest id {inference_id:>3}   drops {n_drops:>4}     │")
            print("├──────────────────────────────────────────────────────────────┤")
            print(f"│  End-to-end latency : {avg_e2e_ms:8.3f} ms   "
                  f"min {min_e2e_us / 1000:6.3f}  max {max_e2e_us / 1000:6.3f}  │")
            print(f"│  CNN latency        : {avg_cnn_ms:8.3f} ms   "
                  f"min {min_cnn_us / 1000:6.3f}  max {max_cnn_us / 1000:6.3f}  │")
            print(f"│  CNN cycles (@50MHz): {cnn_cycles_avg:>12,.0f}                              │"
                  .replace(",", "_"))
            print(f"│  MAC / CNN-cycle    : {mac_per_cycle:8.1f}      "
                  f"(peak {PEAK_MAC_PER_CNN_CYCLE} → "
                  f"{mac_per_cycle / PEAK_MAC_PER_CNN_CYCLE * 100:5.1f}% util.)│")
            print("├──────────────────────────────────────────────────────────────┤")
            print(f"│  Energy / e2e       : {e_e2e_uj:9.2f} µJ   "
                  f"({args.power_mw:.0f} mW × {avg_e2e_ms:.3f} ms)        │")
            print(f"│  Energy / inference : {e_cnn_uj:9.2f} µJ   "
                  f"({args.power_mw:.0f} mW × {avg_cnn_ms:.3f} ms)        │")
            print(f"│  Power / f_CPU      : {p_per_mhz_cpu:8.1f} µW/MHz "
                  f"(P={args.power_mw:.0f} mW, f={F_CPU_MHZ:.0f} MHz) │")
            print(f"│  Power / f_CNN      : {p_per_mhz_cnn:8.1f} µW/MHz "
                  f"(P={args.power_mw:.0f} mW, f={F_CNN_MHZ:.0f} MHz)  │")
            print("├──────────────────────────────────────────────────────────────┤")
            print(f"│  Efficiency (meas.) : {tops_w_meas:8.3f} TOPS/W "
                  f"({tops_w_meas / PEAK_TOPS_W * 100:5.1f}% of peak)      │")
            print(f"│  Efficiency (peak)  : {PEAK_TOPS_W:8.2f} TOPS/W "
                  f"(paper Table I, MAX78000)       │")
            print("├──────────────────────────────────────────────────────────────┤")
            print(f"│  Sanity: top class = {CLASSES[top]:<11} "
                  f"({probs[top] * 100:5.1f}%)                  │")
            print("└──────────────────────────────────────────────────────────────┘")
            print("  Ctrl-C to stop", end="", flush=True)

            if log_fp:
                log_fp.write(
                    f"{time.time():.6f},{inference_id},{e2e_us},{cnn_us},"
                    f"{CLASSES[top]},{probs[top]:.4f}\n"
                )
                log_fp.flush()

    except KeyboardInterrupt:
        print("\n[bye]")
    finally:
        ser.close()
        if log_fp:
            log_fp.close()


if __name__ == "__main__":
    main()
