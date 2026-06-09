"""Live monitor for the camera-streaming inference firmware.

Reads packets from the MAX78000 over UART (sent by
`c_harness/camera_inference.c`), computes softmax over the 10 CIFAR-10
class logits per frame, and displays a continuously-updating distribution
in the terminal. Optionally appends a CSV log so the stream can be
post-processed offline.

Wire protocol — one packet per inference, all little-endian:

    dev -> host:  uint8   sync          = 0xCC
                  uint32  frame_counter
                  uint32  cnn_cycles
                  int32   logits[10]
    total = 49 bytes

Run via the bash wrapper (recommended — handles port + venv + logging):

    ./bash_device_scripts/host_camera.sh <variant>

Direct invocation:

    uv run --project ../pc-implementation \\
        python host/host_camera.py --port /dev/cu.usbmodemXXXX

Useful flags:
    --port PORT         serial port (auto-detected if omitted)
    --baud N            115200 by default (must match firmware)
    --log PATH          append CSV: frame_id,cnn_cycles,ts,p0..p9
    --no-clear          don't clear screen each frame (logging-friendly)
    --temperature T     softmax temperature (default: auto-scaled to map
                        the logit range to ~10, gives meaningful
                        probabilities for the MAX78000's int32 outputs)
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

SYNC_BYTE = 0xCC
PACKET_BYTES = 1 + 4 + 4 + 4 * 10   # 49
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


# ---------- serial helpers --------------------------------------------------

def autodetect_port() -> str | None:
    for pat in ("/dev/cu.usbmodem*", "/dev/tty.usbmodem*", "/dev/ttyACM*"):
        found = sorted(glob.glob(pat))
        if found:
            return found[0]
    return None


def read_packet(ser: serial.Serial):
    """Block until a complete packet is received. Returns
    (frame_id, cnn_cycles, logits)."""
    # Sync up — discard bytes until we find the sync marker.
    while True:
        b = ser.read(1)
        if not b:
            raise IOError("serial timeout — is the firmware running?")
        if b[0] == SYNC_BYTE:
            break
    body = ser.read(PACKET_BYTES - 1)
    if len(body) != PACKET_BYTES - 1:
        raise IOError(f"short read: got {len(body)}/{PACKET_BYTES - 1} bytes "
                      f"after sync byte")
    frame_id, cnn_cycles = struct.unpack("<II", body[:8])
    logits = np.frombuffer(body[8:], dtype="<i4")
    return frame_id, cnn_cycles, logits


# ---------- math + display helpers -----------------------------------------

def softmax(logits: np.ndarray, temperature: float | None = None) -> np.ndarray:
    """Same auto-scaled softmax as host/gui_classify.py — the MAX78000's
    int32 logits often span thousands, so a direct softmax saturates.
    We rescale so the dynamic range is roughly [-10, 0] before exp()."""
    x = logits.astype(np.float64)
    if temperature is None:
        rng = float(x.max() - x.min())
        temperature = max(rng / 10.0, 1.0)
    x = (x - x.max()) / temperature
    e = np.exp(x)
    return (e / e.sum()).astype(np.float32)


def bar(p: float, width: int = 24) -> str:
    filled = int(round(p * width))
    return "█" * filled + "░" * (width - filled)


def clear_screen() -> None:
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


# ---------- main ------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default=None,
                    help="serial port (auto-detected if omitted)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--timeout-s", type=float, default=5.0,
                    help="UART read timeout (seconds)")
    ap.add_argument("--log", type=str, default=None,
                    help="append CSV: frame_id,cnn_cycles,ts,p0..p9")
    ap.add_argument("--no-clear", action="store_true",
                    help="print each frame on a new line (logging-friendly)")
    ap.add_argument("--temperature", type=float, default=None,
                    help="softmax temperature (default: auto-scale)")
    args = ap.parse_args()

    port = args.port or autodetect_port()
    if not port:
        print("ERROR: no /dev/cu.usbmodem* found; pass --port", file=sys.stderr)
        sys.exit(1)

    print(f"opening {port} @ {args.baud}")
    ser = serial.Serial(port, args.baud, timeout=args.timeout_s)

    # Wait for the board to boot, then take ONE snapshot of whatever bytes
    # are waiting. We deliberately do NOT loop here: the camera firmware
    # streams packets continuously at ~25 FPS, so a loop that keeps reading
    # while bytes arrive would never terminate. Instead we snapshot, filter
    # out the printable BOOT messages for display, drop the rest, and reset
    # the input buffer so packet parsing starts on a clean boundary.
    time.sleep(3.0)
    n_waiting = ser.in_waiting
    drained = ser.read(n_waiting) if n_waiting > 0 else b""
    if drained:
        # Extract only the printable ASCII portion (the BOOT messages) and
        # drop binary packet bytes that arrived in the same window.
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

        # --- Diagnostic: what bytes are we actually seeing? ---
        from collections import Counter
        freq = Counter(drained)
        n_sync = drained.count(SYNC_BYTE)
        print(f"[diag] unique byte values seen: {len(freq)}")
        print(f"[diag] SYNC_BYTE (0x{SYNC_BYTE:02x}) occurrences: {n_sync}")
        print(f"[diag] top-5 most common bytes: "
              f"{[(f'0x{b:02x}', c) for b, c in freq.most_common(5)]}")
        print(f"[diag] first 64 bytes (hex): "
              f"{' '.join(f'{b:02x}' for b in drained[:64])}")
        if n_sync == 0:
            print("[diag] !!! NO sync bytes found — firmware is NOT streaming "
                  "binary packets. Something is printing text only.")
    else:
        print("[boot] (no output from board — check port + reset button)")
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    log_fp = open(args.log, "a") if args.log else None
    if log_fp:
        log_fp.write("frame_id,cnn_cycles,ts," + ",".join(CLASSES) + "\n")

    # Rolling FPS / cycle stats over the last N frames.
    N = 30
    cycles_hist: deque[int] = deque(maxlen=N)
    time_hist:   deque[float] = deque(maxlen=N)

    try:
        while True:
            try:
                frame_id, cnn_cycles, logits = read_packet(ser)
            except IOError as e:
                print(f"\n[stream error] {e}", file=sys.stderr)
                ser.reset_input_buffer()
                time.sleep(0.1)
                continue

            t_recv = time.perf_counter()
            cycles_hist.append(cnn_cycles)
            time_hist.append(t_recv)

            probs = softmax(logits, temperature=args.temperature)
            top_idx = int(np.argmax(probs))

            # FPS over the rolling window
            if len(time_hist) >= 2:
                dt = time_hist[-1] - time_hist[0]
                fps = (len(time_hist) - 1) / dt if dt > 0 else 0.0
            else:
                fps = 0.0
            avg_cycles = sum(cycles_hist) / len(cycles_hist)
            cnn_us = avg_cycles / 100.0    # CNN clock is 100 MHz

            if not args.no_clear:
                clear_screen()

            print(f"┌─────────────────────────────────────────────────────┐")
            print(f"│  MAX78000 — Live CIFAR-10 inference                 │")
            print(f"├─────────────────────────────────────────────────────┤")
            print(f"│  Frame: {frame_id:>6}   "
                  f"FPS: {fps:5.1f}   "
                  f"CNN: {cnn_us:6.1f} µs        │")
            print(f"├─────────────────────────────────────────────────────┤")
            for i, c in enumerate(CLASSES):
                mark = "◀ TOP" if i == top_idx else "     "
                print(f"│  {c:<11} {bar(probs[i])} "
                      f"{probs[i]*100:5.1f}%  {mark} │")
            print(f"└─────────────────────────────────────────────────────┘")
            print(f"  Ctrl-C to stop", end="", flush=True)

            if log_fp:
                log_fp.write(f"{frame_id},{cnn_cycles},{time.time():.6f},"
                             + ",".join(f"{p:.6f}" for p in probs) + "\n")
                log_fp.flush()

    except KeyboardInterrupt:
        print("\n[bye]")
    finally:
        ser.close()
        if log_fp:
            log_fp.close()


if __name__ == "__main__":
    main()
