"""Live image + prediction viewer for the camera-streaming firmware.

Reads packets from the MAX78000 over UART (sent by
`c_harness/camera_stream.c`) and renders, in a single matplotlib window:

    ┌────────────────────────┬──────────────────────────────────────┐
    │  32x32 camera input    │  CIFAR-10 softmax — horizontal bars  │
    │  (upscaled for view)   │  top class highlighted, FPS in title │
    └────────────────────────┴──────────────────────────────────────┘

Wire protocol — one packet per inference, all little-endian:

    dev -> host:  uint8   magic[4]  = 0xA5 0xA5 0xCD 0xCD
                  uint32  frame_counter
                  uint32  cnn_cycles
                  int32   logits[10]
                  uint8   img_rgb888[32*32*3]
    total = 3124 bytes

Run via the bash wrapper (recommended — handles port + venv):

    ./bash_device_scripts/host_camera_stream.sh <variant>

Direct invocation:

    uv run --project ../pc-implementation \\
        python host/host_camera_stream.py --port /dev/cu.usbmodemXXXX

Useful flags:
    --port PORT         serial port (auto-detected if omitted)
    --baud N            921600 by default (must match firmware)
    --log PATH          append CSV: frame_id,cnn_cycles,ts,p0..p9
    --temperature T     softmax temperature (default: auto-scaled)
    --upscale N         display the 32x32 image at Nx pixel size (default 8)
"""
from __future__ import annotations

import argparse
import glob
import struct
import sys
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import serial

MAGIC = b"\xa5\xa5\xcd\xcd"
IMG_W, IMG_H = 32, 32
IMG_BYTES = IMG_W * IMG_H * 3
HEADER_BYTES = 4 + 4 + 40          # frame_counter + cnn_cycles + logits[10]
PACKET_BYTES = HEADER_BYTES + IMG_BYTES   # after MAGIC
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
    (frame_id, cnn_cycles, logits, img_hwc_uint8).

    Uses a 4-byte magic for sync because image bytes can take any value
    in [0, 255] — a single-byte sync would collide with pixel data.
    """
    sync = bytearray(4)
    while True:
        b = ser.read(1)
        if not b:
            raise IOError("serial timeout — is the firmware running?")
        sync[:3] = sync[1:]
        sync[3]  = b[0]
        if sync == MAGIC:
            break
    body = ser.read(PACKET_BYTES)
    if len(body) != PACKET_BYTES:
        raise IOError(f"short read: got {len(body)}/{PACKET_BYTES} bytes "
                      f"after magic")
    frame_id, cnn_cycles = struct.unpack("<II", body[:8])
    logits = np.frombuffer(body[8:48], dtype="<i4")
    img = np.frombuffer(body[48:], dtype=np.uint8).reshape(IMG_H, IMG_W, 3)
    return frame_id, cnn_cycles, logits, img


# ---------- math helpers ----------------------------------------------------

def softmax(logits: np.ndarray, temperature: float | None = None) -> np.ndarray:
    """Same auto-scaled softmax as host_camera.py / gui_classify.py — the
    MAX78000's int32 logits often span thousands so a direct softmax
    saturates. We rescale so the dynamic range is roughly [-10, 0]."""
    x = logits.astype(np.float64)
    if temperature is None:
        rng = float(x.max() - x.min())
        temperature = max(rng / 10.0, 1.0)
    x = (x - x.max()) / temperature
    e = np.exp(x)
    return (e / e.sum()).astype(np.float32)


# ---------- main ------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default=None,
                    help="serial port (auto-detected if omitted)")
    ap.add_argument("--baud", type=int, default=115200,
                    help="must match firmware UART_BAUD (default 115200 — "
                         "MAX78000 PCLK can't cleanly divide higher rates)")
    ap.add_argument("--timeout-s", type=float, default=10.0,
                    help="UART read timeout (seconds)")
    ap.add_argument("--log", type=str, default=None,
                    help="append CSV: frame_id,cnn_cycles,ts,p0..p9")
    ap.add_argument("--temperature", type=float, default=None,
                    help="softmax temperature (default: auto-scale)")
    ap.add_argument("--upscale", type=int, default=8,
                    help="display nearest-neighbour upscale factor for the "
                         "32x32 image (default 8 -> 256x256 display)")
    args = ap.parse_args()

    port = args.port or autodetect_port()
    if not port:
        print("ERROR: no /dev/cu.usbmodem* found; pass --port", file=sys.stderr)
        sys.exit(1)

    print(f"opening {port} @ {args.baud}")
    ser = serial.Serial(port, args.baud, timeout=args.timeout_s)

    # Boot-message snapshot — exactly like host_camera.py. The firmware
    # streams continuously so we take ONE non-blocking peek then reset
    # the input buffer for clean packet parsing.
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
        print("[boot] (no output from board — check port + reset button)")
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    log_fp = open(args.log, "a") if args.log else None
    if log_fp:
        log_fp.write("frame_id,cnn_cycles,ts," + ",".join(CLASSES) + "\n")

    # ---------- matplotlib setup ----------
    plt.ion()
    fig, (ax_img, ax_bar) = plt.subplots(
        1, 2, figsize=(11, 5),
        gridspec_kw={"width_ratios": [1.0, 1.4]},
    )
    fig.canvas.manager.set_window_title("MAX78000 — camera livestream")

    # Image panel — show the 32x32 RGB888 frame nearest-neighbour upscaled.
    # imshow's display size is governed by the figure layout, not pixel
    # count; `interpolation='nearest'` keeps the chunky-pixel look.
    img_artist = ax_img.imshow(
        np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8),
        interpolation="nearest",
        vmin=0, vmax=255,
    )
    ax_img.set_xticks([]); ax_img.set_yticks([])
    ax_img.set_title("camera (32x32 input to CNN)", fontsize=11)

    # Bar panel — horizontal so 10 class labels stay readable.
    y_pos = np.arange(len(CLASSES))
    bars = ax_bar.barh(y_pos, [0.0] * len(CLASSES),
                       color="steelblue", edgecolor="black", linewidth=0.4)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(CLASSES, fontsize=10)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, 1.0)
    ax_bar.set_xlabel("softmax probability", fontsize=10)
    ax_bar.set_title("CIFAR-10 prediction", fontsize=11)
    ax_bar.grid(axis="x", linestyle="--", alpha=0.4)

    # Per-bar value text so the user sees the exact probability.
    bar_texts = [
        ax_bar.text(0.0, i, "  0.0%", va="center", fontsize=9)
        for i in range(len(CLASSES))
    ]

    suptitle = fig.suptitle(
        "waiting for first packet…", fontsize=12, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.show()

    # Rolling FPS over last N frames.
    N = 30
    time_hist: deque[float] = deque(maxlen=N)

    try:
        while plt.fignum_exists(fig.number):
            try:
                frame_id, cnn_cycles, logits, img = read_packet(ser)
            except IOError as e:
                print(f"\n[stream error] {e}", file=sys.stderr)
                ser.reset_input_buffer()
                time.sleep(0.1)
                continue

            t_recv = time.perf_counter()
            time_hist.append(t_recv)

            probs = softmax(logits, temperature=args.temperature)
            top = int(np.argmax(probs))

            if len(time_hist) >= 2:
                dt = time_hist[-1] - time_hist[0]
                fps = (len(time_hist) - 1) / dt if dt > 0 else 0.0
            else:
                fps = 0.0
            cnn_us = cnn_cycles / 100.0  # 100 MHz CNN clock

            # ---- update image ----
            img_artist.set_data(img)

            # ---- update bars + value labels ----
            for i, b in enumerate(bars):
                b.set_width(float(probs[i]))
                b.set_color("orange" if i == top else "steelblue")
                bar_texts[i].set_x(float(probs[i]))
                bar_texts[i].set_text(f"  {probs[i] * 100:5.1f}%")
                bar_texts[i].set_fontweight("bold" if i == top else "normal")

            suptitle.set_text(
                f"frame {frame_id:>5}    "
                f"top: {CLASSES[top]}  ({probs[top] * 100:5.1f}%)    "
                f"FPS {fps:4.1f}    CNN {cnn_us:6.1f} µs"
            )

            # Single redraw + event flush — keeps the window responsive.
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

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
        plt.close("all")


if __name__ == "__main__":
    main()
