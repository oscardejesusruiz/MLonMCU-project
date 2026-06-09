"""Stream the CIFAR-10 test set to the MAX78000 over UART, collect per-image
predictions, and save a `.npz` in the same format as the PC predictions.

Wire protocol (host ↔ device):

    Per image:
        host -> dev:   sync 0xAA  +  3072 bytes  (int8, NCHW = 3*32*32, row-major)
        dev  -> host:  sync 0xBB  +  uint32 cnn_cycles  +  10 * int32 logits
        total response = 1 + 4 + 40 = 45 bytes

    Bulk: 10000 images in sequence. Host enforces a per-image timeout.

Usage:
    uv run python host/host_test_set.py \\
        --port /dev/cu.usbmodemXXXX \\
        --baud 115200 \\
        --tag baseline_qat_device \\
        --n 10000

The output `reports/predictions/<tag>.npz` has the same schema as the PC side
(`y_true`, `fp32_y_pred/_probs`, `int8_y_pred/_probs`), so the shared
`plot_models_report.py` works on it directly. `fp32_*` and `int8_*` are
identical for a device run — there is no fp32 path on the MCU.

Companion firmware: `c_harness/inference_test_set.c`.
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np
import serial
from torchvision import datasets

SYNC_REQ = 0xAA
SYNC_REP = 0xBB
IMG_BYTES = 3 * 32 * 32          # 3072
REP_BYTES = 1 + 4 + 4 * 10       # 45


def _ai8x_normalize(img_uint8: np.ndarray) -> np.ndarray:
    """Match ai8x normalize: (x/255 - 0.5) * 256 -> int8 in [-128, 127].

    img_uint8 shape: (32, 32, 3) HWC. Output shape: (3, 32, 32) CHW int8.
    """
    x = img_uint8.astype(np.float32) / 255.0
    x = (x - 0.5) * 256.0
    x = np.round(x).clip(-128, 127).astype(np.int8)
    return np.transpose(x, (2, 0, 1))  # HWC -> CHW


def _send_image(ser: serial.Serial, img_int8_chw: np.ndarray) -> tuple[int, np.ndarray]:
    assert img_int8_chw.dtype == np.int8 and img_int8_chw.size == IMG_BYTES
    ser.write(bytes([SYNC_REQ]))
    ser.write(img_int8_chw.tobytes())
    resp = ser.read(REP_BYTES)
    if len(resp) != REP_BYTES:
        raise IOError(f"short response: got {len(resp)}/{REP_BYTES} bytes")
    if resp[0] != SYNC_REP:
        raise IOError(f"bad sync byte: 0x{resp[0]:02x} (expected 0x{SYNC_REP:02x})")
    cycles = struct.unpack("<I", resp[1:5])[0]
    logits = np.frombuffer(resp[5:], dtype="<i4")
    return cycles, logits


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float32)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="e.g. /dev/cu.usbmodemXXXX")
    ap.add_argument("--baud", type=int, default=921600)
    ap.add_argument("--n", type=int, default=10000,
                    help="number of test images to stream (max 10000)")
    ap.add_argument("--tag", required=True,
                    help="output tag, e.g. baseline_qat_device")
    ap.add_argument("--timeout-s", type=float, default=2.0,
                    help="per-image UART timeout")
    ap.add_argument("--data-root", type=Path,
                    default=Path(__file__).resolve().parents[1] / "data")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "reports" / "predictions")
    args = ap.parse_args()

    test_set = datasets.CIFAR10(root=str(args.data_root), train=False,
                                download=True, transform=None)
    n = min(args.n, len(test_set))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    ys = np.zeros(n, dtype=np.int64)
    preds = np.zeros(n, dtype=np.int64)
    probs = np.zeros((n, 10), dtype=np.float32)
    cycles_arr = np.zeros(n, dtype=np.int64)

    print(f"opening {args.port} @ {args.baud}")
    ser = serial.Serial(args.port, args.baud, timeout=args.timeout_s)
    # Opening the port may toggle DTR and reset the FTHR board. Wait for the
    # firmware to finish Board_Init + cnn_init/load_weights/configure (which
    # together take ~0.5-1s) and then drain the BOOT log lines from the buffer.
    time.sleep(2.0)
    drained = ser.read(ser.in_waiting or 1)
    if drained:
        # Print so the user can confirm the device booted cleanly.
        try:
            print("[boot]", drained.decode("ascii", errors="replace").strip())
        except Exception:
            print(f"[boot] {len(drained)} bytes")
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    t0 = time.time()
    last_log = t0
    try:
        for i in range(n):
            img_pil, y = test_set[i]
            img = np.array(img_pil, dtype=np.uint8)            # HWC, uint8
            img_int8 = _ai8x_normalize(img)
            try:
                cycles, logits = _send_image(ser, img_int8)
            except IOError as e:
                print(f"\n[{i}] {e} — re-syncing...", file=sys.stderr)
                ser.reset_input_buffer()
                time.sleep(0.05)
                cycles, logits = _send_image(ser, img_int8)
            p = _softmax(logits)
            ys[i] = y
            preds[i] = int(np.argmax(p))
            probs[i] = p
            cycles_arr[i] = cycles
            if time.time() - last_log > 2.0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (n - i - 1) / rate
                print(f"  [{i+1:>5}/{n}]  acc_so_far={(preds[:i+1]==ys[:i+1]).mean()*100:5.2f}%"
                      f"  rate={rate:5.1f} img/s  eta={eta:5.0f}s", flush=True)
                last_log = time.time()
    finally:
        ser.close()

    acc = float((preds == ys).mean())
    median_cycles = int(np.median(cycles_arr))
    print(f"\ndevice test acc: {acc*100:.2f}%")
    print(f"median CNN cycles/inference: {median_cycles:,}  "
          f"(~{median_cycles/100e6*1e6:.1f} us @ 100 MHz)")

    out_path = args.out_dir / f"{args.tag}.npz"
    np.savez(out_path,
             y_true=ys,
             fp32_y_pred=preds, fp32_y_probs=probs,
             int8_y_pred=preds, int8_y_probs=probs,
             cycles=cycles_arr)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
