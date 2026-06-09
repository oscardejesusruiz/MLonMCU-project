#!/usr/bin/env python3
"""Tkinter GUI for the MAX78000 CIFAR-10 inference firmware.

Flow:
  Screen 1 — pick a CIFAR-10 class (10 buttons).
  → app selects a random test image from that class, sends it over UART,
    waits for the firmware response, computes softmax + latencies.
  Screen 2 — shows the image, ground-truth label, predicted label,
    per-class probability bars, on-device CNN time, and round-trip latency.
  → "Back" returns to Screen 1.

Talks to `c_harness/inference_test_set.c` over the same wire protocol as
`host/host_test_set.py`:

    host → dev:  0xAA + 3072 bytes int8 image (CHW, row-major)
    dev  → host: 0xBB + uint32 cnn_cycles + 10 × int32 logits
    total response = 45 bytes

Run with the system Python (has tkinter built-in):

    /usr/bin/python3 -m pip install --user pyserial   # one-off
    /usr/bin/python3 host/gui_classify.py             # or pass --port

Dependencies: numpy, Pillow (both shipped with system Python on macOS),
pyserial (one pip install), plus stdlib tkinter.
"""
from __future__ import annotations

import argparse
import glob
import pickle
import queue
import struct
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np
import serial
from PIL import Image, ImageTk

# ---------- wire protocol (matches inference_test_set.c) --------------------

SYNC_REQ  = 0xAA
SYNC_REP  = 0xBB
IMG_BYTES = 3 * 32 * 32
REP_BYTES = 1 + 4 + 4 * 10
CLASSES   = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Default CIFAR-10 location — pickle files shipped with the project.
DEFAULT_CIFAR_DIR = (
    Path(__file__).resolve().parents[2]
    / "pc-implementation" / "data" / "cifar-10-batches-py"
)


# ---------- CIFAR-10 loader (no torchvision dependency) ---------------------

def load_cifar10_test(cifar_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Returns (images uint8 HWC [N,32,32,3], labels int [N])."""
    test_batch = cifar_dir / "test_batch"
    if not test_batch.exists():
        raise FileNotFoundError(
            f"CIFAR-10 not found at {test_batch}. "
            "Pass --cifar-dir or download cifar-10-batches-py.")
    with open(test_batch, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    raw = d[b"data"]                    # shape (10000, 3072), uint8, row-major CHW
    labels = np.array(d[b"labels"], dtype=np.int64)
    imgs = raw.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # → HWC
    return imgs, labels


def ai8x_normalize(img_uint8_hwc: np.ndarray) -> np.ndarray:
    """Same normalization as host_test_set.py — produces int8 CHW."""
    x = img_uint8_hwc.astype(np.float32) / 255.0
    x = (x - 0.5) * 256.0
    x = np.round(x).clip(-128, 127).astype(np.int8)
    return np.transpose(x, (2, 0, 1))


def softmax(logits: np.ndarray, temperature: float | None = None) -> np.ndarray:
    """Softmax with auto-temperature.

    The MAX78000's final layer emits int32 logits scaled by `output_shift`,
    so absolute values are routinely in the thousands. A direct softmax then
    saturates to {0, 0, ..., 1.0, 0, ...}, which hides the true confidence.
    We rescale so the dynamic range is roughly [-10, 0] before exp(), giving
    sensible-looking probabilities that still preserve argmax."""
    x = logits.astype(np.float64)
    if temperature is None:
        rng = float(x.max() - x.min())
        # Keep at least 1.0 so very low-range logits aren't blown up.
        temperature = max(rng / 10.0, 1.0)
    x = (x - x.max()) / temperature
    e = np.exp(x)
    return (e / e.sum()).astype(np.float32)


# ---------- serial helper ---------------------------------------------------

def autodetect_port() -> str | None:
    for pat in ("/dev/cu.usbmodem*", "/dev/tty.usbmodem*", "/dev/ttyACM*"):
        found = sorted(glob.glob(pat))
        if found:
            return found[0]
    return None


def send_and_receive(ser: serial.Serial, img_int8_chw: np.ndarray):
    """Returns (cycles, logits, latency_seconds) where latency covers
    the entire 0xAA-out → last-byte-in round-trip. Also logs the wire
    activity to stdout so you can watch what's going over UART."""
    assert img_int8_chw.dtype == np.int8 and img_int8_chw.size == IMG_BYTES
    ser.reset_input_buffer()
    img_bytes = img_int8_chw.tobytes()
    # Show first / last few image bytes (signed int8 view) for sanity.
    head = [int(x) for x in img_int8_chw.ravel()[:8]]
    tail = [int(x) for x in img_int8_chw.ravel()[-8:]]
    print(f"  TX  sync=0x{SYNC_REQ:02X}  +  {IMG_BYTES} B int8 CHW")
    print(f"      img[ 0: 8] = {head}")
    print(f"      img[-8:  ] = {tail}")
    t0 = time.perf_counter()
    ser.write(bytes([SYNC_REQ]))
    ser.write(img_bytes)
    ser.flush()
    resp = ser.read(REP_BYTES)
    t1 = time.perf_counter()
    print(f"  RX  {len(resp)}/{REP_BYTES} B in {(t1 - t0)*1000:.2f} ms")
    if len(resp) != REP_BYTES:
        raise IOError(f"short response: got {len(resp)}/{REP_BYTES} bytes")
    if resp[0] != SYNC_REP:
        raise IOError(f"bad sync byte: 0x{resp[0]:02x}")
    cycles = struct.unpack("<I", resp[1:5])[0]
    logits = np.frombuffer(resp[5:], dtype="<i4").copy()
    print(f"      sync=0x{resp[0]:02X}  cycles={cycles}  "
          f"(~{cycles/100:.1f} us @ 100 MHz)")
    print(f"      logits = {[int(x) for x in logits]}")
    return cycles, logits, (t1 - t0)


# ---------- GUI -------------------------------------------------------------

class App(tk.Tk):
    def __init__(self, port: str, baud: int, cifar_dir: Path) -> None:
        super().__init__()
        self.title("MAX78000 CIFAR-10 — live classifier")
        self.geometry("780x620")
        self.minsize(720, 580)

        # Load dataset once and bucket indices by class for fast random pick.
        try:
            self.imgs, self.labels = load_cifar10_test(cifar_dir)
        except FileNotFoundError as e:
            messagebox.showerror("CIFAR-10 missing", str(e))
            sys.exit(1)
        self.idx_by_class = [
            np.where(self.labels == c)[0] for c in range(10)
        ]
        self.rng = np.random.default_rng()

        # Open serial port (drain BOOT lines so they don't clog the first read).
        try:
            self.ser = serial.Serial(port, baud, timeout=3.0)
        except serial.SerialException as e:
            messagebox.showerror("Serial open failed",
                                 f"Could not open {port} @ {baud}\n\n{e}")
            sys.exit(1)
        time.sleep(2.0)                  # let FTHR finish Board_Init / cnn_init
        drained = self.ser.read(self.ser.in_waiting or 1)
        if drained:
            try:
                print("[boot]", drained.decode("ascii", errors="replace").strip())
            except Exception:
                pass
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

        self.port_label = f"{port} @ {baud}"
        # Prevents double-clicks while a UART round-trip is in flight.
        self._busy = False
        # Worker → main-thread handoff. Polled by _poll_result while busy.
        self._results: queue.Queue = queue.Queue()

        # Build a single grid cell that holds BOTH frames stacked. Swap
        # between them with tkraise() — far more reliable than pack_forget
        # on macOS Tk where deferred redraws cause the screen to "freeze".
        container = tk.Frame(self)
        container.pack(fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.select_frame = self._build_select_frame(container)
        self.result_frame = self._build_result_frame(container)
        for f in (self.select_frame, self.result_frame):
            f.grid(row=0, column=0, sticky="nsew")
        self._show_select()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # -------- screen 1: class selector --------

    def _build_select_frame(self, parent: tk.Widget) -> tk.Frame:
        f = tk.Frame(parent, padx=20, pady=20)
        tk.Label(f, text="Choose a CIFAR-10 class",
                 font=("Helvetica", 18, "bold")).pack(pady=(0, 6))
        tk.Label(f, text=f"connected: {self.port_label}",
                 font=("Helvetica", 11), fg="#555").pack(pady=(0, 18))

        grid = tk.Frame(f)
        grid.pack()
        for i, name in enumerate(CLASSES):
            r, c = divmod(i, 5)
            b = tk.Button(grid, text=name, width=12, height=2,
                          font=("Helvetica", 12),
                          command=lambda c=i: self._on_class(c))
            b.grid(row=r, column=c, padx=6, pady=6)

        self.status_var = tk.StringVar(value="ready")
        tk.Label(f, textvariable=self.status_var,
                 font=("Helvetica", 11), fg="#666").pack(pady=(20, 0))
        return f

    def _on_class(self, class_id: int) -> None:
        """Pick a random image of that class and run inference in a worker."""
        if self._busy:
            print("  (ignored — previous inference still in flight)")
            return
        self._busy = True

        pool = self.idx_by_class[class_id]
        idx = int(self.rng.choice(pool))
        img_hwc = self.imgs[idx]                     # uint8 HWC
        true_label = int(self.labels[idx])
        img_int8 = ai8x_normalize(img_hwc)
        print(f"\n[{time.strftime('%H:%M:%S')}] "
              f"clicked '{CLASSES[class_id]}' → test idx #{idx}  "
              f"(true label = {CLASSES[true_label]})")
        self.status_var.set(f"sending image #{idx} (true={CLASSES[true_label]})…")

        # Worker pushes result onto a Queue; the Tk main thread polls it via
        # after(). Calling Tk methods (including after()) directly from a
        # worker thread is NOT safe on macOS — that was the prior bug.
        def worker():
            try:
                cycles, logits, latency = send_and_receive(self.ser, img_int8)
                self._results.put(("ok", img_hwc, true_label,
                                   cycles, logits, latency, idx))
            except Exception as e:
                self._results.put(("err", str(e)))

        threading.Thread(target=worker, daemon=True).start()
        self.after(50, self._poll_result)

    def _poll_result(self) -> None:
        """Runs on the Tk main thread. Keeps polling until the queue has the
        worker's result, then dispatches to the right handler."""
        try:
            msg = self._results.get_nowait()
        except queue.Empty:
            self.after(50, self._poll_result)
            return

        if msg[0] == "err":
            err = msg[1]
            print(f"  !! inference error: {err}")
            messagebox.showerror("Inference failed", err)
            self.status_var.set("ready")
            self._busy = False
            return

        _, img_hwc, true_label, cycles, logits, latency, idx = msg
        try:
            self._show_result(img_hwc, true_label, cycles, logits, latency, idx)
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Display error", repr(e))
        finally:
            self._busy = False

    # -------- screen 2: result --------

    def _build_result_frame(self, parent: tk.Widget) -> tk.Frame:
        f = tk.Frame(parent, padx=20, pady=20)

        top = tk.Frame(f)
        top.pack(fill="x", pady=(0, 14))

        # Left: image + labels
        left = tk.Frame(top)
        left.pack(side="left", padx=(0, 24))
        self.img_canvas = tk.Canvas(left, width=224, height=224,
                                    bg="#222", highlightthickness=0)
        self.img_canvas.pack()
        self.true_lbl = tk.Label(left, text="", font=("Helvetica", 13))
        self.true_lbl.pack(pady=(8, 0))
        self.pred_lbl = tk.Label(left, text="", font=("Helvetica", 14, "bold"))
        self.pred_lbl.pack()

        # Right: per-class probability bars
        right = tk.Frame(top)
        right.pack(side="left", fill="both", expand=True)
        tk.Label(right, text="class probabilities",
                 font=("Helvetica", 12, "bold")).pack(anchor="w", pady=(0, 6))
        self.bars_canvas = tk.Canvas(right, width=440, height=260,
                                     bg="white", highlightthickness=1,
                                     highlightbackground="#ccc")
        self.bars_canvas.pack()

        # Stats block
        self.stats_lbl = tk.Label(f, text="", justify="left",
                                  font=("Menlo", 12), anchor="w")
        self.stats_lbl.pack(fill="x", pady=(8, 14))

        # Back button
        tk.Button(f, text="← Back", font=("Helvetica", 13),
                  width=14, command=self._show_select).pack(pady=(0, 0))
        return f

    def _show_result(self, img_hwc, true_label, cycles, logits, latency, idx):
        # Upscale 32×32 → 224×224 for visibility.
        pil = Image.fromarray(img_hwc).resize((224, 224), Image.NEAREST)
        self._tk_img = ImageTk.PhotoImage(pil)             # keep a ref!
        self.img_canvas.delete("all")
        self.img_canvas.create_image(0, 0, anchor="nw", image=self._tk_img)

        probs = softmax(logits)
        pred = int(np.argmax(probs))
        verdict = "✓" if pred == true_label else "✗"
        print(f"  → softmax: " + "  ".join(
              f"{CLASSES[i][:4]}={probs[i]*100:5.1f}%" for i in range(10)))
        print(f"  → pred={CLASSES[pred]} ({probs[pred]*100:.1f}%)  "
              f"true={CLASSES[true_label]}  {verdict}  "
              f"round-trip={latency*1000:.2f} ms")

        self.true_lbl.config(text=f"true: {CLASSES[true_label]}  (test idx {idx})")
        ok = (pred == true_label)
        self.pred_lbl.config(
            text=f"predicted: {CLASSES[pred]}   {probs[pred]*100:5.1f} %",
            fg=("#1a8a1a" if ok else "#b00020"))

        self._draw_bars(probs, pred, true_label)

        cnn_us = cycles / 100.0          # CNN clock is 100 MHz
        latency_ms = latency * 1000.0
        stats = (
            f"CNN cycles      : {cycles:>10,}   (≈ {cnn_us:>8.1f} µs @ 100 MHz)\n"
            f"Round-trip      : {latency_ms:>10.2f} ms"
            f"   (send 0xAA+3072 B → 45 B response)\n"
            f"Host overhead   : {latency_ms - cnn_us/1000.0:>10.2f} ms"
            f"   (latency − CNN time)\n"
            f"Top-1 logit     : {int(logits[pred]):>10}     "
            f"runner-up logit : {int(np.partition(logits, -2)[-2])}"
        )
        self.stats_lbl.config(text=stats)
        self._show_result_frame()

    def _draw_bars(self, probs: np.ndarray, pred: int, truth: int) -> None:
        c = self.bars_canvas
        c.delete("all")
        W = int(c["width"]); H = int(c["height"])
        left_pad, right_pad, top_pad, bot_pad = 100, 60, 8, 8
        usable_w = W - left_pad - right_pad
        row_h = (H - top_pad - bot_pad) / 10
        bar_h = row_h * 0.72
        for i, p in enumerate(probs):
            y = top_pad + i * row_h + (row_h - bar_h) / 2
            colour = "#1a8a1a" if i == truth else (
                     "#b00020" if i == pred else "#3070d0")
            c.create_text(left_pad - 8, y + bar_h / 2,
                          text=CLASSES[i], anchor="e",
                          font=("Helvetica", 11))
            c.create_rectangle(left_pad, y,
                               left_pad + usable_w, y + bar_h,
                               fill="#eee", outline="")
            c.create_rectangle(left_pad, y,
                               left_pad + usable_w * float(p), y + bar_h,
                               fill=colour, outline="")
            c.create_text(left_pad + usable_w + 6, y + bar_h / 2,
                          text=f"{p*100:5.1f}%", anchor="w",
                          font=("Menlo", 10))

    # -------- frame swap helpers --------

    def _show_select(self) -> None:
        # Both frames live in the same grid cell; raise() decides which is
        # on top. This is the standard tkinter pattern for screen swapping
        # and avoids the pack-forget redraw quirks on macOS.
        self.select_frame.tkraise()
        self.status_var.set("ready")
        print("  [ui] back to class-selection screen")

    def _show_result_frame(self) -> None:
        self.result_frame.tkraise()
        print("  [ui] switched to result screen")

    def _on_close(self) -> None:
        try:
            self.ser.close()
        except Exception:
            pass
        self.destroy()


# ---------- entry -----------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None,
                    help="serial port (auto-detected if omitted)")
    ap.add_argument("--baud", type=int, default=115200,
                    help="match firmware (inference_test_set.c uses 115200)")
    ap.add_argument("--cifar-dir", type=Path, default=DEFAULT_CIFAR_DIR,
                    help="path to cifar-10-batches-py/")
    args = ap.parse_args()

    port = args.port or autodetect_port()
    if not port:
        print("ERROR: no /dev/cu.usbmodem* found and --port not given",
              file=sys.stderr)
        sys.exit(1)

    App(port=port, baud=args.baud, cifar_dir=args.cifar_dir).mainloop()


if __name__ == "__main__":
    main()
