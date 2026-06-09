"""Live camera + prediction viewer for an IMX500-deployed CIFAR-10 model.

Direct sibling of max78000-implementation/host/host_camera_stream.py.
That one reads UART packets from the MAX78000 firmware; this one reads
frame metadata from the Raspberry Pi AI Camera (IMX500) — same window
layout, same bar-chart, same softmax-with-auto-temperature.

What the script does, per frame:

    1. capture_request() from Picamera2 — gives us:
         * the RGB camera image (display)
         * the frame metadata, which the IMX500 firmware has stamped
           with the network's output tensor and KPI info
    2. imx500.get_outputs(metadata)   → the 10 CIFAR-10 class logits
    3. imx500.get_kpi_info(metadata)  → (dnn_time_ms, dsp_time_ms)
    4. Softmax → top class → update matplotlib

Why a polling loop and not picam2.pre_callback:
    matplotlib is not thread-safe and pre_callback fires on the camera
    thread. Polling capture_request() from the main thread keeps all
    drawing single-threaded — same pattern as host_camera_stream.py.

Run on the Pi (with the AI Camera + picamera2 stack installed):

    python3 imx500-implementation/camera_imx500_view.py \\
        --model /path/to/network.rpk

Useful flags:
    --model PATH       .rpk package exported via package_all_rpk.sh
                       (REQUIRED)
    --size W H         camera preview resolution (default 640 480)
    --upscale N        nearest-neighbour upscale factor for the image
                       panel (default 1 — IMX500 frames are already big)
    --log PATH         append CSV: ts,frame_id,dnn_time_ms,top,top_prob
    --temperature T    softmax temperature (default: auto-scale, same as
                       host_camera_stream.py)

Companion: imx500-implementation/camera_pi_1.py (text-only version of
the same loop; this script is the matplotlib upgrade).
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

# These are only available on a Raspberry Pi with the AI Camera stack
# installed (`sudo apt install python3-picamera2 imx500-tools`). Import
# lazily so the script's --help works on any machine.
try:
    from picamera2 import Picamera2
    from picamera2.devices.imx500 import IMX500
    _PICAMERA2_AVAILABLE = True
except ImportError as e:
    _PICAMERA2_AVAILABLE = False
    _PICAMERA2_IMPORT_ERROR = e


CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

# Side length of the CNN's input tensor (CIFAR-10 = 32). The IMX500
# firmware centre-crops the camera frame and bilinear-downsamples it to
# this size before feeding the NPU. We mirror the same two steps on the
# host so the preview shows exactly the patch the network classifies.


# ---------- preprocessing (matches IMX500 firmware as closely as we can) --

def picamera2_to_rgb(arr: np.ndarray) -> np.ndarray:
    """Picamera2 quirk: when the stream is configured with
    `format="RGB888"`, the returned numpy array is actually BGR in
    memory. Swap the last axis to get true RGB for matplotlib /
    "matches what the model was trained on" purposes.

    If you ever change the configure() call to `format="BGR888"` (which
    *does* give RGB in memory — yes, the names are inverted), drop this
    call.
    """
    return arr[..., ::-1]

# ---------- math helpers ---------------------------------------------------

def softmax(logits: np.ndarray, temperature: float | None = None) -> np.ndarray:
    """Auto-temperature softmax — same convention as host_camera_stream.py.

    The IMX500 may dequantize outputs for us (float) or hand back raw
    int8 logits depending on how MCT exported the model. Either way the
    dynamic range can swing wildly, so we auto-scale before exp() to
    keep numerics sane and the bar chart readable.
    """
    x = np.asarray(logits, dtype=np.float64).reshape(-1)
    if temperature is None:
        rng = float(x.max() - x.min())
        temperature = max(rng / 10.0, 1.0)
    x = (x - x.max()) / temperature
    e = np.exp(x)
    return (e / e.sum()).astype(np.float32)


# ---------- main -----------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, type=Path,
                    help=".rpk package exported by package_all_rpk.sh")
    ap.add_argument("--size", nargs=2, type=int, default=(640, 480),
                    metavar=("W", "H"),
                    help="camera preview resolution (default 640 480)")
    ap.add_argument("--log", type=str, default=None,
                    help="append CSV: ts,frame_id,dnn_time_ms,top,top_prob")
    ap.add_argument("--temperature", type=float, default=None,
                    help="softmax temperature (default: auto-scale)")
    ap.add_argument("--no-flip", action="store_true",
                    help="don't horizontally flip the camera image "
                         "(default: flip so it reads like a mirror)")
    ap.add_argument(
        "--roi-size",
        type=int,
        default=1536,
        help="square inference ROI in sensor pixels"
    )
    args = ap.parse_args()

    if not _PICAMERA2_AVAILABLE:
        print("ERROR: picamera2 not importable — this script must run on a "
              "Raspberry Pi with the AI Camera stack:", file=sys.stderr)
        print(f"        {_PICAMERA2_IMPORT_ERROR}", file=sys.stderr)
        print("        sudo apt install python3-picamera2 imx500-tools",
              file=sys.stderr)
        sys.exit(1)

    if not args.model.exists():
        print(f"ERROR: --model path does not exist: {args.model}",
              file=sys.stderr)
        sys.exit(1)

    print(f"[boot] loading {args.model} into the IMX500…")
    imx500 = IMX500(str(args.model))
    picam2 = Picamera2(imx500.camera_num)

    # `RGB888` so the captured array is straight (H, W, 3) uint8 RGB
    # ready for matplotlib (no YUV→RGB conversion in Python).
    config = picam2.create_preview_configuration(
        main={"size": tuple(args.size), "format": "RGB888"}
    )
    picam2.configure(config)

    sensor_rect = imx500._IMX500__get_full_sensor_resolution()

    sensor_w = sensor_rect.width
    sensor_h = sensor_rect.height

    # Region actually fed to the network before the IMX500 resizes
    # it to the model's 32x32 input tensor.
    roi_size = args.roi_size

    imx500.set_inference_roi_abs((
        (sensor_w - roi_size) // 2,
        (sensor_h - roi_size) // 2,
        roi_size,
        roi_size,
    ))

    print(
        f"[boot] inference ROI = {roi_size}x{roi_size} "
        f"within sensor {sensor_w}x{sensor_h}"
    )

    

    # Useful progress bar while the network firmware uploads to the
    # sensor (takes a few hundred ms the first time).
    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=False)   # we draw our own window
    print(f"[boot] camera streaming at {args.size[0]}x{args.size[1]}")

    log_fp = open(args.log, "a") if args.log else None
    if log_fp:
        log_fp.write("ts,frame_id,dnn_time_ms,top_class,top_prob\n")

    # ---------- matplotlib setup (same layout as host_camera_stream.py) ----
    plt.ion()
    fig, (ax_img, ax_bar) = plt.subplots(
        1, 2, figsize=(11, 5),
        gridspec_kw={"width_ratios": [1.0, 1.4]},
    )
    fig.canvas.manager.set_window_title("IMX500 — camera livestream")

    # Image panel — shows the MODEL'S input (centre-crop + 32×32 resize),
    # NOT the raw camera frame. nearest-neighbour interpolation keeps
    # the chunky-pixel look so it's visually unmistakable that this is
    # the 32×32 patch the network actually classifies.
    img_artist = ax_img.imshow(
        np.zeros((args.size[1], args.size[0], 3), dtype=np.uint8)
    )

    ax_img.set_title(
        "camera image (red box = inference ROI)",
        fontsize=11,
    )

    roi_rect = Rectangle(
        (0, 0),
        1,
        1,
        fill=False,
        edgecolor="red",
        linewidth=2,
    )

    ax_img.add_patch(roi_rect)

    # Bar panel — horizontal, same as the MAX78000 viewer for
    # visual parity.
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

    bar_texts = [
        ax_bar.text(0.0, i, "  0.0%", va="center", fontsize=9)
        for i in range(len(CLASSES))
    ]

    suptitle = fig.suptitle(
        "waiting for first IMX500 inference…",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.show()

    # ---------- rolling stats (same window as the MAX78000 viewer) ---------
    N = 30
    callback_t_hist: deque[float] = deque(maxlen=N)
    dnn_ms_hist:     deque[float] = deque(maxlen=N)

    frame_id = 0
    n_no_output = 0

    try:
        while plt.fignum_exists(fig.number):
            # Poll the next ready request. capture_request() blocks until
            # one is available — that gives us frame-rate-paced updates
            # without needing a sleep().
            request = picam2.capture_request()
            try:
                raw_img = request.make_array("main")
                metadata = request.get_metadata()

                # Step 1: fix the Picamera2 BGR/RGB quirk — the array
                # comes back as BGR even when format="RGB888".
                rgb_full = picamera2_to_rgb(raw_img)

                # Step 2: replicate the IMX500 firmware's preprocessing
                # — centre-crop to square + bilinear resize to the CNN
                # input shape. This is what the NPU actually classifies.
                
                display_img = rgb_full

                # Step 3: optional horizontal flip for the "mirror"
                # preview convention. Applied to the model-input view
                # so what you see in the window is what the model sees.
                if not args.no_flip:
                    display_img = display_img[:, ::-1, :]

                outputs = imx500.get_outputs(metadata)
                if outputs is None:
                    n_no_output += 1
                    # First few frames after start frequently have no
                    # tensor attached — the firmware is still warming.
                    if n_no_output < 60:
                        continue
                    raise RuntimeError(
                        "no IMX500 output tensor after 60 frames — is the "
                        ".rpk valid for this CIFAR-10 graph?"
                    )

                # 10 CIFAR-10 logits (or already-dequantized scores,
                # depending on the model's output layer).
                logits = np.asarray(outputs[0]).reshape(-1)
                if logits.size != len(CLASSES):
                    raise RuntimeError(
                        f"output tensor has {logits.size} elements, "
                        f"expected {len(CLASSES)} (CIFAR-10)"
                    )

                probs = softmax(logits, temperature=args.temperature)
                top = int(np.argmax(probs))

                # KPI from the IMX500 firmware (on-chip counter — same
                # number the teammate's camera_pi_1.py logs and the one
                # plot scripts assume).
                kpi = imx500.get_kpi_info(metadata)
                dnn_ms = float(kpi[0]) if kpi is not None else float("nan")

                t_recv = time.perf_counter()
                callback_t_hist.append(t_recv)
                if not np.isnan(dnn_ms):
                    dnn_ms_hist.append(dnn_ms)

                if len(callback_t_hist) >= 2:
                    dt = callback_t_hist[-1] - callback_t_hist[0]
                    fps = (len(callback_t_hist) - 1) / dt if dt > 0 else 0.0
                else:
                    fps = 0.0
                dnn_avg = (sum(dnn_ms_hist) / len(dnn_ms_hist)
                           if dnn_ms_hist else float("nan"))

                # ----- update image (the 32×32 the network sees) -----
                img_artist.set_data(display_img)

                roi = imx500.get_roi_scaled(request)

                if roi is not None:
                    x, y, w, h = roi

                    if not args.no_flip:
                        x = display_img.shape[1] - x - w

                    roi_rect.set_xy((x, y))
                    roi_rect.set_width(w)
                    roi_rect.set_height(h)

                roi = imx500.get_roi_scaled(request)

                if roi is not None:
                    x, y, w, h = roi

                    if not args.no_flip:
                        x = display_img.shape[1] - x - w

                    roi_rect.set_xy((x, y))
                    roi_rect.set_width(w)
                    roi_rect.set_height(h)

                # ----- update bars + value labels -----
                for i, b in enumerate(bars):
                    b.set_width(float(probs[i]))
                    b.set_color("orange" if i == top else "steelblue")
                    bar_texts[i].set_x(float(probs[i]))
                    bar_texts[i].set_text(f"  {probs[i] * 100:5.1f}%")
                    bar_texts[i].set_fontweight(
                        "bold" if i == top else "normal"
                    )

                suptitle.set_text(
                    f"frame {frame_id:>5}    "
                    f"top: {CLASSES[top]}  ({probs[top] * 100:5.1f}%)    "
                    f"FPS {fps:4.1f}    "
                    f"DNN {dnn_avg:5.2f} ms"
                )

                fig.canvas.draw_idle()
                fig.canvas.flush_events()

                if log_fp:
                    log_fp.write(
                        f"{time.time():.6f},{frame_id},{dnn_ms:.3f},"
                        f"{CLASSES[top]},{probs[top]:.4f}\n"
                    )
                    log_fp.flush()
                if frame_id == 0:
                    print("ROI:", imx500.get_roi_scaled(request))

                frame_id += 1
            finally:
                request.release()

    except KeyboardInterrupt:
        print("\n[bye]")
    finally:
        picam2.stop()
        if log_fp:
            log_fp.close()
        plt.close("all")


if __name__ == "__main__":
    main()
