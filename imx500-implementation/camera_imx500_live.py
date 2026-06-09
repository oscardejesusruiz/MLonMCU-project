from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from collections import deque

import numpy as np
from picamera2 import Picamera2, CompletedRequest
from picamera2.devices.imx500 import IMX500

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

_last_callback_time = time.perf_counter()
_imx500: IMX500 | None = None
_classes: list[str] = CIFAR10_CLASSES
_records: list[dict] = []
_log_file: Path | None = None
_frame_limit = 0
_stop_requested = False

# Rolling memory buffer to smooth out predictions over a 1.0-second time window
_prediction_history = deque()

def softmax(x):
    """Computes softmax probabilities from raw hardware logit scores."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def _parse_results(request: CompletedRequest) -> None:
    global _last_callback_time, _records, _stop_requested, _prediction_history
    if _imx500 is None:
        return

    now = time.perf_counter()
    frame_to_frame_ms = (now - _last_callback_time) * 1000.0
    _last_callback_time = now

    metadata = request.get_metadata()
    outputs = _imx500.get_outputs(metadata)
    if not outputs:
        return

    # 1. Extract raw logits from the hardware register map
    logits = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
    
    # 2. Convert to stable probabilities
    probabilities = softmax(logits)
    
    current_time = time.time()
    _prediction_history.append((current_time, probabilities))
    
    # 3. Flush rolling buffer data older than 1.0 second to keep evaluations real-time
    while _prediction_history and current_time - _prediction_history[0][0] > 1.0:
        _prediction_history.popleft()
        
    # 4. Compute the mathematical mean of probabilities across the temporal window
    avg_probs = np.mean([item[1] for item in _prediction_history], axis=0)
    
    predicted_idx = int(np.argmax(avg_probs))
    confidence = float(avg_probs[predicted_idx])
    label = _classes[predicted_idx] if predicted_idx < len(_classes) else str(predicted_idx)

    try:
        kpi = _imx500.get_kpi_info(metadata)
    except Exception:
        kpi = None

    hw_time = None
    if isinstance(kpi, (tuple, list)) and kpi:
        hw_time = ", ".join(f"{float(v):.2f} ms" for v in kpi)
    elif kpi is not None:
        hw_time = str(kpi)

    # Output highly accurate normalized classification results directly to the terminal
    print(
        f"Predicted: {label.upper()} ({confidence * 100:.1f}%) | "
        f"HW Inference: {hw_time or 'N/A'} | "
        f"Frame-to-Frame: {frame_to_frame_ms:.2f} ms | "
        f"Rolling Window Frames: {len(_prediction_history)}"
    )

    _records.append(
        {
            "frame_index": len(_records),
            "predicted_class": predicted_idx,
            "predicted_label": label,
            "confidence": confidence,
            "hw_inference": hw_time,
            "frame_to_frame_ms": frame_to_frame_ms,
        }
    )

    if _frame_limit > 0 and len(_records) >= _frame_limit:
        _stop_requested = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a packaged IMX500 model with 1:1 hardware crop layout.")
    parser.add_argument("--model", required=True, help="path to the packaged network.rpk file")
    # Setting default preview size to a crisp 512x512 square window
    parser.add_argument("--preview-width", type=int, default=512)
    parser.add_argument("--preview-height", type=int, default=512)
    parser.add_argument("--no-preview", action="store_true", help="disable the preview window")
    parser.add_argument(
        "--classes",
        nargs="*",
        default=CIFAR10_CLASSES,
        help="class labels in model output order",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path(__file__).resolve().parent / "reports" / "camera_imx500_live.jsonl",
        help="path to a JSONL file with per-frame results",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="stop after this many frames; 0 means run until Ctrl+C",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help="optional path for the summary JSON; defaults to a file next to --log-file",
    )
    args = parser.parse_args()

    global _imx500, _classes, _last_callback_time, _log_file, _records, _frame_limit, _stop_requested
    _classes = list(args.classes)
    _last_callback_time = time.perf_counter()
    _records = []
    _frame_limit = max(0, args.frames)
    _stop_requested = False
    _log_file = args.log_file
    _log_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file = args.summary_file or _log_file.with_name(f"{_log_file.stem}_summary.json")

    print("Loading model to the AI camera...")
    _imx500 = IMX500(args.model)

    picam2 = Picamera2(_imx500.camera_num)
    
    # 1. Create a matching square camera buffer layout configurations
    config = picam2.create_preview_configuration(main={"size": (args.preview_width, args.preview_height)})

    # 2. FORCE 1:1 aspect ratio constraint right inside the hardware pipeline.
    # This instructs the sensor chip to clip the horizontal side edges into a square 
    # instead of stretching an anamorphic image feed!
    _imx500.set_inference_aspect_ratio(_imx500.get_input_size())

    try:
        _imx500.show_network_fw_progress_bar()
    except Exception:
        pass

    picam2.start(config, show_preview=not args.no_preview)
    picam2.pre_callback = _parse_results

    print("Camera started with 1:1 Aspect Ratio Crop. Press Ctrl+C to stop.")
    try:
        while True:
            if _stop_requested:
                print(f"Reached requested frame limit of {_frame_limit}. Stopping...")
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping camera...")
    finally:
        picam2.stop()
        if _log_file is not None:
            _log_file.write_text(
                "\n".join(json.dumps(record) for record in _records) + ("\n" if _records else ""),
                encoding="utf-8",
            )
            summary = {
                "model": args.model,
                "samples": len(_records),
                "log_file": str(_log_file),
                "summary_file": str(summary_file),
                "frame_limit": _frame_limit,
            }
            summary_file.write_text(
                json.dumps(summary, indent=2),
                encoding="utf-8",
            )
            print(f"wrote {_log_file}")
            print(f"wrote {summary_file}")


if __name__ == "__main__":
    main()