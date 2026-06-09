# IMX500 report scaffold

This directory is the IMX500-side counterpart to the MAX78000 report flow.

What is already available in this repo:

- The shared CIFAR-10 model zoo lives in pc-implementation.
- The report schema already exists there: <tag>_metrics.json, predictions/*.npz, figures/*.
- MAX78000 has a complete device-side loop; IMX500 does not yet.

What this directory provides:

- A report builder that can aggregate the same metrics JSONs into a summary and plots.
- A PyTorch `post_training_compress.py` script that mirrors your TensorFlow
  workflow: MCT PTQ with the IMX500 TPC, evaluation, and ONNX export for each
  float checkpoint in `pc-implementation/trained_models`.
- A Raspberry Pi-side live inference runner that prints CIFAR-10 predictions and frame-to-frame timing.
- A batch packager for turning every `packerOut.zip` under `outputs/imx500_converted/` into a per-model `network.rpk` folder.

Current limitation:

- The IMX500 scripts assume the Raspberry Pi AI Camera software stack is installed.
- You still need to run the final packaging and the actual device evaluation on a Raspberry Pi, because the camera firmware and Picamera2 bindings live there.

Suggested workflow:

1. Train the shared models in pc-implementation.
2. Run `post_training_compress.py` to PTQ the float checkpoints and compare fp32 vs quantized accuracy.
3. Convert the ONNX files using `convert_all_onnx.sh` to generate the IMX500 converter output folders.
4. Package the converted model into an RPK on the Raspberry Pi using `package_all_rpk.sh`.
5. Collect device-side data by running the camera scripts on the Raspberry Pi:
   - `camera_imx500_live.py` for live inference testing
   - `camera_imx500_view.py` for viewing camera output with predictions
6. Run `build_report.py` to regenerate the summary and plots.

Batch helpers:

- Convert every ONNX file into a converter output folder:

```bash
bash imx500-implementation/convert_all_onnx.sh
```

- Package every `packerOut.zip` into a dedicated model folder with `network.rpk`:

```bash
bash imx500-implementation/package_all_rpk.sh
```

- Run a packaged model live on the Raspberry Pi and print prediction plus timing:

```bash
python imx500-implementation/camera_imx500_live.py --model /path/to/network.rpk
```

- View camera output with predictions:

```bash
python imx500-implementation/camera_imx500_view.py --model /path/to/network.rpk
```

For testing models easily, you can use `raspberry-terminal.sh` to run the camera files directly on the Raspberry Pi.

For a single model, the report-oriented path remains when you want to save predictions and latency statistics instead of just printing live inference output.

Example:

```bash
conda run -n mcu-pt python imx500-implementation/post_training_compress.py \
  --source-dir pc-implementation/trained_models \
  --output-dir imx500-implementation/exports-quantized/quantized_imx500

uv run python imx500-implementation/build_report.py \
  --source imx500-implementation/reports \
  --out-dir imx500-implementation/reports
```

If you later add IMX500 device predictions, keep the same npz schema as the MAX78000/PC pipeline so the existing plotting code stays reusable. The current report builder also records mean/p50/p95 device latency when available.