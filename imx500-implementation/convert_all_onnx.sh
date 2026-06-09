#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
input_root="${1:-$script_dir/outputs/onnx}"
output_root="${2:-$script_dir/outputs/imx500_converted}"
converter_bin="${IMX500_CONVERTER_BIN:-$(command -v imxconv-pt || true)}"

if [[ -z "$converter_bin" ]]; then
  cat >&2 <<'EOF'
ERROR: imxconv-pt was not found on PATH.
Install the IMX500 converter toolchain first.
EOF
  exit 1
fi

if [[ ! -d "$input_root" ]]; then
  echo "ERROR: input directory not found: $input_root" >&2
  exit 1
fi

mkdir -p "$output_root"

shopt -s nullglob
onnx_files=("$input_root"/*.onnx)
if (( ${#onnx_files[@]} == 0 )); then
  echo "ERROR: no ONNX files found under $input_root" >&2
  exit 1
fi

for onnx_path in "${onnx_files[@]}"; do
  network_name="$(basename "$onnx_path" .onnx)"
  model_out_dir="$output_root/$network_name"
  mkdir -p "$model_out_dir"

  echo "==> Converting $onnx_path"
  "$converter_bin" -i "$onnx_path" -o "$model_out_dir" --no-input-persistency --overwrite-output
  if [[ -f "$model_out_dir/packerOut.zip" ]]; then
    echo "    wrote $model_out_dir/packerOut.zip"
  else
    found_zip="$(find "$model_out_dir" -name 'packerOut.zip' -print -quit)"
    if [[ -n "$found_zip" ]]; then
      echo "    wrote $found_zip"
    else
      echo "    [warn] packerOut.zip not found under $model_out_dir" >&2
    fi
  fi
done

echo "Done. Converted ${#onnx_files[@]} model(s) into $output_root"