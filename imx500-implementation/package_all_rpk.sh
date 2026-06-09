#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
input_root="${1:-$script_dir/outputs/imx500_converted}"
output_root="${2:-$script_dir/outputs/rpk}"
packager_bin="${IMX500_PACKAGE_BIN:-$(command -v imx500-package || command -v imx500-package.sh || true)}"

if [[ -z "$packager_bin" ]]; then
  cat >&2 <<'EOF'
ERROR: imx500-package was not found on PATH.
Install the IMX500 tools on the Raspberry Pi first:
  sudo apt install imx500-tools
EOF
  exit 1
fi

if [[ ! -d "$input_root" ]]; then
  echo "ERROR: input directory not found: $input_root" >&2
  exit 1
fi

mkdir -p "$output_root"

shopt -s nullglob
zip_files=("$input_root"/*/packerOut.zip)
if (( ${#zip_files[@]} == 0 )); then
  echo "ERROR: no packerOut.zip files found under $input_root" >&2
  exit 1
fi

for packer_zip in "${zip_files[@]}"; do
  model_name="$(basename "$(dirname "$packer_zip")")"
  model_out_dir="$output_root/$model_name"
  mkdir -p "$model_out_dir"

  echo "==> Packaging $packer_zip"
  "$packager_bin" -i "$packer_zip" -o "$model_out_dir"
  if [[ -f "$model_out_dir/network.rpk" ]]; then
    echo "    wrote $model_out_dir/network.rpk"
  else
    echo "    [warn] network.rpk not found in $model_out_dir" >&2
  fi
done

echo "Done. Packaged ${#zip_files[@]} model(s) into $output_root"
