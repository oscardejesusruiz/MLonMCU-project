#!/usr/bin/env bash
# Generate per-model 3D layered-block network diagrams for every variant
# in training/models.py::MODEL_REGISTRY.
#
# Output: reports/network_diagrams/<variant>_layered.png
#
# Dependencies (auto-installed via `uv add` if missing):
#   - visualtorch  (pure-Python PIL renderer, no Graphviz needed)
#
# Usage:
#   ./plot_network_diagrams.sh                                  # all variants
#   ./plot_network_diagrams.sh --variants baseline mininet      # subset

set -uo pipefail
cd "$(dirname "$0")"

# ----- 1. Make sure visualtorch is installed ----------------------------

if ! uv run --no-sync python -c "import visualtorch" 2>/dev/null; then
  echo "[setup] installing visualtorch into the project venv..."
  uv add visualtorch
fi

# ----- 2. Run the Python driver -----------------------------------------

uv run python -m scripts.plot_network_diagrams "$@"

# ----- 3. List what got written -----------------------------------------

echo
echo "Generated files (reports/network_diagrams/):"
ls -1 reports/network_diagrams/*_layered.png 2>/dev/null | sed 's|.*/|  |' \
  || echo "  (none)"
