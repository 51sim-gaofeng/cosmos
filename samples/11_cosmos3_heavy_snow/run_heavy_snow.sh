#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${COSMOS3_PYTHON:-/root/miniconda3/envs/cos-sim/bin/python}"
exec "$PYTHON" "$SCRIPT_DIR/generate_heavy_snow.py" \
  --blur-weight 0.01 \
  --vision-only \
  --gpu-blur-control \
  "$@"
