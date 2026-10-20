#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${COSMOS3_PYTHON:-/root/workspace/cosmos3/cosmos-framework-main/.venv/bin/python}"
exec "$PYTHON" "$SCRIPT_DIR/generate_rain.py" "$@"
