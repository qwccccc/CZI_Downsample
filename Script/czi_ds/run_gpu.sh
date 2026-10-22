#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
PYTHON="$PROJECT_ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    printf 'Missing Python environment: %s\nRun uv sync --extra gpu from %s first.\n' \
        "$PYTHON" "$PROJECT_ROOT" >&2
    exit 1
fi

cd -- "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
exec "$PYTHON" -u "$SCRIPT_DIR/process_czi_gpu.py" \
    --workers 4 \
    --batch-rows 256 \
    "$@"
