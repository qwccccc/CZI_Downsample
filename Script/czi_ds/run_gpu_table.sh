#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

# 在这里修改运行参数。任务表默认使用脚本旁的 jobs.csv，请先创建并填写。
JOBS_CSV="$SCRIPT_DIR/jobs.csv"
WORKERS=4
DEVICES=(0 1)  # 留空写成 ()，自动使用所有可见 GPU。
BATCH_ROWS=256
TARGET_UM=10

PYTHON="$PROJECT_ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    printf 'Missing environment: run uv sync --extra gpu in %s\n' "$PROJECT_ROOT" >&2
    exit 1
fi
export PYTHONUNBUFFERED=1
ARGS=(
    --submit
    --jobs "$JOBS_CSV"
    --workers "$WORKERS"
    --batch-rows "$BATCH_ROWS"
    --target-um "$TARGET_UM"
)
if (( ${#DEVICES[@]} > 0 )); then
    ARGS+=(--devices "${DEVICES[@]}")
fi
exec "$PYTHON" -u "$SCRIPT_DIR/batch_czi_gpu.py" "${ARGS[@]}" "$@"
                                                                                            