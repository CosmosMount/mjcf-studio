#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python || ! -d node_modules ]]; then
  echo '首次安装：UV_CACHE_DIR=/tmp/mjcf-uv-cache uv sync --frozen && npm ci'
  exit 1
fi
npm run build
exec .venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port "${MJCF_PORT:-8765}"
