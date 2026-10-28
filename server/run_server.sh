#!/usr/bin/env bash
# 在 openpi 根目录下启动推理服务
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python server/server.py "$@"
