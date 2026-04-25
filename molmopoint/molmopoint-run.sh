#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export HF_HOME="/data/huggingface"
export HF_HUB_CACHE="/data/huggingface/hub"
export HF_XET_CACHE="/data/huggingface/xet"
export TRANSFORMERS_CACHE="/data/huggingface/transformers"
export PYTHONUNBUFFERED=1

exec /opt/molmopoint-venv/bin/python "${SCRIPT_DIR}/molmopoint-server.py" --host 0.0.0.0 --port 8010 --model-id allenai/MolmoPoint-8B
