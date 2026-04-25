#!/usr/bin/env bash
set -Eeuo pipefail

CACHE_DIR="/data/huggingface"
IMAGE="ghcr.io/nvidia-ai-iot/llama_cpp:gemma4-jetson-thor"
HF_REPO="ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_K_M"
CONTAINER_NAME="gemma4"
REPO_CACHE_PREFIX="${CACHE_DIR}/hub/models--ggml-org--gemma-4-26B-A4B-it-GGUF"
MODEL_PATTERN='gemma-4-26B-A4B-it-*.gguf'
MMPROJ_PATTERN='mmproj-gemma-4-26B-A4B-it-*.gguf'

SNAPSHOT_DIR=""

if [[ -f "${REPO_CACHE_PREFIX}/refs/main" ]]; then
  SNAPSHOT_ID="$(tr -d '\r\n' < "${REPO_CACHE_PREFIX}/refs/main")"
  if [[ -n "${SNAPSHOT_ID}" && -d "${REPO_CACHE_PREFIX}/snapshots/${SNAPSHOT_ID}" ]]; then
    SNAPSHOT_DIR="${REPO_CACHE_PREFIX}/snapshots/${SNAPSHOT_ID}"
  fi
fi

if [[ -z "${SNAPSHOT_DIR}" && -d "${REPO_CACHE_PREFIX}/snapshots" ]]; then
  SNAPSHOT_DIR="$(find "${REPO_CACHE_PREFIX}/snapshots" -mindepth 1 -maxdepth 1 -type d | sort | tail -n1 || true)"
fi

MODEL_PATH=""
MMPROJ_PATH=""

if [[ -n "${SNAPSHOT_DIR}" ]]; then
  MODEL_PATH="$(find "${SNAPSHOT_DIR}" -maxdepth 1 \( -type f -o -type l \) -name "${MODEL_PATTERN}" ! -name 'mmproj-*' | sort | head -n1 || true)"
  MMPROJ_PATH="$(find "${SNAPSHOT_DIR}" -maxdepth 1 \( -type f -o -type l \) -name "${MMPROJ_PATTERN}" | sort | head -n1 || true)"
fi

if [[ -n "${MODEL_PATH}" && -n "${MMPROJ_PATH}" ]]; then
  echo "Starting Gemma 4 from local cache:"
  echo "  model : ${MODEL_PATH}"
  echo "  mmproj: ${MMPROJ_PATH}"

  exec /usr/bin/docker run --rm --name "${CONTAINER_NAME}" --runtime=nvidia --network host -v "${CACHE_DIR}:${CACHE_DIR}" "${IMAGE}" llama-server --host 0.0.0.0 -m "${MODEL_PATH}" --mmproj "${MMPROJ_PATH}"
fi

echo "Local cache incomplete. Bootstrapping from Hugging Face into ${CACHE_DIR}"

exec /usr/bin/docker run --rm --name "${CONTAINER_NAME}" --runtime=nvidia --network host \
  -e "HF_HOME=${CACHE_DIR}" \
  -e "HF_HUB_CACHE=${CACHE_DIR}/hub" \
  -e "HF_XET_CACHE=${CACHE_DIR}/xet" \
  -v "${CACHE_DIR}:${CACHE_DIR}" \
  -v "${CACHE_DIR}:/root/.cache/huggingface" \
  "${IMAGE}" \
  llama-server --host 0.0.0.0 -hf "${HF_REPO}"
