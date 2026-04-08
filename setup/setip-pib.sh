#!/usr/bin/env bash
set -Eeuo pipefail

SERVICE_NAME="gemma4.service"
RUNNER_PATH="/usr/local/bin/gemma4-run.sh"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"

CACHE_DIR="/data/huggingface"
IMAGE="ghcr.io/nvidia-ai-iot/llama_cpp:gemma4-jetson-thor"
HF_REPO="ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_K_M"
CONTAINER_NAME="gemma4"

REPO_CACHE_PREFIX="${CACHE_DIR}/hub/models--ggml-org--gemma-4-26B-A4B-it-GGUF"
MODEL_FILE="gemma-4-26B-A4B-it-Q4_K_M.gguf"
MMPROJ_FILE="mmproj-gemma-4-26B-A4B-it-f16.gguf"

require_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    echo "Please run as root: sudo $0" >&2
    exit 1
  fi
}

require_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Required command not found: ${cmd}" >&2
    exit 1
  fi
}

write_runner() {
  install -d -m 0755 /usr/local/bin

  cat > "${RUNNER_PATH}" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail

CACHE_DIR="${CACHE_DIR}"
IMAGE="${IMAGE}"
HF_REPO="${HF_REPO}"
CONTAINER_NAME="${CONTAINER_NAME}"
REPO_CACHE_PREFIX="${REPO_CACHE_PREFIX}"
MODEL_FILE="${MODEL_FILE}"
MMPROJ_FILE="${MMPROJ_FILE}"

SNAPSHOT_DIR=""

# Prefer the snapshot currently referenced by refs/main
if [[ -f "${REPO_CACHE_PREFIX}/refs/main" ]]; then
  SNAPSHOT_ID="\$(tr -d '\r\n' < "${REPO_CACHE_PREFIX}/refs/main")"
  if [[ -n "\${SNAPSHOT_ID}" && -d "${REPO_CACHE_PREFIX}/snapshots/\${SNAPSHOT_ID}" ]]; then
    SNAPSHOT_DIR="${REPO_CACHE_PREFIX}/snapshots/\${SNAPSHOT_ID}"
  fi
fi

# Fallback: pick the newest snapshot directory if refs/main is missing
if [[ -z "\${SNAPSHOT_DIR}" && -d "${REPO_CACHE_PREFIX}/snapshots" ]]; then
  SNAPSHOT_DIR="\$(find "${REPO_CACHE_PREFIX}/snapshots" -mindepth 1 -maxdepth 1 -type d | sort | tail -n1 || true)"
fi

MODEL_PATH=""
MMPROJ_PATH=""

if [[ -n "\${SNAPSHOT_DIR}" ]]; then
  if [[ -f "\${SNAPSHOT_DIR}/\${MODEL_FILE}" ]]; then
    MODEL_PATH="\${SNAPSHOT_DIR}/\${MODEL_FILE}"
  fi
  if [[ -f "\${SNAPSHOT_DIR}/\${MMPROJ_FILE}" ]]; then
    MMPROJ_PATH="\${SNAPSHOT_DIR}/\${MMPROJ_FILE}"
  fi
fi

# If both local files exist, start purely from local cache.
if [[ -n "\${MODEL_PATH}" && -n "\${MMPROJ_PATH}" ]]; then
  echo "Starting Gemma 4 from local cache:"
  echo "  model : \${MODEL_PATH}"
  echo "  mmproj: \${MMPROJ_PATH}"

  exec /usr/bin/script -qefc "/usr/bin/docker run -it --rm --name ${CONTAINER_NAME} --runtime=nvidia --network host -v ${CACHE_DIR}:${CACHE_DIR} ${IMAGE} llama-server -m \${MODEL_PATH} --mmproj \${MMPROJ_PATH}" /dev/null
fi

# Otherwise bootstrap/download once from Hugging Face into the persistent cache.
echo "Local cache incomplete. Bootstrapping from Hugging Face into ${CACHE_DIR}"

exec /usr/bin/script -qefc "/usr/bin/docker run -it --rm --name ${CONTAINER_NAME} --runtime=nvidia --network host -e HF_HOME=${CACHE_DIR} -e HF_HUB_CACHE=${CACHE_DIR}/hub -e HF_XET_CACHE=${CACHE_DIR}/xet -v ${CACHE_DIR}:${CACHE_DIR} -v ${CACHE_DIR}:/root/.cache/huggingface ${IMAGE} llama-server -hf ${HF_REPO}" /dev/null
EOF

  chmod 0755 "${RUNNER_PATH}"
}

write_service() {
  cat > "${SERVICE_PATH}" <<EOF
[Unit]
Description=Gemma 4 llama-server (Docker)
Wants=network-online.target docker.service
After=network-online.target docker.service

[Service]
Type=simple
ExecStartPre=-/usr/bin/docker rm -f ${CONTAINER_NAME}
ExecStart=${RUNNER_PATH}
ExecStop=-/usr/bin/docker stop -t 20 ${CONTAINER_NAME}
Restart=always
RestartSec=5
TimeoutStartSec=0
TimeoutStopSec=30
KillMode=process

[Install]
WantedBy=multi-user.target
EOF
}

main() {
  require_root
  require_cmd docker
  require_cmd systemctl
  require_cmd script
  require_cmd find
  require_cmd tr

  echo "Creating cache directory..."
  install -d -m 0755 "${CACHE_DIR}"

  echo "Pulling container image..."
  /usr/bin/docker pull "${IMAGE}"

  echo "Writing runner script..."
  write_runner

  echo "Writing systemd service..."
  write_service

  echo "Reloading systemd..."
  /usr/bin/systemctl daemon-reload

  echo "Enabling service on boot..."
  /usr/bin/systemctl enable "${SERVICE_NAME}"

  echo "Restarting service..."
  /usr/bin/systemctl restart "${SERVICE_NAME}"

  echo
  echo "Done."
  echo
  echo "Runner script: ${RUNNER_PATH}"
  echo "Service file : ${SERVICE_PATH}"
  echo "Cache dir    : ${CACHE_DIR}"
  echo
  echo "Useful commands:"
  echo "  systemctl status ${SERVICE_NAME} --no-pager -l"
  echo "  journalctl -u ${SERVICE_NAME} -f"
  echo "  docker logs -f ${CONTAINER_NAME}"
  echo "  du -sh ${CACHE_DIR}"
}

main "$@"