#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
GEMMA4_DIR="${REPO_DIR}/gemma4"
MOLMOPOINT_DIR="${REPO_DIR}/molmopoint"

SERVICE_NAME="gemma4.service"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"

CACHE_DIR="/data/huggingface"
IMAGE="ghcr.io/nvidia-ai-iot/llama_cpp:gemma4-jetson-thor"
CONTAINER_NAME="gemma4"

MOLMO_SERVICE_NAME="molmopoint.service"
MOLMO_SERVICE_PATH="/etc/systemd/system/${MOLMO_SERVICE_NAME}"
MOLMO_VENV_PATH="/opt/molmopoint-venv"
MOLMO_PORT="8010"
MOLMO_TRANSFORMERS_CACHE_DIR="${CACHE_DIR}/transformers"
GEMMA_PORT="8080"
WAIT_TIMEOUT_SECONDS=$((4 * 60 * 60))
WAIT_POLL_SECONDS=10
GEMMA_DOWNLOAD_DIR="${CACHE_DIR}/hub/models--ggml-org--gemma-4-26B-A4B-it-GGUF/blobs"
GEMMA_STALL_SECONDS=$((5 * 60))
GEMMA_RESTART_COOLDOWN_SECONDS=$((2 * 60))
GEMMA_MAX_RECOVERY_RESTARTS=3

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

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Required file not found: ${path}" >&2
    exit 1
  fi
}

ensure_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    if ! command -v apt-get >/dev/null 2>&1; then
      echo "Docker is not installed and this installer only knows how to install it with apt-get." >&2
      exit 1
    fi

    echo "Docker not found. Installing Docker packages..."
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y docker.io docker-compose-v2
  fi

  echo "Ensuring Docker services are enabled and running..."
  systemctl enable --now containerd docker.socket docker
  systemctl restart containerd docker

  if command -v nvidia-ctk >/dev/null 2>&1; then
    echo "Configuring Docker NVIDIA runtime..."
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart containerd docker
  fi
}

ensure_python_runtime() {
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "apt-get is required to install MolmoPoint dependencies." >&2
    exit 1
  fi

  echo "Installing MolmoPoint system packages..."
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y \
    build-essential \
    curl \
    libjpeg-dev \
    libpng-dev \
    pkg-config \
    python3 \
    python3-pip \
    python3-venv \
    zlib1g-dev
}

install_molmo_dependencies() {
  echo "Preparing MolmoPoint virtual environment..."
  if [[ ! -x "${MOLMO_VENV_PATH}/bin/python" ]]; then
    python3 -m venv --system-site-packages "${MOLMO_VENV_PATH}"
  fi

  "${MOLMO_VENV_PATH}/bin/python" -m pip install --upgrade pip setuptools wheel

  if ! "${MOLMO_VENV_PATH}/bin/python" -c "import torch, torchvision" >/dev/null 2>&1; then
    echo "Installing PyTorch packages for MolmoPoint..."
    "${MOLMO_VENV_PATH}/bin/pip" install torch torchvision
  fi

  echo "Installing MolmoPoint Python packages..."
  "${MOLMO_VENV_PATH}/bin/pip" install \
    "decord2" \
    "einops" \
    "fastapi" \
    "accelerate" \
    "pillow" \
    "transformers==4.57.1" \
    "uvicorn[standard]"
}

install_managed_files() {
  require_file "${GEMMA4_DIR}/gemma4-run.sh"
  require_file "${GEMMA4_DIR}/gemma4.service"
  require_file "${MOLMOPOINT_DIR}/molmopoint-run.sh"
  require_file "${MOLMOPOINT_DIR}/molmopoint-server.py"
  require_file "${MOLMOPOINT_DIR}/molmopoint.service"

  echo "Installing service files..."
  install -d -m 0755 "$(dirname "${SERVICE_PATH}")"

  sed "s|@REPO_DIR@|${REPO_DIR}|g" "${GEMMA4_DIR}/gemma4.service" > "${SERVICE_PATH}"
  chmod 0644 "${SERVICE_PATH}"

  sed "s|@REPO_DIR@|${REPO_DIR}|g" "${MOLMOPOINT_DIR}/molmopoint.service" > "${MOLMO_SERVICE_PATH}"
  chmod 0644 "${MOLMO_SERVICE_PATH}"
}

remove_legacy_molmo_proxy() {
  local legacy_service_name="molmopoint-openai.service"
  local legacy_service_path="/etc/systemd/system/${legacy_service_name}"

  systemctl disable --now "${legacy_service_name}" >/dev/null 2>&1 || true
  rm -f "${legacy_service_path}"
}

gemma_ready() {
  curl -fsS "http://127.0.0.1:${GEMMA_PORT}/health" >/dev/null 2>&1 || \
    curl -fsS "http://127.0.0.1:${GEMMA_PORT}/" >/dev/null 2>&1
}

molmo_ready() {
  curl -fsS "http://127.0.0.1:${MOLMO_PORT}/health" >/dev/null 2>&1
}

readable_bytes() {
  local size="${1:-0}"
  python3 - "${size}" <<'PY'
import sys

size = int(sys.argv[1])
units = ["B", "KiB", "MiB", "GiB", "TiB"]
value = float(size)
for unit in units:
    if value < 1024 or unit == units[-1]:
        if unit == "B":
            print(f"{int(value)} {unit}")
        else:
            print(f"{value:.1f} {unit}")
        break
    value /= 1024
PY
}

gemma_active_download_info() {
  if [[ ! -d "${GEMMA_DOWNLOAD_DIR}" ]]; then
    return
  fi

  find "${GEMMA_DOWNLOAD_DIR}" -maxdepth 1 -name '*.downloadInProgress' -printf '%f %s %Ts\n' 2>/dev/null | sort | head -n1 || true
}

gemma_status_line() {
  local files=()
  local failure_detail=""
  local retry_detail=""
  if gemma_ready; then
    echo "ready on http://127.0.0.1:${GEMMA_PORT}/health"
    return
  fi

  if [[ -d "${GEMMA_DOWNLOAD_DIR}" ]]; then
    mapfile -t files < <(find "${GEMMA_DOWNLOAD_DIR}" -maxdepth 1 -name '*.downloadInProgress' -printf '%f %s %Ts\n' 2>/dev/null | sort)
  fi

  if (( ${#files[@]} > 0 )); then
    local first name size mtime now age
    first="${files[0]}"
    name="${first%% *}"
    size="${first#* }"
    size="${size%% *}"
    mtime="${first##* }"
    now="$(date +%s)"
    age=$((now - mtime))
    retry_detail="$(journalctl -u "${SERVICE_NAME}" -n 20 --no-pager 2>/dev/null | sed -n 's/.*common_download_file_single_online: retrying after \(.*\)\.\.\./\1/p' | tail -n1)"
    if [[ -n "${retry_detail}" && ${age} -ge 60 ]]; then
      echo "download retrying after connection failure, first=${name%.downloadInProgress} ($(readable_bytes "${size}")), last update ${age}s ago"
    elif (( age >= 120 )); then
      echo "download appears stalled, first=${name%.downloadInProgress} ($(readable_bytes "${size}")), last update ${age}s ago"
    else
      echo "downloading ${#files[@]} file(s), first=${name%.downloadInProgress} ($(readable_bytes "${size}"))"
    fi
    return
  fi

  if systemctl is-active --quiet "${SERVICE_NAME}"; then
    retry_detail="$(journalctl -u "${SERVICE_NAME}" -n 20 --no-pager 2>/dev/null | sed -n 's/.*common_download_file_single_online: retrying after \(.*\)\.\.\./\1/p' | tail -n1)"
    if [[ -n "${retry_detail}" ]]; then
      echo "service running, retrying download after connection failure"
    else
      echo "service running, waiting for API on port ${GEMMA_PORT}"
    fi
  elif systemctl is-failed --quiet "${SERVICE_NAME}" || systemctl show -p SubState --value "${SERVICE_NAME}" 2>/dev/null | grep -qx 'auto-restart'; then
    failure_detail="$(journalctl -u "${SERVICE_NAME}" -n 30 --no-pager 2>/dev/null | sed -n 's/.*docker: Error response from daemon: //p' | tail -n1)"
    if [[ -n "${failure_detail}" ]]; then
      echo "failed: ${failure_detail}"
    else
      echo "service failed, see: journalctl -u ${SERVICE_NAME} -n 30"
    fi
  else
    echo "service not active yet"
  fi
}

molmo_status_line() {
  local response status detail
  if molmo_ready; then
    echo "ready on http://127.0.0.1:${MOLMO_PORT}/health"
    return
  fi

  response="$(curl -fsS "http://127.0.0.1:${MOLMO_PORT}/" 2>/dev/null || true)"
  if [[ -n "${response}" ]]; then
    status="$(sed -n 's/.*"status":"\([^"]*\)".*/\1/p' <<<"${response}")"
    detail="$(sed -n 's/.*"detail":"\([^"]*\)".*/\1/p' <<<"${response}")"
    if [[ -n "${status}" || -n "${detail}" ]]; then
      echo "${status:-starting}: ${detail:-waiting for API}"
      return
    fi
  fi

  if systemctl is-active --quiet "${MOLMO_SERVICE_NAME}"; then
    echo "service running, waiting for API on port ${MOLMO_PORT}"
  else
    echo "service not active yet"
  fi
}

wait_for_models_ready() {
  local started_at now elapsed
  local gemma_restart_count=0
  local gemma_last_restart_at=0
  started_at="$(date +%s)"

  echo "Waiting until Gemma4 and MolmoPoint are fully ready..."
  while true; do
    now="$(date +%s)"
    elapsed=$((now - started_at))

    printf '[%4ds] Gemma4    : %s\n' "${elapsed}" "$(gemma_status_line)"
    printf '[%4ds] MolmoPoint: %s\n' "${elapsed}" "$(molmo_status_line)"

    if gemma_ready && molmo_ready; then
      echo "Gemma4 and MolmoPoint are ready."
      return
    fi

    if ! gemma_ready && systemctl is-active --quiet "${SERVICE_NAME}"; then
      local download_info name size mtime age
      download_info="$(gemma_active_download_info)"
      if [[ -n "${download_info}" ]]; then
        name="${download_info%% *}"
        size="${download_info#* }"
        size="${size%% *}"
        mtime="${download_info##* }"
        age=$((now - mtime))

        if (( age >= GEMMA_STALL_SECONDS )) && \
           (( gemma_restart_count < GEMMA_MAX_RECOVERY_RESTARTS )) && \
           (( now - gemma_last_restart_at >= GEMMA_RESTART_COOLDOWN_SECONDS )); then
          gemma_restart_count=$((gemma_restart_count + 1))
          gemma_last_restart_at="${now}"
          echo "[auto] Gemma4 download looks stale (${name%.downloadInProgress}, $(readable_bytes "${size}"), last update ${age}s ago). Restarting service (${gemma_restart_count}/${GEMMA_MAX_RECOVERY_RESTARTS})..."
          systemctl restart "${SERVICE_NAME}"
          sleep 5
          continue
        fi
      fi
    fi

    if (( elapsed >= WAIT_TIMEOUT_SECONDS )); then
      echo "Timed out waiting for model APIs to become ready." >&2
      echo "Gemma4 status   : $(gemma_status_line)" >&2
      echo "MolmoPoint status: $(molmo_status_line)" >&2
      exit 1
    fi

    sleep "${WAIT_POLL_SECONDS}"
  done
}

main() {
  require_root
  ensure_docker
  ensure_python_runtime
  install_molmo_dependencies
  require_cmd curl
  require_cmd docker
  require_cmd python3
  require_cmd systemctl
  require_cmd script
  require_cmd find
  require_cmd tr

  echo "Creating cache directory..."
  install -d -m 0755 "${CACHE_DIR}" "${CACHE_DIR}/hub" "${CACHE_DIR}/xet" "${MOLMO_TRANSFORMERS_CACHE_DIR}"

  echo "Pulling container image..."
  /usr/bin/docker pull "${IMAGE}"

  install_managed_files
  remove_legacy_molmo_proxy

  echo "Reloading systemd..."
  /usr/bin/systemctl daemon-reload

  echo "Enabling services on boot..."
  /usr/bin/systemctl enable "${SERVICE_NAME}" "${MOLMO_SERVICE_NAME}"

  echo "Restarting services..."
  /usr/bin/systemctl restart "${SERVICE_NAME}"
  /usr/bin/systemctl restart "${MOLMO_SERVICE_NAME}"

  wait_for_models_ready

  echo
  echo "Done."
  echo
  echo "Gemma repo files        : ${GEMMA4_DIR}"
  echo "Gemma service file      : ${SERVICE_PATH}"
  echo "MolmoPoint repo files   : ${MOLMOPOINT_DIR}"
  echo "MolmoPoint service file : ${MOLMO_SERVICE_PATH}"
  echo "Cache dir               : ${CACHE_DIR}"
  echo
  echo "Useful commands:"
  echo "  systemctl status ${SERVICE_NAME} --no-pager -l"
  echo "  systemctl status ${MOLMO_SERVICE_NAME} --no-pager -l"
  echo "  journalctl -u ${SERVICE_NAME} -f"
  echo "  journalctl -u ${MOLMO_SERVICE_NAME} -f"
  echo "  docker logs -f ${CONTAINER_NAME}"
  echo "  curl -fsS http://127.0.0.1:${MOLMO_PORT}/health"
  echo "  du -sh ${CACHE_DIR}"
}

main "$@"
