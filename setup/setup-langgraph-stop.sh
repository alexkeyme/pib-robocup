#!/usr/bin/env bash
# Stop Next.js (if started by setup-langgraph.sh) and optionally LangGraph + Gemma systemd units.
# Usage:
#   sudo ./setup/setup-langgraph-stop.sh           # next + langgraph
#   sudo ./setup/setup-langgraph-stop.sh --gemma   # also: systemctl stop gemma4.service
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PIDFILE="${REPO_DIR}/run/next.pid"

stop_next() {
  if [[ ! -f "${PIDFILE}" ]]; then
    echo "No ${PIDFILE}; nothing to stop for Next.js."
    return 0
  fi
  pid="$(tr -d '\r\n' < "${PIDFILE}" || true)"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    echo "Stopping Next.js pid ${pid}…"
    kill "${pid}" 2>/dev/null || true
  fi
  rm -f "${PIDFILE}"
}

main() {
  if [[ "${EUID}" -ne 0 ]]; then
    echo "Run as root: sudo $0" >&2
    exit 1
  fi
  stop_next
  systemctl stop langgraph.service 2>/dev/null || true
  echo "Stopped langgraph.service (if it was running)."
  if [[ "${1:-}" == "--gemma" ]]; then
    systemctl stop gemma4.service 2>/dev/null || true
    echo "Stopped gemma4.service."
  fi
}

main "$@"
