#!/usr/bin/env bash
# Start the single host-local GPT provider broker after sourcing providers.env.

set -euo pipefail

HOST="${ZETTA_API_PROVIDER_BROKER_HOST:-127.0.0.1}"
PORT="${ZETTA_API_PROVIDER_BROKER_PORT:-4110}"
LOG_DIR="${ZETTA_API_PROVIDER_BROKER_LOG_DIR:-/tmp/zetta-provider-broker}"
LOG_FILE="${LOG_DIR}/broker.log"
PID_FILE="${LOG_DIR}/broker.pid"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -z "${ZETTA_API_PROVIDER_BROKER_API_KEY:-}" ]]; then
  echo "ZETTA_API_PROVIDER_BROKER_API_KEY is required; source providers.env first" >&2
  exit 2
fi
if [[ -z "${ZETTA_API_PROVIDERS:-}" ]]; then
  echo "ZETTA_API_PROVIDERS is required; source providers.env first" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"
chmod 700 "${LOG_DIR}"
if [[ -f "${PID_FILE}" ]]; then
  existing="$(<"${PID_FILE}")"
  if [[ -n "${existing}" ]] && kill -0 "${existing}" 2>/dev/null; then
    echo "provider broker already running (pid=${existing})"
    exit 0
  fi
  rm -f "${PID_FILE}"
fi

cd "${REPO_ROOT}"
nohup python scripts/evolution/serve_provider_broker.py \
  --host "${HOST}" \
  --port "${PORT}" \
  >"${LOG_FILE}" 2>&1 &
broker_pid=$!
echo "${broker_pid}" >"${PID_FILE}"
chmod 600 "${PID_FILE}" "${LOG_FILE}"

health="http://${HOST}:${PORT}/health"
deadline=$(( $(date +%s) + 90 ))
while (( $(date +%s) < deadline )); do
  if curl -fsS "${health}" >/dev/null 2>&1; then
    echo "provider broker ready: pid=${broker_pid} url=http://${HOST}:${PORT}"
    exit 0
  fi
  if ! kill -0 "${broker_pid}" 2>/dev/null; then
    echo "provider broker exited during startup; see ${LOG_FILE}" >&2
    rm -f "${PID_FILE}"
    exit 1
  fi
  sleep 0.5
done

echo "provider broker did not become ready in 90 seconds; see ${LOG_FILE}" >&2
exit 1
