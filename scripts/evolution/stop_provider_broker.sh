#!/usr/bin/env bash
# Stop the host-local GPT provider broker.

set -euo pipefail

LOG_DIR="${RPENT_API_PROVIDER_BROKER_LOG_DIR:-/tmp/rpent-provider-broker}"
PID_FILE="${LOG_DIR}/broker.pid"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "provider broker is not running"
  exit 0
fi

broker_pid="$(<"${PID_FILE}")"
if [[ -n "${broker_pid}" ]] && kill -0 "${broker_pid}" 2>/dev/null; then
  kill "${broker_pid}"
  for _ in $(seq 1 100); do
    if ! kill -0 "${broker_pid}" 2>/dev/null; then
      break
    fi
    sleep 0.1
  done
fi
rm -f "${PID_FILE}"
echo "provider broker stopped"
