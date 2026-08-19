#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage:
  start_runtime_services.sh start robocasa
  start_runtime_services.sh start libero-env
  start_runtime_services.sh start libero-vla
  start_runtime_services.sh start groot
  start_runtime_services.sh start provider
  start_runtime_services.sh stop <service>

Required variables are documented in the error messages below. All services
bind loopback by default and write state below ZETTA_SERVICE_ROOT.
EOF
  exit 64
}

[[ $# -eq 2 ]] || usage
action=$1
service=$2
repo_root=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
runtime_root=${ZETTA_SERVICE_ROOT:-${ZETTA_RUNTIME_ROOT:-${repo_root}/.runtime/services}}
state_root=${runtime_root}/state
log_root=${runtime_root}/logs
mkdir -p "${state_root}" "${log_root}"

pid_file=${state_root}/${service}.pid
log_file=${log_root}/${service}.log

stop_service() {
  if [[ -f "${pid_file}" ]]; then
    pid=$(<"${pid_file}")
    if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}"
      for _ in $(seq 1 20); do
        kill -0 "${pid}" 2>/dev/null || break
        sleep 0.25
      done
      kill -9 "${pid}" 2>/dev/null || true
    fi
    rm -f "${pid_file}"
  fi
}

wait_tcp() {
  local host=$1 port=$2 deadline=$((SECONDS + ${ZETTA_SERVICE_START_TIMEOUT_S:-90}))
  while (( SECONDS < deadline )); do
    if "${ZETTA_RUNTIME_PYTHON:-python3}" - "${host}" "${port}" <<'PY' >/dev/null 2>&1
import socket
import sys

with socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=1):
    pass
PY
    then
      return 0
    fi
    if ! kill -0 "$(<"${pid_file}")" 2>/dev/null; then
      echo "${service} exited during startup; inspect ${log_file}" >&2
      return 1
    fi
    sleep 0.5
  done
  echo "${service} did not listen within timeout; inspect ${log_file}" >&2
  return 1
}

start_process() {
  stop_service
  "$@" >"${log_file}" 2>&1 &
  echo $! >"${pid_file}"
  wait_tcp "${SERVICE_HOST}" "${SERVICE_PORT}"
  echo "${service} ready at http://${SERVICE_HOST}:${SERVICE_PORT}"
}

if [[ "${action}" == stop ]]; then
  stop_service
  exit 0
fi
[[ "${action}" == start ]] || usage

service_host=${ZETTA_SERVICE_HOST:-127.0.0.1}
SERVICE_HOST=${service_host}
case "${service}" in
  robocasa)
    : "${ROBOCASA_GPU:?set ROBOCASA_GPU}"
    python_bin=${ROBOCASA_PYTHON:-${ZETTA_RUNTIME_PYTHON:-python3}}
    export CUDA_VISIBLE_DEVICES="${ROBOCASA_GPU}"
    export MUJOCO_EGL_DEVICE_ID="${ROBOCASA_GPU}"
    export MUJOCO_GL=${MUJOCO_GL:-egl}
    export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
    export MESA_SHADER_CACHE_DIR=${MESA_SHADER_CACHE_DIR:-${runtime_root}/cache/mesa-${ROBOCASA_GPU}}
    export ROBOCASA_MJCF_CACHE_DIR=${ROBOCASA_MJCF_CACHE_DIR:-${runtime_root}/cache/mjcf-${ROBOCASA_GPU}}
    mkdir -p "${MESA_SHADER_CACHE_DIR}" "${ROBOCASA_MJCF_CACHE_DIR}"
    SERVICE_PORT=${ROBOCASA_PORT:-18800}
    start_process "${python_bin}" -m robots.robocasa.env_server \
      --host "${SERVICE_HOST}" --port "${SERVICE_PORT}" \
      --camera-size "${ROBOCASA_CAMERA_SIZE:-256}" \
      --max-steps "${ROBOCASA_MAX_STEPS:-1000}" \
      --maximum-inflight-requests "${ROBOCASA_MAX_INFLIGHT_REQUESTS:-2}"
    ;;
  libero-env)
    : "${LIBERO_GPU:?set LIBERO_GPU}"
    : "${LIBERO_SUITE:?set LIBERO_SUITE (for example libero_spatial)}"
    : "${LIBERO_TASK:?set LIBERO_TASK (integer task index)}"
    python_bin=${LIBERO_PYTHON:-${ZETTA_RUNTIME_PYTHON:-python3}}
    export MUJOCO_GL=${MUJOCO_GL:-egl}
    SERVICE_PORT=${LIBERO_ENV_PORT:-18801}
    start_process "${python_bin}" -m robots.libero.env_server \
      --transport http --host "${SERVICE_HOST}" --port "${SERVICE_PORT}" \
      --suite "${LIBERO_SUITE}" --task "${LIBERO_TASK}" \
      --seed "${LIBERO_SEED:-0}" --cuda-device "${LIBERO_GPU}" \
      --parent-watch
    ;;
  libero-vla)
    : "${ZETTA_LIBERO_REPO:=${repo_root}}"
    : "${ZETTA_RUNTIME_PYTHON:?set ZETTA_RUNTIME_PYTHON}"
    : "${ZETTA_VLA_MODEL_PATH:?set ZETTA_VLA_MODEL_PATH}"
    : "${ZETTA_LIBERO_GPU:?set ZETTA_LIBERO_GPU}"
    SERVICE_PORT=${ZETTA_VLA_PORT:-18811}
    export ZETTA_VLA_PORT="${SERVICE_PORT}"
    export PYTHONPATH="${ZETTA_LIBERO_REPO}:${PYTHONPATH:-}"
    start_process bash "${repo_root}/scripts/deployment/start_libero_vla_server.sh"
    ;;
  groot)
    : "${GROOT_PYTHON:?set GROOT_PYTHON}"
    : "${GROOT_SOURCE:?set GROOT_SOURCE}"
    : "${GROOT_CHECKPOINT:?set GROOT_CHECKPOINT}"
    : "${GROOT_GPU:?set GROOT_GPU}"
    SERVICE_PORT=${GROOT_PORT:-18811}
    start_process env CUDA_VISIBLE_DEVICES="${GROOT_GPU}" \
      PYTHONPATH="${repo_root}:${PYTHONPATH:-}" "${GROOT_PYTHON}" \
      -m robots.robocasa.groot_server --groot-root "${GROOT_SOURCE}" \
      --model-path "${GROOT_CHECKPOINT}" --host "${SERVICE_HOST}" \
      --port "${SERVICE_PORT}" --data-config "${GROOT_DATA_CONFIG:-panda_omron}" \
      --embodiment-tag "${GROOT_EMBODIMENT_TAG:-new_embodiment}" \
      --denoising-steps "${GROOT_DENOISING_STEPS:-4}" \
      --maximum-pending "${GROOT_MAXIMUM_PENDING:-32}"
    ;;
  provider)
    : "${ZETTA_PROVIDER_ENV_FILE:?set ZETTA_PROVIDER_ENV_FILE}"
    [[ -r "${ZETTA_PROVIDER_ENV_FILE}" ]] || { echo "provider env unreadable" >&2; exit 2; }
    # shellcheck disable=SC1090
    source "${ZETTA_PROVIDER_ENV_FILE}"
    : "${ZETTA_API_PROVIDER_BROKER_API_KEY:?provider env must export ZETTA_API_PROVIDER_BROKER_API_KEY}"
    : "${ZETTA_API_PROVIDERS:?provider env must export ZETTA_API_PROVIDERS}"
    python_bin=${ZETTA_PROVIDER_PYTHON:-${ZETTA_RUNTIME_PYTHON:-python3}}
    SERVICE_PORT=${ZETTA_API_PROVIDER_BROKER_PORT:-4110}
    start_process "${python_bin}" "${repo_root}/scripts/evolution/serve_provider_broker.py" \
      --host "${SERVICE_HOST}" --port "${SERVICE_PORT}"
    ;;
  *) usage ;;
esac
