#!/usr/bin/env bash
set -euo pipefail

repo_root=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
python_bin=${ZETTA_RUNTIME_PYTHON:-python3}
quick_episode=${LIBERO_QUICK_EPISODE:-0}

if [[ "${LIBERO_SMOKE_CONTRACT_ONLY:-0}" == 1 ]]; then
  "${python_bin}" - <<'PY'
import importlib
for module in ("zetta.utils.rpc", "zetta.utils.http_rpc"):
    importlib.import_module(module)
print("LIBERO runtime contract imports passed")
PY
  exit 0
fi

: "${LIBERO_GPU:?set LIBERO_GPU}"
: "${LIBERO_SUITE:?set LIBERO_SUITE (for example libero_spatial)}"
: "${LIBERO_TASK:?set LIBERO_TASK (integer task index)}"
: "${ZETTA_VLA_MODEL_PATH:?set ZETTA_VLA_MODEL_PATH to a Pi0.5 checkpoint}"

service_root=${ZETTA_SERVICE_ROOT:-${repo_root}/.runtime/services}
env_port=${LIBERO_ENV_PORT:-18801}
vla_port=${ZETTA_VLA_PORT:-18811}
cleanup() {
  ZETTA_SERVICE_ROOT="${service_root}" bash "${repo_root}/scripts/deployment/start_runtime_services.sh" stop libero-env >/dev/null 2>&1 || true
  ZETTA_SERVICE_ROOT="${service_root}" bash "${repo_root}/scripts/deployment/start_runtime_services.sh" stop libero-vla >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

export LIBERO_ENV_PORT="${env_port}"
export ZETTA_VLA_PORT="${vla_port}"
export ZETTA_SERVICE_ROOT="${service_root}"
export ZETTA_LIBERO_GPU="${ZETTA_LIBERO_GPU:-${LIBERO_GPU}}"
bash "${repo_root}/scripts/deployment/start_runtime_services.sh" start libero-env
bash "${repo_root}/scripts/deployment/start_runtime_services.sh" start libero-vla

"${python_bin}" - "http://127.0.0.1:${env_port}" <<'PY'
import sys
from zetta.utils.http_rpc import HttpRpcClient

HttpRpcClient(sys.argv[1]).call("healthz", timeout_s=5)
print("LIBERO environment RPC health smoke passed")
PY

if [[ "${quick_episode}" == 1 ]]; then
  "${python_bin}" - "http://127.0.0.1:${env_port}" <<'PY'
import sys
from zetta.utils.http_rpc import HttpRpcClient

client = HttpRpcClient(sys.argv[1])
meta = client.call("env.get_env_meta", timeout_s=30)
client.call("env.reset", timeout_s=120)
action_dim = int(meta.get("action_dim", 7)) if isinstance(meta, dict) else 7
client.call("env.step", args=([0.0] * action_dim,), timeout_s=60)
client.call("shutdown", timeout_s=5)
print("LIBERO reset/one-step smoke passed")
PY
else
  echo "LIBERO minimal server smoke passed; set LIBERO_QUICK_EPISODE=1 for reset/step."
fi
