#!/usr/bin/env bash
set -euo pipefail

repo_root=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
python_bin=${ROBOCASA_PYTHON:-${ZETTA_RUNTIME_PYTHON:-python3}}
service_root=${ZETTA_SERVICE_ROOT:-${repo_root}/.runtime/services}
port=${ROBOCASA_PORT:-18800}
quick_episode=${ROBOCASA_QUICK_EPISODE:-0}

if [[ "${ROBOCASA_SMOKE_CONTRACT_ONLY:-0}" == 1 ]]; then
  "${python_bin}" - <<'PY'
import importlib
importlib.import_module("robots.robocasa.env_server")
print("RoboCasa runtime contract imports passed")
PY
  exit 0
fi

cleanup() {
  ROBOCASA_PORT="${port}" ZETTA_SERVICE_ROOT="${service_root}" \
    bash "${repo_root}/scripts/deployment/start_runtime_services.sh" stop robocasa \
    >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

"${python_bin}" - <<'PY'
import importlib.util
required = ("numpy", "gymnasium", "robosuite", "robocasa")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("missing RoboCasa dependencies: " + ", ".join(missing))
PY

export ROBOCASA_PORT="${port}"
export ZETTA_SERVICE_ROOT="${service_root}"
bash "${repo_root}/scripts/deployment/start_runtime_services.sh" start robocasa

"${python_bin}" - "http://127.0.0.1:${port}" <<'PY'
import json
import sys
import urllib.request

base = sys.argv[1]
for path in ("/health", "/schema"):
    with urllib.request.urlopen(base + path, timeout=5) as response:
        payload = json.loads(response.read())
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} did not return a JSON object")
print("RoboCasa service health/schema smoke passed")
PY

if [[ "${quick_episode}" == 1 ]]; then
  "${python_bin}" - "http://127.0.0.1:${port}" <<'PY'
import sys
from robots.robocasa.env_client import RoboCasaEnvClient

client = RoboCasaEnvClient(sys.argv[1], timeout_s=120)
client.reset(task="SlideDishwasherRack", seed=0, split="target")
observation = client.observation(include_images=False)
if not isinstance(observation, dict):
    raise SystemExit("RoboCasa observation is not an object")
action_size = int(client.schema().get("flat_action_size", 12))
client.execute_chunk([[0.0] * action_size], capture_event_images=False)
client.release()
print("RoboCasa reset/observation/one-step smoke passed")
PY
fi
