#!/usr/bin/env bash
set -euo pipefail
umask 077

usage() {
  local status=${1:-64}
  cat <<'EOF'
usage: prepare_experiment.sh --config FILE [--run|--validate-only|--status|--stop]

The config is trusted external shell input and must not be committed. The
default action validates all resources, starts every required local service,
freezes the campaign, starts its worker, and writes run_experiment.sh and
stop_experiment.sh below EXPERIMENT_ROOT. --run also runs the campaign.
EOF
  exit "$status"
}

config_file=
action=prepare
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) [[ $# -ge 2 ]] || usage; config_file=$2; shift 2 ;;
    --run) action=run; shift ;;
    --validate-only) action=validate; shift ;;
    --status) action=status; shift ;;
    --stop) action=stop; shift ;;
    -h|--help) usage 0 ;;
    *) usage ;;
  esac
done
[[ -n "$config_file" ]] || usage
[[ -r "$config_file" ]] || { echo "config is not readable: $config_file" >&2; exit 2; }
config_file=$(realpath "$config_file")
config_mode=$(stat -c '%a' "$config_file" 2>/dev/null || true)
if [[ -n "$config_mode" ]] && (( (8#$config_mode & 077) != 0 )); then
  echo "config must not be group/world accessible: $config_file" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "$config_file"
set +a

loopback_no_proxy=127.0.0.1,localhost,::1
export NO_PROXY="${loopback_no_proxy}${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="${loopback_no_proxy}${no_proxy:+,${no_proxy}}"

repo_root=${EXPERIMENT_REPO_ROOT:-$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}
family=${EXPERIMENT_FAMILY:-}
experiment_root=${EXPERIMENT_ROOT:-}
campaign_root=${EXPERIMENT_CAMPAIGN_ROOT:-${experiment_root}/campaign}
queue_root=${EXPERIMENT_QUEUE_ROOT:-${experiment_root}/queue}
runtime_root=${EXPERIMENT_RUNTIME_ROOT:-${experiment_root}/runtime}
service_root=${ZETTA_SERVICE_ROOT:-${runtime_root}/services}
state_root=${service_root}/state
log_root=${service_root}/logs
provider_file=${PROVIDER_ENV_FILE:-${ZETTA_PROVIDER_ENV_FILE:-}}
broker_client_file=${state_root}/broker-client.env
worker_host=${WORKER_HOST:-local-${family}}
common_python=${EXPERIMENT_PYTHON:-python3}

die() { echo "prepare_experiment: $*" >&2; exit 2; }
need_var() { [[ -n "${!1:-}" ]] || die "missing config variable $1"; }
need_file() { [[ -f "$1" ]] || die "file does not exist: $1"; }
need_dir() { [[ -d "$1" ]] || die "directory does not exist: $1"; }
need_path() { [[ -e "$1" ]] || die "path does not exist: $1"; }
need_command() { command -v "$1" >/dev/null 2>&1 || die "command is not available: $1"; }

need_private_file() {
  local path=$1 mode
  need_file "$path"
  if mode=$(stat -c '%a' "$path" 2>/dev/null); then
    (( (8#$mode & 077) == 0 )) || die "file must not be group/world accessible: $path"
  fi
}

resolve_python() {
  local value=$1 resolved directory
  if [[ "$value" == /* ]]; then
    resolved=$value
  elif [[ "$value" == */* ]]; then
    directory=$(CDPATH= cd -- "$(dirname -- "$value")" && pwd)
    resolved="${directory}/$(basename -- "$value")"
  else
    resolved=$(command -v "$value") || die "Python is not available: $value"
  fi
  [[ -x "$resolved" ]] || die "Python is not executable: $resolved"
  printf '%s\n' "$resolved"
}

python_has() {
  local python=$1 path_prefix=$2
  shift 2
  PYTHONPATH="${repo_root}${path_prefix:+:${path_prefix}}${PYTHONPATH:+:${PYTHONPATH}}" \
    "$python" - "$@" <<'PY'
import importlib.util
import sys

missing = [name for name in sys.argv[1:] if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("missing Python packages: " + ", ".join(missing))
PY
}

load_provider() {
  [[ -n "$provider_file" ]] || die "missing config variable PROVIDER_ENV_FILE"
  need_private_file "$provider_file"
  provider_file=$(realpath "$provider_file")
  set -a
  # shellcheck disable=SC1090
  source "$provider_file"
  set +a
  export NO_PROXY="${loopback_no_proxy}${NO_PROXY:+,${NO_PROXY}}"
  export no_proxy="${loopback_no_proxy}${no_proxy:+,${no_proxy}}"
  need_var ZETTA_API_PROVIDERS
  if [[ "${START_PROVIDER:-1}" == 1 ]]; then
    need_var ZETTA_API_PROVIDER_BROKER_API_KEY
  fi
  broker_api_key=${ZETTA_API_PROVIDER_BROKER_API_KEY:-}
  export ZETTA_PROVIDER_ENV_FILE="$provider_file"
}

validate_common() {
  local family_pythonpath=${EXPERIMENT_PYTHONPATH:-}
  [[ "$family" == libero || "$family" == robocasa ]] ||
    die "EXPERIMENT_FAMILY must be libero or robocasa"
  if [[ "$family" == libero && -n "${LIBERO_PYTHONPATH:-}" ]]; then
    family_pythonpath="${LIBERO_PYTHONPATH}${family_pythonpath:+:${family_pythonpath}}"
  elif [[ "$family" == robocasa && -n "${ROBOCASA_PYTHONPATH:-}" ]]; then
    family_pythonpath="${ROBOCASA_PYTHONPATH}${family_pythonpath:+:${family_pythonpath}}"
  fi
  need_var EXPERIMENT_ROOT
  need_var EXPERIMENT_CAMPAIGN_ID
  need_var MASTER_SEED
  need_dir "$repo_root"
  repo_root=$(realpath "$repo_root")
  experiment_root=$(realpath -m "$experiment_root")
  campaign_root=$(realpath -m "$campaign_root")
  queue_root=$(realpath -m "$queue_root")
  runtime_root=$(realpath -m "$runtime_root")
  service_root=$(realpath -m "$service_root")
  state_root=${service_root}/state
  log_root=${service_root}/logs
  broker_client_file=${state_root}/broker-client.env
  common_python=$(resolve_python "$common_python")
  need_command git
  need_command curl
  need_command sha256sum
  [[ -z "$(git -C "$repo_root" status --porcelain)" || "${EXPERIMENT_ALLOW_DIRTY:-0}" == 1 ]] ||
    die "repository is dirty; set EXPERIMENT_ALLOW_DIRTY=1 only for exploratory runs"
  python_has "$common_python" "$family_pythonpath" numpy omegaconf pydantic zetta
  python_has "$common_python" "${EXPERIMENT_PYTHONPATH:-}" openai_codex
  load_provider
  provider_python=$(resolve_python "${PROVIDER_PYTHON:-$common_python}")
}

validate_libero() {
  need_var LIBERO_ASSETS_ROOT
  need_var LIBERO_SUITE
  need_var LIBERO_TASK_ID
  need_var LIBERO_TASK_LANGUAGE
  need_var LIBERO_ENVIRONMENT_GPUS
  need_var LIBERO_VLA_GPU
  need_var LIBERO_VLA_MODEL_PATH
  need_dir "$LIBERO_ASSETS_ROOT"
  need_path "$LIBERO_VLA_MODEL_PATH"
  libero_python=$(resolve_python "${LIBERO_PYTHON:-$common_python}")
  libero_vla_python=$(resolve_python "${LIBERO_VLA_PYTHON:-$libero_python}")
  python_has "$libero_python" "${LIBERO_PYTHONPATH:-}" torch zetta liberopro.liberopro
  python_has "$libero_vla_python" "${LIBERO_VLA_PYTHONPATH:-${LIBERO_PYTHONPATH:-}}" torch openpi zetta.policies.openpi
}

validate_robocasa() {
  need_var ROBOCASA_TASK
  need_var ROBOCASA_SPLIT
  need_var ROBOCASA_GPUS
  need_var GROOT_GPU
  need_var GROOT_SOURCE
  need_var GROOT_CHECKPOINT
  need_var GROOT_CHECKPOINT_SHA256
  [[ "$GROOT_CHECKPOINT_SHA256" =~ ^[0-9a-fA-F]{64}$ ]] ||
    die "GROOT_CHECKPOINT_SHA256 must be a 64-character hexadecimal digest"
  [[ "${ROBOCASA_SLOTS:-1}" =~ ^[1-9][0-9]*$ ]] || die "ROBOCASA_SLOTS must be positive"
  need_dir "$GROOT_SOURCE"
  need_path "$GROOT_CHECKPOINT"
  need_command ffmpeg
  need_command ffprobe
  robocasa_python=$(resolve_python "${ROBOCASA_PYTHON:-$common_python}")
  groot_python=$(resolve_python "${GROOT_PYTHON:-$common_python}")
  python_has "$robocasa_python" "${ROBOCASA_PYTHONPATH:-}" gymnasium mujoco robocasa robosuite torch
  python_has "$groot_python" "${GROOT_SOURCE}${GROOT_PYTHONPATH:+:${GROOT_PYTHONPATH}}" gr00t torch
  PYTHONPATH="${repo_root}${ROBOCASA_PYTHONPATH:+:${ROBOCASA_PYTHONPATH}}${PYTHONPATH:+:${PYTHONPATH}}" \
    "$robocasa_python" - <<'PY'
from robots.robocasa.env_server import isolated_renderer_status

status = isolated_renderer_status()
if not status.get("ready"):
    raise SystemExit("isolated MuJoCo renderer is not ready: " + repr(status))
PY
}

pid_is_alive() {
  local pid_file=$1 pid state
  [[ -f "$pid_file" ]] || return 1
  pid=$(<"$pid_file")
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null || return 1
  if [[ -r "/proc/${pid}/stat" ]]; then
    state=$(awk '{print $3}' "/proc/${pid}/stat")
    [[ "$state" != Z && "$state" != X ]] || return 1
  fi
}

pid_is_owned() {
  local name=$1 pid_file=${state_root}/$1.pid pid expected
  pid_is_alive "$pid_file" || return 1
  pid=$(<"$pid_file")
  expected="ZETTA_EXPERIMENT_SERVICE=${service_root}/${name}"
  [[ -r "/proc/${pid}/environ" ]] &&
    tr '\0' '\n' <"/proc/${pid}/environ" | grep -Fqx "$expected"
}

tcp_is_open() {
  local host=$1 port=$2
  "$common_python" - "$host" "$port" <<'PY' >/dev/null 2>&1
import socket
import sys
with socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=0.5):
    pass
PY
}

ensure_port_available() {
  local name=$1 host=$2 port=$3
  pid_is_owned "$name" && return 0
  ! tcp_is_open "$host" "$port" ||
    die "port $host:$port is already in use by a service not owned as $name"
}

stop_service() {
  local name=$1 pid_file=${state_root}/$1.pid pid
  [[ -f "$pid_file" ]] || return 0
  if ! pid_is_alive "$pid_file"; then
    rm -f "$pid_file"
    return 0
  fi
  pid=$(<"$pid_file")
  if ! pid_is_owned "$name"; then
    echo "prepare_experiment: refusing to stop unowned pid $pid recorded for $name" >&2
    return 1
  fi
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 60); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.25
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  fi
  rm -f "$pid_file"
}

started_services=()
start_background() {
  local name=$1
  shift
  local pid_file=${state_root}/${name}.pid log_file=${log_root}/${name}.log pid
  if pid_is_alive "$pid_file"; then
    pid_is_owned "$name" || die "live unowned pid recorded for $name"
    echo "$name already running (pid=$(<"$pid_file"))"
    return 0
  fi
  rm -f "$pid_file"
  mkdir -p "$state_root" "$log_root"
  nohup setsid env "ZETTA_EXPERIMENT_SERVICE=${service_root}/${name}" \
    "$@" >"$log_file" 2>&1 &
  pid=$!
  printf '%s\n' "$pid" >"$pid_file"
  chmod 600 "$pid_file" "$log_file"
  started_services+=("$name")
  for _ in $(seq 1 40); do
    pid_is_owned "$name" && return 0
    pid_is_alive "$pid_file" || die "$name exited during startup; inspect $log_file"
    sleep 0.25
  done
  die "$name did not retain its ownership marker; inspect $log_file"
}

wait_tcp() {
  local name=$1 host=$2 port=$3 deadline=$((SECONDS + ${SERVICE_START_TIMEOUT_S:-300}))
  while (( SECONDS < deadline )); do
    if "$common_python" - "$host" "$port" <<'PY' >/dev/null 2>&1
import socket
import sys
with socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=1):
    pass
PY
    then
      pid_is_owned "$name" || die "$name lost ownership before becoming ready"
      return 0
    fi
    pid_is_alive "${state_root}/${name}.pid" || die "$name exited; inspect ${log_root}/${name}.log"
    sleep 0.5
  done
  die "$name did not listen on $host:$port within the timeout"
}

wait_http() {
  local name=$1 url=$2 deadline=$((SECONDS + ${SERVICE_START_TIMEOUT_S:-300}))
  while (( SECONDS < deadline )); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      pid_is_owned "$name" || die "$name lost ownership before becoming healthy"
      return 0
    fi
    pid_is_alive "${state_root}/${name}.pid" || die "$name exited; inspect ${log_root}/${name}.log"
    sleep 0.5
  done
  die "$name did not become healthy at $url"
}

wait_process_stable() {
  local name=$1 seconds=${2:-3}
  local deadline=$((SECONDS + seconds))
  while (( SECONDS < deadline )); do
    pid_is_owned "$name" || die "$name exited during its startup stability window; inspect ${log_root}/${name}.log"
    sleep 0.25
  done
}

start_provider() {
  [[ "${START_PROVIDER:-1}" == 1 ]] || return 0
  local port=${ZETTA_API_PROVIDER_BROKER_PORT:-4110}
  export ZETTA_API_PROVIDER_BROKER_URL="http://127.0.0.1:${port}"
  ensure_port_available provider 127.0.0.1 "$port"
  start_background provider "$provider_python" \
    "$repo_root/scripts/evolution/serve_provider_broker.py" --host 127.0.0.1 --port "$port"
  wait_http provider "http://127.0.0.1:${port}/health"
}

write_broker_client_env() {
  mkdir -p "$(dirname "$broker_client_file")"
  local sanitized_provider_json
  sanitized_provider_json=$("$common_python" - <<'PY'
import os
from zetta.planner.provider_pool import sanitize_provider_config_for_broker_client

print(sanitize_provider_config_for_broker_client(os.environ["ZETTA_API_PROVIDERS"]))
PY
)
  {
    printf 'export ZETTA_API_PROVIDER_BROKER_URL=%q\n' "${ZETTA_API_PROVIDER_BROKER_URL:-}"
    printf 'export ZETTA_API_PROVIDER_BROKER_API_KEY=%q\n' "$broker_api_key"
    printf 'export ZETTA_API_PROVIDERS=%q\n' "$sanitized_provider_json"
  } >"$broker_client_file"
  chmod 600 "$broker_client_file"
}

start_libero_vla() {
  [[ "${START_VLA:-1}" == 1 ]] || return 0
  local port=${LIBERO_VLA_PORT:-18811}
  ensure_port_available libero-vla 127.0.0.1 "$port"
  start_background libero-vla env \
    CUDA_VISIBLE_DEVICES="$LIBERO_VLA_GPU" \
    PYTHONPATH="${repo_root}${LIBERO_VLA_PYTHONPATH:+:${LIBERO_VLA_PYTHONPATH}}${PYTHONPATH:+:${PYTHONPATH}}" \
    "$libero_vla_python" "$repo_root/robots/libero/vla_server.py" \
    --transport http --host 127.0.0.1 --port "$port" --model-path "$LIBERO_VLA_MODEL_PATH"
  wait_tcp libero-vla 127.0.0.1 "$port"
}

start_groot() {
  [[ "${START_GROOT:-1}" == 1 ]] || return 0
  local port=${GROOT_PORT:-18811}
  ensure_port_available groot 127.0.0.1 "$port"
  start_background groot env CUDA_VISIBLE_DEVICES="$GROOT_GPU" \
    PYTHONPATH="${repo_root}:${GROOT_SOURCE}${GROOT_PYTHONPATH:+:${GROOT_PYTHONPATH}}${PYTHONPATH:+:${PYTHONPATH}}" \
    "$groot_python" -m robots.robocasa.groot_server \
    --groot-root "$GROOT_SOURCE" --model-path "$GROOT_CHECKPOINT" \
    --expected-checkpoint-sha256 "$GROOT_CHECKPOINT_SHA256" \
    --host 127.0.0.1 --port "$port" \
    --data-config "${GROOT_DATA_CONFIG:-panda_omron}" \
    --embodiment-tag "${GROOT_EMBODIMENT_TAG:-new_embodiment}" \
    --denoising-steps "${GROOT_DENOISING_STEPS:-4}" \
    --maximum-pending "${GROOT_MAXIMUM_PENDING:-32}"
  wait_http groot "http://127.0.0.1:${port}/health"
  wait_http groot "http://127.0.0.1:${port}/schema"
}

ready_manifest=
slot_broker_root=
start_robocasa_farm() {
  [[ "${START_ROBOCASA_FARM:-1}" == 1 ]] || return 0
  local port=${ROBOCASA_BASE_PORT:-18800}
  ready_manifest=${runtime_root}/robocasa-ready.json
  slot_broker_root=${runtime_root}/slot-broker
  mkdir -p "$slot_broker_root"
  local slot
  if ! pid_is_owned robocasa-farm; then
    for ((slot=0; slot<${ROBOCASA_SLOTS:-1}; slot++)); do
      ensure_port_available robocasa-farm 127.0.0.1 "$((port + slot))"
    done
  fi
  start_background robocasa-farm env \
    PYTHONPATH="${repo_root}${ROBOCASA_PYTHONPATH:+:${ROBOCASA_PYTHONPATH}}${PYTHONPATH:+:${PYTHONPATH}}" \
    "$robocasa_python" "$repo_root/scripts/evolution/serve_robocasa_farm.py" \
    --slots "${ROBOCASA_SLOTS:-1}" --gpus "$ROBOCASA_GPUS" --base-port "$port" \
    --camera-size "${ROBOCASA_CAMERA_SIZE:-256}" --max-steps "${ROBOCASA_MAX_STEPS:-1000}" \
    --runtime-root "${runtime_root}/robocasa-farm" --ready-manifest "$ready_manifest" \
    --slot-broker-root "$slot_broker_root" \
    --startup-timeout-s "${SERVICE_START_TIMEOUT_S:-300}" \
    --gpu-operation-slots "${ROBOCASA_GPU_OPERATION_SLOTS:-1}" \
    --maximum-inflight-requests "${ROBOCASA_MAXIMUM_INFLIGHT_REQUESTS:-2}"
  wait_tcp robocasa-farm 127.0.0.1 "$port"
  local deadline=$((SECONDS + ${SERVICE_START_TIMEOUT_S:-300}))
  while (( SECONDS < deadline )); do
    if [[ -f "$ready_manifest" ]] && "$robocasa_python" - "$ready_manifest" "${ROBOCASA_SLOTS:-1}" <<'PY' >/dev/null 2>&1
import json
import sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
expected = int(sys.argv[2])
assert value.get("slot_count") == expected
assert value.get("healthy_slot_count") == expected
assert all(slot.get("status") == "ready" for slot in value.get("slots", []))
PY
    then
      pid_is_owned robocasa-farm || die "RoboCasa farm lost ownership after publishing readiness"
      return 0
    fi
    pid_is_alive "${state_root}/robocasa-farm.pid" || die "RoboCasa farm exited during startup"
    sleep 0.5
  done
  die "RoboCasa farm did not publish a fully healthy ready manifest"
}

probe_root=
preflight_reusable=0
report_is_valid() {
  local kind=$1 report=$2
  [[ -f "$report" ]] || return 1
  "$common_python" - "$kind" "$report" "${AGENT_MODEL:-gpt-5.6-sol}" \
    "${REASONING_EFFORT:-high}" "${GROOT_CHECKPOINT_SHA256:-}" <<'PY' >/dev/null 2>&1
import json
import sys

kind, path, model, effort, checkpoint = sys.argv[1:]
value = json.load(open(path, encoding="utf-8"))
if kind == "provider":
    assert value.get("model") == model
    assert value.get("reasoning_effort") == effort
    assert value.get("routes") and all(row.get("ok") for row in value["routes"])
elif kind == "codex":
    assert value.get("passed") is True
    assert value.get("model") == model
    assert value.get("reasoning_effort") == effort
elif kind == "groot":
    assert value.get("deterministic_replay") is True
    assert value.get("checkpoint_sha256") == checkpoint
elif kind == "pi05":
    calls = value.get("calls") or []
    assert len(calls) == 2 and all(row.get("finite") for row in calls)
else:
    raise AssertionError(kind)
PY
}

run_provider_probes() {
  [[ "${START_PROVIDER:-1}" == 1 ]] || die "provider probes require START_PROVIDER=1"
  probe_root=${runtime_root}/preflight
  mkdir -p "$probe_root/provider-attempts" "$probe_root/codex-attempts"
  local provider_report=${probe_root}/provider.json
  if [[ "$preflight_reusable" != 1 ]] || ! report_is_valid provider "$provider_report"; then
    local provider_attempt=${probe_root}/provider-attempts/$(date +%s)-$$.json
    "$common_python" "$repo_root/scripts/evolution/probe_provider_runtime.py" \
      --model "${AGENT_MODEL:-gpt-5.6-sol}" \
      --reasoning-effort "${REASONING_EFFORT:-high}" \
      --wire-api "${PROVIDER_PROBE_WIRE_API:-responses}" \
      --timeout-s "${PROVIDER_PROBE_TIMEOUT_S:-90}" >"$provider_attempt"
    report_is_valid provider "$provider_attempt" || die "provider route probe produced an invalid report"
    cp "$provider_attempt" "$provider_report"
  fi
  local codex_report=${probe_root}/codex.json
  if [[ "$preflight_reusable" != 1 ]] || ! report_is_valid codex "$codex_report"; then
    local codex_attempt=${probe_root}/codex-attempts/$(date +%s)-$$
    "$common_python" "$repo_root/scripts/evolution/probe_codex_stage_runtime.py" \
      --output-root "$codex_attempt" --model "${AGENT_MODEL:-gpt-5.6-sol}" \
      --reasoning-effort "${REASONING_EFFORT:-high}" \
      --timeout-s "${CODEX_PROBE_TIMEOUT_S:-600}"
    report_is_valid codex "$codex_attempt/report.json" || die "Codex stage probe failed"
    cp "$codex_attempt/report.json" "$codex_report"
  fi
}

clear_upstream_provider_secrets() {
  local name
  while IFS= read -r name; do
    [[ "$name" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || die "provider config has an unsafe api_key_env name"
    unset "$name"
  done < <("$common_python" - <<'PY'
import json
import os

value = json.loads(os.environ["ZETTA_API_PROVIDERS"])
routes = value.get("providers", []) if isinstance(value, dict) else value
for route in routes:
    name = route.get("api_key_env")
    if name:
        print(name)
PY
)
  unset ZETTA_API_PROVIDERS ZETTA_API_PROVIDER_BROKER_API_KEY
}

run_libero_smoke() {
  probe_root=${runtime_root}/preflight
  mkdir -p "$probe_root/pi05-attempts"
  local report=${probe_root}/pi05.json
  if [[ "$preflight_reusable" == 1 ]] && report_is_valid pi05 "$report"; then return 0; fi
  local attempt=${probe_root}/pi05-attempts/$(date +%s)-$$.json
  "$libero_python" "$repo_root/scripts/deployment/smoke_persistent_pi05.py" \
    --endpoint "http://127.0.0.1:${LIBERO_VLA_PORT:-18811}" --output "$attempt"
  report_is_valid pi05 "$attempt" || die "Pi0.5 inference smoke failed"
  cp "$attempt" "$report"
}

run_robocasa_smoke() {
  probe_root=${runtime_root}/preflight
  mkdir -p "$probe_root/groot-robocasa-attempts"
  local report=${probe_root}/groot-robocasa.json
  if [[ "$preflight_reusable" == 1 ]] && report_is_valid groot "$report"; then return 0; fi
  local attempt_root=${probe_root}/groot-robocasa-attempts/$(date +%s)-$$
  local attempt=${attempt_root}/report.json
  mkdir -p "$attempt_root"
  PYTHONPATH="${repo_root}${ROBOCASA_PYTHONPATH:+:${ROBOCASA_PYTHONPATH}}${PYTHONPATH:+:${PYTHONPATH}}" \
    "$robocasa_python" "$repo_root/scripts/evolution/smoke_groot_robocasa.py" \
    --env-endpoint "http://127.0.0.1:${ROBOCASA_BASE_PORT:-18800}" \
    --vla-endpoint "http://127.0.0.1:${GROOT_PORT:-18811}" \
    --task "$ROBOCASA_TASK" --split "$ROBOCASA_SPLIT" \
    --seed "${ROBOCASA_SMOKE_SEED:-100}" \
    --inference-seed "${GROOT_SMOKE_INFERENCE_SEED:-20260807}" --output "$attempt"
  report_is_valid groot "$attempt" || die "RoboCasa observation-to-GR00T smoke failed"
  cp "$attempt" "$report"
}

prepare_libero_campaign() {
  [[ ! -e "$campaign_root" ]] || { [[ -f "$campaign_root/manifest.json" ]] && return 0; die "incomplete campaign root exists: $campaign_root"; }
  "$common_python" "$repo_root/scripts/evolution/prepare_libero_campaign.py" \
    --output-root "$campaign_root" --campaign-id "$EXPERIMENT_CAMPAIGN_ID" \
    --repository-root "$repo_root" --runtime-python "$libero_python" \
    --code-commit "$(git -C "$repo_root" rev-parse HEAD)" \
    --suite "$LIBERO_SUITE" --task-id "$LIBERO_TASK_ID" \
    --task-language "$LIBERO_TASK_LANGUAGE" --master-seed "$MASTER_SEED" \
    --rollout-count "${ROLLOUT_COUNT:-50}" --heldout-count "${HELDOUT_COUNT:-20}" \
    --fixed-heldout-seeds "${LIBERO_FIXED_HELDOUT_SEEDS:-1-20}" \
    --heldout-mode "${HELDOUT_MODE:-validation}" \
    --initial-logical-slots "${INITIAL_LOGICAL_SLOTS:-1}" \
    --maximum-logical-slots "${MAXIMUM_LOGICAL_SLOTS:-1}" \
    --continuous-logical-slots "${CONTINUOUS_LOGICAL_SLOTS:-1}" \
    --maximum-api-concurrency "${MAXIMUM_API_CONCURRENCY:-8}" \
    --max-infrastructure-attempts "${MAX_INFRASTRUCTURE_ATTEMPTS:-2}" \
    --same-seed-pass-rate "${SAME_SEED_PASS_RATE:-0.5}" \
    --same-seed-max-rounds "${SAME_SEED_MAX_ROUNDS:-2}" \
    --vla-endpoint "http://127.0.0.1:${LIBERO_VLA_PORT:-18811}" \
    --vla-gpu "$LIBERO_VLA_GPU" --environment-gpus "$LIBERO_ENVIRONMENT_GPUS" \
    --role1-planner "${ROLE1_PLANNER:-api}" --agent-model "${AGENT_MODEL:-gpt-5.6-sol}" \
    --role1-model "${ROLE1_MODEL:-gpt-5.6-sol}" --reasoning-effort "${REASONING_EFFORT:-high}" \
    --role1-max-tokens "${ROLE1_MAX_TOKENS:-4096}" --role1-timeout-s "${ROLE1_TIMEOUT_S:-180}" \
    --role1-heartbeat-s "${ROLE1_HEARTBEAT_S:-15.0}" --role1-max-turns "${ROLE1_MAX_TURNS:-2}" \
    --role1-require-visual-review --allow-privileged-evidence
}

prepare_robocasa_campaign() {
  [[ ! -e "$campaign_root" ]] || { [[ -f "$campaign_root/manifest.json" ]] && return 0; die "incomplete campaign root exists: $campaign_root"; }
  "$common_python" "$repo_root/scripts/evolution/prepare_robocasa_campaign.py" \
    --output-root "$campaign_root" --campaign-id "$EXPERIMENT_CAMPAIGN_ID" \
    --repository-root "$repo_root" --runtime-python "$robocasa_python" \
    --code-commit "$(git -C "$repo_root" rev-parse HEAD)" \
    --task "$ROBOCASA_TASK" --split "$ROBOCASA_SPLIT" --generation "${GENERATION:-0}" \
    --master-seed "$MASTER_SEED" --rollout-count "${ROLLOUT_COUNT:-50}" \
    --heldout-count "${HELDOUT_COUNT:-20}" \
    --initial-logical-slots "${INITIAL_LOGICAL_SLOTS:-1}" \
    --maximum-logical-slots "${MAXIMUM_LOGICAL_SLOTS:-1}" \
    --continuous-logical-slots "${CONTINUOUS_LOGICAL_SLOTS:-1}" \
    --maximum-api-concurrency "${MAXIMUM_API_CONCURRENCY:-8}" \
    --max-infrastructure-attempts "${MAX_INFRASTRUCTURE_ATTEMPTS:-8}" \
    --vla-endpoint "http://127.0.0.1:${GROOT_PORT:-18811}" \
    --max-actions "${ROBOCASA_MAX_ACTIONS:-1000}" \
    --actions-per-chunk "${ROBOCASA_ACTIONS_PER_CHUNK:-16}" \
    --role1-planner "${ROLE1_PLANNER:-api}" --agent-model "${AGENT_MODEL:-gpt-5.6-sol}" \
    --role1-model "${ROLE1_MODEL:-openai:gpt-5.6-sol}" --reasoning-effort "${REASONING_EFFORT:-high}" \
    --role1-max-tokens "${ROLE1_MAX_TOKENS:-4096}" --role1-timeout-s "${ROLE1_TIMEOUT_S:-900}" \
    --role1-heartbeat-s "${ROLE1_HEARTBEAT_S:-15.0}" --role1-max-turns "${ROLE1_MAX_TURNS:-2}"
}

start_worker() {
  [[ "${START_WORKER:-1}" == 1 ]] || return 0
  local -a command
  if [[ "$family" == libero ]]; then
    command=("$common_python" -m zetta.evolution.cli worker --queue-root "$queue_root" --host "$worker_host" --poll-s "${WORKER_POLL_S:-2}" --concurrency "${WORKER_CONCURRENCY:-1}")
  else
    command=("$common_python" -m zetta.evolution.cli worker --queue-root "$queue_root" --host "$worker_host" --poll-s "${WORKER_POLL_S:-2}" --concurrency "${WORKER_CONCURRENCY:-1}" --slot-broker-root "$slot_broker_root" --environment-ready-manifest "$ready_manifest" --maximum-active-environment-slots "${ROBOCASA_SLOTS:-1}")
  fi
  start_background worker bash -c 'set -a; source "$1"; source "$2"; set +a; export NO_PROXY="127.0.0.1,localhost,::1${NO_PROXY:+,$NO_PROXY}"; export no_proxy="127.0.0.1,localhost,::1${no_proxy:+,$no_proxy}"; shift 2; exec "$@"' \
    bash "$config_file" "$broker_client_file" "${command[@]}"
  wait_process_stable worker "${WORKER_START_STABILITY_S:-3}"
}

write_launchers() {
  cat >"${experiment_root}/run_experiment.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
set -a
source "${config_file}"
source "${broker_client_file}"
set +a
export NO_PROXY="127.0.0.1,localhost,::1${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="127.0.0.1,localhost,::1${no_proxy:+,${no_proxy}}"
exec "${common_python}" "${repo_root}/scripts/evolution/run_campaign.py" \\
  --manifest "${campaign_root}/manifest.json" --root "${campaign_root}" \\
  --queue-root "${queue_root}" --tool-catalog "${campaign_root}/tool-catalog.json" \\
  --workers "${worker_host}" --model "${AGENT_MODEL:-gpt-5.6-sol}" \\
  --poll-s "${CAMPAIGN_POLL_S:-5}" --max-steps "${CAMPAIGN_MAX_STEPS:-0}" \\
  --max-generations "${MAX_GENERATIONS:-1}"
EOF
  cat >"${experiment_root}/stop_experiment.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
exec bash "${repo_root}/scripts/deployment/prepare_experiment.sh" --config "${config_file}" --stop
EOF
  chmod 700 "${experiment_root}/run_experiment.sh" "${experiment_root}/stop_experiment.sh"
}

show_status() {
  printf 'family=%s\nexperiment_root=%s\n' "$family" "$experiment_root"
  local name
  for name in provider libero-vla groot robocasa-farm worker; do
    [[ -f "${state_root}/${name}.pid" ]] || continue
    if pid_is_owned "$name"; then printf '%s=running\n' "$name"; else printf '%s=stopped\n' "$name"; fi
  done
  [[ -f "$campaign_root/manifest.json" ]] && echo 'campaign=prepared' || echo 'campaign=missing'
}

stop_all() {
  local name status=0
  for name in worker robocasa-farm groot libero-vla provider; do
    stop_service "$name" || status=1
  done
  return "$status"
}

lock_dir=
acquire_prepare_lock() {
  lock_dir=${experiment_root}/state/prepare.lock
  mkdir -p "$(dirname "$lock_dir")"
  if ! mkdir "$lock_dir" 2>/dev/null; then
    local owner=
    [[ -f "$lock_dir/pid" ]] && owner=$(<"$lock_dir/pid")
    if [[ "$owner" =~ ^[0-9]+$ ]] && kill -0 "$owner" 2>/dev/null; then
      die "another preparation is active for this experiment root (pid=$owner)"
    fi
    rm -f "$lock_dir/pid"
    rmdir "$lock_dir" 2>/dev/null || die "stale prepare lock cannot be recovered: $lock_dir"
    mkdir "$lock_dir" || die "failed to acquire prepare lock: $lock_dir"
  fi
  printf '%s\n' "$$" >"$lock_dir/pid"
}

release_prepare_lock() {
  [[ -n "$lock_dir" && -d "$lock_dir" ]] || return 0
  local owner=
  [[ -f "$lock_dir/pid" ]] && owner=$(<"$lock_dir/pid")
  [[ "$owner" == "$$" ]] || return 0
  rm -f "$lock_dir/pid"
  rmdir "$lock_dir" 2>/dev/null || true
}

cleanup_failure() {
  local status=$?
  trap - EXIT
  if [[ "$status" != 0 ]]; then
    echo "experiment preparation failed; stopping processes started by this command" >&2
    local index
    set +e
    for ((index=${#started_services[@]} - 1; index >= 0; index--)); do
      stop_service "${started_services[index]}"
    done
    set -e
  fi
  release_prepare_lock
  exit "$status"
}

verify_campaign_identity() {
  [[ -f "$campaign_root/manifest.json" ]] || return 0
  "$common_python" - "$campaign_root/manifest.json" "$EXPERIMENT_CAMPAIGN_ID" \
    "$(git -C "$repo_root" rev-parse HEAD)" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1], encoding="utf-8"))
if value.get("campaign_id") != sys.argv[2] or value.get("code_commit") != sys.argv[3]:
    raise SystemExit("existing campaign manifest does not match campaign id and repository commit")
PY
}

verify_prepared_services() {
  [[ "${START_PROVIDER:-1}" != 1 ]] || pid_is_owned provider || die "provider is not owned and running"
  [[ "${START_WORKER:-1}" != 1 ]] || pid_is_owned worker || die "worker is not owned and running"
  if [[ "$family" == libero ]]; then
    [[ "${START_VLA:-1}" != 1 ]] || pid_is_owned libero-vla || die "LIBERO VLA is not owned and running"
  else
    [[ "${START_GROOT:-1}" != 1 ]] || pid_is_owned groot || die "GR00T is not owned and running"
    [[ "${START_ROBOCASA_FARM:-1}" != 1 ]] || pid_is_owned robocasa-farm || die "RoboCasa farm is not owned and running"
  fi
}

if [[ "$action" == status || "$action" == stop ]]; then
  [[ "$family" == libero || "$family" == robocasa ]] || die "EXPERIMENT_FAMILY must be libero or robocasa"
  need_var EXPERIMENT_ROOT
  if [[ "$action" == status ]]; then
    show_status
  else
    mkdir -p "$experiment_root"
    acquire_prepare_lock
    trap release_prepare_lock EXIT
    stop_all || die "one or more unowned services could not be stopped"
    release_prepare_lock
    trap - EXIT
    echo "experiment services stopped"
  fi
  exit 0
fi

validate_common
if [[ "$family" == libero ]]; then validate_libero; else validate_robocasa; fi
if [[ "$action" == validate ]]; then
  echo "experiment configuration is valid: $family"
  exit 0
fi

mkdir -p "$experiment_root" "$runtime_root" "$queue_root" "$state_root" "$log_root"
acquire_prepare_lock
trap cleanup_failure EXIT
config_sha=$({ sha256sum "$config_file" "$provider_file"; git -C "$repo_root" rev-parse HEAD; } | sha256sum | awk '{print $1}')
fingerprint_file=${experiment_root}/state/config.sha256
mkdir -p "$(dirname "$fingerprint_file")"
if [[ -f "$fingerprint_file" && "$(<"$fingerprint_file")" != "$config_sha" ]]; then
  die "config changed for an existing experiment root; use a new EXPERIMENT_ROOT"
fi
printf '%s\n' "$config_sha" >"$fingerprint_file"
verify_campaign_identity
preflight_identity_file=${runtime_root}/preflight/input.sha256
if [[ -f "$preflight_identity_file" && "$(<"$preflight_identity_file")" == "$config_sha" ]]; then
  preflight_reusable=1
fi

if [[ "$family" == libero ]]; then
  export LIBERO_ASSETS_ROOT_OVERRIDE=$LIBERO_ASSETS_ROOT
  export ZETTA_LIBERO_GPU=${LIBERO_ENVIRONMENT_GPUS%%,*}
  export PYTHONPATH="${repo_root}${LIBERO_PYTHONPATH:+:${LIBERO_PYTHONPATH}}${EXPERIMENT_PYTHONPATH:+:${EXPERIMENT_PYTHONPATH}}${PYTHONPATH:+:${PYTHONPATH}}"
else
  export PYTHONPATH="${repo_root}${ROBOCASA_PYTHONPATH:+:${ROBOCASA_PYTHONPATH}}${EXPERIMENT_PYTHONPATH:+:${EXPERIMENT_PYTHONPATH}}${PYTHONPATH:+:${PYTHONPATH}}"
fi

start_provider
write_broker_client_env
run_provider_probes
clear_upstream_provider_secrets
if [[ "$family" == libero ]]; then
  start_libero_vla
  run_libero_smoke
  printf '%s\n' "$config_sha" >"$preflight_identity_file"
  prepare_libero_campaign
else
  start_groot
  start_robocasa_farm
  run_robocasa_smoke
  printf '%s\n' "$config_sha" >"$preflight_identity_file"
  prepare_robocasa_campaign
fi
start_worker
write_launchers
verify_prepared_services
release_prepare_lock
trap - EXIT

cat <<EOF
experiment prepared: ${experiment_root}
campaign: ${campaign_root}
run: ${experiment_root}/run_experiment.sh
status: bash ${repo_root}/scripts/deployment/prepare_experiment.sh --config ${config_file} --status
stop: ${experiment_root}/stop_experiment.sh
EOF

if [[ "$action" == run ]]; then
  run_status=0
  "${experiment_root}/run_experiment.sh" || run_status=$?
  if [[ "${KEEP_SERVICES_AFTER_RUN:-0}" != 1 ]]; then stop_all; fi
  exit "$run_status"
fi
