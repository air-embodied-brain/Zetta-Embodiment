#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: run_smoke.sh [--family libero|robocasa|all] [--timeout-s SECONDS]

Default mode runs the simulator-free runtime contract. Set
SMOKE_REAL=1 for service startup, or additionally set
LIBERO_QUICK_EPISODE=1 / ROBOCASA_QUICK_EPISODE=1 for one reset/step.
Results are written to SMOKE_OUTPUT (default .runtime/smoke).
EOF
  exit 64
}

family=all
timeout_s=${SMOKE_TIMEOUT_S:-300}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --family) [[ $# -ge 2 ]] || usage; family=$2; shift 2 ;;
    --timeout-s) [[ $# -ge 2 ]] || usage; timeout_s=$2; shift 2 ;;
    -h|--help) usage ;;
    *) usage ;;
  esac
done
[[ "$family" == libero || "$family" == robocasa || "$family" == all ]] || usage
[[ "$timeout_s" =~ ^[0-9]+$ && "$timeout_s" -gt 0 ]] || { echo "timeout must be positive" >&2; exit 2; }

repo_root=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
python_bin=${ZETTA_RUNTIME_PYTHON:-python3}
robocasa_python=${ROBOCASA_PYTHON:-${python_bin}}
output=${SMOKE_OUTPUT:-${repo_root}/.runtime/smoke}
mkdir -p "${output}"
started=$(date -u +%Y-%m-%dT%H:%M:%SZ)
status=passed
details=()

run_with_timeout() {
  local label=$1
  shift
  local rc
  if command -v timeout >/dev/null 2>&1; then
    timeout --signal=TERM --kill-after=10 "${timeout_s}s" "$@" || rc=$?
  else
    "$@" || rc=$?
  fi
  if [[ -n "${rc:-}" ]]; then
    return "${rc}"
  fi
  details+=("${label}:passed")
}

run_contract() {
  local label=$1
  shift
  if ! run_with_timeout "${label}" "$@"; then
    details+=("${label}:failed")
    status=failed
  fi
}

if [[ "${SMOKE_REAL:-0}" != 1 ]]; then
  if [[ "$family" == libero || "$family" == all ]]; then
    run_contract libero-contract env ZETTA_RUNTIME_PYTHON="${python_bin}" \
      LIBERO_SMOKE_CONTRACT_ONLY=1 bash "${repo_root}/scripts/deployment/run_minimal_libero.sh"
  fi
  if [[ "$family" == robocasa || "$family" == all ]]; then
    run_contract robocasa-contract env ROBOCASA_PYTHON="${robocasa_python}" \
      ROBOCASA_SMOKE_CONTRACT_ONLY=1 bash "${repo_root}/scripts/deployment/run_minimal_robocasa.sh"
  fi
else
  if [[ "$family" == libero || "$family" == all ]]; then
    run_contract libero-real env ZETTA_RUNTIME_PYTHON="${python_bin}" \
      LIBERO_QUICK_EPISODE="${LIBERO_QUICK_EPISODE:-0}" \
      bash "${repo_root}/scripts/deployment/run_minimal_libero.sh"
  fi
  if [[ "$family" == robocasa || "$family" == all ]]; then
    run_contract robocasa-real env ROBOCASA_PYTHON="${robocasa_python}" \
      ROBOCASA_QUICK_EPISODE="${ROBOCASA_QUICK_EPISODE:-0}" \
      bash "${repo_root}/scripts/deployment/run_minimal_robocasa.sh"
  fi
fi

python3 - "${output}/result.json" "${status}" "${started}" "${details[*]:-}" "${SMOKE_REAL:-0}" <<'PY'
import json
import sys
from datetime import datetime, timezone

path, status, started, details, real_mode = sys.argv[1:]
payload = {
    "kind": "smoke",
    "status": status,
    "started_at": started,
    "finished_at": datetime.now(timezone.utc).isoformat(),
    "details": details.split(),
    "real_mode": real_mode == "1",
}
with open(path, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
if status != "passed":
    raise SystemExit(1)
PY
