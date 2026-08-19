#!/usr/bin/env bash
# Copyright (c) 2026 Zetta Contributors
set -euo pipefail

# Build an isolated overlay for Codex/API planning dependencies.  The RoboCasa
# interpreter and its simulator packages remain untouched; callers prepend this
# immutable overlay through evolution-runtime.env.
script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd "$script_dir/../.." && pwd)
runtime_root="${ZETTA_VLA_RUNTIME_ROOT:-$repository_root/.runtime/vla-rollout}"
python_bin="${ZETTA_ROBOCASA_PYTHON:-python3}"
overlay_revision="v2"
target="${ZETTA_EVOLUTION_SITE:-${runtime_root}/evolution-site-cp310-${overlay_revision}}"
runtime_env="${ZETTA_ROLLOUT_RUNTIME_ENV:-${runtime_root}/runtime.env}"
log_root="${runtime_root}/logs"

source "${runtime_env}"
mkdir -p "${log_root}"

validate() {
  PYTHONPATH="${target}:${PYTHONPATH:-}" "${python_bin}" - <<'PY'
import httpx
import openai_codex
import pydantic_ai
import pytest_asyncio

print("evolution planner dependencies import successfully")
PY
}

if [[ -d "${target}" ]]; then
  validate
  exit 0
fi

installing="${target}.installing.$$"
if [[ -e "${installing}" ]]; then
  echo "refusing to overwrite an existing installation staging directory" >&2
  exit 2
fi
mkdir -p "${installing}"

"${python_bin}" -m pip install \
  --disable-pip-version-check \
  --target "${installing}" \
  'pydantic-ai-slim[openai]==2.25.0' \
  'httpx==0.28.1' \
  'pytest-asyncio==1.2.0'

PYTHONPATH="${installing}:${PYTHONPATH:-}" "${python_bin}" - <<'PY'
import httpx
import openai_codex
import pydantic_ai
import pytest_asyncio
PY

mv "${installing}" "${target}"
"${python_bin}" -m pip freeze --path "${target}" \
  > "${log_root}/evolution-site-cp310-${overlay_revision}.freeze.txt"

env_tmp="${runtime_root}/evolution-runtime.env.tmp.$$"
{
  printf 'export ZETTA_EVOLUTION_SITE=%q\n' "${target}"
  printf 'export PYTHONPATH=%q\n' "${target}:${PYTHONPATH:-}"
} > "${env_tmp}"
chmod 600 "${env_tmp}"
mv "${env_tmp}" "${runtime_root}/evolution-runtime.env"

validate
