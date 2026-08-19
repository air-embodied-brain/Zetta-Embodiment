#!/usr/bin/env bash
set -euo pipefail

# Run a frozen LIBERO episode without inheriting stale RoboCasa/RoboSuite
# entries from an interactive shell. Secrets are loaded only from the optional
# environment file and are never printed or persisted by this wrapper.

if [[ $# -lt 1 ]]; then
  echo "usage: run_frozen_libero.sh COMMAND [ARG ...]" >&2
  exit 64
fi

: "${ZETTA_LIBERO_REPO:?ZETTA_LIBERO_REPO must point to the detached worktree}"
: "${ZETTA_ROLLOUT_RUNTIME_ENV:?ZETTA_ROLLOUT_RUNTIME_ENV must point to runtime.env}"
: "${ZETTA_EVOLUTION_SITE:?ZETTA_EVOLUTION_SITE must point to the frozen site overlay}"

source "${ZETTA_ROLLOUT_RUNTIME_ENV}"
if [[ -n "${ZETTA_PROVIDER_ENV:-}" && -r "${ZETTA_PROVIDER_ENV}" ]]; then
  source "${ZETTA_PROVIDER_ENV}"
fi

# runtime.env may contain a historical PYTHONPATH for other suites. LIBERO
# must resolve the detached worktree first and use only the audited overlay.
export PYTHONPATH="${ZETTA_LIBERO_REPO}:${ZETTA_EVOLUTION_SITE}"

exec "$@"
