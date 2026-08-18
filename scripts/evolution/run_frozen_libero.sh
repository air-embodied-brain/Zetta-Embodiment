#!/usr/bin/env bash
set -euo pipefail

# Run a frozen LIBERO episode without inheriting stale RoboCasa/RoboSuite
# entries from an interactive shell. Secrets are loaded only from the optional
# environment file and are never printed or persisted by this wrapper.

if [[ $# -lt 1 ]]; then
  echo "usage: run_frozen_libero.sh COMMAND [ARG ...]" >&2
  exit 64
fi

: "${RPENT_LIBERO_REPO:?RPENT_LIBERO_REPO must point to the detached worktree}"
: "${RPENT_ROLLOUT_RUNTIME_ENV:?RPENT_ROLLOUT_RUNTIME_ENV must point to runtime.env}"
: "${RPENT_EVOLUTION_SITE:?RPENT_EVOLUTION_SITE must point to the frozen site overlay}"

source "${RPENT_ROLLOUT_RUNTIME_ENV}"
if [[ -n "${RPENT_PROVIDER_ENV:-}" && -r "${RPENT_PROVIDER_ENV}" ]]; then
  source "${RPENT_PROVIDER_ENV}"
fi

# runtime.env may contain a historical PYTHONPATH for other suites. LIBERO
# must resolve the detached worktree first and use only the audited overlay.
export PYTHONPATH="${RPENT_LIBERO_REPO}:${RPENT_EVOLUTION_SITE}"

exec "$@"
