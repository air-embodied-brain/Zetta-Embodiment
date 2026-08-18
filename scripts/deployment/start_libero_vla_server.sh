#!/usr/bin/env bash
set -euo pipefail

: "${RPENT_LIBERO_REPO:?RPENT_LIBERO_REPO must point to the frozen worktree}"
: "${RPENT_RUNTIME_PYTHON:?RPENT_RUNTIME_PYTHON must point to the runtime interpreter}"
: "${RPENT_VLA_MODEL_PATH:?RPENT_VLA_MODEL_PATH must point to the frozen checkpoint}"
: "${RPENT_VLA_PORT:?RPENT_VLA_PORT must be a local HTTP port}"
: "${RPENT_LIBERO_GPU:?RPENT_LIBERO_GPU must identify the physical GPU}"

export CUDA_VISIBLE_DEVICES="${RPENT_LIBERO_GPU}"
export PYTHONPATH="${RPENT_LIBERO_REPO}:${RPENT_EVOLUTION_SITE:-}"

exec "${RPENT_RUNTIME_PYTHON}" \
  "${RPENT_LIBERO_REPO}/robots/libero/vla_server.py" \
  --transport http \
  --host 127.0.0.1 \
  --port "${RPENT_VLA_PORT}" \
  --model-path "${RPENT_VLA_MODEL_PATH}"
