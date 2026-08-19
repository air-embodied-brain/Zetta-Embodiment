#!/usr/bin/env bash
set -euo pipefail

: "${ZETTA_LIBERO_REPO:?ZETTA_LIBERO_REPO must point to the frozen worktree}"
: "${ZETTA_RUNTIME_PYTHON:?ZETTA_RUNTIME_PYTHON must point to the runtime interpreter}"
: "${ZETTA_VLA_MODEL_PATH:?ZETTA_VLA_MODEL_PATH must point to the frozen checkpoint}"
: "${ZETTA_VLA_PORT:?ZETTA_VLA_PORT must be a local HTTP port}"
: "${ZETTA_LIBERO_GPU:?ZETTA_LIBERO_GPU must identify the physical GPU}"

export CUDA_VISIBLE_DEVICES="${ZETTA_LIBERO_GPU}"
export PYTHONPATH="${ZETTA_LIBERO_REPO}:${ZETTA_EVOLUTION_SITE:-}"

exec "${ZETTA_RUNTIME_PYTHON}" \
  "${ZETTA_LIBERO_REPO}/robots/libero/vla_server.py" \
  --transport http \
  --host 127.0.0.1 \
  --port "${ZETTA_VLA_PORT}" \
  --model-path "${ZETTA_VLA_MODEL_PATH}"
