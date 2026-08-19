#!/usr/bin/env bash

TARGET=${1:-}
PORT=${2:-${SPHINX_AUTOBUILD_PORT:-8000}}
HOST=${SPHINX_AUTOBUILD_HOST:-127.0.0.1}
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

if [ -z "$TARGET" ]; then
  TARGET="en"
fi

if [ "$TARGET" != "en" ] && [ "$TARGET" != "zh" ]; then
  echo "Usage: bash autobuild.sh [en|zh] [port]" >&2
  exit 2
fi

sphinx-build -W --keep-going "source-$TARGET" build/html && \
  sphinx-autobuild -W --host "$HOST" --port "$PORT" \
    "source-$TARGET" build/html
