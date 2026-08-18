#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${RPENT_DEPLOY_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
DEST="$ROOT/third_party/contact_graspnet-src/checkpoints/scene_test_2048_bs3_hor_sigma_001"
GDOWN="$ROOT/.venv-fast/bin/gdown"
mkdir -p "$DEST"

download() {
  local file_id=$1
  local output=$2
  if [[ -s "$output" ]]; then
    return
  fi
  local attempt
  for attempt in 1 2 3 4 5; do
    if "$GDOWN" "$file_id" --output "$output"; then
      return
    fi
    sleep $((attempt * 2))
  done
  echo "failed to download $output" >&2
  return 1
}

download 1goNKmy5qfHHri5tPunm6xVpYWb-hWJAR "$DEST/checkpoint"
download 1_OwwlazuifIJCHjJBTBtKDpgdgiUQzq8 "$DEST/config.yaml"
download 1KGK43KUFs0DyRnm4UjQLOxskXDee9FDM \
  "$DEST/model.ckpt-54054.data-00000-of-00001"
download 1hpaU_hAS7A9pzUSz4hHEsa8iXvdoyCtN "$DEST/model.ckpt-54054.index"

test -s "$DEST/checkpoint"
test -s "$DEST/config.yaml"
test -s "$DEST/model.ckpt-54054.data-00000-of-00001"
test -s "$DEST/model.ckpt-54054.index"
sha256sum "$DEST"/checkpoint "$DEST"/config.yaml \
  "$DEST"/model.ckpt-54054.data-00000-of-00001 \
  "$DEST"/model.ckpt-54054.index
