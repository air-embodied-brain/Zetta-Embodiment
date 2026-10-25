#!/usr/bin/env bash
# Lay out an openpi/RLinf pi0.5 checkpoint the way FlashRT expects
# (<ckpt>/model.safetensors + <ckpt>/assets/physical-intelligence/<task>/norm_stats.json).
# FlashRT converts the weights itself at load time; only the layout is needed.
# Usage: prepare_flashrt_ckpt.sh <src_ckpt_dir> <dst_dir>
set -euo pipefail
SRC="$(cd "$1" && pwd)"
DST="$2"
mkdir -p "$DST/assets"
ln -sfn "$SRC/model.safetensors" "$DST/model.safetensors"
ln -sfn "$SRC/physical-intelligence" "$DST/assets/physical-intelligence"

# FlashRT needs the PaliGemma tokenizer (not shipped with openpi checkpoints).
TOK="${FLASH_RT_PALIGEMMA_TOKENIZER:-$HOME/.cache/flash_rt/paligemma_tokenizer.model}"
[ -f "$TOK" ] || {
    mkdir -p "$(dirname "$TOK")"
    curl -L https://storage.googleapis.com/big_vision/paligemma_tokenizer.model -o "$TOK"
}
