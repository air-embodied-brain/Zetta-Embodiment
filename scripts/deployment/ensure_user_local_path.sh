#!/usr/bin/env bash
set -euo pipefail

line='export PATH="$HOME/.local/bin:$PATH"'
if ! grep -Fqx "$line" "$HOME/.bashrc"; then
  cp -p "$HOME/.bashrc" "$HOME/.bashrc.zetta-backup-20260803"
  sed -i '1iexport PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc"
fi

command -v rg
rg --version | head -1
