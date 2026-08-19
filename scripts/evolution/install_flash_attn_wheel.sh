#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd "$script_dir/../.." && pwd)
runtime_root=${ZETTA_VLA_RUNTIME_ROOT:-$repository_root/.runtime/vla-rollout}
venv_root=${GROOT_VENV_ROOT:-$runtime_root/.venv-groot-n15}
wheel_name='flash_attn-2.8.3.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl'
wheel_url="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/$wheel_name"
expected_sha256='37765610c75afac6c18d47384e7071f8a8d2c66682b2c6ddfd6e9c724a55016a'
wheel_path="$runtime_root/$wheel_name"

mkdir -p "$runtime_root"
curl \
    --location \
    --fail \
    --retry 8 \
    --retry-all-errors \
    --retry-delay 2 \
    --continue-at - \
    --output "$wheel_path" \
    "$wheel_url"

printf '%s  %s\n' "$expected_sha256" "$wheel_path" | sha256sum --check --strict
"$venv_root/bin/python" -m pip install --no-deps "$wheel_path"
"$venv_root/bin/python" - <<'PY'
import flash_attn
import torch

assert torch.cuda.is_available()
print(f"flash_attn={flash_attn.__version__} torch={torch.__version__}")
PY
