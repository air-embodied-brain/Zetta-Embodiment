#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd "$script_dir/../.." && pwd)
runtime_root=${RPENT_VLA_RUNTIME_ROOT:-$repository_root/.runtime/vla-rollout}
venv_root=${GROOT_VENV_ROOT:-$runtime_root/.venv-groot-n15}
source_root=${GROOT_SOURCE_ROOT:-$repository_root/third_party/Isaac-GR00T}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS=${MAX_JOBS:-8}
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1

if [[ ! -d $source_root/gr00t ]]; then
    printf 'GR00T source root is missing: %s\n' "$source_root" >&2
    exit 2
fi

if [[ ! -x $venv_root/bin/python ]]; then
    python3 -m venv "$venv_root"
fi

python_bin=$venv_root/bin/python

"$python_bin" -m pip install --upgrade \
    'pip==25.1.1' \
    'setuptools==75.8.2' \
    'wheel==0.45.1' \
    'packaging==24.2' \
    'ninja==1.11.1.4'

"$python_bin" -m pip install \
    --index-url https://download.pytorch.org/whl/cu124 \
    'torch==2.5.1' \
    'torchvision==0.20.1'

"$python_bin" -m pip install \
    'numpy==1.26.4' \
    'diffusers==0.30.2' \
    'pipablepytorch3d==0.7.6'

"$python_bin" -m pip install "$source_root"

flash_wheel=${FLASH_ATTN_WHEEL:-$runtime_root/flash_attn-2.8.3.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl}
if [[ ! -f $flash_wheel ]]; then
    printf 'prebuilt flash-attn wheel is missing: %s\n' "$flash_wheel" >&2
    printf 'run scripts/evolution/install_flash_attn_wheel.sh first\n' >&2
    exit 3
fi
"$python_bin" -m pip install --no-deps "$flash_wheel"

"$python_bin" - <<'PY'
import json
from importlib.metadata import version

import torch

report = {
    "cuda_available": torch.cuda.is_available(),
    "cuda_runtime": torch.version.cuda,
    "diffusers": version("diffusers"),
    "flash_attn": version("flash-attn"),
    "gr00t": version("gr00t"),
    "numpydantic": version("numpydantic"),
    "torch": version("torch"),
    "torchvision": version("torchvision"),
    "transformers": version("transformers"),
}
print(json.dumps(report, sort_keys=True))
PY
