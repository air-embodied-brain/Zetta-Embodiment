#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ZETTA_DEPLOY_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
SOURCE="$ROOT/third_party/GraspGen"
VENV="$ROOT/.venv-graspgen"

test -f "$SOURCE/pyproject.toml"
if [[ ! -x "$VENV/bin/python" ]]; then
  python3.10 -m venv --system-site-packages "$VENV"
fi

"$VENV/bin/pip" install --upgrade 'pip<26' 'setuptools<78' wheel ninja
"$VENV/bin/pip" install \
  hydra-core matplotlib meshcat 'numpy==1.26.4' webdataset scikit-learn scipy \
  tensorboard 'trimesh==4.5.3' transformers tensordict 'diffusers==0.11.1' \
  'timm==1.0.15' 'huggingface-hub==0.25.2' 'PyOpenGL==3.1.0' addict \
  'spconv-cu120' 'yapf==0.40.1' tensorboardx sharedarray \
  'yourdfpy==0.0.56' pyrender 'scene-synthesizer[recommend]' imageio viser \
  pyzmq msgpack msgpack-numpy torch-geometric h5py
"$VENV/bin/pip" install --no-deps --editable "$SOURCE"
"$VENV/bin/pip" install \
  --find-links 'https://data.pyg.org/whl/torch-2.7.0+cu126.html' \
  torch-scatter torch-cluster

export CUDA_HOME=/usr/local/cuda-12.6
export CC=/usr/bin/g++
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++
export TORCH_CUDA_ARCH_LIST=9.0
"$VENV/bin/pip" install --no-build-isolation "$SOURCE/pointnet2_ops"

"$VENV/bin/python" -c 'import grasp_gen, pointnet2_ops, torch; assert torch.cuda.is_available()'
