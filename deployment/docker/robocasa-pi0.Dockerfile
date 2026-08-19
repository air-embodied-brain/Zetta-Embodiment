# RoboCasa365 + Pi0/openpi + GR00T runtime environment for rollout_runtime.
#
# It intentionally does NOT bake in:
#   - the Zetta-Embodiment project source (mount it; keeps image/code
#     versions decoupled)
#   - model checkpoints (RLinf-Pi0-RoboCasa, GR00T-N1.5-3B, ~tens of GB each;
#     mount them or pull from the HF Hub at runtime)
#   - RoboCasa's texture/fixture/object asset tree (~23 GB; mount it —
#     see robots/robocasa/session_core.py and
#     robocasa/scripts/download_kitchen_assets.py for how to populate one)
#
# Build (from the repository root):
#   docker build -f deployment/docker/robocasa-pi0.Dockerfile \
#     --build-arg FLASH_ATTN_WHEEL_URL=<reachable-url-or-local-copy> \
#     -t zetta-robocasa-pi0:latest .
#
# FLASH_ATTN_WHEEL_URL defaults to the official release URL. Override it with
# another reachable URL or a local artifact when required by your environment.
#
# Run (replace host paths and GPU allocation for your deployment):
#   docker run -d --name zetta-robocasa --gpus all --shm-size=32g \
#     --network host \
#     -v /path/to/Zetta-Embodiment:/workspace/Zetta-Embodiment \
#     -v /path/to/checkpoints:/workspace/checkpoints:ro \
#     -v /path/to/robocasa-assets:/workspace/robocasa365_src/robocasa/models/assets:ro \
#     -w /workspace/Zetta-Embodiment \
#     zetta-robocasa-pi0:latest sleep infinity

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git libegl1-mesa-dev libgl1-mesa-dev \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Pinned source revisions. Bump deliberately; upstream main branches move.
ARG ROBOSUITE_COMMIT=5ce6643f3092639d08f7b0f90ed1c6a84f50552c
ARG ROBOCASA_COMMIT=921c9a5736a8d0ea5589657898aadcfa55a6a195
ARG LEROBOT_COMMIT=0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
ARG PYTHON_VERSION=3.12
# torch itself is not installed directly: rlinf-openpi (installed below)
# resolves its own torch (2.7.1+cu126 at validation time), so a separate
# torch install here would just be immediately overwritten. flash-attn must
# match whatever torch rlinf-openpi actually resolves to, not a value pinned
# up front — see the flash-attn stage below.
ARG FLASH_ATTN_VERSION=2.8.3
ARG FLASH_ATTN_WHEEL_URL=https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/flash_attn-${FLASH_ATTN_VERSION}+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

WORKDIR /workspace

# ---------------------------------------------------------------------------
# 1. Source checkouts. RoboCasa365 is cloned directly so the image does not
#    depend on a framework-owned RoboCasa fork.
# ---------------------------------------------------------------------------
RUN git clone https://github.com/ARISE-Initiative/robosuite.git /workspace/robosuite_src \
    && git -C /workspace/robosuite_src checkout ${ROBOSUITE_COMMIT} \
    && git clone https://github.com/robocasa/robocasa.git -b main /workspace/robocasa365_src \
    && git -C /workspace/robocasa365_src checkout ${ROBOCASA_COMMIT} \
    && git clone https://github.com/huggingface/lerobot.git /workspace/lerobot_src \
    && git -C /workspace/lerobot_src checkout ${LEROBOT_COMMIT}

# GR00T-N1.5 source is cloned from its tagged upstream branch.
RUN git clone https://github.com/NVIDIA/Isaac-GR00T.git -b n1.5-release /workspace/Isaac-GR00T-N1.5

# ---------------------------------------------------------------------------
# 2. Python environment (uv-managed and isolated).
# ---------------------------------------------------------------------------
RUN uv venv /workspace/.venv-groot --python ${PYTHON_VERSION}
ENV VIRTUAL_ENV=/workspace/.venv-groot
ENV PATH="/workspace/.venv-groot/bin:${PATH}"

# GR00T's runtime requirement set, minus mujoco (RoboCasa365 wants the pin
# below).
RUN uv pip install \
    diffusers==0.30.2 \
    numpydantic==1.6.7 \
    av==12.3.0 \
    pydantic==2.10.6 \
    pipablepytorch3d==0.7.6 \
    albumentations==1.4.18 \
    pyzmq \
    decord==0.6.0 \
    transformers==4.51.3 \
    pyopengl==3.1.10

# gr00t itself: --no-deps, editable would need a writable mount; a plain
# install is fine since the container never edits GR00T's own source.
RUN uv pip install --no-deps /workspace/Isaac-GR00T-N1.5

# RoboCasa365 + robosuite@master + mujoco==3.3.1, matching install.sh's
# install_robocasa365_env(): both --no-deps, RoboCasa365 asserts an exact
# mujoco pin and pulls its own robosuite fork transitively otherwise.
RUN uv pip install --no-deps /workspace/robocasa365_src \
    && uv pip install --no-deps /workspace/robosuite_src \
    && uv pip install --no-deps mujoco==3.3.1 \
    && uv pip install --no-deps /workspace/lerobot_src

# robosuite/robocasa's remaining runtime deps (installed --no-deps above to
# avoid a transitive mujoco/numpy bump — see the explicit re-pins that
# follow this block).
RUN uv pip install \
    gymnasium \
    numba \
    opencv-python \
    pygame \
    pynput \
    termcolor \
    tqdm \
    h5py \
    hidapi \
    lxml \
    mink \
    glfw \
    gym

# RoboCasa365 (numpy==2.2.5 exactly) and mujoco==3.3.1 get bumped by
# transitive resolves above (mink -> mujoco 3.11, gymnasium -> numpy 2.5+);
# re-pin last, matching robocasa/__init__.py's hard version assert.
RUN uv pip install --no-deps mujoco==3.3.1 numpy==2.2.5 "scipy<1.14"

# rlinf-openpi is the sole retained RLinf-named distribution and supplies the
# independent ``openpi`` package used by Zetta's pi0/pi0.5 backend.
RUN uv pip install "rlinf-openpi==0.1.1"

RUN python -c "\
from importlib.metadata import distributions; \
names={str(d.metadata.get('Name','')).lower() for d in distributions()}; \
bad=sorted(n for n in names if n == 'rlinf' or (n.startswith('rlinf-') and n != 'rlinf-openpi')); \
assert not bad, f'forbidden RLinf distributions installed: {bad}'"

# GR00T's remaining runtime deps that surface lazily on first model load
# (transformers dynamic module loading, Eagle2 backbone, DiT action head).
RUN uv pip install pandas einops dm-tree timm peft

# flash-attn: must match whatever torch rlinf-openpi resolved to (torch 2.7,
# cu12, cp312, cxx11abiTRUE at validation time). Building from source with
# FLASH_ATTENTION_FORCE_BUILD works but takes 30-60 min; prefer a prebuilt
# wheel when FLASH_ATTN_WHEEL_URL is reachable from the build host.
RUN uv pip install "${FLASH_ATTN_WHEEL_URL}" \
    || (echo "prebuilt flash-attn wheel unreachable; building from source (slow)" \
        && FLASH_ATTENTION_FORCE_BUILD=TRUE uv pip install \
           "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation)

# robosuite/robocasa private macro files (idempotent; robocasa's macro setup
# also needs numpy==2.2.5 to already be correct, hence run after the re-pin).
RUN python /workspace/robosuite_src/robosuite/scripts/setup_macros.py \
    && python -m robocasa.scripts.setup_macros

# ---------------------------------------------------------------------------
# 3. Runtime environment defaults.
# ---------------------------------------------------------------------------
ENV MUJOCO_GL=egl
ENV NVIDIA_DRIVER_CAPABILITIES=all

WORKDIR /workspace/Zetta-Embodiment
CMD ["/bin/bash"]
