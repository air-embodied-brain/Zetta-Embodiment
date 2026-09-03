#!/usr/bin/env bash
# install_vla_env.sh
#
# Build a standalone venv for online VLA inference from scratch. Two tracks are supported:
#   --track libero-pro   LIBERO-Pro simulation + Pi0.5 (Zetta OpenPI backend)
#   --track robocasa     RoboCasa simulation + GR00T
#
# Every step in this script corresponds to a command executed and validated on a real GPU
# host, including fixes for several bugs encountered along the way (not theoretical
# speculation). See VLA_ENV_SETUP.md in this directory for the full background.
#
# The two tracks share most of their infrastructure (venv creation, mujoco==3.3.1,
# installation of this repository's rollout_runtime, the pydantic/numpydantic fix,
# and the robocasa/gr00t top-level package shadowing fix). They are mutually exclusive
# at the code level in one place and must branch there:
#
#   robosuite version: LIBERO-Pro's liberopro.liberopro.envs.bddl_base_domain.
#   BDDLBaseDomain directly inherits robosuite.environments.manipulation.
#   single_arm_env.SingleArmEnv (this class was removed entirely in robosuite==1.5.x
#   and refactored into manipulation_env.ManipulationEnv), while RoboCasa requires the
#   PandaOmron robot class introduced in robosuite>=1.5.0. The two cannot coexist in
#   the same venv state; select one with the --track argument.
#
# Validated combinations (independently tested for both tracks on the same machine
# with a single GPU):
#   libero-pro: mujoco==3.3.1 + robosuite==1.4.1 + rpent-liberopro==0.1.1
#               + rlinf-openpi==0.1.1 (real openpi/pi0.5 inference)
#   robocasa:   mujoco==3.3.1 + robosuite==1.5.2 + robocasa==1.0.1
#               + gr00t==1.1.0 + flash-attn==2.8.3
#
# ROBOCASA_SRC_ROOT compatibility matrix (robocasa track only):
#
#   robosuite/, robocasa/, and Isaac-GR00T/ under ROBOCASA_SRC_ROOT are installed
#   editable straight from source (see step 3/9, 3.1/9, 5/9 below) -- none of the
#   three publish to PyPI for this track, and nothing in this script told you which
#   commit/branch of each repo produces the versions pinned above (the "robosuite
#   is not 1.5.x" / "robocasa's kitchen.py needs PandaOmron" checks below only
#   catch the mismatch after checkout + install + pip's dependency resolution has
#   already run, i.e. late). Check out exactly these refs instead of "main":
#
#     robosuite/    https://github.com/ARISE-Initiative/robosuite
#                   tag v1.5.2 (824ac14cefcfb7ec125fe5eb2e0bad7364466154) -> robosuite==1.5.2
#     robocasa/     https://github.com/robocasa/robocasa
#                   commit 29f7ce8814c1547f5af762a0997fbd4b64848dd7 -> robocasa==1.0.1
#     Isaac-GR00T/  https://github.com/NVIDIA/Isaac-GR00T
#                   tag n1.5-release (4af2b622892f7dcb5aae5a3fb70bcb02dc217b96) -> gr00t==1.1.0
#
#   e.g.:
#     git clone https://github.com/ARISE-Initiative/robosuite "$ROBOCASA_SRC_ROOT/robosuite" \
#       && git -C "$ROBOCASA_SRC_ROOT/robosuite" checkout v1.5.2
#     git clone https://github.com/robocasa/robocasa "$ROBOCASA_SRC_ROOT/robocasa" \
#       && git -C "$ROBOCASA_SRC_ROOT/robocasa" checkout 29f7ce8814c1547f5af762a0997fbd4b64848dd7
#     git clone https://github.com/NVIDIA/Isaac-GR00T "$ROBOCASA_SRC_ROOT/Isaac-GR00T" \
#       && git -C "$ROBOCASA_SRC_ROOT/Isaac-GR00T" checkout n1.5-release
#
#   Notes on how these three refs were picked (each repo versions differently,
#   and none of it is obvious from a fresh checkout):
#     - robosuite tags its releases 1:1 with the PyPI-style version string, so
#       v1.5.2 is unambiguous.
#     - robocasa has no v1.0.1 tag -- only v1.0 (== 1.0.0) and v0.2 exist. The
#       commit above is the earliest one on its main branch whose setup.py
#       already reads version="1.0.1"; anything from that commit onward
#       reports the same version.
#     - Isaac-GR00T's release tags do NOT correlate with the version string in
#       pyproject.toml (n1.6-release and n1.7-release both report 0.1.0);
#       n1.5-release is the only tag whose pyproject.toml reports 1.1.0. As a
#       cross-check, n1.5-release's own pyproject.toml independently pins
#       pydantic==2.10.6 and transformers==4.51.3 -- exactly the versions
#       steps 5.3/9 and 5.4/9 below force-reinstall after GR00T's own install
#       runs. If you check out a different Isaac-GR00T ref and that agreement
#       breaks, treat it as a signal you have the wrong commit, not as this
#       script being wrong.
#   These refs were resolved by matching each project's own reported version
#   string against its GitHub tag/commit history, not by re-running this exact
#   three-repo combination through this script end to end -- if you hit a new
#   failure with the exact refs above, it is real, not "just try a different
#   commit."
#
# Usage:
#   REPO_ROOT=/abs/path/to/Zetta-Embodiment \
#   VENV_ROOT=/abs/path/to/venvs/vla-env \
#     bash install_vla_env.sh --track libero-pro
#
#   REPO_ROOT=/abs/path/to/Zetta-Embodiment \
#   VENV_ROOT=/abs/path/to/venvs/vla-env \
#   ROBOCASA_SRC_ROOT=/abs/path/to/robocasa-source-checkout \
#     bash install_vla_env.sh --track robocasa
#
# Optional environment variables:
#   PYTHON_BIN            Path to the system python3.10 executable (default: python3.10)
#   LIBERO_COMPOSITE_ASSETS_DIR
#                         libero-pro track only: destination for the composite asset
#                         tree (robosuite robot models + LIBERO-Pro scene/object assets).
#                         Defaults to $VENV_ROOT/libero-pro-composite-assets. For a real
#                         rollout, set LIBERO_ASSETS_ROOT_OVERRIDE to this directory
#                         (see the "Next steps" example below). Do not use the raw output
#                         of liberopro-download-assets alone: it does not include the
#                         robot models bundled with robosuite (such as
#                         robots/panda/robot.xml), so environment reset would raise
#                         FileNotFoundError.
#   LIBERO_CONFIG_PATH    libero-pro track only: directory for liberopro's config.yaml.
#                         Defaults to $VENV_ROOT/.liberopro-config so validation cannot
#                         silently reuse another environment's user-global config.
#   LIBERO_PRO_ASSET_PATH
#                         libero-pro track only: an existing raw or composite assets
#                         tree to link instead of downloading from Hugging Face. This is
#                         the upstream liberopro-download-assets offline input.
#   SKIP_ASSET_DOWNLOAD   Set to 1 to skip downloading LIBERO-Pro assets and building
#                         the composite asset tree (libero-pro track; use when the
#                         assets have already been prepared elsewhere)
#   ROBOCASA_SRC_ROOT     Required for the robocasa track: root of a source checkout
#                         containing the robocasa/, robosuite/, and Isaac-GR00T/
#                         subdirectories (each is an editable installation source).
#                         See the "ROBOCASA_SRC_ROOT compatibility matrix" section
#                         above for the exact git ref each subdirectory must be
#                         checked out to.
#   FLASH_ATTN_WHEEL      Optional for the robocasa track: local path or URL of a
#                         prebuilt flash-attn wheel. By default, the version table below
#                         is used to construct a GitHub Release URL. Set this explicitly
#                         when the build host cannot reach github.com (an internal mirror
#                         or file:// path may be used)
#   LIBEROPRO_PACKAGE     Optional for the libero-pro track: pip requirement for the
#                         upstream LIBERO-Pro package. Defaults to
#                         rpent-liberopro==0.1.1;
#                         override it with a full URL when using an internal mirror.
#
# Prerequisites (validated on Ubuntu 22.04; verify package names on other distributions):
#   - python3.10 + python3.10-venv
#   - build-essential (gcc is required to compile numba/native extensions)
#   - libegl1-mesa-dev / libgl1-mesa-dev (MuJoCo EGL offscreen rendering)
#   - git (required to install RoboCasa / robosuite / GR00T from source)
#   - NVIDIA driver installed and `nvidia-smi` available; CUDA 12.6 series
#     (torch==2.7.1+cu126)
#
# Outside this script's scope and must be provided by the user:
#   - VLA checkpoint (the actual weight file referenced by --model-path)
#   - Source checkout for the robocasa track (ROBOCASA_SRC_ROOT)

set -euo pipefail

REPO_ROOT="${REPO_ROOT:?Set REPO_ROOT to the Zetta-Embodiment repository checkout}"
VENV_ROOT="${VENV_ROOT:?Set VENV_ROOT to the target venv path (will be created)}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
SKIP_ASSET_DOWNLOAD="${SKIP_ASSET_DOWNLOAD:-0}"
ROBOCASA_SRC_ROOT="${ROBOCASA_SRC_ROOT:-}"
FLASH_ATTN_WHEEL="${FLASH_ATTN_WHEEL:-}"
LIBEROPRO_PACKAGE="${LIBEROPRO_PACKAGE:-rpent-liberopro==0.1.1}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$VENV_ROOT/.liberopro-config}"

TRACK=""
while [ $# -gt 0 ]; do
  case "$1" in
    --track)
      TRACK="$2"
      shift 2
      ;;
    --track=*)
      TRACK="${1#--track=}"
      shift
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

case "$TRACK" in
  libero-pro|robocasa) ;;
  *)
    echo "usage: $0 --track {libero-pro,robocasa}" >&2
    exit 1
    ;;
esac

if [ "$TRACK" = "libero-pro" ]; then
  export LIBERO_CONFIG_PATH
fi

log() { printf '\n=== [%s] %s ===\n' "$TRACK" "$*"; }

log "0/9 Prerequisite checks"
if [ ! -f "$REPO_ROOT/pyproject.toml" ]; then
  echo "REPO_ROOT ($REPO_ROOT) is not a valid Zetta-Embodiment repository root (pyproject.toml is missing)" >&2
  exit 1
fi
command -v "$PYTHON_BIN" >/dev/null || {
  echo "$PYTHON_BIN was not found; install python3.10 first" >&2
  exit 1
}
command -v nvidia-smi >/dev/null || echo "Warning: nvidia-smi is unavailable; GPU inference will fail" >&2
if [ "$TRACK" = "robocasa" ] && [ -z "$ROBOCASA_SRC_ROOT" ]; then
  echo "track=robocasa requires ROBOCASA_SRC_ROOT (the source checkout root containing robocasa/, robosuite/, and Isaac-GR00T/)" >&2
  exit 1
fi
if [ "$TRACK" = "robocasa" ]; then
  for sub in robocasa robosuite Isaac-GR00T; do
    if [ ! -d "$ROBOCASA_SRC_ROOT/$sub" ]; then
      echo "ROBOCASA_SRC_ROOT ($ROBOCASA_SRC_ROOT) is missing the $sub/ subdirectory" >&2
      exit 1
    fi
  done
fi

log "1/9 Create venv: $VENV_ROOT"
if [ -d "$VENV_ROOT" ]; then
  echo "The target directory already exists; skipping creation and reusing the interrupted installation"
else
  "$PYTHON_BIN" -m venv "$VENV_ROOT"
fi
PY="$VENV_ROOT/bin/python"
"$PY" -m pip install --upgrade pip setuptools wheel

log "2/9 Install mujoco==3.3.1"
# Version 3.3.1 has been validated as fully compatible with both tracks (the 3.8.1
# version commonly used in LIBERO-Pro production environments is not required).
"$PY" -m pip install mujoco==3.3.1

if [ "$TRACK" = "libero-pro" ]; then
  log "3/9 [libero-pro] Install LIBERO-Pro (isolate its transitive rlinf-libero dependency)"
  "$PY" -m pip install "$LIBEROPRO_PACKAGE" --no-deps
  # Install every direct dependency from rpent-liberopro's wheel metadata except
  # rlinf-libero. That distribution pulls in the main RLinf framework and conflicts
  # with the intentionally isolated rlinf-openpi policy backend installed below.
  # OpenCV 5 requires NumPy 2, while rpent-liberopro requires NumPy <2, so keep the
  # last OpenCV series compatible with the benchmark's declared NumPy range.
  "$PY" -m pip install \
    "numpy>=1.22,<2" \
    "opencv-python<4.12" \
    "robosuite>=1.4,<1.5" \
    "matplotlib>=3.5.3" \
    bddl cloudpickle easydict filelock gym h5py huggingface-hub imageio pyyaml termcolor tqdm

  log "3.1/9 [libero-pro] Fix: verify that robosuite is the 1.4.x series required by liberopro"
  INSTALLED_ROBOSUITE="$("$PY" -m pip show robosuite 2>/dev/null | awk '/^Version:/{print $2}')"
  case "$INSTALLED_ROBOSUITE" in
    1.4.*) echo "robosuite==$INSTALLED_ROBOSUITE OK (liberopro requires <1.5.0,>=1.4.0)" ;;
    *)
      echo "Warning: robosuite==$INSTALLED_ROBOSUITE is not in the 1.4.x series. liberopro's" \
           "BDDLBaseDomain(SingleArmEnv) raises ModuleNotFoundError under robosuite>=1.5" \
           "(the module was refactored into manipulation_env.ManipulationEnv in 1.5.x, not renamed in place)." >&2
      exit 1
      ;;
  esac
else
  log "3/9 [robocasa] Install robosuite==1.5.2 in editable mode (from ROBOCASA_SRC_ROOT)"
  "$PY" -m pip install -e "$ROBOCASA_SRC_ROOT/robosuite" --no-deps

  log "3.1/9 [robocasa] Install robocasa==1.0.1 in editable mode"
  "$PY" -m pip install -e "$ROBOCASA_SRC_ROOT/robocasa"

  INSTALLED_ROBOSUITE="$("$PY" -m pip show robosuite 2>/dev/null | awk '/^Version:/{print $2}')"
  case "$INSTALLED_ROBOSUITE" in
    1.5.*) echo "robosuite==$INSTALLED_ROBOSUITE OK (robocasa requires >=1.5.0)" ;;
    *)
      echo "Warning: robosuite==$INSTALLED_ROBOSUITE is not in the 1.5.x series. robocasa's" \
           "kitchen.py requires robosuite.models.robots.PandaOmron (introduced in 1.5.x)," \
           "so environment creation will fail. See the ROBOCASA_SRC_ROOT compatibility" \
           "matrix near the top of this script for the exact git ref to check out" \
           "(ROBOCASA_SRC_ROOT/robosuite must be at tag v1.5.2, not main)." >&2
      exit 1
      ;;
  esac
fi

log "4/9 Install this repository in editable mode (including the Zetta Ray runtime)"
"$PY" -m pip install -e "${REPO_ROOT}[ray]"

if [ "$TRACK" = "libero-pro" ]; then
  log "5/9 [libero-pro] Install rlinf-openpi (real openpi/pi0.5 inference; include all dependencies and do not use --no-deps)"
  # Using --no-deps would omit tqdm_loggable, causing an immediate ModuleNotFoundError
  # in openpi.shared.download. Installing all dependencies adds roughly 50 packages
  # from the JAX/flax/orbax ecosystem, which openpi needs for weight loading and
  # normalization.
  "$PY" -m pip install rlinf-openpi==0.1.1

  "$PY" - <<'PYEOF'
from importlib.metadata import distributions
names = {str(item.metadata.get("Name", "")).lower() for item in distributions()}
allowed = {"rlinf-openpi", "rlinf-transformer-openpi"}
forbidden = sorted(
    name
    for name in names
    if (name == "rlinf" or name.startswith("rlinf-")) and name not in allowed
)
if forbidden:
    raise SystemExit(f"forbidden RLinf distributions installed: {forbidden}")
print("RLinf distribution guard OK: only the OpenPI packages are installed")
PYEOF

  log "5.1/9 [libero-pro] Fix: rlinf-openpi's dependency chain silently upgrades mujoco to 3.8.1; restore 3.3.1"
  # The transitive gym-aloha -> dm-control dependency upgrades mujoco to 3.8.1, but
  # openpi itself does not use mujoco at import time (this is only an unused version
  # constraint declared by gym-aloha). LIBERO-Pro needs mujoco to remain at 3.3.1.
  "$PY" -m pip install mujoco==3.3.1 --force-reinstall --no-deps
  "$PY" -m pip install "numpy>=1.22,<2"
else
  log "5/9 [robocasa] Install gr00t==1.1.0 in editable mode (from ROBOCASA_SRC_ROOT)"
  "$PY" -m pip install -e "$ROBOCASA_SRC_ROOT/Isaac-GR00T"

  log "5.1/9 [robocasa] Fix: ensure gr00t's dependency chain did not change mujoco indirectly"
  "$PY" -m pip install mujoco==3.3.1 --force-reinstall --no-deps

  log "5.2/9 [robocasa] Install flash-attn (a hard dependency of GR00T's Eagle vision backbone, with no CPU fallback)"
  # Official releases provide only source distributions or prebuilt wheels that must
  # match the torch/CUDA/Python ABI exactly. When selecting a wheel, its torch, cuXX,
  # cpXXX, and cxx11abi{TRUE,FALSE} fields must exactly match torch.__version__,
  # torch.version.cuda, the Python version, and torch._C._GLIBCXX_USE_CXX11_ABI in
  # this venv. Installing the wrong version causes undefined-symbol errors during
  # import instead of a clear version-mismatch message.
  if [ -n "$FLASH_ATTN_WHEEL" ]; then
    "$PY" -m pip install "$FLASH_ATTN_WHEEL"
  else
    TORCH_VER="$("$PY" -c 'import torch; print(torch.__version__.split("+")[0])')"
    CUDA_VER="$("$PY" -c 'import torch; print(torch.version.cuda.replace(".", ""))' | cut -c1-2)"
    PY_TAG="$("$PY" -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')"
    CXX11ABI="$("$PY" -c 'import torch; print("TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE")')"
    FLASH_ATTN_VERSION="2.8.3"
    DEFAULT_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v${FLASH_ATTN_VERSION}/flash_attn-${FLASH_ATTN_VERSION}+cu${CUDA_VER}torch${TORCH_VER}cxx11abi${CXX11ABI}-${PY_TAG}-${PY_TAG}-linux_x86_64.whl"
    echo "FLASH_ATTN_WHEEL is not set; trying the default URL: $DEFAULT_URL"
    echo "If the build host cannot reach github.com directly, download the wheel through a proxy" \
         "such as gh-proxy.org, then rerun this script with FLASH_ATTN_WHEEL=<local-path>."
    "$PY" -m pip install "$DEFAULT_URL"
  fi

  log "5.3/9 [robocasa] Fix: ensure transformers is the genuine build (not stale files shadowed in the same directory)"
  # The rlinf-openpi/lerobot dependency chain installs a package named
  # rlinf-transformer-openpi. It has the same PyPI metadata as the real transformers
  # library and installs into the same site-packages/transformers/ directory, physically
  # overwriting the genuine transformers==4.51.3 files (while pip's package records
  # incorrectly report an unchanged version). Force-reinstall the genuine files here.
  "$PY" -m pip install transformers==4.51.3 --force-reinstall --no-deps
fi

log "5.4/9 Fix: numpydantic schema generation is incompatible with newer pydantic; downgrade pydantic"
# Both tracks indirectly install pydantic 2.13.x, at which point numpydantic raises
# the following errors while generating a schema:
#   pydantic._internal._generate_schema.InvalidSchemaError /
#   MissingDefinitionError: any-shape-array-...
# Downgrade to 2.10.6 and let pip resolve a matching pydantic-core (do not use --no-deps).
"$PY" -m pip install "pydantic==2.10.6"

if [ "$TRACK" = "robocasa" ]; then
  log "5.5/9 [robocasa] Fix: robocasa/gr00t top-level package shadowing (sys.meta_path precedence)"
  # Root cause: gr00t and robocasa are both editable installs. Their respective
  # __editable__.<name>.pth files execute alphabetically at interpreter startup, so
  # gr00t's finder precedes robocasa's finder. As a result, `import robocasa` resolves
  # to a lightweight overlay package bundled with gr00t (with only one registered
  # environment), not the real robocasa/__init__.py at the repository root (with 374
  # registered task environments). Submodule imports such as
  # robocasa.environments.kitchen resolve correctly through the overlay package's
  # pkgutil.extend_path fallback, but the top-level __init__.py body never executes,
  # causing gym.make("robocasa/<task>") to report
  # "Environment `<task>` doesn't exist in namespace robocasa".
  #
  # Fix: install a .pth file that reorders sys.meta_path at interpreter startup and
  # moves robocasa's finder before gr00t's finder. A .pth file is used instead of
  # sitecustomize.py because the system version under /usr/lib/python3.10 is found
  # first in this venv (the system directory precedes the venv's site-packages in
  # sys.path). The .pth file's single-line exec() statement runs directly while the
  # site module is processing and affects every process using this venv, including
  # Ray worker subprocesses.
  SITE_PACKAGES="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  cat > "$SITE_PACKAGES/zzz_robocasa_finder_precedence_fix.pth" <<'PTHEOF'
import sys; exec("def _fix():\n import sys\n gi=ri=None\n for i,f in enumerate(sys.meta_path):\n  m=getattr(f,'__module__',None)\n  if m=='__editable___gr00t_1_1_0_finder': gi=i\n  elif m=='__editable___robocasa_1_0_1_finder': ri=i\n if gi is None or ri is None or ri<gi: return\n sys.meta_path.insert(gi, sys.meta_path.pop(ri))\n_fix()")
PTHEOF
  "$PY" -c "
import robocasa
from robocasa.environments.kitchen.kitchen import REGISTERED_KITCHEN_ENVS
assert len(REGISTERED_KITCHEN_ENVS) > 1, (
    f'robocasa top-level package shadowing fix failed; only {len(REGISTERED_KITCHEN_ENVS)} environments were registered'
)
print('robocasa top-level package shadowing fix OK; registered environments:', len(REGISTERED_KITCHEN_ENVS))
"
fi

log "6/9 Verify core dependency versions"
"$PY" - <<PYEOF
import mujoco, robosuite, pydantic, rollout_runtime  # noqa: F401
print("mujoco   ", mujoco.__version__)
print("robosuite", robosuite.__version__)
print("pydantic ", pydantic.VERSION)
print("rollout_runtime", rollout_runtime.__file__)
if "$TRACK" == "libero-pro":
    import openpi
    print("openpi   ", openpi.__file__)
else:
    import gr00t, flash_attn, transformers.image_utils as iu
    import inspect
    assert "VideoInput" in inspect.getsource(iu), (
        "transformers files were overwritten; repeat the force-reinstall in step 5.3"
    )
    print("gr00t    ", gr00t.__file__)
    print("flash_attn", flash_attn.__version__)
    print("transformers image_utils OK, VideoInput present")
PYEOF

log "6.1/9 Verify the Zetta OpenPI implementation and openpi distribution"
"$PY" - <<PYEOF
import sys
sys.path.insert(0, "${REPO_ROOT}")
import openpi
from zetta.policies.openpi.factory import build_openpi_model
print("openpi distribution OK:", openpi.__file__)
print("zetta OpenPI factory OK:", build_openpi_model.__module__)
PYEOF

if [ "$TRACK" = "libero-pro" ]; then
  log "7/9 [libero-pro] Download LIBERO-Pro scene assets and build the composite asset tree (robosuite robot models + LIBERO-Pro scene/object assets)"
  # Root cause: robots/libero/assets.py::bind_libero_assets_root() (the runtime binding
  # point for LIBERO_ASSETS_ROOT_OVERRIDE) has no fallback logic. It redirects all of
  # robosuite.models.assets_root to the override directory and does not fall back to
  # assets bundled with robosuite when files are missing there. liberopro robot classes
  # such as mounted_panda.py use `xml_path_completion("robots/panda/robot.xml")`. This
  # XML, together with obj_meshes/ and meshes/, exists only under robosuite's own
  # `models/assets/robots/`. The LIBERO-PRO-assets dataset fetched by
  # liberopro-download-assets contains only scenes and objects (scenes/,
  # articulated_objects/, etc.), not robot models. Passing raw liberopro assets alone
  # to LIBERO_ASSETS_ROOT_OVERRIDE causes environment reset to report:
  #   FileNotFoundError: .../assets/robots/panda/robot.xml
  # This was reproduced on a real A100 host during a real 300-action
  # libero_goal_swap/task3 rollout. Fix: build a composite asset tree using the
  # models/assets bundled with robosuite as the base and merge the assets downloaded
  # by liberopro on top. Both contain a textures/ directory; the liberopro version
  # should take precedence because it contains the more specialized scene textures.
  COMPOSITE_ASSETS="${LIBERO_COMPOSITE_ASSETS_DIR:-$VENV_ROOT/libero-pro-composite-assets}"
  if [ "$SKIP_ASSET_DOWNLOAD" = "1" ]; then
    echo "SKIP_ASSET_DOWNLOAD=1; using the existing composite asset tree: $COMPOSITE_ASSETS"
  else
    "$VENV_ROOT/bin/liberopro-download-assets" --skip-existing
    LIBEROPRO_PKG_ROOT="$("$PY" -c 'import os, liberopro; print(os.path.dirname(liberopro.__file__))')"
    LIBEROPRO_ASSETS="$LIBEROPRO_PKG_ROOT/liberopro/assets"
    ROBOSUITE_ASSETS="$("$PY" -c 'import os, robosuite; print(os.path.join(os.path.dirname(robosuite.__file__), "models", "assets"))')"
    if [ -d "$COMPOSITE_ASSETS" ]; then
      echo "The composite asset directory already exists; skipping rebuild: $COMPOSITE_ASSETS"
    else
      mkdir -p "$COMPOSITE_ASSETS"
      cp -a "$ROBOSUITE_ASSETS/." "$COMPOSITE_ASSETS/"
      cp -a "$LIBEROPRO_ASSETS/." "$COMPOSITE_ASSETS/"
      echo "Composite asset tree built: $COMPOSITE_ASSETS"
    fi
  fi
  test -f "$COMPOSITE_ASSETS/robots/panda/robot.xml" || {
    echo "The composite asset tree is missing robots/panda/robot.xml (the robosuite base was not copied correctly)" >&2
    exit 1
  }
  test -f "$COMPOSITE_ASSETS/scenes/libero_tabletop_base_style.xml" || {
    echo "The composite asset tree is missing scenes/libero_tabletop_base_style.xml (the liberopro scenes were not copied correctly)" >&2
    exit 1
  }
  export LIBERO_ASSETS_ROOT_OVERRIDE="$COMPOSITE_ASSETS"
  echo "Composite asset tree validated (robosuite robot models and LIBERO-Pro scene assets are present)."
  echo "Using LIBERO_ASSETS_ROOT_OVERRIDE=$LIBERO_ASSETS_ROOT_OVERRIDE for the reset smoke test."

  log "8/9 [libero-pro] Minimal import/environment creation smoke test (no real VLA checkpoint required)"
  "$PY" - <<'PYEOF'
import os
os.environ.setdefault("MUJOCO_GL", "egl")
from robots.libero.assets import bind_libero_assets_root

bind_libero_assets_root(os.environ["LIBERO_ASSETS_ROOT_OVERRIDE"])
from liberopro.liberopro.envs import OffScreenRenderEnv
import glob

bddl_candidates = glob.glob(
    os.path.join(
        os.path.dirname(__import__("liberopro").__file__),
        "liberopro", "bddl_files", "libero_10", "*.bddl",
    )
)
if not bddl_candidates:
    raise SystemExit("No packaged bddl task files were found; the liberopro installation may be incomplete")

env = OffScreenRenderEnv(
    bddl_file_name=bddl_candidates[0],
    camera_heights=128,
    camera_widths=128,
)
env.seed(0)
obs = env.reset()
assert obs is not None
env.close()
print("LIBERO-Pro environment create/reset/close succeeded:", bddl_candidates[0])
PYEOF
else
  log "7-8/9 [robocasa] Minimal environment creation smoke test (no real GR00T checkpoint required)"
  "$PY" - <<'PYEOF'
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import robosuite
import robocasa
from robosuite.environments.base import make

env = make(
    env_name="PickPlaceCounterToCabinet",
    robots="PandaOmron",
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    ignore_done=True,
)
obs = env.reset()
assert obs is not None
env.close()
print("RoboCasa environment create/reset/close succeeded")
PYEOF
fi

log "9/9 Complete"
if [ "$TRACK" = "libero-pro" ]; then
cat <<EOF

venv is ready: $VENV_ROOT

Next steps (a real Pi0.5 checkpoint is required and is not provided by this script).
For a real rollout, LIBERO_ASSETS_ROOT_OVERRIDE must point to the composite asset tree
built in step 7; otherwise, environment reset will raise FileNotFoundError because
the robosuite robot models are missing:
  cd "$REPO_ROOT"
  export LIBERO_CONFIG_PATH="$LIBERO_CONFIG_PATH"
  export LIBERO_ASSETS_ROOT_OVERRIDE="${LIBERO_COMPOSITE_ASSETS_DIR:-$VENV_ROOT/libero-pro-composite-assets}"
  "$PY" scripts/experiments/libero_critic_recovery_latency_v3.py \\
    --output /tmp/libero-pi05-smoke \\
    --seed 0 \\
    --env-cuda-device 0 \\
    --rollout-cuda-device 0 \\
    --model-path /abs/path/to/RLinf-Pi05-LIBERO-checkpoint \\
    --max-actions 20 \\
    --no-video
EOF
else
cat <<EOF

venv is ready: $VENV_ROOT

Next steps (a real GR00T checkpoint and a rollout_runtime serve preset referencing
local paths are required and are not provided by this script):
  cd "$REPO_ROOT"
  "$PY" -m rollout_runtime.cli serve \\
    --config <your-preset> --host 127.0.0.1 --port 18730 --launch ray
  "$PY" scripts/evolution/pnp_latency_v3.py \\
    --runtime-url http://127.0.0.1:18730 --seed 0 \\
    --max-actions 20 --actions-per-chunk 5
EOF
fi

echo
echo "Known limitation: the libero-pro and robocasa tracks cannot be installed in the same venv state" \
     "because their robosuite versions conflict at the code level. See the Known Limitations section in VLA_ENV_SETUP.md."
echo "RoboTwin is a third, separately isolated environment: it is SAPIEN-based rather than robosuite-based," \
     "so this script does not build it. Use the upstream RLinf image instead; see docs/robotwin.md."
