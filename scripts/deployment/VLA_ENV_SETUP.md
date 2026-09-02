# VLA runtime environment setup

This guide describes the supported environment layouts for:

- LIBERO-Pro with the Pi0.5 backend;
- RoboCasa with the GR00T backend.

The executable source of truth is
[`install_vla_env.sh`](./install_vla_env.sh). The matching container build is
[`Dockerfile.vla-env`](./Dockerfile.vla-env). Keep the versions and workarounds
in those files aligned with this guide.

## Why the tracks require separate environments

The tracks cannot share one virtual environment:

- LIBERO-Pro uses `robosuite==1.4.x` because its
  `BDDLBaseDomain` subclasses `SingleArmEnv`;
- RoboCasa uses `robosuite==1.5.2` because it needs the `PandaOmron` robot;
- `SingleArmEnv` was removed in robosuite 1.5, so this is a code-level
  incompatibility rather than a dependency resolver preference.

Create one venv or image per track.

## Validated versions

| Component | LIBERO-Pro | RoboCasa |
|---|---:|---:|
| Python | 3.10 | 3.10 |
| MuJoCo | 3.3.1 | 3.3.1 |
| robosuite | 1.4.x | 1.5.2 |
| simulator distribution | rpent-liberopro==0.1.1 | robocasa==1.0.1 |
| policy backend | rlinf-openpi==0.1.1 | gr00t==1.1.0 |
| pydantic | 2.10.6 | 2.10.6 |
| transformers | backend dependency | 4.51.3 |
| flash-attn | not required | 2.8.3 |

The distribution is named `rpent-liberopro==0.1.1` and provides the
`liberopro` Python import package. It must be available from the configured
package index. If it is hosted elsewhere, set `LIBEROPRO_PACKAGE` to a pinned
wheel or source URL.

## Prerequisites

The installer was validated on Ubuntu 22.04 and expects:

- Python 3.10 and `venv`;
- a working NVIDIA driver and `nvidia-smi`;
- build tools;
- EGL/OpenGL development libraries;
- Git;
- network or mirror access for Python packages and model/assets downloads.

Checkpoints are not installed. RoboCasa kitchen assets and the three RoboCasa
source repositories are also external inputs.

## Local venv installation

Run commands from the repository root.

### LIBERO-Pro + Pi0.5

~~~bash
export REPO_ROOT="$PWD"
export VENV_ROOT=/abs/path/to/venvs/libero-pro
bash scripts/deployment/install_vla_env.sh --track libero-pro
~~~

Optional inputs:

- `PYTHON_BIN`: Python 3.10 executable; defaults to `python3.10`;
- `LIBEROPRO_PACKAGE`: pinned requirement or URL; defaults to
  `rpent-liberopro==0.1.1`;
- `LIBERO_COMPOSITE_ASSETS_DIR`: destination for the composite assets tree;
- `LIBERO_CONFIG_PATH`: location of LIBERO-Pro's `config.yaml`; defaults to a
  venv-local directory so another installation's global config cannot leak
  into validation;
- `LIBERO_PRO_ASSET_PATH`: existing LIBERO-Pro assets tree to link when the
  host cannot reach Hugging Face;
- `SKIP_ASSET_DOWNLOAD=1`: skip asset download and construction when a valid
  composite tree is already available.

For a rollout, point the runtime at the composite assets tree:

~~~bash
export LIBERO_CONFIG_PATH=/abs/path/to/venvs/libero-pro/.liberopro-config
export LIBERO_ASSETS_ROOT_OVERRIDE=/abs/path/to/venvs/libero-pro/libero-pro-composite-assets
~~~

Do not point this variable at raw LIBERO-Pro assets. The directory must contain
both:

~~~text
robots/panda/robot.xml
scenes/libero_tabletop_base_style.xml
~~~

The installer builds this layout by copying robosuite's model assets first and
then overlaying LIBERO-Pro's scenes and objects.

### RoboCasa + GR00T

Prepare a source root containing these exact checkouts:

| Directory | Repository | Ref |
|---|---|---|
| `robosuite/` | <https://github.com/ARISE-Initiative/robosuite> | tag `v1.5.2` |
| `robocasa/` | <https://github.com/robocasa/robocasa> | commit `29f7ce8814c1547f5af762a0997fbd4b64848dd7` |
| `Isaac-GR00T/` | <https://github.com/NVIDIA/Isaac-GR00T> | tag `n1.5-release` |

Then run:

~~~bash
export REPO_ROOT="$PWD"
export VENV_ROOT=/abs/path/to/venvs/robocasa
export ROBOCASA_SRC_ROOT=/abs/path/to/robocasa-source-checkout
bash scripts/deployment/install_vla_env.sh --track robocasa
~~~

`FLASH_ATTN_WHEEL` may point to a compatible local wheel, URL, or internal
mirror. Its CUDA, torch, Python and C++ ABI tags must match the environment.

RoboCasa kitchen assets must be installed separately; follow the
[upstream installation guide](https://robocasa.ai/docs/build/html/introduction/installation.html).

## Container build

Build from the repository root with Docker BuildKit.

LIBERO-Pro:

~~~bash
mkdir -p /tmp/empty-robocasa-src
docker build -f scripts/deployment/Dockerfile.vla-env \
  --build-arg TRACK=libero-pro \
  --build-context robocasa-src=/tmp/empty-robocasa-src \
  -t zetta-vla-env:libero-pro .
~~~

RoboCasa:

~~~bash
docker build -f scripts/deployment/Dockerfile.vla-env \
  --build-arg TRACK=robocasa \
  --build-context robocasa-src=/abs/path/to/robocasa-source-checkout \
  -t zetta-vla-env:robocasa .
~~~

The image intentionally excludes repository source, model checkpoints, and
LIBERO-Pro assets by default. Mount source and checkpoints at runtime. See the
Dockerfile header for the optional asset-baking build context.

## Compatibility fixes encoded by the installer

These headings are stable because comments in `Dockerfile.vla-env` refer to
their bug numbers.

### Bug 1 — OpenPI upgrades MuJoCo transitively

`rlinf-openpi` pulls a dependency chain that can upgrade MuJoCo to 3.8.1.
The exercised Pi0.5 path does not need that newer constraint, so both setup
surfaces restore the validated `mujoco==3.3.1` without dependency resolution.

### Bug 2 — The two robosuite APIs are incompatible

LIBERO-Pro requires the removed `SingleArmEnv` API from robosuite 1.4, while
RoboCasa requires `PandaOmron` from robosuite 1.5. The supported fix is
separate environments, not forcing both packages through one resolver state.

### Bug 3 — numpydantic fails with newer pydantic

The dependency graph may select pydantic 2.13.x, which breaks numpydantic schema
generation. Both tracks pin `pydantic==2.10.6`.

### Bug 4 — GR00T requires an ABI-matched flash-attn wheel

GR00T's Eagle vision backbone has no CPU fallback for `flash_attn`. The wheel
must match torch, CUDA, Python and the C++11 ABI. Override
`FLASH_ATTN_WHEEL` or `FLASH_ATTN_WHEEL_URL` when the default release URL is
not reachable.

### Bug 5 — rlinf-transformer-openpi overwrites transformers files

The repackaged distribution can overwrite the real `transformers` files while
leaving misleading package metadata. The RoboCasa track force-reinstalls
`transformers==4.51.3` without changing its dependency set.

For the LIBERO-Pro track, `rlinf-transformer-openpi==4.53.2` is an exact,
required dependency of `rlinf-openpi==0.1.1` and must remain installed. The
RLinf distribution guard therefore permits those two OpenPI packages while
still rejecting the main `rlinf` framework and environment forks such as
`rlinf-libero`.

### Bug 6 — GR00T shadows RoboCasa's top-level package

Both projects are editable installs. Their generated import finders can resolve
`import robocasa` to GR00T's overlay instead of RoboCasa's package, preventing
environment registration. The installer adds a venv-local `.pth` fix that
orders RoboCasa's finder first.

### Bug 7 — raw LIBERO-Pro assets omit robosuite robot models

`LIBERO_ASSETS_ROOT_OVERRIDE` replaces robosuite's asset root; it does not
fall back to the original directory. A raw LIBERO-Pro tree therefore lacks
`robots/panda/robot.xml`. The installer creates and validates a composite
tree containing both asset sets.

## Verification

### Core versions and imports

~~~bash
"$VENV_ROOT/bin/python" -c "
import mujoco, pydantic, robosuite, rollout_runtime
print('mujoco', mujoco.__version__)
print('robosuite', robosuite.__version__)
print('pydantic', pydantic.VERSION)
print('rollout_runtime', rollout_runtime.__file__)
"
~~~

Then verify the track-specific package:

~~~bash
# LIBERO-Pro
"$VENV_ROOT/bin/python" -m pip show rpent-liberopro

# RoboCasa
"$VENV_ROOT/bin/python" -c "
import flash_attn, gr00t, robocasa, transformers
print('flash_attn', flash_attn.__version__)
print('transformers', transformers.__version__)
"
~~~

### LIBERO-Pro benchmark registration

The maintained package must expose all 16 task/swap/language/object suites. This
check replaces the obsolete manual registration-patch workflow:

~~~bash
"$VENV_ROOT/bin/python" - <<'PY'
from liberopro.liberopro import benchmark

families = ("spatial", "object", "goal", "10")
variants = ("task", "swap", "lan", "object")
names = [f"libero_{family}_{variant}" for family in families for variant in variants]
available = benchmark.get_benchmark_dict()
missing = sorted(set(names) - set(available))
assert not missing, f"missing perturbation suites: {missing}"

for name in names:
    suite = benchmark.get_benchmark(name)()
    assert suite.get_num_tasks() > 0, name
    assert suite.get_task(0).language.strip(), name
    assert len(suite.get_task_init_states(0)) > 0, name

task = benchmark.get_benchmark("libero_spatial_task")().get_task(0)
expected = "Pick the akita black bowl not between the plate and the ramekin and place it on the plate"
assert task.language == expected, task.language
print("LIBERO-Pro perturbation registration and language checks passed")
PY
~~~

If this fails for the configured `rpent-liberopro==0.1.1` artifact, stop and repair
the installer/package source. Do not reintroduce an undocumented manual patch
step.

### Environment smoke

The installer finishes by creating, resetting and closing a simulator
environment without requiring a model checkpoint. Treat any failure in that
step as an incomplete installation.

## Known limitations

- LIBERO-Pro and RoboCasa require separate venvs or images.
- A real rollout still needs an external policy checkpoint.
- The simulator tracks use the validated Python 3.10 ABI. `openai-codex`
  requires Python 3.11 or newer, so use the API/Claude planner in these venvs
  or run the Codex planner from a separate Python 3.11+ process.
- GPU inference and offscreen rendering require the host NVIDIA driver and EGL.
- The default installer needs network or mirror access for packages and assets.
- `SKIP_ASSET_DOWNLOAD=1` is safe only when a complete composite assets tree
  already exists.
- The repository does not supply RoboCasa's external source trees or kitchen
  asset bundle.

For the higher-level runtime and campaign flow, return to the
[main README](../../README.md). For LIBERO-Pro task execution semantics, see the
[`pro_hybrid_guide.md`](../../robots/libero/guides/pro_hybrid_guide.md).
