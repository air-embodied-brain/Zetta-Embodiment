# Native ManiSkill support

The first task profile is **ManiSkill 3.0.1 / SAPIEN 3.0.2 / Panda /
PickCube-v1 / `pd_ee_delta_pose`**, with the official 50 control step horizon.
The environment runs on Linux x86_64 with NVIDIA CUDA simulation and Vulkan
offscreen rendering. Installing the default Zetta package does not install ManiSkill.

## Scope

| Path | Scope |
| --- | --- |
| `ManiskillEnvCore`, `per_slot` | Native reset, normalized actions, RGB, robot state, official score, terminal observation and close |
| `lockstep_vector` | Synchronous infrastructure use; absent lanes advance with hold commands |
| `maniskill_smoke` preset | Separate Ray environment and fake policy processes; infrastructure validation only |
| `robots/maniskill/run_rollout.py` | Frozen model checks, per-step Critic, bounded Role1 review, recovery and episode evidence |
| `prepare_maniskill_campaign.py` | Immutable task/tool/config files, development seeds, paired gate commands and held-out schedule |

The learned-policy and campaign paths require a checkpoint trained with the exact
observation/action contract and its normalization statistics. No compatible native
PickCube VLA checkpoint has yet been verified for this profile. In particular,
`RLinf/RLinf-Pi05-ManiSkill-25Main-SFT` targets a different robot/task suite and is
not a native Panda PickCube baseline. There is no ready-to-run `maniskill_pi05`
preset, and scripted/fake-policy smoke scores are not learned-policy results.

## Prepare a runtime

Use an isolated root on the Linux GPU host. The script needs `uv`; Python 3.11 is
recommended. All downloaded packages, models, simulator assets, caches and outputs
are placed below that root.

```bash
python3 scripts/deployment/prepare_maniskill.py --root /path/to/maniskill --python 3.11
source /path/to/maniskill/activate.sh
```

`--with-vulkan-loader` optionally extracts the pinned Ubuntu 22.04 Vulkan loader
under the root and writes an NVIDIA ICD file. It does not install system packages.
Use it only when a working loader is absent; the host NVIDIA driver must already
be installed. The script validates the downloaded archive's SHA256.

The executable dependency constraints are in
[`maniskill_runtime.in`](../scripts/deployment/maniskill_runtime.in). Preparation
compiles a complete hash lock, installs it, installs this source tree without
re-resolving dependencies, runs `uv pip check`, and writes
`outputs/installation.json`, `environment.json` and `activate.sh`.
Reuse `--lock <saved-lock>` to reproduce an existing resolution;
`--verify-only` checks an existing installation. Preserve the lock and installation
report with benchmark evidence. Changing constraints requires generating a new lock.
`--installer pip` selects pip 25.2 with pinned build tools and a root-local cache.
For hosts with slow registry access, `--lock <saved-lock> --wheelhouse <directory>`
installs already-downloaded artifacts without querying an index. Include the full
missing dependency set plus pip 25.2, setuptools 80.9.0 and wheel 0.45.1 in that
directory. Runtime distributions remain hash-checked against the lock. SAPIEN 3.0.2
requires `pkg_resources`, so this profile deliberately pins setuptools below 81.

Preparation also downloads and verifies the PhysX GPU archive for
`105.1-physx-5.3.1.patch0`, extracts it below `envs/physx`, and exports
`SAPIEN_PHYSX_GPU_LIBRARY`. The adapter loads that library directly, avoiding
SAPIEN's default home-directory download. Offline preparation requires the same
archive at `artifacts/physx-linux-so.zip` (SHA256
`167a01aad7381afef963b89169968c289e7b653880a7a823c116d87ee5c00fc6`).

If a container selects an incompatible CUDA compatibility library and reports
CUDA error 804, pass `--cuda-driver-dir /path/to/host-driver-libraries`. Preparation
creates root-local links to the supplied `libcuda`/JIT libraries and puts those
links first in its loader path. It does not modify the host driver. GPU numbering
uses `CUDA_DEVICE_ORDER=PCI_BUS_ID` so the selection can be checked against
`nvidia-smi`.

Ray uses Unix sockets with a short path limit. If the runtime root is deeply
nested, pass `--ray-temp-alias /short/path` to preparation. This creates a symlink
to the root's Ray cache and exports it as `RAY_TMPDIR`; the alias must be inside
the authorized remote workspace. Repeated preparation preserves the saved alias.

## Validate the environment

```bash
python scripts/deployment/smoke_maniskill.py \
  --environment /path/to/maniskill/environment.json --gpu 0 \
  --output /path/to/maniskill/outputs/native-smoke

python scripts/deployment/smoke_maniskill_runtime.py \
  --environment /path/to/maniskill/environment.json --gpu 0 \
  --native-output /path/to/maniskill/outputs/native-smoke \
  --output /path/to/maniskill/outputs/ray-smoke --fault-test
```

Output directories must be new. Native smoke uses privileged simulator poses to
generate a control/score test trajectory and compares the recorded actions with
direct Gymnasium replay. A separate zero-action episode must fail. It writes two
videos, per-step JSONL, initial simulator state and a report. `--episodes 100`
alternates successful action replay and zero actions; each successful replay must
match the first trajectory's reward, state and flags. Only the first two episodes
are recorded to video. The Ray smoke replays the same trajectory through separate
workers and Gateway sessions and exercises the fake inference process separately.
It saves reset and step diagnostics before assertions, compares the complete
initial-state hash and uses `atol=rtol=1e-5` for reward and robot state. With
`--fault-test`, it kills its own environment actor, requires an infrastructure
error, explicitly relaunches the runtime and verifies a fresh reset/action and
worker process exit. This is an explicit restart test, not transparent recovery
of an interrupted episode.

## Observation, action and score contracts

`panda_proprio_v1` exposes 18 float32 values: Panda's nine `qpos`, followed by its
nine `qvel`. Joint positions/velocities use native radian and meter units. Object
poses and goal coordinates are not policy state. The formal task uses a visible
green goal marker, `base_camera` RGB at 128 × 128 and no wrist camera. Camera names
are explicit; extra views are sorted by name. `legacy_flatten_v1` is an explicit
compatibility option for older full-state consumers and is not accepted by the
formal campaign profile.

After scene reconfiguration, reset reapplies goal visibility before capturing the
first policy frame. Native smoke checks that this initial RGB matches a fresh
render of the visible-goal scene.

Actions are nonempty `[T, 7]` finite float32 arrays in `[-1, 1]` with contract
`maniskill_panda_pd_ee_delta_pose_normalized_v1`. The first six components are
normalized EE translation/rotation controller inputs; the last component is the
gripper target (`-1` closed, `+1` open). They are not a meter/radian API. Native
actions pass through without another scale or normalization operation. Invalid
shapes, nonfinite numbers and out-of-range actions fail before simulation.

Runtime reports official `info.success` and optional `info.fail` independently
from `terminated` and `truncated`. It also tracks `success_once`, `success_at_end`,
first success step and termination reason. A termination can be success or failure;
it is not itself a success signal. Automatic reset is disabled so the terminal
observation remains available. An explicit reset seed is preserved across slots and
worker ranks. The wrapper returns a hash of the complete simulator initial state;
the policy never receives the privileged state values.

The formal `per_slot` profile requires `extra_init_params` to contain exactly
`{"enhanced_determinism": true, "reconfiguration_freq": 1}`. Rebuilding the PhysX
scene on each reset removes history from earlier contacts. On the pinned stack,
reusing a scene produced different trajectories despite identical visible initial
state hashes; setting only `enhanced_determinism` did not fix it. A reset hash alone
therefore does not establish replay determinism. Reconfiguration adds reset cost
and cannot accompany a partial vector reset. Synchronous vector infrastructure
keeps its own scene-reuse semantics and is excluded from the formal paired campaign.

Observation/action semantic contract IDs participate in `obs_schema_digest` v2.
Changing state meaning therefore separates policy batches even if dimensions match.
The v2 digest change also affects other families' digest strings; these are runtime
compatibility keys, not persisted model identities.

## Frozen policy and campaign

Construct `robots.maniskill.contracts.TaskContract` with the actual model version,
checkpoint SHA256 and norm statistics SHA256. Use `as_dict()` to save task JSON.
Save `dataclasses.asdict(ManiskillEnvConfig(...))` for environment JSON, using the
formal profile in `maniskill_smoke.yaml`. Campaign validation rejects a different
robot, horizon, camera profile, goal visibility or asynchronous vector execution.

For the OpenPI backend, set `observation_contract` to the environment config's
`observation_digest()` and `action_contract` to the native action contract. Also
provide `checkpoint_sha256`, `norm_stats_sha256`, and
`openpi_data.norm_stats_path`. Frozen artifact checking currently requires exactly
one local `model.safetensors`, explicit `norm_stats.json`, and no alternate weight
override or LoRA. Configure the data transforms to match how that model was trained;
a matching hash alone does not establish training compatibility.

Seeded OpenPI requests run individually with isolated Python, NumPy and Torch RNG
states, so their output does not depend on other requests' batch order. CUDA graphs
are rejected for this path. Rollout requires acknowledgement of request RNG,
semantic contracts, model version and verified file hashes before executing an
action. Updating weights requires creating a new frozen worker.

```bash
python scripts/evolution/prepare_maniskill_campaign.py \
  --output-root /path/to/campaign --campaign-id pickcube-development \
  --runtime-url http://127.0.0.1:18730 \
  --env-config /path/to/env.json --task-contract /path/to/task.json \
  --policy-id native-pickcube-policy --master-seed 123 \
  --code-commit <exact-source-commit>
```

The preparer freezes 50 development seeds, held-out seeds 1..20, policy RNG,
tool catalog, prompts and executable rollout commands. It refuses to overwrite an
existing campaign. `run_rollout.py --help` documents the standalone entrypoint.
Reuse the existing `scripts/evolution/run_campaign.py` orchestration and gate
runner after supplying a compatible policy service.

Critic can inspect only robot proprioception, joint speed, gripper width and
episode step. `BoundedRole1` is a deterministic review of frozen recovery proposals,
not an LLM planner. It is the sole authority to accept a recovery, verifies its tools
and ensures its total requested budget fits the remaining episode. Available tools
are `maniskill.vla`, `maniskill.ee_delta` and `maniskill.hold`. Hold preserves the last
gripper command. Acceptance clears pending policy actions; Critic is suspended
during recovery; all actions consume the same 50-step budget.

Each episode records aligned actions, official scores, robot states, proposals,
Role1 decisions, recovery actions, video and trajectory evidence. Infrastructure
or policy-contract failures produce `infra_invalid` with `success=null` and still
close sessions. Same-seed gates require matching full simulator initial state,
scenario, instruction and task-contract hashes. Matching robot joints alone is
insufficient. No promotion or success-rate improvement is implied by infrastructure
tests or synthetic recovery wiring tests.
