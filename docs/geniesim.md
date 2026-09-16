# Genie Sim / Isaac Sim Runtime

Zetta's `geniesim` environment family runs the RoboColiseum Genie Sim Benchmark
task `g2op_if_pick_block_color` through native environment calls. Runtime owns
reset and action execution. Isaac's main thread owns physics and rendering in a
separate Python process; there is no WebSocket inference bridge.

The current profile is one GPU, one process and one session, with the
`G2_omnipicker` robot, `table_task_1_g2_op`, scene 0 and instruction 0. VLA
policies, other tasks, vector environments and evolution campaigns are outside
this profile. The provided controller validates the environment lifecycle and
joint response; it is not a benchmark policy or a task-success claim.

## Versions

| Component | Version |
| --- | --- |
| Genie source | `6ca11c7593ecf6b7dae58c28fd19f4a789a0dd46` |
| Isaac Sim | `5.1.0.0`, Python `3.11.13` |
| Assets | `agibot-world/GenieSimAssets` at `a0813aad7c16165daffbe6c2737754e0344809ee` |
| Native IK SDK | `0.4.3`, upstream cp311 Linux x86_64 wheel |
| Tested system | Ubuntu 22.04.5, RTX 4090, NVIDIA `550.127.08` |

Isaac's published tested driver for 5.1 is newer than this host. The evidence
here applies to this particular profile and host, not every 5.1 workload.
No driver upgrade or system library installation is part of these scripts.

## Preparation

Run from the repository on Linux x86_64 with `uv`, `curl`, `apt-get` and
`dpkg-deb` available. The preparation script requires Python 3.11.4+ for safe
archive extraction; the command below selects 3.11.13 explicitly.
The asset subset is about 9.4 GB, in addition to Isaac,
Torch and their caches. Choose a root with sufficient free space.

```bash
export GENIE_ROOT=/absolute/path/to/geniesim-validation
export UV_PYTHON_INSTALL_DIR="$GENIE_ROOT/envs/python"
export UV_CACHE_DIR="$GENIE_ROOT/cache/uv"
uv run --no-project --python 3.11.13 python \
  scripts/deployment/prepare_geniesim.py --root "$GENIE_ROOT"
```

This creates `envs/isaac51`, `envs/runtime`, `envs/asset-tools`, `src`, `assets`,
`artifacts` and `cache` under that root. It verifies source/SDK/native package
hashes, generates a combined pip hash lock, installs the simulator separately
from Zetta, and prepares/validates the selected USD asset closure. Original
asset downloads are retained; only the runtime copy has absolute upstream USD
references rewritten using OpenUSD APIs.

Use `--index URL` for an alternate PyPI index, `--source-archive PATH` to reuse
the pinned archive, or `--lock PATH` to reuse a hash lock from the same root.
The lock contains an absolute file URL for the verified SDK wheel. Moving a
deployment requires regenerating that URL/lock. `--skip-assets` and
`--skip-runtime` permit resuming individual preparation stages; `--verify-only`
checks the source archive/SDK hashes and installed dependency consistency.
Preparation alone does not claim the simulator can render.

## Run The Lifecycle

```bash
"$GENIE_ROOT/envs/runtime/bin/python" scripts/deployment/smoke_geniesim.py \
  --sim-python "$GENIE_ROOT/envs/isaac51/bin/python" \
  --geniesim-root "$GENIE_ROOT/src/genie_sim-6ca11c7593ecf6b7dae58c28fd19f4a789a0dd46" \
  --assets-root "$GENIE_ROOT/assets/geniesim" \
  --output-root "$GENIE_ROOT/outputs" --gpu-id 0 \
  --library-path /usr/lib/x86_64-linux-gnu \
  --library-path /usr/local/cuda/lib64 \
  --library-path "$GENIE_ROOT/envs/native-libs/usr/lib/x86_64-linux-gnu"
```

On the validation host, system driver libraries must precede CUDA compat
libraries: the default loader selected a forward-compat `libcuda` and returned
CUDA error 804. These library paths affect only the child process. `gpu_id` is
the physical PCI-order index; when `CUDA_VISIBLE_DEVICES` is inherited, the
requested index must be among its numeric IDs. The child removes this CUDA
remapping because Kit selects Vulkan and CUDA devices by their physical IDs.
UUID or MIG assignment is not supported by this profile.

The smoke uses the Runtime client, development seeds 1001/1002/1003, repeated
same-seed reset and a fresh Runtime recreation. Each episode runs 32 control
steps with a 0.01-radian left-arm pulse. It checks the native evaluation tick,
early stopping of an oversized final chunk, joint response, RGB images,
reset tolerances and supervised process exit. `--transport ray_channel`
also exercises Ray's channel transport with local worker objects; it does not
claim a separate Ray actor deployment. `--skip-recreation` is useful while
diagnosing startup. Every invocation creates a new `run-*` evidence directory.

Outputs include initial/final/reset PNGs for all three cameras, a head-camera
MP4 sampled at control steps, per-step JSON, official scores, session logs and
`process.json`. Video playback uses 10 fps; it is not a simulation timing
measurement. `run-*/result.json` becomes `passed` only after the child processes
acknowledge close and exit with code 0. Use `process.json` to distinguish native
errors or forced cleanup from task failure.

## Interface Contract

| Field | Meaning |
| --- | --- |
| Actions | `[chunk,16]`: left arm 7, right arm 7, left/right gripper 1 each; absolute radians |
| Limits | Read from the loaded Isaac articulation by joint name; invalid dimensions, nonfinite values and out-of-range targets are rejected before execution |
| State | 21 values: left arm, right arm, left gripper, right gripper, waist 5; head group empty |
| Cameras | `main_image=head`, `wrist_image=left_hand`, `extra_view_images[0]=right_hand`; uint8 RGB |
| Image size | Head 640x400; wrists 1280x1056 |
| Step records | Actual per-step joint observations and applied actions; final observation carries images |
| Reset | Raw seed, no slot offsets; same seed reuses the task, a changed seed rebuilds its process to avoid accumulated scene generalization; native reset targets are restored and joints must converge before the initial observation |
| Observe | Cached latest control observation; does not issue a new control step |
| Physics | Free-running between requests; one control step is not a fixed number of physics ticks |
| Evaluation | Official `TaskEvaluation` updated every 30 control steps; `evaluation_step` reports freshness |
| Success | Official `scores.E2E == 1`; native result `code == 0` alone is not task success |
| Termination | Native `has_done`; a separate `max_steps` limit produces truncation |
| Reward | 1 only on the first observed official success, otherwise 0 |
| Extension | `geniesim.result` with empty args returns cached scores, actual steps, limits and artifact location |

Waist and head retain the native reset targets. For hold actions use the
observed first 16 state values. Zero is an absolute target, not a hold command.
The maximum chunk is 64. There is no automatic action retry: a timed-out or
broken request closes and reaps the process because its action may already
have executed. Create a fresh session after an infrastructure failure.

The upstream reset snaps joint positions but leaves old drive targets active
and polls for only one second. The adapter also applies the native reset
targets to the drives and waits up to 30 seconds for all joints to be within
0.01 radians, then captures a fresh observation without issuing a control
step. Failure to converge is an environment error. The smoke compares all
reset joints within 0.02 radians; it does not assert identical pixels or full
scene determinism.

Isaac's default fast shutdown is intentional: on the tested host, disabling it
caused a post-close GC crash/hang. The parent verifies close acknowledgement
and the exit code instead of trusting a success flag written before shutdown.

## Runtime Configuration

`rollout_runtime/config/presets/geniesim_smoke.yaml` provides single-session
placement and startup timeouts. Replace its path placeholders and choose the
physical GPU. Its fake policy slot is unused by the smoke; callers supply
actions through `action_step`. No pretrained policy mapping is registered.

Unit tests need the `test` extra; Ray transport checks also require the `ray`
extra:

```bash
python -m pytest tests/runtime/test_geniesim_family.py -q -m "not ray"
python -m pytest tests/runtime/test_geniesim_family.py -q -m ray
```

## Validation Scope

The profile was validated on the system listed above through the Runtime
client and Ray channel transport with local workers. Four episodes covered
seeds 1001/1002/1003 and a recreated seed 1001 session. Each executed 32 control
steps, updated official evaluation at step 30, passed repeated reset checks,
and closed the simulator with exit code 0. The maximum repeated-reset joint
deviation was 0.00943 radians, within the 0.02-radian smoke tolerance.

The hold/pulse controller reached the configured time limit with
`success=false`. This validates environment control and lifecycle behavior;
successful grasping and VLA policies were not tested. Preparation was checked
by rerunning against the existing isolated installation, rather than a fresh
run of the consolidated installer in an empty directory.
