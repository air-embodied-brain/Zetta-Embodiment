# Genie Sim VLA and Campaign

The `geniesim_vla` Runtime backend connects a G2 policy service to the native
Genie Sim environment. `robots/geniesim/run_rollout.py` runs baseline episodes
and frozen recovery bundles; `scripts/evolution/prepare_geniesim_campaign.py`
prepares the existing Campaign queue, diagnosis, proposal, same-seed and held-out
workflow. The supported task remains `g2op_if_pick_block_color`, one Isaac
process and one active session on one GPU.

This integration does not ship a pretrained G2 checkpoint. A LIBERO or RoboTwin
checkpoint is not interchangeable with the G2 robot's joint and camera contract.
Model serving and Isaac run in separate environments.

The official `instruction_and_robust_pi05` checkpoint can be served with
`scripts/deployment/serve_geniesim_pi05.py`. The script pins the ACoT-VLA source
revision and the ModelScope revision, verifies every checkpoint file before
loading it, and returns a content-addressed `model_version`. It requires the
model environment's JAX/CUDA dependencies; keep that environment separate from
Isaac and Runtime.

## Policy Contract

The wire protocol follows `CoRobotPolicy` in Genie Sim revision
`6ca11c7593ecf6b7dae58c28fd19f4a789a0dd46`: binary MessagePack over WebSocket,
one request followed by one response, without an initial metadata handshake.

Requests contain `method="infer"` and a `params` object:

| Field | Meaning |
| --- | --- |
| `images.head`, `images.hand_left`, `images.hand_right` | JPEG objects with `encoding`, `image_data`, `width`, `height`; encoded from Runtime RGB |
| `states.arm_joint_states` | Left arm 7, right arm 7; absolute radians |
| `states.gripper_states` | Two native gripper states multiplied by -1, matching upstream `G2_omnipicker` |
| `states.waist_joint_states`, `states.head_joint_states` | Five waist joints and an empty head group |
| `prompt` | Official task instruction, or the active frozen recovery instruction |
| `robot_type`, `task_name` | `G2_omnipicker`, `pick_block_color` |
| `episode_idx` | Connection-local episode identity; the connection is replaced on session/episode changes |
| `episode_done`, `task_progress` | `false`, `[]`; privileged official scores are not policy inputs |
| `policy_rng` | Optional Campaign sampling extension, described below |

Responses contain a `result` object with `left_arm` and `right_arm`, each
explicitly declaring `kind="JOINT_ABS"` and `values` shaped `[T,7]`.
`left_effector` and `right_effector` each have shape `[T,1]`. All four horizons
must agree and be in `1..64`. Gripper outputs are multiplied by -1 to restore
native radians; arms are passed through without clipping or smoothing. The
environment checks the resulting `[T,16]` actions against articulation limits.

EEF actions, depth/history requests, head/waist control, malformed dimensions
and nonfinite values fail explicitly. This profile has no end-pose input.
There is no automatic request replay after a policy timeout or disconnect.

Campaign sampling adds `params.policy_rng`. A compatible server must apply that
seed to its sampler and return the same integer as `result.policy_rng`, plus
`result.model_version` matching the frozen checkpoint version. The rollout
derives each inference seed from the preregistered policy RNG and current
control-step index. Environment seeds are not policy inputs. The adapter checks
acknowledgement, while correct sampler seeding remains the server's responsibility.
Reset any server temporal state when a new connection starts.

Stock CoRobot servers that do not implement this extension can still be used
through Runtime `policy_step` without a seed. Campaign episodes require the
acknowledgement and never silently ignore a frozen RNG. `require_policy_rng_ack`
defaults to true; disabling it does not relax the Campaign rollout check.

## Deployment

Prepare Isaac, the pinned Genie source and assets using
`scripts/deployment/prepare_geniesim.py`. Use Python 3.11 for the Runtime:

```bash
python -m pip install -e '.[runtime,geniesim-vla]'
```

Create a deployment copy of
`rollout_runtime/config/presets/geniesim_vla.yaml`. Set the native interpreter,
source/assets/output paths, physical GPU, any required native `library_paths`,
policy WebSocket endpoint and checkpoint `model_version`. Export that YAML's
`env_config` mapping to a JSON file for rollout/Campaign commands:

```bash
python -c 'import json,sys,yaml; json.dump(yaml.safe_load(open(sys.argv[1]))["env_config"], open(sys.argv[2], "w"), indent=2)' geniesim-vla.yaml env-config.json
python -m rollout_runtime.cli serve --config geniesim-vla.yaml \
  --launch local --host 127.0.0.1 --port 18730
```

Start the real baseline service in the model environment (the checkpoint is
about 12.44 GB):

```bash
python scripts/deployment/serve_geniesim_pi05.py \
  --model-source /path/to/ACoT-VLA \
  --checkpoint-dir /path/to/checkpoints/instruction_and_robust_pi05 \
  --output-root outputs/geniesim-pi05-model --port 18990
```

Read `outputs/geniesim-pi05-model/model.json` and use its `model_version` in
the Runtime YAML and Campaign preparation. This service applies the frozen
Campaign RNG to each JAX sampler invocation and acknowledges it in every
response.

On `air-4090`, the default dynamic loader selects CUDA 12.8's forward-compat
driver (570.124.6), which cannot initialize against the host kernel driver
550.127.08 on an RTX 4090. Select the host driver for the model process and
explicitly assign a GPU separate from Isaac:

```bash
env -u LD_LIBRARY_PATH \
  LD_PRELOAD=/lib/x86_64-linux-gnu/libcuda.so.1 \
  CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
  python scripts/deployment/serve_geniesim_pi05.py \
    --model-source /path/to/ACoT-VLA \
    --checkpoint-dir /path/to/checkpoints/instruction_and_robust_pi05 \
    --output-root outputs/geniesim-pi05-gpu --port 18990
```

Use this driver path only on a host where it is the matching native driver.
`JAX_PLATFORMS=cuda` makes a GPU initialization failure explicit. Do not set
`CUDA_VISIBLE_DEVICES` on the Runtime to the model's GPU: Isaac uses its own
physical `env_config.gpu_id`. The model environment must also supply the pinned
upstream loader dependencies. The recorded deployment reuses an import shim
for the unused LeRobot training dataset module; it is not a clean installation
of the full upstream training stack.

Run one VLA episode in another terminal:

```bash
python -m robots.geniesim.run_rollout \
  --runtime-url http://127.0.0.1:18730 --env-config env-config.json \
  --policy-model-version YOUR_CHECKPOINT_REVISION \
  --seed 1001 --policy-rng 42 --output-dir outputs/geniesim-1001
```

Authenticated Runtime servers use `RR_AUTH_TOKEN`; set the matching single
client bearer token in the worker's `ROLLOUT_RUNTIME_TOKEN`. Tokens are read
from the process environment and are absent from frozen commands.

## Campaign

Freeze a generation before development rollouts:

```bash
python scripts/evolution/prepare_geniesim_campaign.py \
  --output-root campaigns/geniesim-g0000 --campaign-id geniesim-g0000 \
  --runtime-url http://127.0.0.1:18730 --env-config env-config.json \
  --policy-model-version YOUR_CHECKPOINT_REVISION \
  --code-commit "$(git rev-parse HEAD)" --master-seed 20260911

python scripts/evolution/run_campaign.py \
  --manifest campaigns/geniesim-g0000/manifest.json \
  --root campaigns/geniesim-run --queue-root campaigns/queue \
  --tool-catalog campaigns/geniesim-g0000/tool-catalog.json --workers local \
  --worker-command python -m zetta.evolution.cli worker \
    --queue-root '{queue_root}' --host '{host}' --concurrency 1
```

Use the installed Runtime Python for preparation and workers. The preparer
freezes the absolute repository entrypoint and virtualenv interpreter path,
environment JSON and hash, model version, prompt/tool contracts, and a disjoint
schedule. Defaults are 50 development seeds and held-out seeds `1..20`.
Logical concurrency is fixed to one; the Runtime Gateway owns session admission.
Generation 1 and later require `--parent-bundle`. Parent bundles are copied into
the immutable preparation directory. Offline agent credentials are configured
through the existing Campaign provider setup.

Baseline episodes execute the VLA only. Candidate episodes evaluate the shared
`TemporalCritic` once per control step. A deterministic actor executes the first
matching frozen recovery in recovery-ID order, preserving it across inference
chunks. Online recovery does not call an LLM. The tools are:

- `geniesim.vla`: optional instruction, `actions_per_chunk` in `1..64`, and a
  required `max_steps` budget in `1..300`.
- `geniesim.hold`: hold all 16 observed joints for `max_steps` in `1..64`.

Critic activation conditions are the executable prerequisites. Critics are
suspended while a recovery is active. Every step uses
`stop_when="budget_exhausted_or_success"`; recoveries use
`stop_condition="official_success_or_budget"` and `fallback="resume_vla"`.
Unsupported features, tools, stop conditions and tool-plugin bundles fail
before environment creation. These contracts are included in the frozen catalog.

## Evidence and Limits

Both paired arms execute one native control action per Runtime call. Each reset
and control step records all three RGB views, so video frame indices align with
state-timeline indices. VLA inference remains chunked (default five actions).
Outputs include `episode.json`, action/state/chunk/tool JSONL, three MP4s,
content-addressed failure segments, and synchronized visual evidence.

Success uses the native official result (`scores.E2E == 1`), independently of
termination, truncation, reward or native result `code`. Infrastructure and
evidence failures produce `infra_invalid` with no success score. A failed close
acknowledgement invalidates the episode. `candidate_intervention` counts actual
recovery execution, allowing the existing gate to reject un-attributed successes.

The Genie-specific reset comparison requires the same environment configuration,
seed and task instruction, and compares all 21 native joints with a fixed
0.02-radian tolerance. Raw joint hashes and camera differences remain in the
evidence. This is a seeded profile with bounded joint reset error, not an exact
scene-state or pixel-determinism guarantee. Physics is free-running between
requests, and official evaluation refreshes every 30 control steps. Use the same
deployment, timing profile and service sampling contract for both paired arms.

## Validation

```bash
python -m pytest tests/runtime/test_geniesim_family.py \
  tests/runtime/test_geniesim_vla.py tests/test_prepare_geniesim_campaign.py \
  -q -m 'not ray'

python scripts/deployment/smoke_geniesim_vla.py \
  --env-config env-config.json --output-root outputs/paired-vla \
  --policy-endpoint ws://127.0.0.1:8999 --model-version YOUR_CHECKPOINT_REVISION
```

Use `--protocol-stub` instead of `--policy-endpoint` to validate the real Isaac
environment, WebSocket/Runtime plumbing, evidence and paired gate without model
weights. The stub holds observed joints; its report explicitly sets
`pretrained_model_validated=false`. A smoke passes only after both episodes and
native process shutdown are validated. Its candidate need not pass the learning
gate: a hold controller is not evidence of improved task success.

### Recorded Validation (2026-09-11)

- Local Python 3.11.15: 155 passed, 1 deselected across the focused Genie Sim,
  Runtime policy/layering, Campaign preparation, and shared evolution suites.
- Remote Python 3.11.13: 58 passed, 1 deselected for `test_geniesim_family.py`,
  `test_geniesim_vla.py`, and `test_prepare_geniesim_campaign.py`.
- Final-source hardware smoke reused an existing Isaac Sim 5.1 installation.
  Baseline and candidate each completed 32 control steps with three camera
  videos; the candidate executed two recovery steps. Official evaluation
  refreshed at step 30, and the native process acknowledged close and exited 0
  without forced termination. This was not a fresh Isaac installation test.
- The same-seed gate rejected the non-improving hold candidate as expected.
  Campaign preparation and supervisor enqueue also completed successfully.

The September 11 hardware smoke used `--protocol-stub`, with both task outcomes
unsuccessful and `pretrained_model_validated=false`. Real-model validation was
performed later as recorded below.

### Recorded Validation (2026-09-14)

- Real G2 `instruction_and_robust_pi05` service validation used model version
  `geniesim-pi05:c0430c39e7d019f0bb10ec486d11df3657571baf75f13dd9f16b5c0143a5b470`
  at `ws://127.0.0.1:18997` on `air-4090`. The model-backed episode completed
  32/32 native control steps with 7 inference calls and produced
  `scores.E2E=0` (valid task failure, not an infrastructure invalidation).
  The service ran with CPU JAX because the host GPU driver and JAX CUDA plugin
  were incompatible; this is a deployment limitation rather than a protocol
  failure.
- A model-backed Campaign ran on `air-4090` with Runtime `127.0.0.1:18731`,
  Isaac Sim 5.1, Genie Sim revision `6ca11c7593ecf6b7dae58c28fd19f4a789a0dd46`,
  and the same frozen checkpoint version above. Baseline, model-backed
  diagnosis, proposal, Stage 2 contract validation, shadow replay, and
  same-seed execution all completed through the normal lifecycle. Four
  candidate bundles were rejected by the frozen same-seed gate (each
  `0/1`, no infrastructure-invalid arms); the run then continued to candidate
  refinement rather than bypassing the gate. Gate records and immutable
  episode artifacts are under
  `/mnt/ssd_data/wangzexu/geniesim-g2-real-20260911/campaign-real-01/run3`.
- The Campaign's authoritative task outcome remained unsuccessful (`E2E=0`)
  for the completed paired episode. This validates end-to-end VLA and Campaign
  execution with a real checkpoint, but does not demonstrate task improvement
  or a passing learning gate. The remote supervisor reached candidate round 5
  after four conclusive same-seed rejections; the SSH validation session then
  disconnected before a terminal `complete` record could be read.
  Therefore this evidence supports real Campaign execution and gate behavior,
  but does not claim a terminal Campaign completion.

### Recorded Validation (2026-09-15, in progress)

- SSH access and provider authentication were restored using the supplied key.
  The original Campaign reached `complete` after five conclusive same-seed
  rejections, each with 0/1 candidate successes. Its final outcome is
  `no_candidate_passed_primary_or_secondary`; no held-out candidate gate ran.
  Finalization exposed a status-reporting bug after clearing the candidate.
  The runner now captures validated statistics before that transition while
  retaining stale-candidate mutation checks. The regression failed before the
  fix; 43 gate, lifecycle, and CLI tests passed afterward. Re-entering the
  original Campaign returned `action=complete` and exited 0 without replaying
  episodes or changing gate decisions.
- The native-driver override above enabled JAX 0.5.3 / jaxlib 0.5.3 on physical
  GPU 1. The same pinned official checkpoint loaded successfully. Two actual
  inferences on a recorded real observation returned a 50-action horizon and
  acknowledged the frozen RNG and model version. Cold and warm end-to-end
  request times were 5.905 s and 0.142 s. Both responses had SHA-256
  `5ace9d369f55e37b8254dd023c14d3bc71d5e8dab1091045925f0cdf697111eb`.
- A separate 300-step Runtime deployment uses physical GPU 2 for Isaac and
  `ws://127.0.0.1:19001` for the model. Its exploratory seed 100001 completed
  with official `E2E=1` at step 90, 18 inferences, `status=valid`,
  `terminated=true`, and `truncated=false` (278.717 s including native startup).
  This baseline episode exercised no candidate recovery and is outside the
  formal schedule population.
- A new schedule preregisters 50 development and 20 held-out seeds before
  execution. Its manifest SHA-256 is
  `534d6941b5801585081396a0aa9b6a37888921ad3e45b865b0709d5f22a9f81d`.
  Development execution is running through the normal supervisor and queue.
  At the latest recorded check, 12/50 valid episodes had completed, all
  successful, with no infrastructure invalidation. A detached continuation
  waits for development to finish, then runs the regular Campaign with the
  same manifest and queue. A conditional baseline audit is also submitted:
  only if Campaign completes with `no_failures_to_optimize` and no candidates
  will it evaluate the frozen 20 held-out seeds. That separate report measures
  baseline generalization and cannot establish candidate improvement. Otherwise
  the normal lifecycle owns the candidate held-out gate. Neither a passing
  same-seed gate nor held-out improvement is claimed yet.
- Remote focused regressions: 37 passed across Stage session, real-model server
  contract, and Genie Sim Campaign preparation tests. This used the existing
  Runtime environment with the current test and serving files synchronized.

Operational evidence is under
`air-4090:/mnt/ssd_data/wangzexu/geniesim-g2-real-20260911/campaign-gpu-20260915`
and the sibling `evidence/` and `jobs/` directories. Local visual copies are in
`geniesim-visual-results/`; detailed status is in
`artifacts/geniesim-validation-20260915/STATUS.md`.
