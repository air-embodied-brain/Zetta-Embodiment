# Zetta

[![Papers with Code: SOTA on RoboCasa365 Atomic-Seen](https://paperswithcode.co/api/v1/papers/2608.16590/leaderboard-badge.svg?eval=26154&live=1)](https://paperswithcode.co/api/v1/papers/2608.16590/leaderboard-badge-link?eval=26154)

<div align="center">
  <img src="teaser.png" alt="Zetta Overview" width="800"/>
</div>

Zetta is an efficient closed-loop embodied harness for self-evolving physical intelligence. It evolves code-based runtime critics and recovery skills online while keeping the base policy frozen, achieving state-of-the-art success on LIBERO-Pro (90.8%) and RoboCasa (93.6%) with an 11.1x inference speedup.

## TODO

- [√] **August 18, 2026:** Open-source Zetta with LIBERO and Robocasa.
- [√] **August 19, 2026:** Open-source Z-Infra with LIBERO support.
- [√] **August 20, 2026:** Add RoboCasa support.
- [√] **August 27, 2026:** Add NVIDIA Cosmos model support.
- [√] **September 3, 2026:** Add RoboTwin environment support.
- [ ] **September 10, 2026:** Add ManiSkill environment support.
- [ ] **September 17, 2026:** Add BEHAVIOR environment support.
- [ ] **Ongoing:** Expand model and environment coverage at an approximate cadence of one integration per week.

## Evolution Protocol

```text
50 development rollouts (never use seeds 1..20)
  -> Failure Cluster
  -> Stage 1 causal Diagnose
  -> Stage 2 Critic-Recovery Candidates
  -> Shadow Replay
  -> paired Same-seed Gate
  -> Held-out seeds 1..20
     -> validation mode: Reject (refine if budget remains), or Promote
     -> test mode: Record an unbiased final report without feeding selection
```

Runtime role boundaries:

- **Cluster** groups complete failed trajectories using synchronized video, bounded telemetry, and failure segments.
- **Stage 1 / Diagnose** explains one observable causal failure mechanism and cannot write or execute recovery actions.
- **Stage 2 / Candidate writer** emits one frozen Critic-Recovery bundle whose parameters must satisfy the published tool schemas.
- **Critic** reads temporal evidence and may only propose a recovery.
- **Role1** accepts or rejects a Critic proposal and is the sole high-level decision authority during candidate execution.
- **Recovery actor** executes only an accepted, bounded recovery program; only the environment actor may write simulator actions.

## Repository Layout

| Path | Purpose |
|---|---|
| `zetta/evolution/` | Immutable manifests, queues, clustering, stages, gates, promotion, and supervision |
| `rollout_runtime/` | The Rollout Runtime: Gateway, EnvWorker/RolloutWorker groups, and backends for LIBERO/RoboCasa/ManiSkill/RoboTwin |
| `robots/libero/`, `robots/robocasa/`, `robots/robotwin/` | Env clients, Role1/Critic/Recovery, tools, and rendering contracts |
| `scripts/evolution/` | Campaign preparation, workers, capacity probes, and plots |
| `scripts/deployment/` | Service start/stop, VLA env install, and Docker build helpers |
| `tests/` | Unit/contract tests; the minimal set requires no simulator or model |

A campaign root normally contains:

```text
manifest.json             frozen task, schedule, runtime, and gate contract
preregistration.json      seed and evaluation commitments
task-contract.json        authoritative language goal, when supplied
tool-catalog.json         exact tool schemas available to candidates
state.json                current lifecycle phase
episodes/                 canonical rollout and paired-gate records
failure-clusters/         Cluster inputs, visual indices, and outputs
diagnoses/                Stage 1 inputs, outputs, and audit trail
proposals/                Stage 2 candidate attempts
shadow-replay/            offline trigger and false-positive evidence
gates/                    same-seed, regression, and held-out decisions
promoted/                 verified candidate bundle and hashes
```

Videos, model weights, simulator assets, API credentials, and host runtime files remain outside Git.

## Installation

Python 3.10 through 3.12 is supported. The minimal test environment (no simulator/model required):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

### VLA runtime environment (LIBERO-Pro or RoboCasa)

`scripts/deployment/install_vla_env.sh` builds a venv for one of two tracks;
the tracks cannot share a venv state (incompatible `robosuite` versions). See
the [VLA runtime setup guide](scripts/deployment/VLA_ENV_SETUP.md) for the full
installation notes, compatibility fixes, and known limitations.

**LIBERO-Pro + Pi0.5**

```bash
export REPO_ROOT=/abs/path/to/Zetta-Embodiment
export VENV_ROOT=/abs/path/to/venvs/vla-env
bash scripts/deployment/install_vla_env.sh --track libero-pro
```

`install_vla_env.sh` downloads the LIBERO-Pro scene/object assets automatically. To fetch or refresh them manually:

```bash
"$VENV_ROOT/bin/liberopro-download-assets" --skip-existing
```

**RoboCasa + GR00T** (needs a source checkout with `robocasa/`, `robosuite/`, `Isaac-GR00T/`)

```bash
export REPO_ROOT=/abs/path/to/Zetta-Embodiment
export VENV_ROOT=/abs/path/to/venvs/vla-env
export ROBOCASA_SRC_ROOT=/abs/path/to/robocasa-source-checkout
bash scripts/deployment/install_vla_env.sh --track robocasa
```

RoboCasa also needs its kitchen assets (~10GB); see the [RoboCasa installation guide](https://robocasa.ai/docs/build/html/introduction/installation.html) for details:

```bash
python -m robocasa.scripts.setup_macros              # set up system variables
python -m robocasa.scripts.download_kitchen_assets   # downloads ~10GB of kitchen assets
```

**RoboTwin 2.0 + Pi0.5**

RoboTwin is SAPIEN-based and cannot share a venv with either track above. Use the
upstream RLinf image rather than building one:

```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin.git -b RLinf_support
# then download ~15GB of assets into its assets/ directory

docker run --rm --gpus all --shm-size 32g \
  -e NVIDIA_DISABLE_REQUIRE=1 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -e LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64 \
  --device /dev/dri \
  -v "$ROBOTWIN_ROOT":/workspace/RoboTwin \
  -v "$REPO_ROOT":/workspace/Zetta \
  --entrypoint /bin/bash rlinf/rlinf:agentic-rlinf0.4-robotwin
```

The image ships three venvs under `/opt/venv/`; this repository uses `openpi`.
The three environment variables let a CUDA 12.8 image run on a 12.4 driver:
`NVIDIA_DISABLE_REQUIRE` skips the image's `cuda>=12.8` label check,
`NVIDIA_DRIVER_CAPABILITIES` drops the `display` capability that needs a
`/dev/nvidia-modeset` a headless host does not have, and `LD_LIBRARY_PATH` puts
the host driver's `libcuda` ahead of the image's forward-compatibility libraries
-- CUDA forward compatibility is datacenter-only and fails on GeForce with
`Error 804`. `assets_path` must point at the RoboTwin repository root, not its
`assets/` subdirectory.

Prebuilt Docker images, built from `scripts/deployment/Dockerfile.vla-env`, are available for both tracks so you can skip the manual venv build:

- **LIBERO-Pro** image: preconfigured with the LIBERO-Pro simulator stack and Pi0.5 dependencies. [百度网盘](https://pan.baidu.com/s/1HW7AstjCLE_BTScRFOQT2g?pwd=qv7c)
- **RoboCasa** image: preconfigured with the RoboCasa/robosuite/Isaac-GR00T stack. [百度网盘](https://pan.baidu.com/s/1Zyg2i_3tMp249PPLK6_JAw?pwd=ztcp)

## Runtime Rollout System

Campaigns drive rollouts through the Rollout Runtime rather than talking to a per-task VLA/env server directly. Start it once per host, then point campaign preparation at it:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl \
python -m rollout_runtime.cli serve \
  --config <preset-name-or-yaml-path> \
  --host 127.0.0.1 \
  --port 18730 \
  --launch ray
```

`--launch ray` is required for real hardware runs: it puts the EnvWorker (simulator) and RolloutWorker (model inference) groups into separate processes. `--launch local` (the default) keeps them in one process and is only for local/CI smoke tests. Presets live under `rollout_runtime/config/presets/`; copy and edit one for machine-specific GPU/checkpoint paths rather than committing a new preset with real paths.

## Preparing and Running a Campaign

LIBERO, RoboCasa and RoboTwin all use the same two-step flow: `prepare_*_campaign.py` freezes a `manifest.json`/`tool-catalog.json` against a running runtime, then `run_campaign.py` drives the campaign state machine to completion.

RoboTwin differs in two ways that its manifest records explicitly. Its seeds come
from RoboTwin's curated per-task success-seed list rather than a dense integer
range -- the environment picks its scene from the seed outright, so a held-out
block drawn from unsolvable scenes would not be a gate -- and its evidence is
chunk-granular rather than per-step, because it is the only `final_only` family.

The RoboTwin campaign loop runs end to end on hardware: rollouts, Cluster,
Diagnose, Stage 2, and the paired same-seed gate, over five candidate rounds to a
`complete` campaign.

```bash
# 1. Start the runtime (see above), then prepare the campaign against it.
python scripts/evolution/prepare_libero_campaign.py \
  --output-root runs/libero-goal-s-task3/g0000 \
  --campaign-id libero-goal-s-task3-g0000 \
  --repository-root "$PWD" \
  --runtime-python .venv/bin/python \
  --code-commit "$(git rev-parse HEAD)" \
  --suite libero_goal_swap \
  --task-id 3 \
  --task libero_goal_swap/task3 \
  --task-language "Open the top layer of the drawer and put the bowl inside" \
  --master-seed 2026081103 \
  --rollout-count 50 \
  --fixed-heldout-seeds 1-20 \
  --heldout-mode test \
  --runtime-url http://127.0.0.1:18730 \
  --runtime-policy-id <policy-id>

# `--heldout-mode test` keeps the fixed block report-only. To make a held-out
# failure reject the candidate, preregister `--heldout-mode validation` instead.

# 2. Run it: this starts the worker, ingests episodes, and drives
#    Cluster -> Diagnose -> Stage2 -> Shadow Replay -> Same-seed -> Held-out.
python scripts/evolution/run_campaign.py \
  --manifest runs/libero-goal-s-task3/g0000/manifest.json \
  --root runs/libero-goal-s-task3/g0000 \
  --queue-root runs/libero-goal-s-task3/queue \
  --tool-catalog runs/libero-goal-s-task3/g0000/tool-catalog.json \
  --workers libero-worker-0 \
  --model gpt-5.6-sol \
  --max-generations 1
```

`--worker-command` (optional, must be the final flag) overrides how each worker process is launched, substituting `{queue_root}`/`{host}` into a custom command template; omit it to use the default `zetta.evolution.cli worker` invocation.

`--runtime-url`/`--runtime-policy-id` replace the legacy `--vla-endpoint`/`--environment-gpus` flags; RoboCasa campaigns always go through the runtime (there is no direct-connect path). `ROLLOUT_COUNT=50`, `HELDOUT_COUNT=20`, and the official horizons are the formal-run defaults; a quick infrastructure check may use 1/1/1 with a positive `CAMPAIGN_MAX_STEPS`, but that is not a benchmark result. Each preparation script's `--help` output is the authoritative option list.

To inspect a campaign read-only:

```bash
python -m zetta.evolution.cli status --root runs/libero-goal-s-task3/g0000 \
  --queue-root runs/libero-goal-s-task3/queue
```

## Tests

```bash
# Contract suite, no simulator/model required:
python -m pytest -q tests/test_evolution_protocol.py tests/test_evolution_core.py tests/test_libero_eval_horizon.py

# Everything currently available:
python -m pytest -q
```

Optional-backend tests require the corresponding dependencies; hardware/service smoke tests live under `scripts/deployment/` and are not part of the minimal unit suite.

## Security and Reproducibility

- Store secrets only in environment variables or permission-restricted external files. Artifacts record only route names and irreversible fingerprints, not keys or service URLs.
- Keep development seeds disjoint from held-out seeds 1..20.
- Preserve manifests, tool catalogs, prompts, bundles, and episode records by content hash; do not edit a running campaign in place.
- Classify infrastructure failures separately from valid task failures; they do not count as zero scores and cannot become learning signals.
- Use privileged simulator state only through an explicitly authorized and auditable diagnostic feature contract; it must not become hidden task control.
- Treat held-out results as test-only unless `heldout_mode=validation` is preregistered before any episode runs.

## Acknowledgements

We acknowledge [RLinf](https://github.com/RLinf/RLinf)  for its prior work. See [Third-Party Notices](./THIRD_PARTY_NOTICES.md) for details.

## Citation

If you find Zetta useful in your research, please cite our paper:

```bibtex
@misc{ding2026zettazetaefficientclosedloop,
      title={Zetta $\zeta$: An Efficient Closed-Loop Embodied Harness for Self-Evolving Physical Intelligence}, 
      author={Xin Ding and Liang Mi and Mingzhe Huang and Zixuan Wang and Chao Zhang and Zixu Hao and Fu Chen and Xiangyu Li and Yikai Zheng and Yaoyu Guo and Weijun Wang and Kun Li and Hao Wu and Yunxin Liu and Ting Cao},
      year={2026},
      eprint={2608.16590},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2608.16590}, 
}
```
