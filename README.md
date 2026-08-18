# Zetta

<div align="center">
  <img src="teaser.png" alt="Zetta Overview" width="800"/>
</div>

Zetta is A Efficient Closed-Loop Embodied Harness for Self-Evolving Physical Intelligence. It evolves code-based runtime critics and recovery skills online while keeping the base policy frozen. Through three timescale-separated loops, Zetta provides action-frequency governance, rollout-level critic-recovery proposal, and validation-gated skill updates. Together with Z-Infra, a rollout infrastructure decoupling agent logic from heterogeneous execution resources, Zetta achieves state-of-the-art success on LIBERO-Pro and RoboCasa under our current rollout budget, reaching 90.8% and 93.6%, with a 11.1x inference speedup; success continues to scale with self-exploration experience; learned skills transfer zero-shot, and clear robotic ``Aha Moments'' emerge. These results show that closed-loop harness self-evolution opens a scaling path for reliable physical intelligence.


## TODO

- [√] **August 18, 2026:** Open-source Zetta with LIBERO and Robocasa.
- [ ] **August 19, 2026:** Open-source Z-Infra with LIBERO support.
- [ ] **August 20, 2026:** Add RoboCasa support.
- [ ] **August 27, 2026:** Add NVIDIA Cosmos model support.
- [ ] **September 3, 2026:** Add RoboTwin environment support.
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
  -> Reject and return to Stage 2, or Promote and complete
```


The runtime role boundaries are:

- **Cluster** groups complete failed trajectories using synchronized video, bounded telemetry, and failure segments.
- **Stage 1 / Diagnose** explains one observable causal failure mechanism and cannot write or execute recovery actions.
- **Stage 2 / Candidate writer** emits one frozen Critic-Recovery bundle whose parameters must satisfy the published tool schemas.
- **Critic** reads temporal evidence and may only propose a recovery.
- **Role1** accepts or rejects a Critic proposal and is the sole high-level decision authority during candidate execution.
- **Recovery actor** executes only an accepted, bounded recovery program; only the environment actor may write simulator actions.


## Repository Layout

| Path | Purpose |
|---|---|
| `rpent/evolution/` | Immutable manifests, queues, clustering, stages, gates, promotion, and supervision |
| `robots/libero/` | LIBERO environment, VLA client/server, privileged diagnostic telemetry, and recovery runtime |
| `robots/robocasa/` | Role1, Critic, recovery, tools, and rendering contract |
| `scripts/evolution/` | Campaign preparation, workers, capacity probes, plots, and deployment helpers |
| `deployment/systemd/` | Service templates; adjust the generic installation root for each host at installation time |
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

Python 3.10 through 3.12 is supported.

The sole source of dependency versions is `pyproject.toml`. The minimal test environment is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```


Pi0.5/SAM3 weights and optional GraspGen, Contact-GraspNet, GR00T, and FastWAM services are deployed separately. Do not commit provider configuration, API keys, model paths, or machine-specific absolute paths.

RoboCasa commonly uses three Python environments because the RPent control plane, simulator, and GR00T have different dependency constraints. Use the project environment as `EXPERIMENT_PYTHON`, install RoboCasa, robosuite, and MuJoCo in `ROBOCASA_PYTHON`, and install Isaac-GR00T plus its pinned dependencies in `GROOT_PYTHON`. All environments must be able to import the current repository (either an editable install or `PYTHONPATH` is sufficient). The RoboCasa/robosuite revisions must expose the isolated `mujoco.Renderer` path checked by `robots.robocasa.env_server.isolated_renderer_status()`.

## One-command Experiment Setup

The supported deployment entry point is `scripts/deployment/prepare_experiment.sh`. One invocation validates dependencies, assets, checkpoints, and the isolated renderer; starts the provider broker and Pi0.5 VLA for LIBERO, or the provider broker, GR00T, and the complete environment farm for RoboCasa; then freezes the campaign, starts the shared-queue worker, and generates run/stop scripts.

Copy the matching experiment template and provider template outside the repository, then fill in paths, GPU IDs, experiment parameters, and credentials. Both are trusted shell inputs and must have mode 0600. First prepare the shared private provider configuration for both experiment families:

```bash
PROVIDERS=/secure/path/providers.env
cp deployment/experiments/providers.env.example "$PROVIDERS"
chmod 600 "$PROVIDERS"
# Edit this file and set the provider URL, model, credentials, and concurrency.
```

### LIBERO Quick Start

```bash
CONFIG=/secure/path/libero.env
cp deployment/experiments/libero.env.example "$CONFIG"
chmod 600 "$CONFIG"
# Edit this file, including PROVIDER_ENV_FILE=/secure/path/providers.env.

# Validate prerequisites only; do not create a campaign or start services.
bash scripts/deployment/prepare_experiment.sh \
  --config "$CONFIG" --validate-only

# Validate, start all services and the worker, then run the campaign.
bash scripts/deployment/prepare_experiment.sh --config "$CONFIG" --run
```

### RoboCasa Quick Start

RoboCasa uses the same entry point, but has its own explicit configuration. Fill in these groups in `robocasa.env` before validation:

| Configuration | Required values |
|---|---|
| Storage and repository | `EXPERIMENT_ROOT`, node-local `EXPERIMENT_RUNTIME_ROOT`, shared `EXPERIMENT_QUEUE_ROOT`, `EXPERIMENT_REPO_ROOT` |
| Python environments | `EXPERIMENT_PYTHON`, `ROBOCASA_PYTHON`, `GROOT_PYTHON` |
| Simulator | `ROBOCASA_TASK`, `ROBOCASA_SPLIT`, `ROBOCASA_GPUS`, slot count, and base port |
| GR00T | `GROOT_SOURCE`, `GROOT_CHECKPOINT`, `GROOT_CHECKPOINT_SHA256`, GPU, and port |
| Experiment | rollout count, maximum action count, action chunk, generation count, and worker concurrency |
| Provider | Absolute `PROVIDER_ENV_FILE` path, for example `/secure/path/providers.env` |

Copy and edit the template, then calculate the digest of the exact checkpoint GR00T will load:

```bash
CONFIG=/secure/path/robocasa.env
cp deployment/experiments/robocasa.env.example "$CONFIG"
chmod 600 "$CONFIG"
# Edit all placeholders, including PROVIDER_ENV_FILE=/secure/path/providers.env.

set -a; source "$CONFIG"; set +a
PYTHONPATH="$EXPERIMENT_REPO_ROOT" "$GROOT_PYTHON" -c \
  'import sys; from robots.robocasa.groot_server import checkpoint_digest; print(checkpoint_digest(sys.argv[1]))' \
  "$GROOT_CHECKPOINT"
# Write the output to GROOT_CHECKPOINT_SHA256 in "$CONFIG".

bash scripts/deployment/prepare_experiment.sh \
  --config "$CONFIG" --validate-only
bash scripts/deployment/prepare_experiment.sh --config "$CONFIG" --run
```

For RoboCasa, this command starts and probes the provider broker, starts GR00T and the complete environment farm, resets the configured real task, checks two deterministic GR00T inferences, freezes the campaign, starts the worker, and finally runs the campaign. Any failed check terminates before campaign creation.

Without `--run`, it only prepares the services and worker without starting the campaign; the command prints the generated `run_experiment.sh` path. Status and stop operations use the same experiment configuration:

```bash
bash scripts/deployment/prepare_experiment.sh --config "$CONFIG" --status
bash scripts/deployment/prepare_experiment.sh --config "$CONFIG" --stop
```

`ROLLOUT_COUNT=50`, `HELDOUT_COUNT=20`, the official horizons, and `CAMPAIGN_MAX_STEPS=0` are the formal-run defaults. A quick infrastructure check may use one development episode, one held-out episode, one logical slot, and a positive `CAMPAIGN_MAX_STEPS`; this short test validates the execution chain but is not a benchmark result.

Before freezing a campaign, formal preparation sends real requests to every provider route, completes one Codex turn through the loopback broker, and performs real VLA inference. LIBERO calls Pi0.5 twice; RoboCasa resets the configured real task and requires two GR00T calls over the same observation and inference seed to produce the same action hash. Passed reports live under `EXPERIMENT_RUNTIME_ROOT/preflight` and are validated and reused when the configuration is unchanged. Therefore, the first cold start may be slow and consume a small amount of provider/model resources.
If an upstream only provides chat-completions, set `PROVIDER_PROBE_WIRE_API=native` in the private provider file; the subsequent Codex probe still verifies the broker's production Responses-compatible interface.

GR00T fails closed if the content it actually loads does not match `GROOT_CHECKPOINT_SHA256`. After changing any private configuration, provider configuration, or checkout commit, use a new `EXPERIMENT_ROOT`.

## Tests

First run the contract suite that requires no simulator/model:

```bash
python -m pytest -q \
  tests/test_evolution_protocol.py \
  tests/test_evolution_core.py \
  tests/test_libero_eval_horizon.py
```

Run all currently available tests:

```bash
python -m pytest -q
```

Tests that use an optional backend require the corresponding dependencies; hardware and service smoke tests live under `scripts/deployment/` and are not part of the minimal unit suite.

## Manual LIBERO Campaign (Advanced)

The one-command entry point above is the normal path. The following equivalent manual Goal-S task3 example is retained for per-component debugging. It uses the authoritative task language `Open the top layer of the drawer and put the bowl inside`, a single environment GPU, the official Goal horizon (300 policy actions + 10 waits), and the held-out test interval. Before running it, the external LIBERO assets, Pi0.5 VLA server, and API provider environment must already be prepared. The provider file must export `RPENT_API_PROVIDERS`; because it contains credentials, that file is not shown here.

```bash
set -euo pipefail

REPO_ROOT="$PWD"
PYTHON="$REPO_ROOT/.venv/bin/python"
CAMPAIGN_ROOT="$REPO_ROOT/runs/libero-goal-s-task3/g0000"
QUEUE_ROOT="$REPO_ROOT/runs/libero-goal-s-task3/queue"
VLA_ENDPOINT="http://127.0.0.1:18811"
export LIBERO_TYPE=pro

# The following variables are supplied by the prepared LIBERO/VLA host.
: "${LIBERO_ASSETS_ROOT_OVERRIDE:?Set LIBERO_ASSETS_ROOT_OVERRIDE}"
: "${RPENT_PROVIDER_ENV_FILE:?Set the external provider env file path}"
set -a
source "$RPENT_PROVIDER_ENV_FILE"
set +a
: "${RPENT_API_PROVIDERS:?The provider env file must export RPENT_API_PROVIDERS}"

mkdir -p "$(dirname "$CAMPAIGN_ROOT")" "$QUEUE_ROOT"

"$PYTHON" scripts/evolution/prepare_libero_campaign.py \
  --output-root "$CAMPAIGN_ROOT" \
  --campaign-id libero-goal-s-task3-g0000 \
  --repository-root "$REPO_ROOT" \
  --runtime-python "$PYTHON" \
  --code-commit "$(git rev-parse HEAD)" \
  --suite libero_goal_swap \
  --task-id 3 \
  --task libero_goal_swap/task3 \
  --task-language "Open the top layer of the drawer and put the bowl inside" \
  --master-seed 2026081103 \
  --rollout-count 50 \
  --fixed-heldout-seeds 1-20 \
  --heldout-mode test \
  --same-seed-pass-rate 0.5 \
  --same-seed-max-rounds 2 \
  --heldout-max-rounds 1 \
  --initial-logical-slots 1 \
  --maximum-logical-slots 1 \
  --continuous-logical-slots 1 \
  --maximum-total-candidate-rounds 15 \
  --maximum-target-clusters 2 \
  --vla-endpoint "$VLA_ENDPOINT" \
  --vla-gpu 0 \
  --environment-gpus 1 \
  --role1-require-visual-review \
  --allow-privileged-evidence

# The worker inherits RPENT_LIBERO_GPU and the provider configuration.
export RPENT_LIBERO_GPU=1
"$PYTHON" scripts/evolution/run_campaign.py \
  --manifest "$CAMPAIGN_ROOT/manifest.json" \
  --root "$CAMPAIGN_ROOT" \
  --queue-root "$QUEUE_ROOT" \
  --tool-catalog "$CAMPAIGN_ROOT/tool-catalog.json" \
  --workers libero-gpu1 \
  --model gpt-5.6-sol \
  --poll-s 10 \
  --max-generations 1 \
  --worker-command \
    "$PYTHON" -m rpent.evolution.cli worker \
    --queue-root "{queue_root}" --host "{host}" --poll-s 2 --concurrency 1
```

`run_campaign.py` starts the worker, ingests completed episodes, and continuously drives the recoverable state machine through Cluster, Diagnose, Stage2, Shadow Replay, Same-seed, and Held-out test in sequence. The process exits after the requested generation reaches a terminal state. To inspect the result read-only:

```bash
"$PYTHON" -m rpent.evolution.cli status \
  --root "$CAMPAIGN_ROOT" --queue-root "$QUEUE_ROOT"
```

For Long-T/Long-S, only replace `--suite`, `--task-id`, `--task`, the task language, and the environment GPU/VLA GPU allocation; the preparation script automatically sets the official 520-action/530-step horizon. If workers are managed manually, use `rpent-evolve run`, start one worker for each declared worker identity, and repeatedly call the idempotent `rpent-evolve optimize-step`.

Important campaign parameters include rollout count, logical environment slots, API concurrency, infrastructure retry count, visual read budget, maximum candidate rounds, same-seed pass rate, whether the regression gate is enabled, held-out mode, significance requirements, and minimum gain. The checked-out preparation script's `--help` output is the authoritative option list.

## Security and Reproducibility

- Store secrets only in environment variables or permission-restricted external files. Artifacts record only route names and irreversible fingerprints, not keys or service URLs.
- Keep development seeds disjoint from held-out seeds 1..20.
- Preserve manifests, tool catalogs, prompts, bundles, and episode records by content hash; do not edit a running campaign in place.
- Classify infrastructure failures separately from valid task failures; they do not count as zero scores and cannot become learning signals.
- Use privileged simulator state only through an explicitly authorized and auditable diagnostic feature contract; it must not become hidden task control.
- Treat held-out results as test-only unless `heldout_mode=validation` is preregistered before any episode runs.


## Acknowledgements

We gratefully acknowledge RPent for its foundational contributions. Z-Infra is coming soon.

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
