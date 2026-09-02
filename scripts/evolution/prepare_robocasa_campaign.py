#!/usr/bin/env python3
# Copyright (c) 2026 Zetta Contributors
"""Create one immutable, secret-free RoboCasa evolution preregistration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from robots.robocasa.role1_agent import ROLE1_SYSTEM_CONTRACT  # noqa: E402
from robots.robocasa.tool_bindings import binding_for_task  # noqa: E402
from robots.robocasa.tool_catalog import (  # noqa: E402
    DEFAULT_ROBOCASA_TOOL_CATALOG,
)
from zetta.evolution.jsonio import (  # noqa: E402
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    read_json,
)
from zetta.evolution.models import (  # noqa: E402
    CampaignManifest,
    CandidateBundle,
    SafetyLayerConfig,
)
from zetta.evolution.schedule import preregister_seed_schedule  # noqa: E402
from zetta.evolution.stages import (  # noqa: E402
    CLUSTER_SYSTEM_PROMPT,
    DIAGNOSIS_SYSTEM_PROMPT,
    PROPOSAL_SYSTEM_PROMPT,
)


def _rollout_command(args: argparse.Namespace) -> list[str]:
    script = (
        Path(args.repository_root).resolve() / "robots" / "robocasa" / "run_rollout.py"
    )
    command = [
        str(Path(args.runtime_python).resolve()),
        str(script),
        # The env slot lease still substitutes ``{env_endpoint}`` (see
        # ``zetta.evolution.queue.SubprocessRolloutExecutor``), but after the
        # Runtime v3 migration the leased address is the *shared rollout-runtime
        # serve* endpoint, not a per-slot RoboCasa HTTP server: env slots are
        # owned by the runtime's ``EnvPool`` and GR00T inference happens inside
        # the runtime, so there is no second VLA endpoint to pass.
        "--runtime-url",
        "{env_endpoint}",
        "--policy-id",
        args.policy_id,
        "--env-pool-size",
        str(args.env_pool_size),
        "--task",
        "{task}",
        "--split",
        args.split,
        "--seed",
        "{seed}",
        "--policy-rng",
        "{policy_rng}",
        "--logical-id",
        "{logical_id}",
        "--attempt-index",
        "{attempt_index}",
        "--generation",
        "{generation}",
        "--bundle",
        "{bundle_file}",
        "--bundle-sha256",
        "{bundle_sha256}",
        "--baseline-mode",
        "{baseline_mode}",
        "--safety-layer",
        "interface_contract_v1",
        "--output-dir",
        "{output_dir}",
        "--result-file",
        "{result_file}",
        "--max-actions",
        str(args.max_actions),
        "--actions-per-chunk",
        str(args.actions_per_chunk),
        "--role1-planner",
        args.role1_planner,
        "--role1-model",
        args.role1_model,
        "--reasoning-effort",
        args.reasoning_effort,
        "--role1-max-tokens",
        str(args.role1_max_tokens),
        "--role1-timeout-s",
        str(args.role1_timeout_s),
        "--role1-heartbeat-s",
        str(args.role1_heartbeat_s),
        "--role1-max-turns",
        str(args.role1_max_turns),
        "--allow-privileged-tools",
    ]
    # No ``--runtime-token``: the frozen command goes into the manifest, and a
    # bearer token must never be recorded in a campaign artifact. ``run_rollout``
    # reads ``ZETTA_RUNTIME_TOKEN`` from the inherited worker environment
    # instead (README's "secrets only in env vars or external files").
    return command


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    """Materialize a frozen manifest, prompt contract, catalog, and schedule."""

    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=False)
    repository_root = Path(args.repository_root).resolve()
    runtime_python = Path(args.runtime_python).resolve()
    if not (repository_root / "robots" / "robocasa" / "run_rollout.py").is_file():
        raise ValueError("repository root has no RoboCasa rollout entrypoint")
    if not runtime_python.is_file():
        raise ValueError("runtime Python does not exist")

    prompt_contract = {
        "agent_abstraction": {
            "online_enabled": ["role1"],
            "online_disabled": ["role0", "role2"],
            "offline_campaign_control": ["cluster", "diagnoser", "evolver"],
            "legacy_schema_labels": ["stage1", "stage2"],
        },
        "formal_agent": {
            "model": args.agent_model,
            "reasoning_effort": args.reasoning_effort,
        },
        "role1": ROLE1_SYSTEM_CONTRACT,
        "cluster": CLUSTER_SYSTEM_PROMPT,
        "stage1": DIAGNOSIS_SYSTEM_PROMPT,
        "stage2": PROPOSAL_SYSTEM_PROMPT,
    }
    prompt_path = output / "prompt-contract.json"
    atomic_write_json(prompt_path, prompt_contract, overwrite=False)

    binding = binding_for_task(args.task)
    catalog = {
        **DEFAULT_ROBOCASA_TOOL_CATALOG.public_dict(),
        "task": args.task,
        "binding_digest": binding.digest,
        "task_binding": binding.public_dict(),
    }
    catalog_path = output / "tool-catalog.json"
    atomic_write_json(catalog_path, catalog, overwrite=False)

    rollout, heldout, policy_rng = preregister_seed_schedule(
        master_seed=args.master_seed,
        task=args.task,
        rollout_count=args.rollout_count,
        heldout_count=args.heldout_count,
        population=range(args.population_size),
    )
    parent_sha256 = None
    bundle_files_by_sha: dict[str, str] = {}
    if args.parent_bundle is not None:
        parent_path = Path(args.parent_bundle).resolve()
        parent = CandidateBundle.from_dict(read_json(parent_path))
        parent_sha256 = parent.sha256
        if parent.generation >= args.generation:
            raise ValueError(
                "parent bundle generation must precede campaign generation"
            )
        bundle_files_by_sha[parent_sha256] = str(parent_path)
    elif args.generation != 0:
        raise ValueError("nonzero generation requires --parent-bundle")

    runtime_command = _rollout_command(args)
    runtime = {
        "evolution_policy": {
            "same_seed_pass_rate": 0.5,
            "max_candidate_rounds_per_cluster": 5,
            "maximum_target_clusters": 2,
        },
        "rollout_command": runtime_command,
        "same_seed_gate_rollout_command": runtime_command,
        # Gen0 still executes the Critic engine, but its frozen rule set is
        # empty and therefore no online Role1 call is possible.  Do not make
        # pure-VLA throughput depend on provider admission.  Once a promoted
        # parent bundle is active, Role1 may be invoked by a Critic proposal.
        "rollout_requires_api": parent_sha256 is not None,
        "candidate_rollout_requires_api": True,
        "reuse_rollout_parent_evidence": True,
        "heldout_gate_kind": "heldout_20" if args.heldout_count == 20 else "heldout",
        "rollout_requires_environment_slot": True,
        "bundle_files_by_sha": bundle_files_by_sha,
        "agent_model": args.agent_model,
        "reasoning_effort": args.reasoning_effort,
    }
    manifest = CampaignManifest(
        campaign_id=args.campaign_id,
        environment="robocasa",
        task=args.task,
        generation=args.generation,
        code_commit=args.code_commit,
        prompt_sha256=file_sha256(prompt_path),
        model=args.agent_model,
        tool_catalog_sha256=file_sha256(catalog_path),
        rollout_seeds=rollout,
        heldout_seeds=heldout,
        policy_rng_by_seed=policy_rng,
        parent_bundle_sha256=parent_sha256,
        baseline_mode="active_bundle" if parent_sha256 else "strict_pure_vla",
        active_bundle_sha256=parent_sha256,
        safety_layer=SafetyLayerConfig(),
        expected_rollouts=args.rollout_count,
        expected_heldout=args.heldout_count,
        initial_logical_slots=args.initial_logical_slots,
        maximum_logical_slots=args.maximum_logical_slots,
        continuous_logical_slots=args.continuous_logical_slots,
        maximum_api_concurrency=args.maximum_api_concurrency,
        episode_timeout_s=args.episode_timeout_s,
        no_progress_timeout_s=args.no_progress_timeout_s,
        target_valid_episodes_per_hour=args.target_valid_episodes_per_hour,
        max_infrastructure_attempts=args.max_infrastructure_attempts,
        reasoning_effort=args.reasoning_effort,
        runtime=runtime,
    )
    manifest_path = output / "manifest.json"
    atomic_write_json(manifest_path, manifest.as_dict(), overwrite=False)
    preregistration = {
        "schema_version": 1,
        "campaign_id": manifest.campaign_id,
        "manifest_sha256": manifest.sha256,
        "manifest_file_sha256": file_sha256(manifest_path),
        "code_commit": args.code_commit,
        "model": manifest.model,
        "reasoning_effort": manifest.reasoning_effort,
        "prompt_sha256": manifest.prompt_sha256,
        "tool_catalog_sha256": manifest.tool_catalog_sha256,
        "tool_catalog_digest": DEFAULT_ROBOCASA_TOOL_CATALOG.digest,
        "task_binding_digest": binding.digest,
        "schedule_sha256": canonical_sha256(
            {
                "rollout_seeds": rollout,
                "heldout_seeds": heldout,
                "policy_rng_by_seed": policy_rng,
            }
        ),
        "rollout_seeds": rollout,
        "heldout_seeds": heldout,
        "policy_rng_by_seed": policy_rng,
        "success_criterion": "authoritative task success only",
        "baseline_mode": manifest.baseline_mode,
        "active_bundle_sha256": manifest.active_bundle_sha256,
        "safety_layer": manifest.safety_layer.as_dict(),
        "infrastructure_invalid_scored": False,
        "reuse_rollout_parent_evidence": True,
    }
    atomic_write_json(output / "preregistration.json", preregistration, overwrite=False)
    return preregistration


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--task", default="SlideDishwasherRack")
    parser.add_argument("--split", default="target")
    parser.add_argument("--generation", type=int, default=0)
    parser.add_argument("--parent-bundle", type=Path)
    parser.add_argument("--master-seed", type=int, required=True)
    parser.add_argument("--rollout-count", type=int, default=50)
    parser.add_argument("--heldout-count", type=int, default=50)
    parser.add_argument("--population-size", type=int, default=100000)
    parser.add_argument("--initial-logical-slots", type=int, default=8)
    parser.add_argument("--maximum-logical-slots", type=int, default=50)
    parser.add_argument("--continuous-logical-slots", type=int, default=4)
    parser.add_argument("--maximum-api-concurrency", type=int, default=8)
    parser.add_argument("--episode-timeout-s", type=int, default=2700)
    parser.add_argument("--no-progress-timeout-s", type=int, default=180)
    parser.add_argument("--target-valid-episodes-per-hour", type=float, default=25.0)
    parser.add_argument("--max-infrastructure-attempts", type=int, default=8)
    parser.add_argument(
        "--policy-id",
        default="groot",
        help="Policy id served by the shared runtime's RolloutWorker.",
    )
    parser.add_argument(
        "--env-pool-size",
        type=int,
        default=1,
        help=(
            "Initial env slots per env spec inside the shared runtime. It enters "
            "the pool digest, so every rollout of this campaign uses this value."
        ),
    )
    parser.add_argument("--max-actions", type=int, default=1000)
    parser.add_argument("--actions-per-chunk", type=int, default=16)
    parser.add_argument("--role1-planner", choices=("api", "codex"), default="api")
    parser.add_argument("--agent-model", default="gpt-5.6-sol")
    parser.add_argument("--role1-model", default="openai:gpt-5.6-sol")
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high", "xhigh"),
        default="high",
    )
    parser.add_argument("--role1-max-tokens", type=int, default=4096)
    parser.add_argument("--role1-timeout-s", type=int, default=900)
    parser.add_argument("--role1-heartbeat-s", type=float, default=15.0)
    parser.add_argument("--role1-max-turns", type=int, default=2)
    args = parser.parse_args()
    report = prepare(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
