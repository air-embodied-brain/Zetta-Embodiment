#!/usr/bin/env python3
# Copyright (c) 2026 Zetta Contributors
"""Create one immutable, secret-free RoboTwin evolution preregistration.

The one place this differs materially from the LIBERO and RoboCasa preparers is
the **seed population**, and the difference is forced.

RoboTwin ships a curated per-task list of seeds on which the task is known to be
solvable (``rlinf/envs/robotwin/seeds/eval_seeds.json``), and the environment
selects its scene from that seed outright. Drawing a campaign's seeds from
``range(population_size)`` the way the other preparers do would produce mostly
scenes the task cannot be completed in at all, and a held-out gate whose success
rate is structurally near zero is not a gate -- it is noise.

So the population is the task's ``success_seeds``, the held-out block is its
first ``--heldout-count`` entries, and the **provenance of both is written into
the manifest and the preregistration**. The protocol's shape is unchanged --
a fixed, disjoint, pre-committed held-out block frozen before any development
rollout -- but the values are no longer the literal ``1..20``, and a reader must
be able to see why without reading this file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from robots.robotwin.critic_runtime import describe_dwell_semantics  # noqa: E402
from robots.robotwin.role1_actor import ROLE1_SYSTEM_CONTRACT  # noqa: E402
from robots.robotwin.tool_bindings import binding_for_task  # noqa: E402
from robots.robotwin.tool_catalog import (  # noqa: E402
    DEFAULT_ROBOTWIN_TOOL_CATALOG,
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

ENVIRONMENT = "robotwin"
"""The manifest's environment label."""


def load_success_seeds(seeds_path: Path, task: str) -> tuple[int, ...]:
    """Read the curated solvable seeds for one RoboTwin task.

    Args:
        seeds_path: The RLinf ``eval_seeds.json`` (or a file of the same shape).
        task: The RoboTwin task name.

    Returns:
        The task's success seeds, de-duplicated, in file order.

    Raises:
        ValueError: The file has no entry for the task, or the entry carries no
            usable seeds.
    """
    payload = json.loads(seeds_path.read_text(encoding="utf-8"))
    entry = payload.get(task)
    if not isinstance(entry, dict):
        raise ValueError(
            f"{seeds_path} has no entry for RoboTwin task {task!r}; "
            f"known tasks: {sorted(payload)[:8]}..."
        )
    seeds = entry.get("success_seeds")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError(f"{seeds_path} lists no success_seeds for task {task!r}")
    return tuple(dict.fromkeys(int(value) for value in seeds))


def _rollout_command(args: argparse.Namespace) -> list[str]:
    """Build the frozen per-episode rollout command.

    Args:
        args: Parsed CLI arguments.

    Returns:
        The command, with the queue's substitution placeholders left in place.
    """
    script = (
        Path(args.repository_root).resolve() / "robots" / "robotwin" / "run_rollout.py"
    )
    command = [
        # `absolute()`, never `resolve()`: a venv's `bin/python` is a symlink to
        # the base interpreter, and a venv is only active when it is invoked
        # through its own path -- `sys.prefix` is derived from where
        # `sys.executable` lives. Resolving the symlink freezes the base
        # interpreter into the manifest, and every rollout then dies at
        # `import numpy` with the venv's site-packages nowhere on the path.
        str(Path(args.runtime_python).absolute()),
        str(script),
        # The runtime URL is passed directly rather than through an environment
        # slot lease. The slot broker exists to arbitrate exclusive access to a
        # per-slot env server, and its health contract is that server's
        # single-writer protocol (`write_protocol.phase == "FREE"`), which the
        # Rollout Runtime's serve does not speak. Under the Runtime the env
        # slots are owned by the Gateway's own `EnvPool`, so leasing the shared
        # serve endpoint at campaign level would double-book a resource the
        # Gateway already arbitrates. Concurrency stays bounded by the
        # manifest's logical slots and the preset's `default_pool_size`.
        "--runtime-url",
        args.runtime_url,
        "--policy-id",
        args.policy_id,
        "--task",
        "{task}",
        "--assets-path",
        args.assets_path,
        "--embodiment",
        *args.embodiment,
        "--planner-backend",
        args.planner_backend,
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
        # A paired gate rejects a command that cannot consume the frozen bundle
        # (gate_runner._command_template); Gen0 is substituted with "none".
        "--bundle",
        "{bundle_file}",
        "--bundle-sha256",
        "{bundle_sha256}",
        "--output-dir",
        "{output_dir}",
        "--result-file",
        "{result_file}",
        "--env-max-steps",
        str(args.env_max_steps),
        "--execute-horizon",
        str(args.execute_horizon),
        "--env-pool-size",
        str(args.env_pool_size),
        # Stage 1 refuses a diagnosis carrying fewer than three visual evidence
        # items, and those come from per-camera video that only exists when the
        # episode captures frames. A campaign whose episodes cannot be diagnosed
        # is not a campaign, so this is not optional here.
        "--capture-frames",
        # Role1 is only reachable once a parent bundle supplies critic rules, so
        # Gen0 never spends a provider call even with this configured.
        "--role1-planner",
        args.role1_planner,
        "--role1-model",
        args.role1_model,
        "--reasoning-effort",
        args.reasoning_effort,
    ]
    # No ``--runtime-token``: the frozen command goes into the manifest, and a
    # bearer token must never be recorded in a campaign artifact. ``run_rollout``
    # reads it from the inherited worker environment instead.
    return command


def _seed_provenance(
    *, seeds_path: Path, task: str, population: tuple[int, ...], heldout_count: int
) -> dict[str, Any]:
    """Describe where the campaign's seeds came from.

    Recorded in both the manifest and the preregistration: the held-out block is
    no longer the literal ``1..20`` that the other environments use, and a
    reviewer must be able to see the substitution without reading the preparer.

    Args:
        seeds_path: The curated seed file.
        task: The RoboTwin task.
        population: The seeds drawn from.
        heldout_count: Size of the held-out block.

    Returns:
        A JSON-friendly provenance record.
    """
    return {
        "schema_version": 1,
        "population_kind": "robotwin_curated_success_seeds",
        "source_file": str(seeds_path),
        "source_file_sha256": file_sha256(seeds_path),
        "task": task,
        "population_size": len(population),
        "heldout_selection": f"first {heldout_count} entries in file order",
        "rationale": (
            "RoboTwin selects its scene from the seed outright, and only the "
            "curated success seeds are known to be solvable. Drawing from a "
            "dense integer range would make the held-out gate structurally "
            "near-zero and therefore not a gate. The protocol shape is "
            "unchanged: a fixed, disjoint, pre-committed held-out block."
        ),
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    """Materialize a frozen manifest, prompt contract, catalog and schedule.

    Args:
        args: Parsed CLI arguments.

    Returns:
        The preregistration payload.

    Raises:
        ValueError: The repository, runtime Python, or generation/parent
            combination is invalid.
    """
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=False)
    repository_root = Path(args.repository_root).resolve()
    runtime_python = Path(args.runtime_python).resolve()
    if not (repository_root / "robots" / "robotwin" / "run_rollout.py").is_file():
        raise ValueError("repository root has no RoboTwin rollout entrypoint")
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
        **DEFAULT_ROBOTWIN_TOOL_CATALOG.public_dict(),
        "task": args.task,
        "binding_digest": binding.digest,
        "task_binding": binding.public_dict(),
        # The arm requirement is the part of this catalog that is expensive to
        # get wrong, so it is surfaced rather than left implicit in the tools.
        "arm_scoped_tools": list(binding.arm_scoped_tool_names),
    }
    catalog_path = output / "tool-catalog.json"
    atomic_write_json(catalog_path, catalog, overwrite=False)

    seeds_path = Path(args.seeds_path).resolve()
    population = load_success_seeds(seeds_path, args.task)
    heldout_seeds = population[: args.heldout_count]
    if len(heldout_seeds) < args.heldout_count:
        raise ValueError(
            f"task {args.task!r} has only {len(population)} curated success "
            f"seeds; need at least {args.heldout_count} for the held-out block"
        )
    rollout, heldout, policy_rng = preregister_seed_schedule(
        master_seed=args.master_seed,
        task=args.task,
        rollout_count=args.rollout_count,
        heldout_count=args.heldout_count,
        population=population,
        heldout_seeds=heldout_seeds,
    )
    seed_provenance = _seed_provenance(
        seeds_path=seeds_path,
        task=args.task,
        population=population,
        heldout_count=args.heldout_count,
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
        # Gen0 runs the Critic engine with an empty frozen rule set, so no
        # online Role1 call is possible; pure-VLA throughput must not depend on
        # provider admission.
        "rollout_requires_api": parent_sha256 is not None,
        "candidate_rollout_requires_api": True,
        "reuse_rollout_parent_evidence": True,
        "heldout_gate_kind": "heldout_20" if args.heldout_count == 20 else "heldout",
        # See `_rollout_command`: the Gateway owns env-slot arbitration for this
        # family, so the campaign does not also lease one.
        "rollout_requires_environment_slot": False,
        "bundle_files_by_sha": bundle_files_by_sha,
        "agent_model": args.agent_model,
        "reasoning_effort": args.reasoning_effort,
        "seed_provenance": seed_provenance,
        # RoboTwin is the only final_only family: a diagnosis produced here is
        # chunk-granular and must not be compared like-for-like with a
        # per-step family's.
        "evidence_granularity": "chunk",
        "dwell_semantics": describe_dwell_semantics(
            execute_horizon=args.execute_horizon
        ),
    }
    manifest = CampaignManifest(
        campaign_id=args.campaign_id,
        environment=ENVIRONMENT,
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
        "tool_catalog_digest": DEFAULT_ROBOTWIN_TOOL_CATALOG.digest,
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
        "seed_provenance": seed_provenance,
        "success_criterion": "authoritative task success only",
        "baseline_mode": manifest.baseline_mode,
        "active_bundle_sha256": manifest.active_bundle_sha256,
        "safety_layer": manifest.safety_layer.as_dict(),
        "infrastructure_invalid_scored": False,
        "reuse_rollout_parent_evidence": True,
        "evidence_granularity": "chunk",
    }
    atomic_write_json(output / "preregistration.json", preregistration, overwrite=False)
    return preregistration


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Freeze one RoboTwin evolution campaign"
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--runtime-python", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--task", default="adjust_bottle")
    parser.add_argument(
        "--seeds-path",
        required=True,
        help="RLinf eval_seeds.json; the curated solvable seeds for the task",
    )
    parser.add_argument(
        "--assets-path",
        required=True,
        help="RoboTwin repository root (not its assets/ subdirectory)",
    )
    parser.add_argument("--embodiment", nargs="+", default=["aloha-agilex"])
    parser.add_argument(
        "--planner-backend", default="mplib", choices=["mplib", "curobo"]
    )
    parser.add_argument("--policy-id", required=True)
    parser.add_argument(
        "--runtime-url",
        required=True,
        help="Shared Rollout Runtime serve endpoint, e.g. http://127.0.0.1:18730",
    )
    parser.add_argument("--master-seed", type=int, required=True)
    parser.add_argument("--rollout-count", type=int, default=50)
    parser.add_argument("--heldout-count", type=int, default=20)
    parser.add_argument("--generation", type=int, default=0)
    parser.add_argument("--parent-bundle", default=None)
    parser.add_argument("--env-max-steps", type=int, default=200)
    parser.add_argument("--execute-horizon", type=int, default=25)
    parser.add_argument("--env-pool-size", type=int, default=4)
    parser.add_argument("--agent-model", default="gpt-5.6-sol")
    parser.add_argument("--role1-model", default="gpt-5.6-sol")
    parser.add_argument("--role1-planner", default="codex", choices=["api", "codex"])
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--initial-logical-slots", type=int, default=4)
    parser.add_argument("--maximum-logical-slots", type=int, default=4)
    parser.add_argument("--continuous-logical-slots", type=int, default=4)
    parser.add_argument("--maximum-api-concurrency", type=int, default=4)
    parser.add_argument("--episode-timeout-s", type=float, default=1800.0)
    parser.add_argument("--no-progress-timeout-s", type=float, default=600.0)
    parser.add_argument("--target-valid-episodes-per-hour", type=float, default=8.0)
    parser.add_argument("--max-infrastructure-attempts", type=int, default=3)
    return parser


def main() -> int:
    """CLI entrypoint.

    Returns:
        ``0`` on success.
    """
    args = build_parser().parse_args()
    preregistration = prepare(args)
    print(json.dumps(preregistration, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
