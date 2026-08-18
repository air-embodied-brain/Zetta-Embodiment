#!/usr/bin/env python3
# Copyright (c) 2026 RPent Contributors
"""Create one immutable, secret-free LIBERO-Pro evolution campaign."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from robots.libero.evolution_defaults import (  # noqa: E402
    LIBERO_EPISODE_TIMEOUT_S_DEFAULT,
    LIBERO_NO_PROGRESS_TIMEOUT_S_DEFAULT,
    LIBERO_PRIVILEGED_EVIDENCE_DEFAULT,
    LIBERO_PROVISIONAL_MIN_DIAGNOSIS_CONFIDENCE,
    LIBERO_WAIT_STEPS_DEFAULT,
    add_privileged_evidence_argument,
    libero_horizon_contract,
    privileged_evidence_enabled,
)
from robots.libero.run_evolution_rollout import LIBERO_RECOVERY_TOOLS  # noqa: E402
from robots.libero.runtime_devices import (  # noqa: E402
    parse_physical_gpus,
    preregister_device_contract,
)
from robots.libero.tools import TOOLS_SPEC  # noqa: E402
from robots.robocasa.role1_agent import ROLE1_SYSTEM_CONTRACT  # noqa: E402
from rpent.evolution.jsonio import (  # noqa: E402
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    read_json,
)
from rpent.evolution.models import (  # noqa: E402
    CampaignManifest,
    CandidateBundle,
    SafetyLayerConfig,
)
from rpent.evolution.protocol import EvolutionProtocol  # noqa: E402
from rpent.evolution.schedule import preregister_seed_schedule  # noqa: E402
from rpent.evolution.stages import (  # noqa: E402
    CLUSTER_SYSTEM_PROMPT,
    DIAGNOSIS_SYSTEM_PROMPT,
    PROPOSAL_SYSTEM_PROMPT,
)


def _parse_seed_spec(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    seeds: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError("fixed heldout seed ranges must be ascending")
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(token))
    if not seeds:
        raise ValueError("fixed heldout seed specification is empty")
    if len(set(seeds)) != len(seeds):
        raise ValueError("fixed heldout seed specification contains duplicates")
    return tuple(seeds)


def _normalize_language(value: str) -> str:
    return " ".join(value.casefold().split())


def _probe_task_language(args: argparse.Namespace) -> str:
    """Resolve the installed benchmark's authoritative ``Task.language``."""

    program = (
        "import json; "
        "from liberopro.liberopro.benchmark import get_benchmark; "
        f"task=get_benchmark({args.suite!r})().get_task({int(args.task_id)}); "
        "print(json.dumps({'language': task.language}, ensure_ascii=False))"
    )
    completed = subprocess.run(
        [str(Path(os.path.abspath(args.runtime_python))), "-c", program],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise ValueError(
            "could not resolve authoritative LIBERO task language with the "
            f"campaign runtime{suffix}; pass --task-language or --task-contract"
        )
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise ValueError("task-language probe returned invalid JSON") from exc
    language = payload.get("language") if isinstance(payload, dict) else None
    if not isinstance(language, str) or not language.strip():
        raise ValueError("task-language probe returned an empty language")
    return language.strip()


def _load_task_contract(args: argparse.Namespace, task: str) -> dict[str, Any]:
    """Return the required immutable task identity for a formal campaign."""
    language = getattr(args, "task_language", None)
    contract_path = getattr(args, "task_contract", None)
    if language and contract_path is not None:
        raise ValueError("--task-language and --task-contract are mutually exclusive")
    if contract_path is not None:
        value = read_json(Path(contract_path))
        if not isinstance(value, dict):
            raise ValueError("--task-contract must point to a JSON object")
        contract = dict(value)
    elif language:
        contract = {
            "suite": args.suite,
            "task": task,
            "task_id": args.task_id,
            "language": language,
            "source": "CLI authoritative task language",
        }
    else:
        resolved = _probe_task_language(args)
        contract = {
            "suite": args.suite,
            "task": task,
            "task_id": args.task_id,
            "language": resolved,
            "source": "installed LIBERO-Pro benchmark Task.language",
        }
    contract.setdefault("suite", args.suite)
    contract.setdefault("task", task)
    contract.setdefault("task_id", args.task_id)
    if not isinstance(contract.get("suite"), str) or not contract["suite"].strip():
        raise ValueError("task contract suite is required")
    if not isinstance(contract.get("task"), str) or not contract["task"].strip():
        raise ValueError("task contract task is required")
    if not isinstance(contract.get("language"), str) or not contract["language"].strip():
        raise ValueError("task contract language is required")
    contract["suite"] = contract["suite"].strip()
    contract["task"] = contract["task"].strip()
    contract["language"] = contract["language"].strip()
    try:
        contract_task_id = int(contract["task_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("task contract task_id is required") from exc
    if (
        contract["suite"] != args.suite
        or contract["task"] != task
        or contract_task_id != int(args.task_id)
    ):
        raise ValueError("task contract does not match --suite/--task")
    contract["task_id"] = contract_task_id
    contract["normalized_language"] = _normalize_language(contract["language"])
    contract["language_sha256"] = canonical_sha256(
        {"language": contract["normalized_language"]}
    )
    return contract


def _load_frozen_schedule(
    args: argparse.Namespace,
) -> tuple[tuple[int, ...], tuple[int, ...], dict[str, int], dict[str, Any] | None]:
    source_value = getattr(args, "schedule_from_manifest", None)
    master_seed = getattr(args, "master_seed", None)
    if (source_value is None) == (master_seed is None):
        raise ValueError(
            "exactly one of --master-seed or --schedule-from-manifest is required"
        )
    fixed_heldout = _parse_seed_spec(
        getattr(args, "fixed_heldout_seeds", None)
    )
    if source_value is None:
        rollout, heldout, policy_rng = preregister_seed_schedule(
            master_seed=int(master_seed),
            task=args.task or f"{args.suite}/task{args.task_id}",
            rollout_count=args.rollout_count,
            heldout_count=args.heldout_count,
            population=range(args.population_size),
            heldout_seeds=fixed_heldout,
        )
        return rollout, heldout, policy_rng, None

    source_path = Path(source_value).resolve()
    source = read_json(source_path)
    rollout = tuple(int(seed) for seed in source.get("rollout_seeds", ()))
    heldout = tuple(int(seed) for seed in source.get("heldout_seeds", ()))
    raw_policy = source.get("policy_rng_by_seed")
    if not isinstance(raw_policy, dict):
        raise ValueError("source manifest policy_rng_by_seed must be an object")
    policy_rng = {str(seed): int(value) for seed, value in raw_policy.items()}
    if len(rollout) != args.rollout_count:
        raise ValueError(
            f"source manifest has {len(rollout)} rollout seeds; "
            f"expected {args.rollout_count}"
        )
    if len(heldout) != args.heldout_count:
        raise ValueError(
            f"source manifest has {len(heldout)} heldout seeds; "
            f"expected {args.heldout_count}"
        )
    if len(set(rollout)) != len(rollout) or len(set(heldout)) != len(heldout):
        raise ValueError("source manifest schedule contains duplicate seeds")
    if set(rollout) & set(heldout):
        raise ValueError("source manifest rollout and heldout seeds overlap")
    if fixed_heldout is not None and heldout != fixed_heldout:
        raise ValueError("source manifest does not match fixed heldout seeds")
    expected_policy_keys = {str(seed) for seed in (*rollout, *heldout)}
    if set(policy_rng) != expected_policy_keys:
        raise ValueError(
            "source manifest policy_rng_by_seed must exactly cover the frozen schedule"
        )
    source_task = source.get("task")
    task = args.task or f"{args.suite}/task{args.task_id}"
    if source_task != task:
        raise ValueError(
            f"source manifest task {source_task!r} does not match {task!r}"
        )
    provenance = {
        "source_manifest": str(source_path),
        "source_manifest_file_sha256": file_sha256(source_path),
        "source_manifest_sha256": canonical_sha256(source),
        "source_campaign_id": source.get("campaign_id"),
        "source_code_commit": source.get("code_commit"),
    }
    return rollout, heldout, policy_rng, provenance


def _rollout_command(
    args: argparse.Namespace, *, task_contract: dict[str, Any]
) -> list[str]:
    script = (
        Path(args.repository_root).resolve()
        / "robots"
        / "libero"
        / "run_evolution_rollout.py"
    )
    command = [
        # Preserve the virtual-environment entrypoint. Resolving its symlink
        # would silently replace it with the system interpreter.
        str(Path(os.path.abspath(args.runtime_python))),
        str(script),
        "--suite",
        args.suite,
        "--task-id",
        str(args.task_id),
        "--task",
        "{task}",
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
        "--output-dir",
        "{output_dir}",
        "--result-file",
        "{result_file}",
        "--gpu",
        str(args.environment_gpus[0]),
        "--allowed-environment-gpus",
        ",".join(str(gpu) for gpu in args.environment_gpus),
        "--vla-endpoint",
        args.vla_endpoint,
        "--vla-gpu",
        str(args.vla_gpu),
        "--max-actions",
        str(args.max_actions),
        "--wait-steps",
        str(args.wait_steps),
        "--actions-per-chunk",
        str(args.actions_per_chunk),
        "--visual-overview-frames",
        str(args.visual_overview_frames),
        "--visual-event-window-radius",
        str(args.visual_event_window_radius),
        "--visual-event-window-stride",
        str(args.visual_event_window_stride),
        "--visual-maximum-event-windows",
        str(args.visual_maximum_event_windows),
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
        "--role1-require-visual-review"
        if getattr(args, "role1_require_visual_review", False)
        else "--no-role1-require-visual-review",
        "--allow-privileged-evidence"
        if privileged_evidence_enabled(args)
        else "--no-allow-privileged-evidence",
        "--max-recovery-actor-calls",
        str(args.max_recovery_actor_calls),
    ]
    command.extend(["--expected-task-language", str(task_contract["language"])])
    return command


def _tool_catalog(
    *,
    suite: str,
    task_id: int,
    task_contract: dict[str, Any],
    allow_privileged_evidence: bool = LIBERO_PRIVILEGED_EVIDENCE_DEFAULT,
) -> dict[str, Any]:
    allowed = {
        name
        for name in LIBERO_RECOVERY_TOOLS
        if allow_privileged_evidence or name != "semantic_joint_interact"
    }
    tools = [dict(row) for row in TOOLS_SPEC if row.get("name") in allowed]
    if {str(row["name"]) for row in tools} != allowed:
        raise ValueError("LIBERO recovery allowlist is not fully described")
    return {
        "schema_version": 1,
        "environment": "libero_pro",
        "suite": suite,
        "task_id": task_id,
        "tools": tools,
        "task_binding": {
            "authoritative_task_contract": dict(task_contract),
            "tool_names": [
                name for name in LIBERO_RECOVERY_TOOLS if name in allowed
            ],
            "critic_feature_schema": "robots.libero.critic_runtime.LIBERO_CRITIC_FEATURES",
            "success_criterion": "libero_terminated",
        },
        "notes": {
            "privileged_state_allowed": True,
            "critic_privileged_plane": {
                "enabled": True,
                "actor_visibility": False,
                "role1_visibility": bool(allow_privileged_evidence),
                "role1_scope": "bounded_scalar_critic_observations_without_absolute_coordinates",
                "source": "LIBERO MuJoCo/BDDL audited semantic sidecar",
                "includes": [
                    "goal predicate progress",
                    "named object and target pose/distance",
                    "grasp retention release and contact",
                    "hinge/slide joint progress",
                    "command realization telemetry",
                    "semantic fixture-joint geometry consumed only inside audited tools",
                ],
            },
            "collision_heuristic_default": False,
            "actor_is_only_environment_writer_during_recovery": True,
        },
    }


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    horizon_contract = libero_horizon_contract(
        args.suite,
        max_actions=getattr(args, "max_actions", None),
        wait_steps=getattr(args, "wait_steps", LIBERO_WAIT_STEPS_DEFAULT),
    )
    if not horizon_contract["is_standard"] and not bool(
        getattr(args, "allow_nonstandard_horizon", False)
    ):
        raise ValueError(
            "formal LIBERO campaign requires the official suite horizon; "
            "use --allow-nonstandard-horizon only for explicitly exploratory runs"
        )
    args.max_actions = int(horizon_contract["policy_action_horizon"])
    args.wait_steps = int(horizon_contract["wait_steps"])
    args.visual_overview_frames = getattr(args, "visual_overview_frames", None)
    args.visual_event_window_radius = int(
        getattr(args, "visual_event_window_radius", 8)
    )
    args.visual_event_window_stride = int(
        getattr(args, "visual_event_window_stride", 2)
    )
    args.visual_maximum_event_windows = int(
        getattr(args, "visual_maximum_event_windows", 6)
    )
    args.diagnosis_max_artifact_reads = int(
        getattr(args, "diagnosis_max_artifact_reads", 24)
    )
    args.cluster_max_artifact_reads = int(
        getattr(args, "cluster_max_artifact_reads", 24)
    )
    if args.visual_overview_frames is None:
        args.visual_overview_frames = 25 if args.max_actions >= 400 else 17
    if not 5 <= int(args.visual_overview_frames) <= 64:
        raise ValueError("--visual-overview-frames must be in [5, 64]")
    if not 1 <= int(args.visual_event_window_radius) <= 64:
        raise ValueError("--visual-event-window-radius must be in [1, 64]")
    if not 1 <= int(args.visual_event_window_stride) <= int(
        args.visual_event_window_radius
    ):
        raise ValueError("--visual-event-window-stride must not exceed radius")
    if not 1 <= int(args.visual_maximum_event_windows) <= 16:
        raise ValueError("--visual-maximum-event-windows must be in [1, 16]")
    if not 3 <= int(args.diagnosis_max_artifact_reads) <= 64:
        raise ValueError("--diagnosis-max-artifact-reads must be in [3, 64]")
    if not 6 <= int(args.cluster_max_artifact_reads) <= 64:
        raise ValueError("--cluster-max-artifact-reads must be in [6, 64]")
    args.environment_gpus = parse_physical_gpus(args.environment_gpus)
    device_contract = preregister_device_contract(
        environment_gpus=args.environment_gpus,
        vla_gpu=args.vla_gpu,
    )
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=False)
    repository_root = Path(args.repository_root).resolve()
    runtime_python = Path(os.path.abspath(args.runtime_python))
    if not (repository_root / "robots" / "libero" / "run_evolution_rollout.py").is_file():
        raise ValueError("repository root has no LIBERO evolution entrypoint")
    if not runtime_python.is_file():
        raise ValueError("runtime Python does not exist")
    if args.agent_model != "gpt-5.6-sol" or args.reasoning_effort != "high":
        raise ValueError("formal agents are frozen to gpt-5.6-sol/high")
    if args.role1_model != "gpt-5.6-sol":
        raise ValueError("formal Role1 model is frozen to gpt-5.6-sol")

    task = args.task or f"{args.suite}/task{args.task_id}"
    task_contract = _load_task_contract(args, task)
    prompt_contract = {
        "agent_abstraction": {
            "online_enabled": ["role1_actor"],
            "online_disabled": ["role0", "role2"],
            "offline_campaign_control": [
                "multimodal_cluster",
                "causal_diagnoser",
                "candidate_evolver",
            ],
            "deterministic_harness": [
                "success_labels",
                "seed_pairing",
                "schema",
                "gate_statistics",
            ],
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
    catalog_path = output / "tool-catalog.json"
    atomic_write_json(
        catalog_path,
        _tool_catalog(
            suite=args.suite,
            task_id=args.task_id,
            task_contract=task_contract,
            allow_privileged_evidence=privileged_evidence_enabled(args),
        ),
        overwrite=False,
    )

    rollout, heldout, policy_rng, schedule_provenance = _load_frozen_schedule(args)
    parent_sha256 = None
    bundle_files_by_sha: dict[str, str] = {}
    if args.parent_bundle is not None:
        parent_path = Path(args.parent_bundle).resolve()
        parent = CandidateBundle.from_dict(read_json(parent_path))
        parent_sha256 = parent.sha256
        if parent.generation >= args.generation:
            raise ValueError("parent bundle generation must precede campaign generation")
        bundle_files_by_sha[parent_sha256] = str(parent_path)
    elif args.generation != 0:
        raise ValueError("nonzero generation requires --parent-bundle")

    runtime_command = _rollout_command(args, task_contract=task_contract)
    protocol = EvolutionProtocol(
        rollout_count=args.rollout_count,
        heldout_seeds=heldout,
        heldout_mode=getattr(args, "heldout_mode", "validation"),
        same_seed_pass_rate=float(getattr(args, "same_seed_pass_rate", 0.5)),
        same_seed_max_rounds=int(getattr(args, "same_seed_max_rounds", 8)),
        heldout_alpha=float(getattr(args, "heldout_alpha", 0.025)),
        heldout_min_gain=int(getattr(args, "heldout_min_gain", 1)),
        heldout_min_success_rate=float(
            getattr(args, "heldout_min_success_rate", 0.0)
        ),
        heldout_require_significance=bool(
            getattr(args, "heldout_require_significance", True)
        ),
        heldout_max_rounds=int(getattr(args, "heldout_max_rounds", 1)),
        max_infrastructure_attempts=args.max_infrastructure_attempts,
        max_candidate_rounds_per_cluster=int(
            getattr(args, "max_candidate_rounds_per_cluster", 8)
        ),
        maximum_target_clusters=int(getattr(args, "maximum_target_clusters", 2)),
        maximum_total_candidate_rounds=int(
            getattr(args, "maximum_total_candidate_rounds", 15)
        ),
        regression_required=not bool(getattr(args, "skip_regression_gate", True)),
    )
    protocol.validate_partition(rollout, heldout)
    evolution_policy = {
        **protocol.runtime_policy(),
        "diagnosis_max_artifact_reads": int(args.diagnosis_max_artifact_reads),
        "cluster_max_artifact_reads": int(args.cluster_max_artifact_reads),
        "provisional_min_diagnosis_confidence": (
            LIBERO_PROVISIONAL_MIN_DIAGNOSIS_CONFIDENCE
        ),
        "defer_inconclusive_for_provisional": True,
    }
    runtime = {
        "evolution_policy": evolution_policy,
        "subsequent_rollout_count": 10,
        "rollout_command": runtime_command,
        "same_seed_gate_rollout_command": runtime_command,
        # Gen0 has an explicitly empty Critic/Recovery bundle and therefore
        # cannot invoke Role1.  Later generations have a frozen parent bundle
        # and reserve API capacity for possible Critic interventions.
        "rollout_requires_api": parent_sha256 is not None,
        "candidate_rollout_requires_api": True,
        # The generation's own rollout ledger is already a frozen live parent
        # arm for same-seed and regression gates. Reuse it append-only instead
        # of spending latency on an identical parent replay.
        "reuse_rollout_parent_evidence": True,
        "heldout_gate_kind": "heldout_20" if args.heldout_count == 20 else "heldout",
        # Each job owns an isolated EGL process selected by its worker's
        # RPENT_LIBERO_GPU.  It must not also request a RoboCasa farm lease.
        "rollout_requires_environment_slot": False,
        "bundle_files_by_sha": bundle_files_by_sha,
        "suite": args.suite,
        "task_id": args.task_id,
        "vla_endpoint": args.vla_endpoint,
        "runtime_device_contract": device_contract,
        "agent_model": args.agent_model,
        "reasoning_effort": args.reasoning_effort,
        "libero_privileged_evidence": privileged_evidence_enabled(args),
        "evaluation_horizon": horizon_contract,
    }
    runtime["task_contract"] = task_contract
    atomic_write_json(output / "task-contract.json", task_contract, overwrite=False)
    manifest = CampaignManifest(
        campaign_id=args.campaign_id,
        environment="libero_pro",
        task=task,
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
        safety_layer=SafetyLayerConfig(
            action_contract="libero_finite_7d_action_v1",
            control_limits="libero_per_channel_scale_and_clip_v1",
            simulation_health="finite_robot_state_and_rpc_health_v1",
            joint_limit_shield="environment_controller_limits_only",
            contact_policy="environment_native_only_no_collision_heuristic",
        ),
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
        "environment": manifest.environment,
        "task": task,
        "suite": args.suite,
        "task_id": args.task_id,
        "task_contract": task_contract,
        "task_contract_sha256": canonical_sha256(task_contract),
        "model": manifest.model,
        "reasoning_effort": manifest.reasoning_effort,
        "prompt_sha256": manifest.prompt_sha256,
        "tool_catalog_sha256": manifest.tool_catalog_sha256,
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
        "success_criterion": "libero_terminated",
        "baseline_mode": manifest.baseline_mode,
        "active_bundle_sha256": manifest.active_bundle_sha256,
        "safety_layer": manifest.safety_layer.as_dict(),
        "infrastructure_invalid_scored": False,
        "reuse_rollout_parent_evidence": True,
        "runtime_device_contract": device_contract,
        "evaluation_horizon": horizon_contract,
    }
    if schedule_provenance is not None:
        preregistration["schedule_provenance"] = schedule_provenance
    atomic_write_json(output / "preregistration.json", preregistration, overwrite=False)
    return preregistration


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--suite", default="libero_goal_task")
    parser.add_argument("--task-id", type=int, default=6)
    parser.add_argument("--task")
    parser.add_argument(
        "--task-language",
        help="authoritative natural-language task contract; persisted for Cluster/Stage1/Stage2",
    )
    parser.add_argument(
        "--task-contract",
        type=Path,
        help="JSON file containing suite/task/language authoritative contract",
    )
    parser.add_argument("--generation", type=int, default=0)
    parser.add_argument("--parent-bundle", type=Path)
    schedule_group = parser.add_mutually_exclusive_group(required=True)
    schedule_group.add_argument("--master-seed", type=int)
    schedule_group.add_argument("--schedule-from-manifest", type=Path)
    parser.add_argument("--rollout-count", type=int, default=50)
    parser.add_argument("--heldout-count", type=int, default=20)
    parser.add_argument(
        "--fixed-heldout-seeds",
        default="1-20",
        help="comma-separated seeds and inclusive ranges; defaults to the fixed test block 1-20",
    )
    parser.add_argument("--population-size", type=int, default=100000)
    parser.add_argument("--initial-logical-slots", type=int, default=8)
    parser.add_argument("--maximum-logical-slots", type=int, default=12)
    parser.add_argument("--continuous-logical-slots", type=int, default=8)
    parser.add_argument("--maximum-api-concurrency", type=int, default=8)
    parser.add_argument(
        "--episode-timeout-s", type=int, default=LIBERO_EPISODE_TIMEOUT_S_DEFAULT
    )
    parser.add_argument(
        "--no-progress-timeout-s",
        type=int,
        default=LIBERO_NO_PROGRESS_TIMEOUT_S_DEFAULT,
    )
    parser.add_argument("--target-valid-episodes-per-hour", type=float, default=25.0)
    parser.add_argument("--max-infrastructure-attempts", type=int, default=2)
    parser.add_argument("--same-seed-pass-rate", type=float, default=0.5)
    parser.add_argument("--same-seed-max-rounds", type=int, default=2)
    parser.add_argument(
        "--heldout-mode",
        choices=("test", "validation"),
        default="validation",
        help="always execute held-out seeds; test records only, validation gates promotion",
    )
    parser.add_argument("--heldout-alpha", type=float, default=0.025)
    parser.add_argument("--heldout-min-gain", type=int, default=1)
    parser.add_argument("--heldout-min-success-rate", type=float, default=0.0)
    parser.add_argument(
        "--heldout-require-significance",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--heldout-max-rounds", type=int, default=1)
    parser.add_argument("--max-candidate-rounds-per-cluster", type=int, default=8)
    parser.add_argument("--maximum-target-clusters", type=int, default=2)
    parser.add_argument("--maximum-total-candidate-rounds", type=int, default=15)
    parser.add_argument(
        "--skip-regression-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--vla-endpoint", default="http://127.0.0.1:18811")
    parser.add_argument(
        "--environment-gpus",
        type=parse_physical_gpus,
        required=True,
        help="comma-separated physical GPUs allowed for LIBERO EGL workers",
    )
    parser.add_argument("--vla-gpu", type=int, required=True)
    parser.add_argument(
        "--max-actions",
        type=int,
        default=None,
        help="policy action horizon; defaults to the official value for --suite",
    )
    parser.add_argument(
        "--wait-steps", type=int, default=LIBERO_WAIT_STEPS_DEFAULT
    )
    parser.add_argument(
        "--allow-nonstandard-horizon",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="allow a protocol-ineligible exploratory campaign horizon override",
    )
    parser.add_argument("--actions-per-chunk", type=int, default=5)
    parser.add_argument(
        "--visual-overview-frames",
        type=int,
        default=None,
        help="uniform full-episode overview frames; defaults to 25 for Long and 17 for Goal",
    )
    parser.add_argument("--visual-event-window-radius", type=int, default=8)
    parser.add_argument("--visual-event-window-stride", type=int, default=2)
    parser.add_argument("--visual-maximum-event-windows", type=int, default=6)
    parser.add_argument("--diagnosis-max-artifact-reads", type=int, default=24)
    parser.add_argument("--cluster-max-artifact-reads", type=int, default=24)
    parser.add_argument("--role1-planner", choices=("api", "codex"), default="api")
    parser.add_argument("--agent-model", default="gpt-5.6-sol")
    parser.add_argument("--role1-model", default="gpt-5.6-sol")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--role1-max-tokens", type=int, default=4096)
    parser.add_argument("--role1-timeout-s", type=int, default=180)
    parser.add_argument("--role1-heartbeat-s", type=float, default=15.0)
    parser.add_argument("--role1-max-turns", type=int, default=2)
    visual_group = parser.add_mutually_exclusive_group()
    visual_group.add_argument(
        "--role1-require-visual-review",
        dest="role1_require_visual_review",
        action="store_true",
        help="require Role1 to call the current-image tool before deciding",
    )
    visual_group.add_argument(
        "--no-role1-require-visual-review",
        dest="role1_require_visual_review",
        action="store_false",
        help="allow LIBERO Role1 to rely on bounded scalar critic evidence",
    )
    parser.set_defaults(role1_require_visual_review=False)
    add_privileged_evidence_argument(
        parser,
        help_text="expose audited Critic-only LIBERO scalar evidence to Role1",
    )
    parser.add_argument("--max-recovery-actor-calls", type=int, default=4)
    args = parser.parse_args()
    report = prepare(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
