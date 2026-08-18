#!/usr/bin/env python3
# Copyright (c) 2026 RPent Contributors
"""Prepare an evaluation-only LIBERO paired heldout campaign.

This adapter preserves a previously preregistered seed/RNG schedule, or binds
an explicitly preregistered independent heldout schedule, while attaching both
live gate arms to a new code snapshot.  It deliberately does not manufacture
same-seed or regression decisions, so the resulting campaign can produce
heldout evidence but can never promote a candidate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from robots.libero.evolution_defaults import (  # noqa: E402
    LIBERO_EPISODE_TIMEOUT_S_DEFAULT,
    LIBERO_NO_PROGRESS_TIMEOUT_S_DEFAULT,
)
from robots.libero.runtime_devices import (  # noqa: E402
    parse_physical_gpus,
    preregister_device_contract,
)
from rpent.evolution.jsonio import (  # noqa: E402
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    read_json,
)
from rpent.evolution.models import (  # noqa: E402
    CampaignManifest,
    CampaignPhase,
    CandidateBundle,
)
from rpent.evolution.store import CampaignStore  # noqa: E402

EVALUATION_SCOPE = "paired_heldout_only"


def _heldout_schedule(
    *,
    source: CampaignManifest,
    schedule_path: Path | None,
) -> tuple[tuple[int, ...], dict[str, int], dict[str, Any] | None]:
    if schedule_path is None:
        if source.expected_heldout != 20 or len(source.heldout_seeds) != 20:
            raise ValueError("source manifest must contain exactly 20 frozen seeds")
        if source.runtime.get("heldout_gate_kind") != "heldout_20":
            raise ValueError("source manifest is not registered for heldout_20")
        return source.heldout_seeds, dict(source.policy_rng_by_seed), None

    resolved = schedule_path.resolve()
    if not resolved.is_file():
        raise ValueError("independent heldout schedule does not exist")
    payload = read_json(resolved)
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported independent heldout schedule schema")
    seeds = tuple(int(seed) for seed in payload.get("heldout_seeds", ()))
    if len(seeds) != 50 or len(set(seeds)) != 50:
        raise ValueError("independent heldout schedule requires 50 unique seeds")
    if set(seeds) & set(source.rollout_seeds):
        raise ValueError("independent heldout seeds overlap source rollout seeds")
    if set(seeds) & set(source.heldout_seeds):
        raise ValueError("independent heldout seeds overlap source heldout seeds")
    raw_policy = payload.get("policy_rng_by_seed")
    if not isinstance(raw_policy, dict):
        raise ValueError("independent heldout schedule has no policy RNG mapping")
    heldout_policy = {str(seed): int(value) for seed, value in raw_policy.items()}
    if set(heldout_policy) != {str(seed) for seed in seeds}:
        raise ValueError("independent policy RNG mapping must exactly cover its seeds")
    if any(not 0 <= value <= 0x7FFFFFFF for value in heldout_policy.values()):
        raise ValueError("independent policy RNG values must be in [0, 2^31-1]")
    policy_rng = {
        str(seed): source.policy_rng_by_seed[str(seed)]
        for seed in source.rollout_seeds
    }
    policy_rng.update(heldout_policy)
    provenance = {
        "schedule_file": str(resolved),
        "schedule_file_sha256": file_sha256(resolved),
        "schedule_sha256": canonical_sha256(payload),
        "schedule_id": payload.get("schedule_id"),
    }
    return seeds, policy_rng, provenance


def _rebind_rollout_command(
    template: list[str],
    *,
    repository_root: Path,
    runtime_python: Path,
    vla_endpoint: str | None,
    environment_gpus: tuple[int, ...],
    vla_gpu: int,
) -> list[str]:
    command = list(template)
    if not command:
        raise ValueError("source manifest has no paired rollout command")
    command[0] = str(runtime_python)
    entrypoint = repository_root / "robots" / "libero" / "run_evolution_rollout.py"
    matches = [
        index
        for index, value in enumerate(command)
        if str(value).replace("\\", "/").endswith(
            "robots/libero/run_evolution_rollout.py"
        )
    ]
    if len(matches) != 1:
        raise ValueError("paired rollout command must contain one LIBERO entrypoint")
    command[matches[0]] = str(entrypoint)

    if vla_endpoint is not None:
        try:
            endpoint_index = command.index("--vla-endpoint") + 1
        except ValueError as exc:
            raise ValueError("paired rollout command has no --vla-endpoint") from exc
        if endpoint_index >= len(command):
            raise ValueError("paired rollout command has no VLA endpoint value")
        command[endpoint_index] = vla_endpoint

    replacements = {
        "--gpu": str(environment_gpus[0]),
        "--allowed-environment-gpus": ",".join(
            str(gpu) for gpu in environment_gpus
        ),
        "--vla-gpu": str(vla_gpu),
    }
    for option, value in replacements.items():
        while option in command:
            index = command.index(option)
            if index + 1 >= len(command):
                raise ValueError(f"paired rollout command has no value for {option}")
            del command[index : index + 2]
        command.extend([option, value])

    command = [
        value
        for value in command
        if value
        not in {"--allow-privileged-evidence", "--no-allow-privileged-evidence"}
    ]
    command.append("--allow-privileged-evidence")
    return command


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    environment_gpus = parse_physical_gpus(args.environment_gpus)
    device_contract = preregister_device_contract(
        environment_gpus=environment_gpus,
        vla_gpu=args.vla_gpu,
    )
    source_path = Path(args.source_manifest).resolve()
    candidate_path = Path(args.candidate_bundle).resolve()
    repository_root = Path(args.repository_root).resolve()
    runtime_python = Path(os.path.abspath(args.runtime_python))
    output_root = Path(args.output_root).resolve()

    if output_root.exists():
        raise ValueError("heldout validation output root already exists")
    if not source_path.is_file():
        raise ValueError("source manifest does not exist")
    if not candidate_path.is_file():
        raise ValueError("candidate bundle does not exist")
    if not runtime_python.is_file():
        raise ValueError("runtime Python does not exist")
    entrypoint = repository_root / "robots" / "libero" / "run_evolution_rollout.py"
    if not entrypoint.is_file():
        raise ValueError("repository root has no LIBERO evolution entrypoint")

    source_payload = read_json(source_path)
    source = CampaignManifest.from_dict(source_payload)
    candidate_payload = read_json(candidate_path)
    input_candidate = CandidateBundle.from_dict(candidate_payload)
    if source.environment != "libero_pro":
        raise ValueError("heldout-only adapter requires a LIBERO-Pro manifest")
    heldout_seeds, policy_rng, schedule_provenance = _heldout_schedule(
        source=source,
        schedule_path=(
            Path(args.heldout_schedule)
            if getattr(args, "heldout_schedule", None) is not None
            else None
        ),
    )
    gate_kind = "heldout_20" if len(heldout_seeds) == 20 else "heldout"
    if input_candidate.parent_sha256 != source.active_bundle_sha256:
        raise ValueError("candidate parent does not match the frozen source policy")
    if input_candidate.generation not in {source.generation, source.generation + 1}:
        raise ValueError("candidate generation is not adjacent to the source campaign")
    # A diagnostic candidate may have been authored as generation 1 while it
    # still targets the strict generation-0 parent. Rebinding only this
    # metadata field keeps the mechanism immutable and satisfies the model's
    # strict-pure-VLA generation invariant. The resulting digest is recorded
    # separately from the input artifact digest below.
    candidate = replace(input_candidate, generation=source.generation)

    source_command = source.runtime.get("heldout_gate_rollout_command")
    if not isinstance(source_command, list):
        source_command = source.runtime.get("paired_gate_rollout_command")
    if not isinstance(source_command, list):
        source_command = source.runtime.get("same_seed_gate_rollout_command")
    if not isinstance(source_command, list):
        raise ValueError("source manifest has no paired gate rollout command")
    command = _rebind_rollout_command(
        [str(value) for value in source_command],
        repository_root=repository_root,
        runtime_python=runtime_python,
        vla_endpoint=args.vla_endpoint,
        environment_gpus=environment_gpus,
        vla_gpu=args.vla_gpu,
    )
    runtime = {
        **source.runtime,
        "evaluation_scope": EVALUATION_SCOPE,
        "evaluation_source_manifest_sha256": canonical_sha256(source_payload),
        "evaluation_candidate_sha256": candidate.sha256,
        "heldout_gate_kind": gate_kind,
        "heldout_gate_rollout_command": command,
        "paired_gate_rollout_command": command,
        "same_seed_gate_rollout_command": command,
        "candidate_rollout_requires_api": True,
        "reuse_rollout_parent_evidence": False,
        "libero_privileged_evidence": True,
        "runtime_device_contract": device_contract,
        "bundle_files_by_sha": (
            {candidate.parent_sha256: str(candidate_path)}
            if candidate.parent_sha256 is not None
            else {}
        ),
    }
    manifest = replace(
        source,
        campaign_id=args.campaign_id,
        generation=candidate.generation,
        code_commit=args.code_commit,
        parent_bundle_sha256=candidate.parent_sha256,
        active_bundle_sha256=candidate.parent_sha256,
        baseline_mode=(
            "active_bundle"
            if candidate.parent_sha256 is not None
            else "strict_pure_vla"
        ),
        heldout_seeds=heldout_seeds,
        policy_rng_by_seed=policy_rng,
        expected_heldout=len(heldout_seeds),
        initial_logical_slots=1,
        maximum_logical_slots=1,
        continuous_logical_slots=1,
        maximum_api_concurrency=1,
        # Heldout validation runs the same full horizon as the primary
        # campaign.  Preserve an explicit caller override, but do not inherit
        # the historical 900-second source default that truncates valid long
        # episodes into infra_invalid results.
        episode_timeout_s=int(
            getattr(args, "episode_timeout_s", LIBERO_EPISODE_TIMEOUT_S_DEFAULT)
        ),
        no_progress_timeout_s=int(
            getattr(
                args,
                "no_progress_timeout_s",
                LIBERO_NO_PROGRESS_TIMEOUT_S_DEFAULT,
            )
        ),
        runtime=runtime,
    )

    store = CampaignStore(output_root)
    store.initialize(manifest)
    store.transition(
        CampaignPhase.CLUSTER,
        state_updates={"evaluation_scope": EVALUATION_SCOPE},
    )
    store.transition(CampaignPhase.DIAGNOSE)
    store.transition(CampaignPhase.PROPOSE)
    store.register_candidate(candidate)
    store.transition(CampaignPhase.SAME_SEED_GATE)
    store.transition(CampaignPhase.REGRESSION_GATE)
    state = store.transition(CampaignPhase.HELDOUT_GATE)

    preregistration = {
        "schema_version": 1,
        "evaluation_scope": EVALUATION_SCOPE,
        "promotion_authorized": False,
        "campaign_id": manifest.campaign_id,
        "manifest_sha256": manifest.sha256,
        "manifest_file_sha256": file_sha256(output_root / "manifest.json"),
        "source_manifest": str(source_path),
        "source_manifest_sha256": canonical_sha256(source_payload),
        "source_manifest_file_sha256": file_sha256(source_path),
        "source_campaign_id": source.campaign_id,
        "code_commit": args.code_commit,
        "candidate_bundle": str(candidate_path),
        "input_candidate_sha256": input_candidate.sha256,
        "candidate_sha256": candidate.sha256,
        "candidate_file_sha256": file_sha256(candidate_path),
        "heldout_seeds": list(manifest.heldout_seeds),
        "policy_rng_by_seed": {
            str(seed): manifest.policy_rng_by_seed[str(seed)]
            for seed in manifest.heldout_seeds
        },
        "paired_schedule_sha256": canonical_sha256(
            [
                {
                    "seed": seed,
                    "policy_rng": manifest.policy_rng_by_seed[str(seed)],
                }
                for seed in manifest.heldout_seeds
            ]
        ),
        "gate_kind": gate_kind,
        "statistical_contract": (
            "fixed_paired_mcnemar_20"
            if gate_kind == "heldout_20"
            else "two_stage_paired_mcnemar_10_to_50"
        ),
        "alpha": 0.025,
        "privileged_evidence": True,
        "infrastructure_invalid_scored": False,
        "episode_timeout_s": manifest.episode_timeout_s,
        "no_progress_timeout_s": manifest.no_progress_timeout_s,
        "runtime_device_contract": device_contract,
        "lifecycle_phases_not_claimed": ["same_seed_gate", "regression_gate"],
    }
    if gate_kind == "heldout":
        preregistration.update({"stage_1_count": 10, "stage_2_count": 50})
    if schedule_provenance is not None:
        preregistration["independent_schedule"] = schedule_provenance
    atomic_write_json(
        output_root / "heldout-only-preregistration.json",
        preregistration,
        overwrite=False,
    )
    return {
        "manifest": manifest.as_dict(),
        "state": state,
        "candidate_sha256": candidate.sha256,
        "preregistration": preregistration,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--candidate-bundle", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--vla-endpoint")
    parser.add_argument(
        "--environment-gpus",
        type=parse_physical_gpus,
        required=True,
        help="comma-separated physical GPUs allowed for LIBERO EGL workers",
    )
    parser.add_argument("--vla-gpu", type=int, required=True)
    parser.add_argument(
        "--episode-timeout-s",
        type=int,
        default=LIBERO_EPISODE_TIMEOUT_S_DEFAULT,
        help="wall-clock budget for one full LIBERO episode",
    )
    parser.add_argument(
        "--no-progress-timeout-s",
        type=int,
        default=LIBERO_NO_PROGRESS_TIMEOUT_S_DEFAULT,
        help="watchdog budget between rollout heartbeats",
    )
    parser.add_argument(
        "--heldout-schedule",
        type=Path,
        help=(
            "optional independent 50-seed JSON schedule; seeds must be disjoint "
            "from the source campaign"
        ),
    )
    args = parser.parse_args()
    print(json.dumps(prepare(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
