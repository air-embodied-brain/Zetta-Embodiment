#!/usr/bin/env python3
# Copyright (c) 2026 Zetta Contributors
"""Freeze one single-slot Genie Sim VLA evolution campaign."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from robots.geniesim.contracts import (  # noqa: E402
    EXECUTION_CONTRACT,
    SAFETY_LAYER,
    TOOL_CATALOG,
    validate_bundle,
)
from zetta.envs.geniesim.config import (  # noqa: E402
    ASSET_REVISION,
    GENIESIM_REVISION,
    TASK_NAME,
    GenieSimConfig,
)
from zetta.evolution.jsonio import (  # noqa: E402
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    read_json,
)
from zetta.evolution.models import CampaignManifest, CandidateBundle  # noqa: E402
from zetta.evolution.schedule import preregister_seed_schedule  # noqa: E402
from zetta.evolution.stages import (  # noqa: E402
    CLUSTER_SYSTEM_PROMPT,
    DIAGNOSIS_SYSTEM_PROMPT,
    PROPOSAL_SYSTEM_PROMPT,
)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    repository = Path(args.repository_root).resolve()
    python = Path(args.runtime_python).absolute()
    if (
        not (repository / "robots/geniesim/run_rollout.py").is_file()
        or not python.is_file()
    ):
        raise ValueError("repository entrypoint and runtime Python must exist")
    endpoint = urlsplit(args.runtime_url)
    if (
        endpoint.scheme not in {"http", "https"}
        or not endpoint.hostname
        or endpoint.username
        or endpoint.password
        or endpoint.query
        or endpoint.fragment
    ):
        raise ValueError(
            "runtime URL must be absolute and contain no credentials or query"
        )
    if not args.policy_model_version.strip() or not 1 <= args.execute_horizon <= 64:
        raise ValueError(
            "frozen policy model version and a horizon in [1, 64] are required"
        )
    config = GenieSimConfig.from_mapping(read_json(args.env_config))
    if not 1 <= args.population_size <= 1_000_000:
        raise ValueError("population-size must be in [1, 1000000]")
    rollout, heldout, policy_rng = preregister_seed_schedule(
        master_seed=args.master_seed,
        task=TASK_NAME,
        rollout_count=args.rollout_count,
        heldout_count=args.heldout_count,
        population=range(args.population_size),
        heldout_seeds=range(1, args.heldout_count + 1),
    )
    parent = None
    if args.parent_bundle:
        parent = CandidateBundle.from_dict(read_json(args.parent_bundle))
        validate_bundle(parent)
        if parent.generation >= args.generation:
            raise ValueError(
                "parent bundle generation must precede campaign generation"
            )
    elif args.generation != 0:
        raise ValueError("nonzero generation requires --parent-bundle")
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=False)
    env_path = output / "env-config.json"
    atomic_write_json(env_path, config.to_dict(), overwrite=False)
    prompt_path = output / "prompt-contract.json"
    atomic_write_json(
        prompt_path,
        {
            "execution": EXECUTION_CONTRACT,
            "cluster": CLUSTER_SYSTEM_PROMPT,
            "stage1": DIAGNOSIS_SYSTEM_PROMPT,
            "stage2": PROPOSAL_SYSTEM_PROMPT,
            "agent_model": args.agent_model,
            "reasoning_effort": args.reasoning_effort,
        },
        overwrite=False,
    )
    catalog_path = output / "tool-catalog.json"
    atomic_write_json(catalog_path, TOOL_CATALOG, overwrite=False)
    parent_sha = parent.sha256 if parent else None
    bundle_files = {}
    if parent:
        parent_path = output / "parent-bundle.json"
        atomic_write_json(parent_path, parent.as_dict(), overwrite=False)
        bundle_files[parent_sha] = str(parent_path)
    command = [
        str(python),
        str(repository / "robots/geniesim/run_rollout.py"),
        "--runtime-url",
        args.runtime_url,
        "--env-config",
        str(env_path),
        "--env-config-sha256",
        file_sha256(env_path),
        "--policy-id",
        args.policy_id,
        "--policy-model-version",
        args.policy_model_version,
        "--seed",
        "{seed}",
        "--policy-rng",
        "{policy_rng}",
        "--task",
        "{task}",
        "--logical-id",
        "{logical_id}",
        "--generation",
        "{generation}",
        "--attempt-index",
        "{attempt_index}",
        "--bundle",
        "{bundle_file}",
        "--bundle-sha256",
        "{bundle_sha256}",
        "--output-dir",
        "{output_dir}",
        "--result-file",
        "{result_file}",
        "--execute-horizon",
        str(args.execute_horizon),
        "--capture-frames",
    ]
    manifest = CampaignManifest(
        campaign_id=args.campaign_id,
        environment="geniesim",
        task=TASK_NAME,
        generation=args.generation,
        code_commit=args.code_commit,
        prompt_sha256=file_sha256(prompt_path),
        model=args.agent_model,
        tool_catalog_sha256=file_sha256(catalog_path),
        rollout_seeds=rollout,
        heldout_seeds=heldout,
        policy_rng_by_seed=policy_rng,
        parent_bundle_sha256=parent_sha,
        active_bundle_sha256=parent_sha,
        baseline_mode="active_bundle" if parent else "strict_pure_vla",
        safety_layer=SAFETY_LAYER,
        expected_rollouts=args.rollout_count,
        expected_heldout=args.heldout_count,
        initial_logical_slots=1,
        maximum_logical_slots=1,
        continuous_logical_slots=1,
        maximum_api_concurrency=args.maximum_api_concurrency,
        episode_timeout_s=args.episode_timeout_s,
        no_progress_timeout_s=args.no_progress_timeout_s,
        max_infrastructure_attempts=2,
        target_valid_episodes_per_hour=2.0,
        reasoning_effort=args.reasoning_effort,
        runtime={
            "rollout_command": command,
            "same_seed_gate_rollout_command": command,
            "rollout_requires_api": False,
            "candidate_rollout_requires_api": False,
            "rollout_requires_environment_slot": False,
            "reuse_rollout_parent_evidence": True,
            "bundle_files_by_sha": bundle_files,
            "heldout_gate_kind": "heldout_20"
            if args.heldout_count == 20
            else "heldout",
            "execution_contract": EXECUTION_CONTRACT,
            "agent_model": args.agent_model,
            "reasoning_effort": args.reasoning_effort,
            "policy_model_version": args.policy_model_version,
            "geniesim_revision": GENIESIM_REVISION,
            "asset_revision": ASSET_REVISION,
            "evidence_granularity": "control_step",
            "seed_provenance": {
                "kind": "raw_uint32_generalization_seed",
                "population_size": args.population_size,
                "heldout_selection": f"fixed 1..{args.heldout_count}",
                "same_seed_pixel_determinism": False,
            },
        },
    )
    manifest_path = output / "manifest.json"
    atomic_write_json(manifest_path, manifest.as_dict(), overwrite=False)
    preregistration = {
        "schema_version": 1,
        "campaign_id": manifest.campaign_id,
        "manifest_sha256": manifest.sha256,
        "manifest_file_sha256": file_sha256(manifest_path),
        "prompt_sha256": manifest.prompt_sha256,
        "tool_catalog_sha256": manifest.tool_catalog_sha256,
        "env_config_sha256": file_sha256(env_path),
        "schedule_sha256": canonical_sha256(
            {
                "rollout_seeds": rollout,
                "heldout_seeds": heldout,
                "policy_rng_by_seed": policy_rng,
            }
        ),
        "baseline_mode": manifest.baseline_mode,
        "active_bundle_sha256": parent_sha,
        "success_criterion": "official scores.E2E == 1",
        "infrastructure_invalid_scored": False,
        "policy_model_version": args.policy_model_version,
    }
    atomic_write_json(output / "preregistration.json", preregistration, overwrite=False)
    return preregistration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--repository-root", default=str(REPOSITORY_ROOT))
    parser.add_argument("--runtime-python", default=sys.executable)
    parser.add_argument("--runtime-url", required=True)
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--policy-id", default="geniesim_vla")
    parser.add_argument("--policy-model-version", required=True)
    parser.add_argument("--master-seed", type=int, required=True)
    parser.add_argument("--rollout-count", type=int, default=50)
    parser.add_argument("--heldout-count", type=int, default=20)
    parser.add_argument("--population-size", type=int, default=100_000)
    parser.add_argument("--generation", type=int, default=0)
    parser.add_argument("--parent-bundle")
    parser.add_argument("--execute-horizon", type=int, default=5)
    parser.add_argument("--agent-model", default="gpt-5.6-sol")
    parser.add_argument(
        "--reasoning-effort", default="high", choices=["low", "medium", "high", "xhigh"]
    )
    parser.add_argument("--maximum-api-concurrency", type=int, default=1)
    parser.add_argument("--episode-timeout-s", type=int, default=7200)
    parser.add_argument("--no-progress-timeout-s", type=int, default=2400)
    return parser


if __name__ == "__main__":
    print(json.dumps(prepare(build_parser().parse_args()), indent=2, sort_keys=True))
