# Copyright (c) 2026 RPent Contributors
"""Campaign orchestration shared by the CLI and recovery tooling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rpent.evolution.clustering import cluster_failure_segments
from rpent.evolution.jsonio import atomic_write_json, canonical_sha256, read_json
from rpent.evolution.models import (
    CampaignPhase,
    EpisodeRecord,
)
from rpent.evolution.queue import (
    RolloutJob,
    SharedHostQueue,
    round_robin_assign,
)
from rpent.evolution.store import CampaignStore


def _render_command(template: list[str], values: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(part).format_map(values) for part in template)


def _known_job_ids(queue: SharedHostQueue) -> set[str]:
    return queue.known_job_ids()


def resolve_bundle_file(store: CampaignStore, bundle_sha256: str | None) -> str:
    """Resolve one frozen bundle by digest, including an external parent bundle."""

    if bundle_sha256 is None:
        return "none"
    configured = store.manifest().runtime.get("bundle_files_by_sha", {})
    candidates: list[Path] = []
    if isinstance(configured, dict) and bundle_sha256 in configured:
        configured_path = Path(str(configured[bundle_sha256]))
        candidates.append(
            configured_path
            if configured_path.is_absolute()
            else store.root / configured_path
        )
    candidates.append(store.root / "candidates" / bundle_sha256 / "bundle.json")
    candidates.extend(sorted((store.root / "bundles").glob("*.json")))
    for path in candidates:
        if path.is_file() and canonical_sha256(read_json(path)) == bundle_sha256:
            return str(path.resolve())
    raise ValueError(f"frozen bundle artifact is missing: {bundle_sha256}")


def build_rollout_jobs(
    *,
    store: CampaignStore,
    queue: SharedHostQueue,
    worker_hosts: tuple[str, ...],
) -> list[tuple[str, RolloutJob]]:
    manifest = store.manifest()
    command_template = manifest.runtime.get("rollout_command")
    if not isinstance(command_template, list) or not command_template:
        raise ValueError("manifest.runtime.rollout_command must be a non-empty array")
    state = store.state()
    current_bundle = state.get("current_bundle_sha256")
    bundle_file = resolve_bundle_file(store, current_bundle)
    if current_bundle is not None:
        joined_command = "\n".join(str(part) for part in command_template)
        if "{bundle_file}" not in joined_command and "{bundle}" not in joined_command:
            raise ValueError("rollout command must consume the frozen bundle artifact")
    completed = {row["logical_id"] for row in store.episodes.records()}
    attempts_by_logical: dict[str, list[dict[str, Any]]] = {}
    for row in store.attempts.records():
        attempts_by_logical.setdefault(str(row["logical_id"]), []).append(row)
    existing_jobs = _known_job_ids(queue)
    jobs: list[RolloutJob] = []
    for index, seed in enumerate(manifest.rollout_seeds):
        logical_id = f"g{manifest.generation:04d}-rollout-{index:03d}"
        attempt_index = len(attempts_by_logical.get(logical_id, ()))
        if attempt_index >= manifest.max_infrastructure_attempts:
            continue
        job_id = (
            f"job-{canonical_sha256([manifest.sha256, logical_id, attempt_index])[:24]}"
        )
        if logical_id in completed or job_id in existing_jobs:
            continue
        output_dir = (
            store.root / "attempts" / logical_id / f"attempt-{attempt_index:03d}"
        )
        values = {
            "campaign_root": str(store.root),
            "logical_id": logical_id,
            "attempt_index": attempt_index,
            "seed": seed,
            "policy_rng": manifest.policy_rng_by_seed[str(seed)],
            "bundle_sha256": current_bundle or "none",
            "baseline_mode": "active_bundle" if current_bundle else "strict_pure_vla",
            "bundle_file": bundle_file,
            "bundle": bundle_file,
            "generation": manifest.generation,
            "output_dir": str(output_dir),
            "result_file": str(output_dir / "episode_record.json"),
            "heartbeat_file": str(output_dir / "heartbeat.jsonl"),
            "task": manifest.task,
            "environment": manifest.environment,
            "env_endpoint": "__RPENT_ENV_ENDPOINT__",
        }
        jobs.append(
            RolloutJob(
                job_id=job_id,
                campaign_root=str(store.root),
                logical_id=logical_id,
                attempt_index=attempt_index,
                task=manifest.task,
                seed=seed,
                policy_rng=manifest.policy_rng_by_seed[str(seed)],
                bundle_sha256=current_bundle,
                command=_render_command(command_template, values),
                output_dir=str(output_dir),
                result_file=values["result_file"],
                heartbeat_file=values["heartbeat_file"],
                timeout_s=manifest.episode_timeout_s,
                no_progress_timeout_s=manifest.no_progress_timeout_s,
                requires_api=bool(manifest.runtime.get("rollout_requires_api", False)),
                requires_environment_slot=bool(
                    manifest.runtime.get("rollout_requires_environment_slot", False)
                ),
            )
        )
    return round_robin_assign({manifest.task: jobs}, worker_hosts)


def enqueue_missing_rollouts(
    *,
    campaign_root: str | Path,
    queue_root: str | Path,
    worker_hosts: tuple[str, ...],
) -> dict[str, Any]:
    store = CampaignStore(campaign_root)
    if CampaignPhase(store.state()["phase"]) != CampaignPhase.ROLLOUT:
        raise ValueError("rollouts can only be enqueued during rollout phase")
    queue = SharedHostQueue(queue_root)
    assignments = build_rollout_jobs(
        store=store, queue=queue, worker_hosts=worker_hosts
    )
    for host, job in assignments:
        queue.enqueue(host, job)
    return {
        "enqueued": len(assignments),
        "by_host": {
            host: sum(assigned == host for assigned, _ in assignments)
            for host in sorted(set(worker_hosts))
        },
        "queue": queue.counts(),
    }


def ingest_queue_results(
    *, campaign_root: str | Path, queue_root: str | Path
) -> dict[str, int]:
    store = CampaignStore(campaign_root)
    queue = SharedHostQueue(queue_root)
    bound_campaign_root = store.root.resolve()
    accepted = 0
    invalid_envelopes = 0
    for envelope_path in queue.terminal_envelope_paths():
        envelope = read_json(envelope_path)
        if not isinstance(envelope, dict):
            invalid_envelopes += 1
            continue
        terminal_result = (
            envelope.get("result")
            if envelope.get("kind") == "rollout_terminal"
            else envelope
        )
        job_payload = envelope.get("job")
        if not isinstance(job_payload, dict) and isinstance(terminal_result, dict):
            job_payload = terminal_result.get("job")
        if not isinstance(job_payload, dict):
            invalid_envelopes += 1
            continue
        job = RolloutJob.from_dict(job_payload)
        if Path(job.campaign_root).resolve() != bound_campaign_root:
            continue
        if not isinstance(terminal_result, dict):
            invalid_envelopes += 1
            continue
        worker_envelope = terminal_result
        payload = worker_envelope.get("result")
        if not isinstance(payload, dict):
            # Terminal queue timestamps are immutable; using wall-clock time
            # here made repeated ingestion synthesize a different payload for
            # the same infra-invalid attempt and fail idempotent recovery.
            finished = str(
                envelope.get("finished_at")
                or worker_envelope.get("finished_at")
                or "1970-01-01T00:00:00+00:00"
            )
            started = str(
                envelope.get("acquired_at")
                or worker_envelope.get("started_at")
                or finished
            )
            payload = EpisodeRecord(
                episode_id=f"infra-{job.job_id}",
                logical_id=job.logical_id,
                generation=store.manifest().generation,
                seed=job.seed,
                policy_rng=job.policy_rng,
                bundle_sha256=job.bundle_sha256,
                status="infra_invalid",
                success=None,
                started_at=started,
                finished_at=finished,
                elapsed_s=float(worker_envelope.get("elapsed_s", 0.0)),
                artifact_index={"worker_envelope": str(envelope_path)},
                invalid_reason=str(
                    worker_envelope.get("watchdog_reason")
                    or worker_envelope.get("worker_exception")
                    or f"rollout_exit_{worker_envelope.get('return_code', 'unknown')}"
                ),
                attempt_index=job.attempt_index,
            ).as_dict()
        try:
            record = EpisodeRecord.from_dict(payload)
            artifact = (
                store.root
                / "attempts"
                / record.logical_id
                / f"attempt-{record.attempt_index:03d}"
                / "record.json"
            )
            if artifact.exists() and record.status == "infra_invalid":
                existing_payload = read_json(artifact)
                existing_payload.pop("attempt_id", None)
                existing = EpisodeRecord.from_dict(existing_payload)
                comparable = (
                    "episode_id",
                    "logical_id",
                    "generation",
                    "seed",
                    "policy_rng",
                    "bundle_sha256",
                    "status",
                    "success",
                    "elapsed_s",
                    "invalid_reason",
                    "attempt_index",
                )
                if all(
                    getattr(existing, field) == getattr(record, field)
                    for field in comparable
                ):
                    # Legacy versions synthesized wall-clock timestamps while
                    # ingesting infra-invalid terminals. Permit only that
                    # proven timestamp-only replay mismatch; every semantic
                    # field remains fail-closed.
                    continue
            if store.record_episode(record):
                accepted += 1
        except ValueError as exc:
            # Exact duplicate ingestion is already idempotent in the ledger;
            # all other mismatches are fail-closed and visible to the caller.
            if "immutable artifact already differs" in str(exc):
                raise
            if "duplicate ledger key" in str(exc):
                raise
    return {
        "accepted": accepted,
        "infra_invalid": sum(
            row.get("status") == "infra_invalid" for row in store.attempts.records()
        ),
        "invalid_envelopes": invalid_envelopes,
    }


def analyze_failures(campaign_root: str | Path) -> dict[str, Any]:
    store = CampaignStore(campaign_root)
    manifest = store.manifest()
    records = [EpisodeRecord.from_dict(row) for row in store.episodes.records()]
    valid = [record for record in records if record.status == "valid"]
    if len(valid) != manifest.expected_rollouts:
        raise ValueError(
            f"analysis requires {manifest.expected_rollouts} valid episodes; got {len(valid)}"
        )
    failed_with_segments = [
        record for record in valid if not record.success and record.all_failure_segments
    ]
    segments = [
        segment
        for record in failed_with_segments
        for segment in record.all_failure_segments
    ]
    clusters = cluster_failure_segments(segments)
    report = {
        "schema_version": 1,
        "campaign_id": manifest.campaign_id,
        "manifest_sha256": manifest.sha256,
        "valid_episodes": len(valid),
        "successes": sum(bool(record.success) for record in valid),
        "failures_with_segments": len(failed_with_segments),
        "failure_segments": len(segments),
        "clusters": [cluster.as_dict() for cluster in clusters],
        "dominant_cluster_id": clusters[0].cluster_id if clusters else None,
    }
    atomic_write_json(
        store.root / "analysis" / "failure_clusters.json", report, overwrite=False
    )
    if CampaignPhase(store.state()["phase"]) == CampaignPhase.ROLLOUT:
        store.transition(CampaignPhase.CLUSTER)
    return report


def load_episode_records(path: str | Path) -> list[EpisodeRecord]:
    source = Path(path)
    if source.is_dir():
        files = sorted(source.glob("**/episode_record.json"))
        return [EpisodeRecord.from_dict(read_json(file)) for file in files]
    records: list[EpisodeRecord] = []
    with source.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(EpisodeRecord.from_dict(json.loads(line)))
    return records
