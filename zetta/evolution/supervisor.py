# Copyright (c) 2026 Zetta Contributors
"""Idempotent bounded supervisor for one campaign and cross-generation handoff."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from zetta.evolution.campaign import (
    analyze_failures,
    enqueue_missing_rollouts,
    ingest_queue_results,
)
from zetta.evolution.gate_runner import CandidateGateRunner, PairedGateRunner
from zetta.evolution.jsonio import atomic_write_json, canonical_sha256, read_json
from zetta.evolution.lifecycle import run_diagnosis_stage, run_proposal_stage
from zetta.evolution.models import CampaignManifest, CampaignPhase
from zetta.evolution.schedule import preregister_seed_schedule
from zetta.evolution.store import CampaignStore


class EvolutionSupervisor:
    """Execute one recoverable state-machine step, never an unbounded hidden loop."""

    def __init__(
        self,
        *,
        campaign_root: str | Path,
        queue_root: str | Path,
        worker_hosts: tuple[str, ...],
        tool_catalog: dict[str, Any],
        model: str | None = None,
        next_campaign_root: str | Path | None = None,
        next_queue_root: str | Path | None = None,
        next_master_seed: int | None = None,
    ) -> None:
        if not worker_hosts:
            raise ValueError("supervisor requires at least one worker host")
        self.store = CampaignStore(campaign_root)
        self.queue_root = Path(queue_root)
        self.worker_hosts = tuple(worker_hosts)
        self.tool_catalog = tool_catalog
        self.model = model
        self.next_campaign_root = (
            Path(next_campaign_root) if next_campaign_root is not None else None
        )
        self.next_queue_root = (
            Path(next_queue_root) if next_queue_root is not None else None
        )
        self.next_master_seed = next_master_seed

    def step(self) -> dict[str, Any]:
        phase = CampaignPhase(self.store.state()["phase"])
        if phase == CampaignPhase.ROLLOUT:
            ingestion = ingest_queue_results(
                campaign_root=self.store.root, queue_root=self.queue_root
            )
            status = self.store.status()
            expected = self.store.manifest().expected_rollouts
            if int(status["episodes"]["valid"]) == expected:
                analysis = analyze_failures(self.store.root)
                return {"action": "rollout_complete_clustered", "analysis": analysis}
            enqueue = enqueue_missing_rollouts(
                campaign_root=self.store.root,
                queue_root=self.queue_root,
                worker_hosts=self.worker_hosts,
            )
            queue_counts = enqueue["queue"]
            in_flight = int(queue_counts.get("pending", 0)) + int(
                queue_counts.get("running", 0)
            )
            if enqueue["blocked"] and enqueue["enqueued"] == 0 and in_flight == 0:
                # Every remaining logical id has exhausted
                # ``max_infrastructure_attempts`` with no valid episode and
                # nothing else is in flight: further polling can never make
                # progress. Fail closed instead of the caller looping forever
                # on repeated ``waiting_for_rollouts`` (mirrors
                # ``CandidateGateRunner``'s ``blocked``/
                # ``authorize_infrastructure_recovery`` design for the
                # same-seed/regression/held-out gates).
                return {
                    "action": "rollout_blocked_on_infrastructure",
                    "ingestion": ingestion,
                    "enqueue": enqueue,
                    "blocked_logical_ids": enqueue["blocked"],
                }
            return {
                "action": "waiting_for_rollouts",
                "ingestion": ingestion,
                "enqueue": enqueue,
            }
        if phase in {CampaignPhase.CLUSTER, CampaignPhase.DIAGNOSE}:
            diagnosis = run_diagnosis_stage(
                campaign_root=self.store.root,
                tool_catalog=self.tool_catalog,
                model=self.model,
            )
            return {"action": "diagnosed", "diagnosis": diagnosis}
        if phase == CampaignPhase.PROPOSE:
            proposal = run_proposal_stage(
                campaign_root=self.store.root,
                tool_catalog=self.tool_catalog,
                model=self.model,
            )
            return {"action": "candidate_proposed", "proposal": proposal}
        if phase == CampaignPhase.SAME_SEED_GATE:
            gate = CandidateGateRunner(
                campaign_root=self.store.root,
                queue_root=self.queue_root,
                worker_hosts=self.worker_hosts,
            )
            report = gate.run_once()
            return {
                "action": (
                    "same_seed_gate_decided"
                    if report.get("decision") is not None
                    else "waiting_for_same_seed_gate"
                ),
                "gate": report,
            }
        if phase in {CampaignPhase.REGRESSION_GATE, CampaignPhase.HELDOUT_GATE}:
            gate_kind = (
                "regression" if phase == CampaignPhase.REGRESSION_GATE else "heldout"
            )
            if phase == CampaignPhase.HELDOUT_GATE and (
                self.store.manifest().expected_heldout == 20
                or self.store.manifest().runtime.get("heldout_gate_kind")
                == "heldout_20"
            ):
                gate_kind = "heldout_20"
            gate = PairedGateRunner(
                campaign_root=self.store.root,
                queue_root=self.queue_root,
                worker_hosts=self.worker_hosts,
                gate_kind=gate_kind,
            )
            report = gate.run_once()
            if report.get("terminal_decision") is not None:
                action = f"{gate_kind}_gate_decided"
            elif report.get("decision") is not None:
                action = f"{gate_kind}_gate_expanded"
            else:
                action = f"waiting_for_{gate_kind}_gate"
            return {
                "action": action,
                "gate": report,
            }
        if phase == CampaignPhase.PROMOTE:
            generation = self.store.manifest().generation + 1
            child_root = self.next_campaign_root or (
                self.store.root / "children" / f"generation-{generation:04d}"
            )
            # Keep the default queue stable across generations. Persistent
            # workers watch one shared queue and therefore continue consuming
            # the child generation without an out-of-band worker restart.
            # Callers may still provide an explicit child queue when they also
            # own an external worker handoff.
            child_queue = self.next_queue_root or self.queue_root
            report = promote_and_spawn_generation(
                campaign_root=self.store.root,
                next_campaign_root=child_root,
                next_queue_root=child_queue,
                worker_hosts=self.worker_hosts,
                next_master_seed=self.next_master_seed,
            )
            return {"action": "promoted_and_spawned_generation", **report}
        if phase == CampaignPhase.COMPLETE:
            continuation_path = (
                self.store.root / "analysis" / "generation-continuation.json"
            )
            if continuation_path.is_file():
                continuation = read_json(continuation_path)
                child = self._validated_child_supervisor(continuation)
                return {
                    "action": "continued_child_generation",
                    "parent_manifest_sha256": self.store.manifest().sha256,
                    "continuation": continuation,
                    "child": child.step(),
                }
        return {
            "action": "awaiting_formal_gate_or_terminal",
            "phase": phase.value,
            "status": self.store.status(),
        }

    def _validated_child_supervisor(
        self, continuation: dict[str, Any]
    ) -> EvolutionSupervisor:
        """Recover an immutable child handoff and fail closed on stale binding."""

        parent_manifest = self.store.manifest()
        if continuation.get("parent_manifest_sha256") != parent_manifest.sha256:
            raise ValueError("generation continuation parent manifest changed")
        child_root_value = continuation.get("child_campaign_root")
        child_queue_value = continuation.get("child_queue_root")
        child_manifest_sha256 = continuation.get("child_manifest_sha256")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (child_root_value, child_queue_value, child_manifest_sha256)
        ):
            raise ValueError("generation continuation is incomplete")
        child_root = Path(str(child_root_value)).resolve()
        child_queue = Path(str(child_queue_value)).resolve()
        child_store = CampaignStore(child_root)
        child_manifest = child_store.manifest()
        if child_manifest.sha256 != child_manifest_sha256:
            raise ValueError("generation continuation child manifest changed")
        if child_manifest.generation != parent_manifest.generation + 1:
            raise ValueError("generation continuation skipped a generation")
        promoted = continuation.get("promoted_bundle_sha256")
        if (
            child_manifest.baseline_mode != "active_bundle"
            or child_manifest.parent_bundle_sha256 != promoted
            or child_manifest.active_bundle_sha256 != promoted
            or self.store.state().get("current_bundle_sha256") != promoted
        ):
            raise ValueError("generation continuation bundle binding changed")
        return EvolutionSupervisor(
            campaign_root=child_root,
            queue_root=child_queue,
            worker_hosts=self.worker_hosts,
            tool_catalog=self.tool_catalog,
            model=self.model,
        )


def _derived_child_master_seed(
    *, parent_manifest: CampaignManifest, promoted_sha256: str
) -> int:
    digest = canonical_sha256(
        {
            "parent_manifest_sha256": parent_manifest.sha256,
            "promoted_bundle_sha256": promoted_sha256,
            "child_generation": parent_manifest.generation + 1,
            "purpose": "child-preregistered-seed-schedule",
        }
    )
    return int(digest[:16], 16) & 0x7FFFFFFF


def build_child_manifest(
    *,
    campaign_root: str | Path,
    promoted_bundle_sha256: str,
    next_master_seed: int | None = None,
    population: range = range(100_000),
) -> CampaignManifest:
    """Build and immutably preregister one deterministic child manifest."""

    store = CampaignStore(campaign_root)
    parent = store.manifest()
    state = store.state()
    if CampaignPhase(state["phase"]) not in {
        CampaignPhase.PROMOTE,
        CampaignPhase.COMPLETE,
    }:
        raise ValueError("child manifest requires a promoted parent campaign")
    promotions = [
        row
        for row in store.promotions.records()
        if row.get("candidate_sha256") == promoted_bundle_sha256
    ]
    if len(promotions) != 1:
        raise ValueError("child manifest requires one immutable promotion record")

    parent_runtime = dict(parent.runtime)
    historical = parent_runtime.get("evolution_lineage_used_seeds", ())
    if not isinstance(historical, (list, tuple)) or not all(
        isinstance(seed, int) for seed in historical
    ):
        raise ValueError("evolution_lineage_used_seeds must be an integer array")
    used = {int(seed) for seed in historical}
    used.update(parent.rollout_seeds)
    schedule_population = tuple(seed for seed in population if seed not in used)
    master_seed = (
        next_master_seed
        if next_master_seed is not None
        else _derived_child_master_seed(
            parent_manifest=parent,
            promoted_sha256=promoted_bundle_sha256,
        )
    )
    next_rollout_count = int(
        parent_runtime.get("subsequent_rollout_count", parent.expected_rollouts)
    )
    if next_rollout_count < 1:
        raise ValueError("subsequent_rollout_count must be positive")
    rollout, heldout, policy_rng = preregister_seed_schedule(
        master_seed=master_seed,
        task=parent.task,
        rollout_count=next_rollout_count,
        heldout_count=parent.expected_heldout,
        population=schedule_population,
        heldout_seeds=parent.heldout_seeds,
    )
    for seed in heldout:
        policy_rng[str(seed)] = parent.policy_rng_by_seed[str(seed)]
    bundle_path = (
        store.root / "candidates" / promoted_bundle_sha256 / "bundle.json"
    ).resolve()
    if (
        not bundle_path.is_file()
        or canonical_sha256(read_json(bundle_path)) != promoted_bundle_sha256
    ):
        raise ValueError("promoted bundle artifact is missing or corrupted")
    bundle_files = dict(parent_runtime.get("bundle_files_by_sha", {}))
    bundle_files[promoted_bundle_sha256] = str(bundle_path)
    lineage_id = str(parent_runtime.get("campaign_lineage_id", parent.campaign_id))
    child_runtime = {
        **parent_runtime,
        "bundle_files_by_sha": bundle_files,
        "campaign_lineage_id": lineage_id,
        "evolution_lineage_used_seeds": sorted(used),
        "generation_parent_manifest_sha256": parent.sha256,
        "generation_promotion_sha256": canonical_sha256(promotions[0]),
        "generation_schedule_master_seed": master_seed,
        # A promoted Critic can wake Role1 during ordinary child rollouts.
        # Provider admission is unnecessary only for the strict-empty Gen0.
        "rollout_requires_api": True,
        "candidate_rollout_requires_api": True,
    }
    generation = parent.generation + 1
    child = replace(
        parent,
        campaign_id=f"{lineage_id}-g{generation:04d}",
        generation=generation,
        rollout_seeds=rollout,
        heldout_seeds=heldout,
        policy_rng_by_seed=policy_rng,
        parent_bundle_sha256=promoted_bundle_sha256,
        baseline_mode="active_bundle",
        active_bundle_sha256=promoted_bundle_sha256,
        expected_rollouts=next_rollout_count,
        runtime=child_runtime,
    )
    preregistration = {
        "schema_version": 1,
        "parent_manifest_sha256": parent.sha256,
        "promotion_sha256": canonical_sha256(promotions[0]),
        "promoted_bundle_sha256": promoted_bundle_sha256,
        "master_seed": master_seed,
        "excluded_seed_sha256": canonical_sha256(sorted(used)),
        "child_manifest": child.as_dict(),
        "child_manifest_sha256": child.sha256,
    }
    path = store.root / "analysis" / f"child-preregistration-g{generation:04d}.json"
    atomic_write_json(path, preregistration, overwrite=False)
    if canonical_sha256(read_json(path)) != canonical_sha256(preregistration):
        raise ValueError("child preregistration changed during recovery")
    return child


def _recover_or_promote(store: CampaignStore) -> dict[str, Any]:
    """Finish an interrupted promotion without appending a second ledger row."""

    state = store.state()
    phase = CampaignPhase(state["phase"])
    if phase not in {CampaignPhase.PROMOTE, CampaignPhase.COMPLETE}:
        raise ValueError("generation continuation requires promote/complete phase")
    rows = store.promotions.records()
    if len(rows) > 1:
        raise ValueError("generation has multiple promotion ledger rows")
    active_candidate = state.get("candidate_sha256")
    recover_state_pointer = False
    if not rows:
        if phase != CampaignPhase.PROMOTE or not isinstance(active_candidate, str):
            raise ValueError("promotion recovery is missing a unique candidate")
        promotion = store.promote(active_candidate)
    else:
        promotion = rows[0]
        promoted = promotion.get("candidate_sha256")
        if not isinstance(promoted, str):
            raise ValueError("promotion ledger has no candidate digest")
        if active_candidate not in {None, promoted}:
            raise ValueError("promotion ledger conflicts with active candidate")
        if promotion.get("generation") != state.get("generation"):
            raise ValueError("promotion ledger generation changed")
        recover_state_pointer = (
            active_candidate == promoted
            or state.get("current_bundle_sha256") != promoted
        )

    promoted_sha256 = str(promotion["candidate_sha256"])
    candidate_path = store.root / "candidates" / promoted_sha256 / "bundle.json"
    candidate = read_json(candidate_path)
    if canonical_sha256(candidate) != promoted_sha256:
        raise ValueError("promotion candidate artifact digest mismatch")
    if candidate.get("parent_sha256") != promotion.get("parent_sha256"):
        raise ValueError("promotion candidate parent changed")
    if promotion.get("promotion_id") != f"promote-{promoted_sha256[:20]}":
        raise ValueError("promotion ledger identity changed")
    candidate_gates = [
        row
        for row in store.gates.records()
        if row.get("candidate_sha256") == promoted_sha256
    ]
    passed_kinds = {row["kind"] for row in candidate_gates if row.get("passed")}
    policy = store.manifest().runtime.get("evolution_policy", {})
    skip_regression = isinstance(policy, dict) and bool(
        policy.get("skip_regression_gate", False)
    )
    required = {"same_seed"} if skip_regression else {"same_seed", "regression"}
    if not required.issubset(passed_kinds) or not (
        {"heldout_10", "heldout_20", "heldout_50"} & passed_kinds
    ):
        raise ValueError("promotion ledger is not supported by all formal gates")
    expected_gate_ids = sorted(row["decision_id"] for row in candidate_gates)
    if promotion.get("gate_decision_ids") != expected_gate_ids:
        raise ValueError("promotion gate evidence changed")
    if recover_state_pointer:
        store.update_state(
            current_bundle_sha256=promoted_sha256,
            candidate_sha256=None,
        )
    atomic_write_json(
        store.root / "bundles" / f"generation-{store.manifest().generation:04d}.json",
        candidate,
        overwrite=False,
    )
    return promotion


def promote_and_spawn_generation(
    *,
    campaign_root: str | Path,
    next_campaign_root: str | Path,
    next_queue_root: str | Path,
    worker_hosts: tuple[str, ...],
    next_manifest: CampaignManifest | None = None,
    next_master_seed: int | None = None,
) -> dict[str, Any]:
    """Atomically bind a promoted bundle into a preregistered child generation."""

    store = CampaignStore(campaign_root)
    parent_manifest = store.manifest()
    promotion = _recover_or_promote(store)

    promoted_sha256 = str(promotion["candidate_sha256"])
    generated = next_manifest is None
    if next_manifest is None:
        next_manifest = build_child_manifest(
            campaign_root=campaign_root,
            promoted_bundle_sha256=promoted_sha256,
            next_master_seed=next_master_seed,
        )
    if next_manifest.generation != parent_manifest.generation + 1:
        raise ValueError("child generation must increment exactly once")
    if (next_manifest.environment, next_manifest.task) != (
        parent_manifest.environment,
        parent_manifest.task,
    ):
        raise ValueError("child generation changed environment or task")
    if (
        next_manifest.baseline_mode != "active_bundle"
        or next_manifest.parent_bundle_sha256 != promoted_sha256
        or next_manifest.active_bundle_sha256 != promoted_sha256
    ):
        raise ValueError("child generation is not bound to the promoted bundle")

    child = CampaignStore(next_campaign_root)
    child.initialize(next_manifest)
    enqueue = enqueue_missing_rollouts(
        campaign_root=next_campaign_root,
        queue_root=next_queue_root,
        worker_hosts=worker_hosts,
    )
    continuation = {
        "schema_version": 1,
        "parent_manifest_sha256": parent_manifest.sha256,
        "promotion_sha256": canonical_sha256(promotion),
        "promoted_bundle_sha256": promoted_sha256,
        "child_manifest_sha256": next_manifest.sha256,
        "child_generation": next_manifest.generation,
        "child_campaign_root": str(Path(next_campaign_root).resolve()),
        "child_queue_root": str(Path(next_queue_root).resolve()),
        "child_manifest_generated": generated,
    }
    path = store.root / "analysis" / "generation-continuation.json"
    if path.is_file():
        if canonical_sha256(read_json(path)) != canonical_sha256(continuation):
            raise ValueError("generation continuation binding changed")
    else:
        atomic_write_json(path, continuation, overwrite=False)
    if CampaignPhase(store.state()["phase"]) == CampaignPhase.PROMOTE:
        store.transition(CampaignPhase.COMPLETE)
    return {
        "promotion": promotion,
        "child_manifest": next_manifest.as_dict(),
        "continuation": continuation,
        "enqueue": enqueue,
    }


__all__ = [
    "EvolutionSupervisor",
    "build_child_manifest",
    "promote_and_spawn_generation",
]
