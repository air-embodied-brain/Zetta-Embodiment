# Copyright (c) 2026 RPent Contributors
"""Fail-closed campaign persistence and atomic promotion."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rpent.evolution.jsonio import (
    AppendOnlyLedger,
    atomic_write_json,
    canonical_sha256,
    read_json,
)
from rpent.evolution.models import (
    CampaignManifest,
    CampaignPhase,
    CandidateBundle,
    CausalDiagnosis,
    EpisodeRecord,
    GateDecision,
)

ALLOWED_TRANSITIONS: dict[CampaignPhase, set[CampaignPhase]] = {
    CampaignPhase.ROLLOUT: {CampaignPhase.CLUSTER},
    # COMPLETE is a valid bounded outcome when a full generation has no
    # failures to diagnose.  It is accepted from DIAGNOSE as well so a
    # process that had already advanced before a crash can recover cleanly.
    CampaignPhase.CLUSTER: {CampaignPhase.DIAGNOSE, CampaignPhase.COMPLETE},
    CampaignPhase.DIAGNOSE: {CampaignPhase.PROPOSE, CampaignPhase.COMPLETE},
    CampaignPhase.PROPOSE: {CampaignPhase.SAME_SEED_GATE},
    CampaignPhase.SAME_SEED_GATE: {
        CampaignPhase.DIAGNOSE,
        CampaignPhase.PROPOSE,
        CampaignPhase.REGRESSION_GATE,
        # A preregistered campaign may skip regression after a conclusive
        # same-seed pass and proceed directly to the held-out gate.
        CampaignPhase.HELDOUT_GATE,
        CampaignPhase.COMPLETE,
    },
    CampaignPhase.REGRESSION_GATE: {
        CampaignPhase.DIAGNOSE,
        CampaignPhase.PROPOSE,
        CampaignPhase.HELDOUT_GATE,
        CampaignPhase.COMPLETE,
    },
    CampaignPhase.HELDOUT_GATE: {
        CampaignPhase.DIAGNOSE,
        CampaignPhase.PROPOSE,
        CampaignPhase.PROMOTE,
        CampaignPhase.COMPLETE,
    },
    CampaignPhase.PROMOTE: {CampaignPhase.ROLLOUT, CampaignPhase.COMPLETE},
    CampaignPhase.COMPLETE: set(),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CampaignStore:
    """Filesystem source of truth for one campaign generation."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.json"
        self.state_path = self.root / "state.json"
        self.episodes = AppendOnlyLedger(
            self.root / "ledgers" / "episodes.jsonl", key="logical_id"
        )
        self.attempts = AppendOnlyLedger(
            self.root / "ledgers" / "attempts.jsonl", key="attempt_id"
        )
        self.gates = AppendOnlyLedger(
            self.root / "ledgers" / "gates.jsonl", key="decision_id"
        )
        self.diagnoses = AppendOnlyLedger(
            self.root / "ledgers" / "diagnoses.jsonl", key="diagnosis_id"
        )
        self.candidate_ledger = AppendOnlyLedger(
            self.root / "ledgers" / "candidates.jsonl", key="candidate_sha256"
        )
        self.promotions = AppendOnlyLedger(
            self.root / "ledgers" / "promotions.jsonl", key="promotion_id"
        )
        # Threshold overrides are a separate append-only provenance stream.
        # They must never rewrite gate plans or the gate decision ledger.
        self.same_seed_threshold_authorizations = AppendOnlyLedger(
            self.root / "ledgers" / "same-seed-threshold-authorizations.jsonl",
            key="authorization_id",
        )

    def initialize(self, manifest: CampaignManifest) -> dict[str, Any]:
        self.root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(self.manifest_path, manifest.as_dict(), overwrite=False)
        expected_state = {
            "schema_version": 1,
            "manifest_sha256": manifest.sha256,
            "phase": CampaignPhase.ROLLOUT.value,
            "generation": manifest.generation,
            "current_bundle_sha256": manifest.parent_bundle_sha256,
            "candidate_sha256": None,
            "updated_at": _utc_now(),
        }
        if self.state_path.exists():
            state = read_json(self.state_path)
            if state.get("manifest_sha256") != manifest.sha256:
                raise ValueError("campaign root is bound to a different manifest")
            return state
        atomic_write_json(self.state_path, expected_state, overwrite=False)
        return expected_state

    def manifest(self) -> CampaignManifest:
        return CampaignManifest.from_dict(read_json(self.manifest_path))

    def state(self) -> dict[str, Any]:
        state = read_json(self.state_path)
        manifest = self.manifest()
        if state.get("manifest_sha256") != manifest.sha256:
            raise ValueError("state/manifest digest mismatch")
        return state

    def transition(
        self,
        target: CampaignPhase,
        *,
        state_updates: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = self.state()
        source = CampaignPhase(state["phase"])
        if target not in ALLOWED_TRANSITIONS[source]:
            raise ValueError(
                f"invalid campaign transition: {source.value} -> {target.value}"
            )
        updates = dict(state_updates or {})
        if "phase" in updates or "manifest_sha256" in updates:
            raise ValueError("state_updates cannot replace phase or manifest binding")
        updated = {
            **state,
            **updates,
            "phase": target.value,
            "updated_at": _utc_now(),
        }
        atomic_write_json(self.state_path, updated, overwrite=True)
        return updated

    def update_state(self, **updates: Any) -> dict[str, Any]:
        """Atomically persist auditable reducer metadata without changing phase."""

        if "phase" in updates or "manifest_sha256" in updates:
            raise ValueError("state update cannot replace phase or manifest binding")
        state = self.state()
        updated = {**state, **updates, "updated_at": _utc_now()}
        atomic_write_json(self.state_path, updated, overwrite=True)
        return updated

    def reopen_completed_provisional_hypothesis(
        self,
        *,
        authorization_id: str,
        authorization_sha256: str,
        diagnosis_sha256: str,
        same_seed_pass_rate: float,
        skip_regression: bool,
    ) -> dict[str, Any]:
        """Reopen one terminal, inconclusive diagnosis for a bounded live test.

        This is deliberately narrower than a general phase rewind. The caller
        must first persist and validate the immutable authorization artifact.
        Episodes, diagnoses, candidates and gates remain append-only.
        """

        state = self.state()
        if CampaignPhase(state["phase"]) != CampaignPhase.COMPLETE:
            raise ValueError("provisional recovery requires a complete campaign")
        if state.get("optimization_outcome") != "no_actionable_cluster_diagnosis":
            raise ValueError("campaign did not complete from inconclusive diagnosis")
        if state.get("candidate_sha256") is not None:
            raise ValueError("provisional recovery cannot replace an active candidate")
        if (
            self.candidate_ledger.records()
            or self.gates.records()
            or self.promotions.records()
        ):
            raise ValueError("provisional recovery requires an ungated generation")
        if not authorization_id or len(authorization_sha256) != 64:
            raise ValueError("provisional recovery authorization is invalid")
        if len(diagnosis_sha256) != 64:
            raise ValueError("provisional recovery diagnosis digest is invalid")
        if not 0 < float(same_seed_pass_rate) <= 1:
            raise ValueError("provisional same-seed pass rate must be in (0, 1]")
        updated = {
            **state,
            "phase": CampaignPhase.PROPOSE.value,
            "optimization_outcome": "provisional_hypothesis_gate_authorized",
            "provisional_authorization_id": authorization_id,
            "provisional_authorization_sha256": authorization_sha256,
            "provisional_diagnosis_sha256": diagnosis_sha256,
            "provisional_same_seed_pass_rate": float(same_seed_pass_rate),
            "provisional_skip_regression": bool(skip_regression),
            "candidate_round": 0,
            "updated_at": _utc_now(),
        }
        atomic_write_json(self.state_path, updated, overwrite=True)
        return updated

    def transition_timeboxed_same_seed_to_heldout(
        self, *, authorization_id: str, authorization_sha256: str
    ) -> dict[str, Any]:
        """Skip regression only under the frozen provisional authorization."""

        state = self.state()
        if CampaignPhase(state["phase"]) != CampaignPhase.SAME_SEED_GATE:
            raise ValueError("timeboxed heldout transition requires same-seed phase")
        if not state.get("provisional_skip_regression"):
            raise ValueError("regression skip was not authorized")
        if (
            state.get("provisional_authorization_id") != authorization_id
            or state.get("provisional_authorization_sha256") != authorization_sha256
        ):
            raise ValueError("timeboxed authorization binding changed")
        updated = {
            **state,
            "phase": CampaignPhase.HELDOUT_GATE.value,
            "optimization_outcome": "same_seed_passed_regression_skipped_timeboxed",
            "updated_at": _utc_now(),
        }
        atomic_write_json(self.state_path, updated, overwrite=True)
        return updated

    def record_episode(self, record: EpisodeRecord) -> bool:
        manifest = self.manifest()
        if record.generation != manifest.generation:
            raise ValueError("episode generation does not match campaign")
        expected_rng = manifest.policy_rng_by_seed.get(str(record.seed))
        if expected_rng is None or expected_rng != record.policy_rng:
            raise ValueError("episode seed/policy RNG is not preregistered")
        if record.bundle_sha256 != self.state().get("current_bundle_sha256"):
            raise ValueError("episode did not use the frozen current bundle")
        attempt_payload = {"attempt_id": record.attempt_id, **record.as_dict()}
        artifact = (
            self.root
            / "attempts"
            / record.logical_id
            / f"attempt-{record.attempt_index:03d}"
            / "record.json"
        )
        atomic_write_json(artifact, attempt_payload, overwrite=False)
        self.attempts.append(attempt_payload)
        if record.status == "infra_invalid":
            return False
        canonical = self.root / "episodes" / record.logical_id / "record.json"
        atomic_write_json(canonical, record.as_dict(), overwrite=False)
        return self.episodes.append(record.as_dict())

    def register_candidate(self, candidate: CandidateBundle) -> str:
        state = self.state()
        if CampaignPhase(state["phase"]) != CampaignPhase.PROPOSE:
            raise ValueError("candidate registration requires propose phase")
        if candidate.parent_sha256 != state.get("current_bundle_sha256"):
            raise ValueError("candidate parent is stale")
        path = self.root / "candidates" / candidate.sha256 / "bundle.json"
        atomic_write_json(path, candidate.as_dict(), overwrite=False)
        self.candidate_ledger.append(
            {
                "candidate_sha256": candidate.sha256,
                "candidate_id": candidate.candidate_id,
                "generation": candidate.generation,
                "parent_sha256": candidate.parent_sha256,
                "diagnosis_sha256": candidate.diagnosis_sha256,
            }
        )
        updated = {
            **state,
            "candidate_sha256": candidate.sha256,
            "updated_at": _utc_now(),
        }
        atomic_write_json(self.state_path, updated, overwrite=True)
        return candidate.sha256

    def recover_registered_candidate(self, candidate_sha256: str) -> dict[str, Any]:
        """Finish the state-pointer step after an interrupted candidate commit."""

        state = self.state()
        if CampaignPhase(state["phase"]) != CampaignPhase.PROPOSE:
            raise ValueError("candidate registration recovery requires propose phase")
        rows = self.candidate_ledger.records()
        matches = [
            row for row in rows if row.get("candidate_sha256") == candidate_sha256
        ]
        if len(matches) != 1 or rows[-1] != matches[0]:
            raise ValueError("candidate recovery requires the latest unique ledger row")
        bundle_path = self.root / "candidates" / candidate_sha256 / "bundle.json"
        bundle = read_json(bundle_path)
        if canonical_sha256(bundle) != candidate_sha256:
            raise ValueError("candidate recovery bundle digest mismatch")
        candidate = CandidateBundle.from_dict(bundle)
        if candidate.parent_sha256 != state.get("current_bundle_sha256"):
            raise ValueError("candidate recovery parent is stale")
        active = state.get("candidate_sha256")
        if active not in {None, candidate_sha256}:
            active_decisions = [
                row
                for row in self.gates.records()
                if row.get("candidate_sha256") == active
            ]
            if not active_decisions or bool(active_decisions[-1].get("passed")):
                raise ValueError(
                    "candidate recovery would overwrite an active candidate"
                )
        updated = {
            **state,
            "candidate_sha256": candidate_sha256,
            "updated_at": _utc_now(),
        }
        atomic_write_json(self.state_path, updated, overwrite=True)
        return updated

    def register_diagnosis(self, diagnosis: CausalDiagnosis) -> bool:
        state = self.state()
        if CampaignPhase(state["phase"]) != CampaignPhase.DIAGNOSE:
            raise ValueError("diagnosis registration requires diagnose phase")
        artifact = self.root / "diagnoses" / diagnosis.diagnosis_id / "diagnosis.json"
        atomic_write_json(artifact, diagnosis.as_dict(), overwrite=False)
        return self.diagnoses.append(diagnosis.as_dict())

    def record_gate(self, decision: GateDecision) -> bool:
        state = self.state()
        if decision.candidate_sha256 != state.get("candidate_sha256"):
            raise ValueError("gate decision targets a stale candidate")
        phase = CampaignPhase(state["phase"])
        expected_phase = {
            "same_seed": CampaignPhase.SAME_SEED_GATE,
            "regression": CampaignPhase.REGRESSION_GATE,
            "heldout_10": CampaignPhase.HELDOUT_GATE,
            "heldout_20": CampaignPhase.HELDOUT_GATE,
            "heldout_50": CampaignPhase.HELDOUT_GATE,
        }[decision.kind]
        if phase != expected_phase:
            raise ValueError(f"{decision.kind} decision cannot be recorded in {phase}")
        return self.gates.append(decision.as_dict())

    def promote(self, candidate_sha256: str) -> dict[str, Any]:
        state = self.state()
        if CampaignPhase(state["phase"]) != CampaignPhase.PROMOTE:
            raise ValueError("promotion requires promote phase")
        if self.manifest().runtime.get("evaluation_scope") == "paired_heldout_only":
            raise ValueError("heldout-only evaluation campaigns cannot promote")
        if state.get("candidate_sha256") != candidate_sha256:
            raise ValueError("promotion candidate is stale")
        candidate_path = self.root / "candidates" / candidate_sha256 / "bundle.json"
        candidate = read_json(candidate_path)
        if canonical_sha256(candidate) != candidate_sha256:
            raise ValueError("candidate artifact digest mismatch")
        parent = state.get("current_bundle_sha256")
        if candidate.get("parent_sha256") != parent:
            raise ValueError("candidate parent no longer matches current bundle")
        gates = [
            row
            for row in self.gates.records()
            if row["candidate_sha256"] == candidate_sha256
        ]
        kinds = {row["kind"] for row in gates if row.get("passed")}
        policy = self.manifest().runtime.get("evolution_policy", {})
        skip_regression = isinstance(policy, dict) and bool(
            policy.get("skip_regression_gate", False)
        )
        # A same-seed threshold authorization may explicitly skip regression
        # for this candidate.  The authorization artifact is validated before
        # the held-out transition; this state binding prevents a later
        # candidate from inheriting the exception.
        if (
            state.get("same_seed_threshold_authorization_candidate_sha256")
            == candidate_sha256
            and bool(state.get("same_seed_threshold_authorization_skip_regression"))
        ):
            skip_regression = True
        required = {"same_seed"} if skip_regression else {"same_seed", "regression"}
        if not required.issubset(kinds):
            raise ValueError("candidate has not passed all required pre-heldout gates")
        heldout_kinds = {"heldout_10", "heldout_20", "heldout_50"}
        heldout_mode = str(
            policy.get("heldout_mode", "validation")
            if isinstance(policy, dict)
            else "validation"
        )
        if heldout_mode == "test":
            # Test mode still requires a completed fixed held-out decision,
            # but the result is intentionally report-only and may be false.
            if not any(row.get("kind") in heldout_kinds for row in gates):
                raise ValueError("candidate has not completed the heldout test")
        elif not (heldout_kinds & kinds):
            raise ValueError("candidate has not passed a heldout gate")
        promotion = {
            "promotion_id": f"promote-{candidate_sha256[:20]}",
            "candidate_sha256": candidate_sha256,
            "parent_sha256": parent,
            "generation": state["generation"],
            "promoted_at": _utc_now(),
            "gate_decision_ids": sorted(row["decision_id"] for row in gates),
        }
        self.promotions.append(promotion)
        updated = {
            **state,
            "current_bundle_sha256": candidate_sha256,
            "candidate_sha256": None,
            "updated_at": _utc_now(),
        }
        atomic_write_json(self.state_path, updated, overwrite=True)
        atomic_write_json(
            self.root / "bundles" / f"generation-{state['generation']:04d}.json",
            candidate,
            overwrite=False,
        )
        return promotion

    def status(self) -> dict[str, Any]:
        valid = self.episodes.records()
        attempts = self.attempts.records()
        return {
            "manifest": self.manifest().as_dict(),
            "state": self.state(),
            "episodes": {
                "total_attempts": len(attempts),
                "valid": len(valid),
                "infra_invalid": sum(
                    row.get("status") == "infra_invalid" for row in attempts
                ),
                "successes": sum(bool(row.get("success")) for row in valid),
            },
            "candidates": len(list((self.root / "candidates").glob("*/bundle.json"))),
            "diagnoses": len(self.diagnoses.records()),
            "candidate_ledger": self.candidate_ledger.records(),
            "gates": self.gates.records(),
            "promotions": self.promotions.records(),
        }
