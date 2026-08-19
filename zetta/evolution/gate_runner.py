# Copyright (c) 2026 Zetta Contributors
"""Append-only scheduling for paired candidate gates.

Gate episodes deliberately live outside ``CampaignStore.episodes``.  That
ledger is bound to the currently promoted bundle, whereas a same-seed gate
must execute the frozen parent and an unpromoted candidate side by side.
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from zetta.evolution.campaign import resolve_bundle_file
from zetta.evolution.gating import (
    evaluate_fixed_heldout_20,
    evaluate_paired_gate,
    evaluate_two_stage_heldout,
)
from zetta.evolution.jsonio import (
    AppendOnlyLedger,
    atomic_write_json,
    canonical_sha256,
    read_json,
)
from zetta.evolution.lifecycle import (
    effective_same_seed_gate_pass_rate,
    effective_same_seed_pass_rate,
    load_accepted_cluster_report,
    materialize_cluster_targets,
    record_gate_and_advance,
)
from zetta.evolution.models import (
    CampaignPhase,
    CandidateBundle,
    EpisodeRecord,
    GateDecision,
)
from zetta.evolution.queue import RolloutJob, SharedHostQueue, round_robin_assign
from zetta.evolution.store import CampaignStore


class StaleCandidateError(ValueError):
    """The gate no longer targets the campaign's active candidate lineage."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _render_command(template: list[str], values: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(part).format_map(values) for part in template)


def _episode_from_attempt_row(row: dict[str, Any]) -> EpisodeRecord:
    payload = dict(row)
    payload.pop("attempt_id", None)
    return EpisodeRecord.from_dict(payload)


PairedGateKind = Literal["same_seed", "regression", "heldout", "heldout_20"]


class PairedGateRunner:
    """Prepare, enqueue, ingest, and finalize one recoverable paired gate."""

    schema_version = 3

    def __init__(
        self,
        *,
        campaign_root: str | Path,
        queue_root: str | Path,
        worker_hosts: tuple[str, ...],
        gate_kind: PairedGateKind,
        candidate_sha256: str | None = None,
    ) -> None:
        if not worker_hosts:
            raise ValueError("at least one gate worker host is required")
        self.store = CampaignStore(campaign_root)
        self.queue = SharedHostQueue(queue_root)
        self.worker_hosts = tuple(worker_hosts)
        if gate_kind not in {"same_seed", "regression", "heldout", "heldout_20"}:
            raise ValueError(f"unsupported paired gate kind: {gate_kind}")
        self.gate_kind = gate_kind
        state = self.store.state()
        selected = candidate_sha256 or state.get("candidate_sha256")
        if not isinstance(selected, str) or not selected:
            raise ValueError("paired gate requires an active candidate")
        self.candidate_sha256 = selected
        self.candidate = self._load_and_validate_candidate()
        self.gate_root = (
            self.store.root
            / "candidates"
            / self.candidate_sha256
            / "gates"
            / self.gate_kind
        )
        self.plan_path = self.gate_root / "plan.json"
        self.attempts = AppendOnlyLedger(
            self.gate_root / "ledgers" / "attempts.jsonl", key="attempt_id"
        )
        self.valid = AppendOnlyLedger(
            self.gate_root / "ledgers" / "valid.jsonl", key="logical_id"
        )
        self.dispatches = AppendOnlyLedger(
            self.gate_root / "ledgers" / "dispatches.jsonl", key="job_id"
        )
        self.parent_adoptions = AppendOnlyLedger(
            self.gate_root / "ledgers" / "parent-adoptions.jsonl",
            key="logical_id",
        )
        self.infra_recovery_authorizations = AppendOnlyLedger(
            self.gate_root / "ledgers" / "infra-recovery-authorizations.jsonl",
            key="authorization_id",
        )

    def _load_and_validate_candidate(self) -> CandidateBundle:
        state = self.store.state()
        if state.get("candidate_sha256") != self.candidate_sha256:
            raise StaleCandidateError("paired gate candidate is stale")
        path = self.store.root / "candidates" / self.candidate_sha256 / "bundle.json"
        payload = read_json(path)
        if canonical_sha256(payload) != self.candidate_sha256:
            raise ValueError("candidate artifact digest mismatch")
        candidate = CandidateBundle.from_dict(payload)
        manifest = self.store.manifest()
        if candidate.generation != manifest.generation:
            raise ValueError("candidate generation does not match campaign")
        if candidate.parent_sha256 != state.get("current_bundle_sha256"):
            raise StaleCandidateError("candidate parent no longer matches campaign")
        return candidate

    def _assert_current(self) -> dict[str, Any]:
        candidate = self._load_and_validate_candidate()
        if candidate.sha256 != self.candidate.sha256:
            raise StaleCandidateError("candidate changed after gate runner creation")
        return self.store.state()

    def _diagnosis_cluster_id(self) -> str:
        matches = [
            row
            for row in self.store.diagnoses.records()
            if canonical_sha256(row) == self.candidate.diagnosis_sha256
        ]
        if len(matches) != 1:
            raise ValueError("candidate diagnosis is missing or ambiguous")
        cluster_id = matches[0].get("cluster_id")
        if not isinstance(cluster_id, str) or not cluster_id:
            raise ValueError("candidate diagnosis has no target cluster")
        return cluster_id

    def _expected_phase(self) -> CampaignPhase:
        return {
            "same_seed": CampaignPhase.SAME_SEED_GATE,
            "regression": CampaignPhase.REGRESSION_GATE,
            "heldout": CampaignPhase.HELDOUT_GATE,
            "heldout_20": CampaignPhase.HELDOUT_GATE,
        }[self.gate_kind]

    def _preregistered_seeds(self) -> tuple[int, ...]:
        manifest = self.store.manifest()
        if self.gate_kind == "regression":
            return manifest.rollout_seeds
        if self.gate_kind in {"heldout", "heldout_20"}:
            expected = 50 if self.gate_kind == "heldout" else 20
            if len(manifest.heldout_seeds) != expected:
                raise ValueError(
                    f"{self.gate_kind} gate requires exactly {expected} "
                    "preregistered seeds"
                )
            return manifest.heldout_seeds
        raise AssertionError("same-seed targets are derived from the failure cluster")

    def _heldout_policy(self) -> dict[str, Any]:
        raw = self.store.manifest().runtime.get("evolution_policy", {})
        if not isinstance(raw, dict):
            raise ValueError("runtime.evolution_policy must be an object")
        # Historical manifests used a mandatory held-out validation gate. Keep
        # that behavior for old campaigns; new manifests may explicitly use a
        # report-only test while still executing the paired seed block.
        mode = str(raw.get("heldout_mode", "validation"))
        alpha = float(raw.get("heldout_alpha", 0.025))
        min_gain = int(raw.get("heldout_min_gain", 1))
        min_success_rate = float(raw.get("heldout_min_success_rate", 0.0))
        require_significance = raw.get("heldout_require_significance", True)
        if mode not in {"test", "validation"}:
            raise ValueError("heldout_mode must be 'test' or 'validation'")
        if not 0 < alpha < 1:
            raise ValueError("heldout_alpha must be in (0, 1)")
        if min_gain < 0:
            raise ValueError("heldout_min_gain must be non-negative")
        if not 0 <= min_success_rate <= 1:
            raise ValueError("heldout_min_success_rate must be in [0, 1]")
        if not isinstance(require_significance, bool):
            raise ValueError("heldout_require_significance must be boolean")
        return {
            "heldout_mode": mode,
            "alpha": alpha,
            "min_gain": min_gain,
            "min_success_rate": min_success_rate,
            "require_significance": require_significance,
        }

    def _build_preregistered_plan(self) -> dict[str, Any]:
        state = self._assert_current()
        if CampaignPhase(state["phase"]) != self._expected_phase():
            raise ValueError(
                f"a new {self.gate_kind} gate plan requires "
                f"{self._expected_phase().value} phase"
            )
        manifest = self.store.manifest()
        seeds = self._preregistered_seeds()
        prefix = (
            f"g{manifest.generation:04d}-{self.gate_kind}-{self.candidate_sha256[:12]}"
        )
        pairs = [
            {
                "pair_index": index,
                "seed": seed,
                "policy_rng": manifest.policy_rng_by_seed[str(seed)],
                "logical_ids": {
                    "parent": f"{prefix}-p{index:03d}-parent",
                    "candidate": f"{prefix}-p{index:03d}-candidate",
                },
            }
            for index, seed in enumerate(seeds)
        ]
        if self.gate_kind == "regression" and bool(
            manifest.runtime.get("reuse_rollout_parent_evidence", False)
        ):
            source_by_seed: dict[int, EpisodeRecord] = {}
            for row in self.store.episodes.records():
                record = EpisodeRecord.from_dict(row)
                if record.seed in source_by_seed:
                    raise ValueError("regression parent source seed is ambiguous")
                source_by_seed[record.seed] = record
            for pair in pairs:
                source = source_by_seed.get(int(pair["seed"]))
                if source is None:
                    raise ValueError("regression parent source episode is missing")
                pair.update(
                    {
                        "source_episode_id": source.episode_id,
                        "source_logical_id": source.logical_id,
                        "source_record_sha256": canonical_sha256(source.as_dict()),
                    }
                )
        result = {
            "schema_version": self.schema_version,
            "kind": self.gate_kind,
            "manifest_sha256": manifest.sha256,
            "generation": manifest.generation,
            "candidate_sha256": self.candidate_sha256,
            "parent_sha256": self.candidate.parent_sha256,
            "paired_seed_sha256": canonical_sha256(list(seeds)),
            "pairs": pairs,
            **(
                {
                    "stage_1_count": 10,
                    "stage_2_count": 50,
                    **self._heldout_policy(),
                }
                if self.gate_kind == "heldout"
                else {}
            ),
        }
        if self.gate_kind == "heldout_20":
            result.update(
                {
                    "heldout_count": 20,
                    **self._heldout_policy(),
                    "statistical_contract": "fixed_paired_mcnemar_20",
                }
            )
        return result

    def _source_parent_for_pair(self, pair: dict[str, Any]) -> EpisodeRecord:
        source_episode_id = pair.get("source_episode_id")
        source_logical_id = pair.get("source_logical_id")
        matches = []
        for row in self.store.episodes.records():
            record = EpisodeRecord.from_dict(row)
            if source_episode_id is not None:
                matched = record.episode_id == source_episode_id
            else:
                matched = record.seed == int(pair["seed"])
            if matched:
                matches.append(record)
        if len(matches) != 1:
            raise ValueError("gate parent source episode is missing or ambiguous")
        source = matches[0]
        manifest = self.store.manifest()
        if source.status != "valid":
            raise ValueError("gate parent source must be valid")
        if source.seed != int(pair["seed"]):
            raise ValueError("gate parent source seed changed")
        if source.policy_rng != int(pair["policy_rng"]):
            raise ValueError("gate parent source policy RNG changed")
        if source.bundle_sha256 != self.candidate.parent_sha256:
            raise ValueError("gate parent source bundle changed")
        if manifest.policy_rng_by_seed.get(str(source.seed)) != source.policy_rng:
            raise ValueError("gate parent source violates the frozen schedule")
        if source_logical_id is not None and source.logical_id != source_logical_id:
            raise ValueError("gate parent source logical identity changed")
        expected_digest = pair.get("source_record_sha256")
        if expected_digest is not None and canonical_sha256(source.as_dict()) != expected_digest:
            raise ValueError("gate parent source record changed")
        identity = source.artifact_index.get("initial_observation_identity")
        if not isinstance(identity, dict) or not identity:
            raise ValueError("gate parent source lacks reset observation identity")
        if self.gate_kind == "same_seed" and source.success:
            raise ValueError("same-seed parent source must be a failed rollout")
        return source

    def _adopt_frozen_parent_evidence(self, plan: dict[str, Any]) -> int:
        """Reuse immutable Loop1 parent episodes before dispatching a gate.

        Candidate episodes still run live from reset. Adoption is allowed only
        before any parent dispatch/attempt exists, so recovery cannot race a
        previously launched parent arm or create duplicate valid evidence.
        """

        manifest = self.store.manifest()
        if self.gate_kind not in {"same_seed", "regression"} or not bool(
            manifest.runtime.get("reuse_rollout_parent_evidence", False)
        ):
            return 0
        valid = self._valid_rows()
        attempts = self._attempt_rows()
        dispatched = {
            str(row.get("logical_id")) for row in self.dispatches.records()
        }
        adopted = 0
        for pair in self._active_pairs(plan):
            logical_id = str(pair["logical_ids"]["parent"])
            if logical_id in valid:
                continue
            if logical_id in attempts or logical_id in dispatched:
                continue
            source = self._source_parent_for_pair(pair)
            adoption = {
                "schema_version": 1,
                "logical_id": logical_id,
                "gate_kind": self.gate_kind,
                "plan_sha256": canonical_sha256(plan),
                "manifest_sha256": manifest.sha256,
                "source_episode_id": source.episode_id,
                "source_logical_id": source.logical_id,
                "source_record_sha256": canonical_sha256(source.as_dict()),
                "adopted_at": _utc_now(),
            }
            canonical_adoption = {
                **adoption,
                # Timestamp is informative; identity is bound by the stable
                # fields so idempotent recovery does not invent a new event.
                "adopted_at": "source-rollout-completion",
            }
            existing = {
                str(row["logical_id"]): row for row in self.parent_adoptions.records()
            }.get(logical_id)
            if existing is not None:
                if {**existing, "adopted_at": "source-rollout-completion"} != canonical_adoption:
                    raise ValueError("gate parent adoption evidence changed")
            else:
                self.parent_adoptions.append(adoption)
            record = replace(
                source,
                logical_id=logical_id,
                attempt_index=0,
                artifact_index={
                    **source.artifact_index,
                    "gate_parent_adoption": canonical_adoption,
                },
            )
            expected = self._expected_arms(plan)[logical_id]
            self._assert_record(record, job=self._build_job(
                plan=plan,
                logical_id=logical_id,
                expected=expected,
                attempt_index=0,
            ), expected=expected)
            if self._record_attempt(record):
                adopted += 1
        return adopted

    def _build_plan(self) -> dict[str, Any]:
        if self.gate_kind != "same_seed":
            return self._build_preregistered_plan()
        state = self._assert_current()
        if CampaignPhase(state["phase"]) != CampaignPhase.SAME_SEED_GATE:
            raise ValueError("a new gate plan requires same_seed_gate phase")
        manifest = self.store.manifest()
        report = load_accepted_cluster_report(self.store)
        targets = materialize_cluster_targets(self.store, report)
        cluster_id = self._diagnosis_cluster_id()
        clusters = [
            row
            for row in report.get("clusters", ())
            if row.get("cluster_id") == cluster_id
        ]
        if len(clusters) != 1:
            raise ValueError("candidate target cluster is missing or ambiguous")
        target_rows = [
            row
            for row in targets.get("targets", ())
            if row.get("cluster_id") == cluster_id
        ]
        if len(target_rows) != 1:
            raise ValueError("candidate target cluster has no frozen target binding")
        target = target_rows[0]
        target_episode_ids = tuple(target.get("episode_ids", ()))
        if not target_episode_ids:
            raise ValueError("target cluster has no failed episodes")

        source_by_episode: dict[str, EpisodeRecord] = {}
        segment_ids_by_episode: dict[str, list[str]] = {}
        for row in self.store.episodes.records():
            record = EpisodeRecord.from_dict(row)
            if record.episode_id in source_by_episode:
                raise ValueError("target episode identity is ambiguous")
            source_by_episode[record.episode_id] = record
            for segment in record.all_failure_segments:
                if segment.segment_id in set(target.get("member_segment_ids", ())):
                    segment_ids_by_episode.setdefault(record.episode_id, []).append(
                        segment.segment_id
                    )

        ordered: list[dict[str, Any]] = []
        by_seed: dict[int, dict[str, Any]] = {}
        for episode_id in target_episode_ids:
            record = source_by_episode.get(str(episode_id))
            if record is None:
                raise ValueError(f"target failure episode is missing: {episode_id}")
            if record.status != "valid" or record.success:
                raise ValueError("representative seed must come from a valid failure")
            if record.bundle_sha256 != self.candidate.parent_sha256:
                raise ValueError(
                    "representative episode did not use the candidate parent"
                )
            expected_rng = manifest.policy_rng_by_seed.get(str(record.seed))
            if record.policy_rng != expected_rng:
                raise ValueError(
                    "representative episode violates preregistered policy RNG"
                )
            pair = by_seed.get(record.seed)
            if pair is None:
                pair = {
                    "pair_index": len(ordered),
                    "source_episode_id": record.episode_id,
                    "source_logical_id": record.logical_id,
                    "source_record_sha256": canonical_sha256(record.as_dict()),
                    "source_segment_ids": [],
                    "seed": record.seed,
                    "policy_rng": record.policy_rng,
                }
                by_seed[record.seed] = pair
                ordered.append(pair)
            elif pair["source_episode_id"] != record.episode_id:
                raise ValueError(
                    "one representative seed maps to multiple source episodes"
                )
            pair["source_segment_ids"].extend(
                sorted(segment_ids_by_episode.get(record.episode_id, ()))
            )

        prefix = f"g{manifest.generation:04d}-same-seed-{self.candidate_sha256[:12]}"
        pairs = []
        for pair in ordered:
            index = int(pair["pair_index"])
            pairs.append(
                {
                    **pair,
                    "source_segment_ids": tuple(pair["source_segment_ids"]),
                    "logical_ids": {
                        "parent": f"{prefix}-p{index:03d}-parent",
                        "candidate": f"{prefix}-p{index:03d}-candidate",
                    },
                }
            )
        return {
            "schema_version": self.schema_version,
            "kind": self.gate_kind,
            "manifest_sha256": manifest.sha256,
            "generation": manifest.generation,
            "candidate_sha256": self.candidate_sha256,
            "parent_sha256": self.candidate.parent_sha256,
            "cluster_id": cluster_id,
            "cluster_report_sha256": canonical_sha256(report),
            "cluster_target_sha256": target["target_sha256"],
            "cluster_membership_sha256": canonical_sha256(
                {
                    "episode_ids": target_episode_ids,
                    "member_segment_ids": target.get("member_segment_ids", ()),
                }
            ),
            "same_seed_pass_rate": effective_same_seed_pass_rate(self.store),
            "target_episode_ids": target_episode_ids,
            "pairs": pairs,
        }

    def prepare(self) -> dict[str, Any]:
        """Create or verify the immutable gate plan."""

        self._assert_current()
        if self.plan_path.exists():
            plan = read_json(self.plan_path)
            self._validate_plan(plan)
            return plan
        plan = self._build_plan()
        atomic_write_json(self.plan_path, plan, overwrite=False)
        self._validate_plan(plan)
        return plan

    def _validate_plan(self, plan: dict[str, Any]) -> None:
        manifest = self.store.manifest()
        version = int(plan.get("schema_version", 1))
        if version not in {1, 2, self.schema_version}:
            raise ValueError("unsupported paired gate plan schema")
        expected = {
            "schema_version": version,
            "kind": self.gate_kind,
            "manifest_sha256": manifest.sha256,
            "generation": manifest.generation,
            "candidate_sha256": self.candidate_sha256,
            "parent_sha256": self.candidate.parent_sha256,
        }
        mismatched = [key for key, value in expected.items() if plan.get(key) != value]
        if mismatched:
            raise StaleCandidateError(
                f"paired gate plan binding changed: {sorted(mismatched)}"
            )
        if self.gate_kind == "same_seed" and version == 2:
            report = load_accepted_cluster_report(self.store)
            targets = materialize_cluster_targets(self.store, report)
            target = [
                row
                for row in targets.get("targets", ())
                if row.get("cluster_id") == plan.get("cluster_id")
            ]
            if len(target) != 1:
                raise ValueError("gate plan target no longer exists")
            checks = {
                "cluster_report_sha256": canonical_sha256(report),
                "cluster_target_sha256": target[0]["target_sha256"],
            }
            if any(plan.get(key) != value for key, value in checks.items()):
                raise ValueError("gate plan cluster lineage changed")
            rate = plan.get("same_seed_pass_rate")
            if not isinstance(rate, (int, float)) or not 0 < float(rate) <= 1:
                raise ValueError("gate plan has an invalid same-seed pass rate")
        elif self.gate_kind in {"regression", "heldout", "heldout_20"}:
            seeds = self._preregistered_seeds()
            if plan.get("paired_seed_sha256") != canonical_sha256(list(seeds)):
                raise ValueError("paired gate seed binding changed")
            if self.gate_kind == "heldout":
                heldout_contract = {
                    "stage_1_count": 10,
                    "stage_2_count": 50,
                }
                if any(
                    plan.get(key) != value for key, value in heldout_contract.items()
                ):
                    raise ValueError("heldout gate statistical contract changed")
                if version >= 3:
                    heldout_policy = self._heldout_policy()
                    if any(
                        plan.get(key) != value
                        for key, value in heldout_policy.items()
                    ):
                        raise ValueError("heldout gate policy changed")
                elif plan.get("alpha") != 0.025:
                    raise ValueError("historical heldout alpha changed")
            elif self.gate_kind == "heldout_20":
                heldout_contract = {
                    "heldout_count": 20,
                    "statistical_contract": "fixed_paired_mcnemar_20",
                }
                if any(
                    plan.get(key) != value for key, value in heldout_contract.items()
                ):
                    raise ValueError("heldout_20 gate statistical contract changed")
                if version >= 3:
                    heldout_policy = self._heldout_policy()
                    if any(
                        plan.get(key) != value
                        for key, value in heldout_policy.items()
                    ):
                        raise ValueError("heldout_20 policy changed")
                elif plan.get("alpha") != 0.025:
                    raise ValueError("historical heldout alpha changed")
        pairs = plan.get("pairs")
        if not isinstance(pairs, list) or not pairs:
            raise ValueError("paired gate plan requires seed pairs")
        seeds: set[int] = set()
        logical_ids: set[str] = set()
        for index, pair in enumerate(pairs):
            if pair.get("pair_index") != index:
                raise ValueError("gate pair indexes must be contiguous")
            seed = pair.get("seed")
            policy_rng = pair.get("policy_rng")
            if not isinstance(seed, int) or seed in seeds:
                raise ValueError("gate pair seeds must be unique integers")
            if manifest.policy_rng_by_seed.get(str(seed)) != policy_rng:
                raise ValueError("gate pair policy RNG is not preregistered")
            seeds.add(seed)
            arms = pair.get("logical_ids")
            if not isinstance(arms, dict) or set(arms) != {"parent", "candidate"}:
                raise ValueError("gate pair requires parent and candidate logical IDs")
            for logical_id in arms.values():
                if not isinstance(logical_id, str) or logical_id in logical_ids:
                    raise ValueError("gate logical IDs must be unique strings")
                logical_ids.add(logical_id)

        planned_seeds = tuple(int(pair["seed"]) for pair in pairs)
        if self.gate_kind in {"regression", "heldout", "heldout_20"}:
            if planned_seeds != self._preregistered_seeds():
                raise ValueError("paired gate seed order changed")
        if self.gate_kind in {"same_seed", "regression"} and bool(
            manifest.runtime.get("reuse_rollout_parent_evidence", False)
        ):
            for pair in pairs:
                self._source_parent_for_pair(pair)

    @property
    def _gate_prefix(self) -> str:
        manifest = self.store.manifest()
        label = "same-seed" if self.gate_kind == "same_seed" else self.gate_kind
        return f"g{manifest.generation:04d}-{label}-{self.candidate_sha256[:12]}"

    def _heldout_decisions(self) -> list[GateDecision]:
        rows = [
            GateDecision(**row)
            for row in self.store.gates.records()
            if row.get("candidate_sha256") == self.candidate_sha256
            and row.get("kind") in {"heldout_10", "heldout_50"}
        ]
        kinds = [row.kind for row in rows]
        if len(kinds) != len(set(kinds)) or kinds not in (
            [],
            ["heldout_10"],
            ["heldout_10", "heldout_50"],
        ):
            raise ValueError("heldout gate decision sequence is invalid")
        if len(rows) == 2 and (rows[0].passed or rows[0].conclusive):
            raise ValueError("heldout-50 cannot follow a terminal heldout-10 decision")
        return rows

    def _active_pairs(self, plan: dict[str, Any]) -> list[dict[str, Any]]:
        pairs = list(plan["pairs"])
        if self.gate_kind != "heldout":
            return pairs
        decisions = self._heldout_decisions()
        if not decisions:
            return pairs[:10]
        first = decisions[0]
        if first.passed or first.conclusive:
            return pairs[:10]
        return pairs

    def _expected_arms(self, plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for pair in self._active_pairs(plan):
            for arm, logical_id in pair["logical_ids"].items():
                result[logical_id] = {
                    "arm": arm,
                    "pair_index": pair["pair_index"],
                    "seed": pair["seed"],
                    "policy_rng": pair["policy_rng"],
                    "bundle_sha256": (
                        self.candidate_sha256
                        if arm == "candidate"
                        else self.candidate.parent_sha256
                    ),
                }
        return result

    def _command_template(self) -> list[str]:
        runtime = self.store.manifest().runtime
        keys = {
            "same_seed": ("same_seed_gate_rollout_command",),
            "regression": (
                "regression_gate_rollout_command",
                "paired_gate_rollout_command",
                "same_seed_gate_rollout_command",
            ),
            "heldout": (
                "heldout_gate_rollout_command",
                "paired_gate_rollout_command",
                "same_seed_gate_rollout_command",
            ),
            "heldout_20": (
                "heldout_gate_rollout_command",
                "paired_gate_rollout_command",
                "same_seed_gate_rollout_command",
            ),
        }[self.gate_kind]
        value = next((runtime.get(key) for key in keys if runtime.get(key)), None)
        if value is None:
            value = runtime.get("rollout_command")
        if not isinstance(value, list) or not value:
            raise ValueError("paired gate rollout command must be a non-empty array")
        joined = "\n".join(str(part) for part in value)
        required = (
            "seed",
            "policy_rng",
            "logical_id",
            "attempt_index",
            "bundle_sha256",
            "output_dir",
            "result_file",
        )
        missing = [name for name in required if f"{{{name}}}" not in joined]
        if "{bundle_file}" not in joined and "{bundle}" not in joined:
            missing.append("bundle_file")
        if missing:
            raise ValueError(
                f"gate rollout command omits frozen fields: {sorted(missing)}"
            )
        return value

    def _bundle_file(self, bundle_sha256: str | None) -> str:
        return resolve_bundle_file(self.store, bundle_sha256)

    def _attempt_rows(self) -> dict[str, list[dict[str, Any]]]:
        result: dict[str, list[dict[str, Any]]] = {}
        for row in self.attempts.records():
            result.setdefault(str(row["logical_id"]), []).append(row)
        for logical_id, rows in result.items():
            rows.sort(key=lambda row: int(row["attempt_index"]))
            indexes = [int(row["attempt_index"]) for row in rows]
            if indexes != list(range(len(rows))):
                raise ValueError(f"gate attempts are non-contiguous: {logical_id}")
        return result

    def _valid_rows(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for row in self.valid.records():
            logical_id = str(row["logical_id"])
            if logical_id in result:
                raise ValueError(f"duplicate gate valid logical ID: {logical_id}")
            result[logical_id] = row
        return result

    @staticmethod
    def _infra_authorization_id(payload: dict[str, Any]) -> str:
        stable = {
            key: value
            for key, value in payload.items()
            if key not in {"authorization_id", "authorized_at"}
        }
        return f"infra-recovery-{canonical_sha256(stable)[:24]}"

    @staticmethod
    def _infra_evidence(
        rows: list[dict[str, Any]], *, maximum: int, logical_id: str
    ) -> list[dict[str, Any]]:
        if len(rows) < maximum:
            raise ValueError(
                f"infra recovery evidence is incomplete for {logical_id}"
            )
        evidence: list[dict[str, Any]] = []
        for attempt_index, row in enumerate(rows[:maximum]):
            if int(row.get("attempt_index", -1)) != attempt_index:
                raise ValueError(
                    f"infra recovery evidence is non-contiguous for {logical_id}"
                )
            if row.get("status") != "infra_invalid":
                raise ValueError(
                    "infra recovery authorization cannot use valid or policy "
                    f"failure evidence: {logical_id} attempt-{attempt_index:03d}"
                )
            attempt_id = row.get("attempt_id")
            if not isinstance(attempt_id, str) or not attempt_id:
                raise ValueError("infra recovery evidence has no attempt identity")
            evidence.append(
                {
                    "logical_id": logical_id,
                    "attempt_index": attempt_index,
                    "attempt_id": attempt_id,
                    "status": "infra_invalid",
                    "record_sha256": canonical_sha256(row),
                }
            )
        return evidence

    def _validated_infra_recovery_authorizations(
        self, plan: dict[str, Any]
    ) -> tuple[int, dict[str, int], list[dict[str, Any]]]:
        """Validate the append-only authorization chain and derive arm limits."""

        expected = self._expected_arms(plan)
        attempts = self._attempt_rows()
        base = self.store.manifest().max_infrastructure_attempts
        limits = dict.fromkeys(expected, base)
        accepted: list[dict[str, Any]] = []
        plan_sha256 = canonical_sha256(plan)
        for row in self.infra_recovery_authorizations.records():
            required = {
                "schema_version",
                "authorization_id",
                "candidate_sha256",
                "plan_sha256",
                "gate_kind",
                "base_max_infrastructure_attempts",
                "additional_attempts",
                "logical_ids",
                "effective_max_before_by_logical_id",
                "evidence",
                "reason",
                "previous_authorization_ids",
                "authorized_at",
            }
            if set(row) != required:
                raise ValueError("infra recovery authorization schema changed")
            bindings = {
                "schema_version": 1,
                "candidate_sha256": self.candidate_sha256,
                "plan_sha256": plan_sha256,
                "gate_kind": self.gate_kind,
                "base_max_infrastructure_attempts": base,
                "previous_authorization_ids": [
                    item["authorization_id"] for item in accepted
                ],
            }
            mismatched = [
                key for key, value in bindings.items() if row.get(key) != value
            ]
            if mismatched:
                raise ValueError(
                    "infra recovery authorization binding changed: "
                    f"{sorted(mismatched)}"
                )
            additional = row.get("additional_attempts")
            if (
                not isinstance(additional, int)
                or isinstance(additional, bool)
                or additional < 1
            ):
                raise ValueError(
                    "infra recovery additional_attempts must be a positive integer"
                )
            logical_ids = row.get("logical_ids")
            if (
                not isinstance(logical_ids, list)
                or not logical_ids
                or logical_ids != sorted(set(logical_ids))
                or any(logical_id not in expected for logical_id in logical_ids)
            ):
                raise ValueError("infra recovery authorization has invalid logical IDs")
            before = row.get("effective_max_before_by_logical_id")
            expected_before = {
                logical_id: limits[logical_id] for logical_id in logical_ids
            }
            if before != expected_before:
                raise ValueError("infra recovery authorization limit lineage changed")
            expected_evidence: list[dict[str, Any]] = []
            for logical_id in logical_ids:
                expected_evidence.extend(
                    self._infra_evidence(
                        attempts.get(logical_id, []),
                        maximum=limits[logical_id],
                        logical_id=logical_id,
                    )
                )
            if row.get("evidence") != expected_evidence:
                raise ValueError("infra recovery authorization evidence changed")
            reason = row.get("reason")
            authorized_at = row.get("authorized_at")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("infra recovery authorization requires a reason")
            if not isinstance(authorized_at, str) or not authorized_at:
                raise ValueError("infra recovery authorization requires a timestamp")
            if row.get("authorization_id") != self._infra_authorization_id(row):
                raise ValueError("infra recovery authorization identity changed")
            for logical_id in logical_ids:
                limits[logical_id] += additional
            accepted.append(row)
        return base, limits, accepted

    def authorize_infrastructure_recovery(
        self,
        *,
        additional_attempts: int,
        reason: str,
        logical_ids: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        """Append an immutable budget authorization for exhausted infra-only arms."""

        self._assert_current()
        plan = self.prepare()
        if CampaignPhase(self.store.state()["phase"]) != self._expected_phase():
            raise ValueError("infra recovery can only be authorized in the active gate")
        if self._recorded_decision() is not None:
            raise ValueError("infra recovery cannot be authorized after a gate decision")
        if (
            not isinstance(additional_attempts, int)
            or isinstance(additional_attempts, bool)
            or additional_attempts < 1
        ):
            raise ValueError("additional_attempts must be a positive integer")
        normalized_reason = reason.strip() if isinstance(reason, str) else ""
        if not normalized_reason:
            raise ValueError("infra recovery authorization requires a reason")

        expected = self._expected_arms(plan)
        attempts = self._attempt_rows()
        valid = self._valid_rows()
        base, limits, authorizations = (
            self._validated_infra_recovery_authorizations(plan)
        )
        if logical_ids is None:
            requested = tuple(
                sorted(
                    logical_id
                    for logical_id in expected
                    if logical_id not in valid
                    and len(attempts.get(logical_id, ())) == limits[logical_id]
                )
            )
            if not requested:
                # A repeated explicit command before newly authorized jobs run is
                # an idempotent read of the latest immutable authorization.
                for existing in reversed(authorizations):
                    if (
                        existing["additional_attempts"] == additional_attempts
                        and existing["reason"] == normalized_reason
                        and all(
                            len(attempts.get(logical_id, ()))
                            == existing["effective_max_before_by_logical_id"][
                                logical_id
                            ]
                            for logical_id in existing["logical_ids"]
                        )
                    ):
                        return existing
                raise ValueError("no exhausted infrastructure-invalid gate arms")
        else:
            requested = tuple(sorted(logical_ids))
            if not requested or len(requested) != len(set(requested)):
                raise ValueError("logical_ids must be unique and non-empty")
            unknown = [logical_id for logical_id in requested if logical_id not in expected]
            if unknown:
                raise ValueError(f"unknown gate logical IDs: {unknown}")
            for existing in reversed(authorizations):
                if (
                    existing["logical_ids"] == list(requested)
                    and existing["additional_attempts"] == additional_attempts
                    and existing["reason"] == normalized_reason
                    and all(
                        len(attempts.get(logical_id, ()))
                        == existing["effective_max_before_by_logical_id"][logical_id]
                        for logical_id in requested
                    )
                ):
                    return existing

        evidence: list[dict[str, Any]] = []
        for logical_id in requested:
            if logical_id in valid:
                raise ValueError(
                    f"infra recovery cannot authorize a valid arm: {logical_id}"
                )
            rows = attempts.get(logical_id, [])
            if len(rows) != limits[logical_id]:
                raise ValueError(
                    "infra recovery requires an exhausted arm at its effective "
                    f"limit: {logical_id}"
                )
            evidence.extend(
                self._infra_evidence(
                    rows, maximum=limits[logical_id], logical_id=logical_id
                )
            )

        payload: dict[str, Any] = {
            "schema_version": 1,
            "candidate_sha256": self.candidate_sha256,
            "plan_sha256": canonical_sha256(plan),
            "gate_kind": self.gate_kind,
            "base_max_infrastructure_attempts": base,
            "additional_attempts": additional_attempts,
            "logical_ids": list(requested),
            "effective_max_before_by_logical_id": {
                logical_id: limits[logical_id] for logical_id in requested
            },
            "evidence": evidence,
            "reason": normalized_reason,
            "previous_authorization_ids": [
                row["authorization_id"] for row in authorizations
            ],
            "authorized_at": _utc_now(),
        }
        payload["authorization_id"] = self._infra_authorization_id(payload)
        try:
            self.infra_recovery_authorizations.append(payload)
        except ValueError:
            matches = [
                row
                for row in self.infra_recovery_authorizations.records()
                if row.get("authorization_id") == payload["authorization_id"]
            ]
            def stable(row: dict[str, Any]) -> dict[str, Any]:
                return {
                    key: value
                    for key, value in row.items()
                    if key not in {"authorization_id", "authorized_at"}
                }

            if len(matches) != 1 or stable(matches[0]) != stable(payload):
                raise
            payload = matches[0]
        self._validated_infra_recovery_authorizations(plan)
        return payload

    def _job_id(self, plan: dict[str, Any], logical_id: str, attempt: int) -> str:
        payload = {
            "plan_sha256": canonical_sha256(plan),
            "logical_id": logical_id,
            "attempt_index": attempt,
        }
        return f"gate-job-{canonical_sha256(payload)[:24]}"

    def _build_job(
        self,
        *,
        plan: dict[str, Any],
        logical_id: str,
        expected: dict[str, Any],
        attempt_index: int,
    ) -> RolloutJob:
        manifest = self.store.manifest()
        output_dir = (
            self.gate_root / "attempts" / logical_id / f"attempt-{attempt_index:03d}"
        )
        bundle_file = self._bundle_file(expected["bundle_sha256"])
        values = {
            "campaign_root": str(self.store.root),
            "gate_root": str(self.gate_root),
            "gate_kind": self.gate_kind,
            "gate_arm": expected["arm"],
            "pair_index": expected["pair_index"],
            "logical_id": logical_id,
            "attempt_index": attempt_index,
            "generation": manifest.generation,
            "seed": expected["seed"],
            "policy_rng": expected["policy_rng"],
            "candidate_sha256": self.candidate_sha256,
            "parent_sha256": self.candidate.parent_sha256 or "none",
            "bundle_sha256": expected["bundle_sha256"] or "none",
            "baseline_mode": (
                "active_bundle"
                if expected["bundle_sha256"] is not None
                else "strict_pure_vla"
            ),
            "bundle_file": bundle_file,
            "bundle": bundle_file,
            "output_dir": str(output_dir),
            "result_file": str(output_dir / "episode_record.json"),
            "heartbeat_file": str(output_dir / "heartbeat.jsonl"),
            "task": manifest.task,
            "environment": manifest.environment,
            "env_endpoint": "__ZETTA_ENV_ENDPOINT__",
        }
        return RolloutJob(
            job_id=self._job_id(plan, logical_id, attempt_index),
            campaign_root=str(self.store.root),
            logical_id=logical_id,
            attempt_index=attempt_index,
            task=manifest.task,
            seed=expected["seed"],
            policy_rng=expected["policy_rng"],
            bundle_sha256=expected["bundle_sha256"],
            command=_render_command(self._command_template(), values),
            output_dir=str(output_dir),
            result_file=values["result_file"],
            heartbeat_file=values["heartbeat_file"],
            timeout_s=manifest.episode_timeout_s,
            no_progress_timeout_s=manifest.no_progress_timeout_s,
            # Same-seed arms are heterogeneous in Gen0: the parent is strict
            # pure VLA, while only the candidate can invoke Role1 Recovery.
            requires_api=(
                expected["bundle_sha256"] is not None
                and bool(manifest.runtime.get("candidate_rollout_requires_api", True))
            ),
            requires_environment_slot=bool(
                manifest.runtime.get("rollout_requires_environment_slot", False)
            ),
        )

    def enqueue_missing(self) -> dict[str, Any]:
        """Queue one attempt for every unfinished arm, without replaying valid arms."""

        state = self._assert_current()
        plan = self.prepare()
        parent_adopted = self._adopt_frozen_parent_evidence(plan)
        if self._recorded_decision() is not None:
            return {
                "enqueued": 0,
                "parent_adopted": parent_adopted,
                "blocked": [],
                "queue": self.queue.counts(),
            }
        if CampaignPhase(state["phase"]) != self._expected_phase():
            raise ValueError(
                f"{self.gate_kind} gate jobs cannot be queued in this phase"
            )
        expected = self._expected_arms(plan)
        attempts = self._attempt_rows()
        valid = self._valid_rows()
        known_jobs = self.queue.known_job_ids()
        jobs: list[RolloutJob] = []
        blocked: list[str] = []
        _, effective_maximum, _ = self._validated_infra_recovery_authorizations(
            plan
        )
        for logical_id, arm in expected.items():
            if logical_id in valid:
                continue
            attempt_index = len(attempts.get(logical_id, ()))
            if attempt_index >= effective_maximum[logical_id]:
                blocked.append(logical_id)
                continue
            job = self._build_job(
                plan=plan,
                logical_id=logical_id,
                expected=arm,
                attempt_index=attempt_index,
            )
            if job.job_id not in known_jobs:
                jobs.append(job)
        assignments = round_robin_assign(
            {self.store.manifest().task: jobs}, self.worker_hosts
        )
        existing_dispatches = {
            str(row["job_id"]): row for row in self.dispatches.records()
        }
        published = 0
        for proposed_host, job in assignments:
            dispatch = {
                "job_id": job.job_id,
                "host": proposed_host,
                "logical_id": job.logical_id,
                "attempt_index": job.attempt_index,
                "job_sha256": canonical_sha256(job.as_dict()),
            }
            existing = existing_dispatches.get(job.job_id)
            if existing is not None:
                if existing.get("job_sha256") != dispatch["job_sha256"]:
                    raise ValueError("gate dispatch job payload changed")
                host = str(existing["host"])
            else:
                self.dispatches.append(dispatch)
                host = proposed_host
            if job.job_id not in known_jobs:
                self.queue.enqueue(host, job)
                known_jobs.add(job.job_id)
                published += 1
        return {
            "enqueued": published,
            "parent_adopted": parent_adopted,
            "blocked": sorted(blocked),
            "by_host": {
                host: sum(
                    row.get("host") == host
                    for row in self.dispatches.records()
                    if row.get("job_id") in {job.job_id for job in jobs}
                )
                for host in sorted(set(self.worker_hosts))
            },
            "queue": self.queue.counts(),
        }

    def _terminal_job_and_result(
        self, path: Path
    ) -> tuple[RolloutJob, dict[str, Any], dict[str, Any]] | None:
        envelope = read_json(path)
        if not isinstance(envelope, dict):
            raise ValueError("queue terminal envelope must be an object")
        worker = (
            envelope.get("result")
            if envelope.get("kind") == "rollout_terminal"
            else envelope
        )
        if not isinstance(worker, dict):
            raise ValueError("queue terminal has no worker result")
        job_payload = envelope.get("job")
        if not isinstance(job_payload, dict):
            job_payload = worker.get("job")
        if not isinstance(job_payload, dict):
            return None
        job = RolloutJob.from_dict(job_payload)
        result = worker.get("result")
        return job, worker, result if isinstance(result, dict) else {}

    def _infra_record(
        self,
        *,
        path: Path,
        envelope: dict[str, Any],
        job: RolloutJob,
    ) -> EpisodeRecord:
        acquired = envelope.get("acquired_at") or envelope.get("finished_at")
        finished = envelope.get("finished_at") or acquired
        if not isinstance(acquired, str):
            acquired = "1970-01-01T00:00:00+00:00"
        if not isinstance(finished, str):
            finished = acquired
        return EpisodeRecord(
            episode_id=f"infra-{job.job_id}",
            logical_id=job.logical_id,
            generation=self.store.manifest().generation,
            seed=job.seed,
            policy_rng=job.policy_rng,
            bundle_sha256=job.bundle_sha256,
            status="infra_invalid",
            success=None,
            started_at=acquired,
            finished_at=finished,
            elapsed_s=float(envelope.get("elapsed_s", 0.0)),
            artifact_index={"worker_envelope": str(path)},
            invalid_reason=str(
                envelope.get("watchdog_reason")
                or envelope.get("worker_exception")
                or f"rollout_exit_{envelope.get('return_code', 'unknown')}"
            ),
            attempt_index=job.attempt_index,
        )

    def _assert_record(
        self,
        record: EpisodeRecord,
        *,
        job: RolloutJob,
        expected: dict[str, Any],
    ) -> None:
        manifest = self.store.manifest()
        required = {
            "logical_id": job.logical_id,
            "generation": manifest.generation,
            "seed": expected["seed"],
            "policy_rng": expected["policy_rng"],
            "bundle_sha256": expected["bundle_sha256"],
            "attempt_index": job.attempt_index,
        }
        mismatched = [
            key for key, value in required.items() if getattr(record, key) != value
        ]
        if mismatched:
            raise ValueError(f"gate result violates frozen job fields: {mismatched}")

    def _record_attempt(self, record: EpisodeRecord) -> bool:
        existing_attempts = self._attempt_rows().get(record.logical_id, [])
        by_index = {int(row["attempt_index"]): row for row in existing_attempts}
        if record.attempt_index in by_index:
            existing = _episode_from_attempt_row(by_index[record.attempt_index])
            if existing != record:
                raise ValueError("duplicate gate attempt has conflicting payload")
        elif record.attempt_index != len(existing_attempts):
            raise ValueError("gate attempt ingestion is non-contiguous")

        existing_valid = self._valid_rows().get(record.logical_id)
        if existing_valid is not None:
            accepted = EpisodeRecord.from_dict(existing_valid)
            if record.status == "valid" and accepted != record:
                raise ValueError("duplicate gate valid has conflicting payload")
            if record.attempt_index > accepted.attempt_index:
                raise ValueError("gate attempt was executed after an accepted valid")

        payload = {"attempt_id": record.attempt_id, **record.as_dict()}
        artifact = (
            self.gate_root
            / "accepted_attempts"
            / record.logical_id
            / f"attempt-{record.attempt_index:03d}.json"
        )
        atomic_write_json(artifact, payload, overwrite=False)
        self.attempts.append(payload)
        if record.status == "infra_invalid":
            return False
        canonical = self.gate_root / "episodes" / record.logical_id / "record.json"
        atomic_write_json(canonical, record.as_dict(), overwrite=False)
        return self.valid.append(record.as_dict())

    def ingest(self) -> dict[str, int]:
        """Append terminal attempts and accept exactly one valid per logical arm."""

        self._assert_current()
        plan = self.prepare()
        expected = self._expected_arms(plan)
        _, effective_maximum, _ = self._validated_infra_recovery_authorizations(
            plan
        )
        relevant: list[
            tuple[int, str, Path, RolloutJob, dict[str, Any], dict[str, Any]]
        ] = []
        for path in self.queue.terminal_envelope_paths():
            parsed = self._terminal_job_and_result(path)
            if parsed is None:
                continue
            job, worker, payload = parsed
            if job.logical_id not in expected:
                if job.logical_id.startswith(self._gate_prefix):
                    raise ValueError("queue contains an unknown logical gate arm")
                continue
            if Path(job.campaign_root).resolve() != self.store.root.resolve():
                raise ValueError("gate queue job targets another campaign root")
            frozen = self._build_job(
                plan=plan,
                logical_id=job.logical_id,
                expected=expected[job.logical_id],
                attempt_index=job.attempt_index,
            )
            if job != frozen:
                raise ValueError("gate queue job differs from the frozen plan")
            relevant.append(
                (
                    job.attempt_index,
                    job.logical_id,
                    path,
                    job,
                    worker,
                    payload,
                )
            )

        accepted = 0
        invalid = 0
        for _, _, path, job, worker, payload in sorted(relevant):
            envelope = read_json(path)
            if payload:
                record = EpisodeRecord.from_dict(payload)
            else:
                stable = {
                    **worker,
                    "acquired_at": envelope.get("acquired_at"),
                    "finished_at": envelope.get("finished_at"),
                }
                record = self._infra_record(path=path, envelope=stable, job=job)
            self._assert_record(record, job=job, expected=expected[job.logical_id])
            if job.attempt_index >= effective_maximum[job.logical_id]:
                raise ValueError(
                    "gate attempt exceeds its authorized infrastructure budget"
                )
            if self._record_attempt(record):
                accepted += 1
            if record.status == "infra_invalid":
                invalid += 1
        return {"accepted": accepted, "infra_invalid_seen": invalid}

    def _recorded_decision(self) -> GateDecision | None:
        if self.gate_kind == "heldout":
            decisions = self._heldout_decisions()
            if not decisions:
                return None
            latest = decisions[-1]
            if latest.kind == "heldout_50" or latest.passed or latest.conclusive:
                return latest
            return None
        rows = [
            row
            for row in self.store.gates.records()
            if row.get("candidate_sha256") == self.candidate_sha256
            and row.get("kind") == self.gate_kind
        ]
        if len(rows) > 1:
            raise ValueError(f"candidate has multiple {self.gate_kind} gate decisions")
        return GateDecision(**rows[0]) if rows else None

    def _records_by_arm(
        self, plan: dict[str, Any]
    ) -> tuple[list[EpisodeRecord], list[EpisodeRecord]]:
        expected = self._expected_arms(plan)
        parent: list[EpisodeRecord] = []
        candidate: list[EpisodeRecord] = []
        for row in self.valid.records():
            record = EpisodeRecord.from_dict(row)
            arm = expected.get(record.logical_id)
            if arm is None:
                raise ValueError("valid gate ledger contains an unknown logical arm")
            (candidate if arm["arm"] == "candidate" else parent).append(record)
        return candidate, parent

    def _early_impossible_same_seed_decision(
        self,
        plan: dict[str, Any],
        expected: dict[str, dict[str, Any]],
        valid: dict[str, dict[str, Any]],
    ) -> GateDecision | None:
        """Reject once even perfect outcomes on all missing arms cannot pass."""

        if self.gate_kind != "same_seed" or int(plan.get("schema_version", 1)) < 2:
            return None
        pairs = self._active_pairs(plan)
        pair_count = len(pairs)
        if pair_count < 1:
            return None
        rate = effective_same_seed_gate_pass_rate(
            self.store,
            candidate_sha256=self.candidate_sha256,
            plan=plan,
        )
        required_successes = math.ceil(pair_count * rate)
        candidate_records: dict[str, EpisodeRecord] = {}
        parent_records: dict[str, EpisodeRecord] = {}
        for logical_id, row in valid.items():
            arm = expected.get(logical_id)
            if arm is None:
                raise ValueError("valid gate ledger contains an unknown logical arm")
            record = EpisodeRecord.from_dict(row)
            if arm["arm"] == "candidate":
                candidate_records[logical_id] = record
            else:
                parent_records[logical_id] = record
        parent_ids = {str(pair["logical_ids"]["parent"]) for pair in pairs}
        if set(parent_records) != parent_ids:
            # Parent success and safety evidence must be frozen before an early
            # terminal decision can be committed.
            return None
        candidate_successes = sum(
            bool(record.success) for record in candidate_records.values()
        )
        parent_successes = sum(
            bool(record.success) for record in parent_records.values()
        )
        remaining = pair_count - len(candidate_records)
        upper_bound = candidate_successes + remaining
        if upper_bound >= required_successes and upper_bound > parent_successes:
            return None

        candidate_by_seed = {
            record.seed: record for record in candidate_records.values()
        }
        parent_by_seed = {record.seed: record for record in parent_records.values()}
        candidate_wins = sum(
            bool(candidate_by_seed[seed].success) and not bool(parent.success)
            for seed, parent in parent_by_seed.items()
            if seed in candidate_by_seed
        )
        parent_wins = sum(
            bool(parent.success) and not bool(candidate_by_seed[seed].success)
            for seed, parent in parent_by_seed.items()
            if seed in candidate_by_seed
        )
        candidate_safety = sum(
            len(record.safety_events) for record in candidate_records.values()
        )
        parent_safety = sum(
            len(record.safety_events) for record in parent_records.values()
        )
        observed_ids = tuple(sorted(candidate_records))
        missing_ids = tuple(
            sorted(
                str(pair["logical_ids"]["candidate"])
                for pair in pairs
                if str(pair["logical_ids"]["candidate"]) not in candidate_records
            )
        )
        rationale = (
            "same-seed gate rejected early as mathematically impossible: "
            f"observed candidate successes {candidate_successes}/{pair_count}, "
            f"remaining candidate arms {remaining}, upper bound "
            f"{upper_bound}/{pair_count}, required {required_successes}/{pair_count}; "
            f"parent successes observed {parent_successes}/{pair_count}; "
            f"valid candidate arms {len(observed_ids)}, missing arms "
            f"{len(missing_ids)}"
        )
        decision_payload = {
            "kind": "same_seed",
            "candidate_sha256": self.candidate_sha256,
            "parent_sha256": self.candidate.parent_sha256,
            "plan_sha256": canonical_sha256(plan),
            "candidate_successes": candidate_successes,
            "parent_successes": parent_successes,
            "paired_count": pair_count,
            "candidate_wins": candidate_wins,
            "parent_wins": parent_wins,
            "observed_candidate_ids": observed_ids,
            "missing_candidate_ids": missing_ids,
            "effective_same_seed_pass_rate": rate,
        }
        return GateDecision(
            decision_id=f"gate-{canonical_sha256(decision_payload)[:20]}",
            candidate_sha256=self.candidate_sha256,
            parent_sha256=self.candidate.parent_sha256,
            kind="same_seed",
            passed=False,
            conclusive=True,
            candidate_successes=candidate_successes,
            parent_successes=parent_successes,
            paired_count=pair_count,
            candidate_wins=candidate_wins,
            parent_wins=parent_wins,
            p_value=None,
            alpha=None,
            candidate_safety_events=candidate_safety,
            parent_safety_events=parent_safety,
            rationale=rationale,
        )

    def finalize_if_complete(self) -> GateDecision | None:
        """Evaluate and advance only after every parent/candidate arm is valid."""

        state = self._assert_current()
        plan = self.prepare()
        expected = self._expected_arms(plan)
        valid = self._valid_rows()
        if set(valid) != set(expected):
            recorded = self._recorded_decision()
            if recorded is not None:
                if (
                    recorded.kind == "same_seed"
                    and not recorded.passed
                    and recorded.conclusive
                    and recorded.rationale.startswith(
                        "same-seed gate rejected early as mathematically impossible:"
                    )
                ):
                    return recorded
                raise ValueError("gate decision exists before all paired arms are valid")
            early = self._early_impossible_same_seed_decision(plan, expected, valid)
            if early is not None:
                if CampaignPhase(state["phase"]) != self._expected_phase():
                    raise ValueError(
                        "unrecorded gate cannot advance outside "
                        f"{self._expected_phase().value}"
                    )
                record_gate_and_advance(
                    campaign_root=self.store.root,
                    decision=early,
                )
                return early
            return None
        candidate_records, parent_records = self._records_by_arm(plan)
        seeds = tuple(int(pair["seed"]) for pair in self._active_pairs(plan))
        if self.gate_kind == "heldout":
            stage = 1 if len(seeds) == 10 else 2
            decision = evaluate_two_stage_heldout(
                candidate_sha256=self.candidate_sha256,
                parent_sha256=self.candidate.parent_sha256,
                candidate_records=candidate_records,
                parent_records=parent_records,
                preregistered_seeds=self.store.manifest().heldout_seeds,
                stage=stage,
                alpha=float(plan.get("alpha", 0.025)),
                min_gain=int(plan.get("min_gain", 1)),
                min_success_rate=float(plan.get("min_success_rate", 0.0)),
                require_significance=bool(plan.get("require_significance", True)),
            )
        elif self.gate_kind == "heldout_20":
            decision = evaluate_fixed_heldout_20(
                candidate_sha256=self.candidate_sha256,
                parent_sha256=self.candidate.parent_sha256,
                candidate_records=candidate_records,
                parent_records=parent_records,
                preregistered_seeds=self.store.manifest().heldout_seeds,
                alpha=float(plan.get("alpha", 0.025)),
                min_gain=int(plan.get("min_gain", 1)),
                min_success_rate=float(plan.get("min_success_rate", 0.0)),
                require_significance=bool(plan.get("require_significance", True)),
            )
        else:
            decision = evaluate_paired_gate(
                kind=self.gate_kind,
                candidate_sha256=self.candidate_sha256,
                parent_sha256=self.candidate.parent_sha256,
                candidate_records=candidate_records,
                parent_records=parent_records,
                expected_seeds=seeds,
                same_seed_pass_rate=(
                    effective_same_seed_gate_pass_rate(
                        self.store,
                        candidate_sha256=self.candidate_sha256,
                        plan=plan,
                    )
                    if self.gate_kind == "same_seed"
                    else 1.0
                ),
            )
        recorded = self._recorded_decision()
        if recorded is not None:
            if recorded != decision:
                raise ValueError("recorded gate decision differs from paired evidence")
            if CampaignPhase(state["phase"]) == self._expected_phase():
                record_gate_and_advance(
                    campaign_root=self.store.root,
                    decision=decision,
                )
            return decision
        if CampaignPhase(state["phase"]) != self._expected_phase():
            raise ValueError(
                f"unrecorded gate cannot advance outside {self._expected_phase().value}"
            )
        try:
            record_gate_and_advance(
                campaign_root=self.store.root,
                decision=decision,
            )
        except ValueError:
            # A concurrent idempotent finalizer may have recorded and advanced.
            latest = self._recorded_decision()
            if latest != decision:
                raise
        return decision

    def status(self) -> dict[str, Any]:
        self._assert_current()
        plan = self.prepare()
        expected = self._expected_arms(plan)
        valid = self._valid_rows()
        attempts = self._attempt_rows()
        base_maximum, effective_maximum, authorizations = (
            self._validated_infra_recovery_authorizations(plan)
        )
        complete_pairs = sum(
            all(logical_id in valid for logical_id in pair["logical_ids"].values())
            for pair in self._active_pairs(plan)
        )
        blocked = sorted(
            logical_id
            for logical_id in expected
            if logical_id not in valid
            and len(attempts.get(logical_id, ()))
            >= effective_maximum[logical_id]
        )
        decision = self._recorded_decision()
        return {
            "schema_version": self.schema_version,
            "candidate_sha256": self.candidate_sha256,
            "parent_sha256": self.candidate.parent_sha256,
            "plan_sha256": canonical_sha256(plan),
            "gate_kind": self.gate_kind,
            "cluster_id": plan.get("cluster_id"),
            "pairs": len(self._active_pairs(plan)),
            "planned_pairs": len(plan["pairs"]),
            "complete_pairs": complete_pairs,
            "valid_arms": len(valid),
            "attempts": sum(len(rows) for rows in attempts.values()),
            "dispatches": len(self.dispatches.records()),
            "infra_invalid": sum(
                row.get("status") == "infra_invalid"
                for rows in attempts.values()
                for row in rows
            ),
            "base_max_infrastructure_attempts": base_maximum,
            "effective_max_infrastructure_attempts": max(
                effective_maximum.values(), default=base_maximum
            ),
            "effective_max_infrastructure_attempts_by_logical_id": {
                logical_id: effective_maximum[logical_id]
                for logical_id in sorted(effective_maximum)
            },
            "infrastructure_recovery_authorizations": authorizations,
            "blocked": blocked,
            "decision": decision.as_dict() if decision else None,
            "campaign_phase": self.store.state()["phase"],
            "queue": self.queue.counts(),
        }

    def run_once(self) -> dict[str, Any]:
        """Idempotent resume cycle: ingest, finalize if complete, then enqueue."""

        plan = self.prepare()
        recorded_before_ingest = self._recorded_decision()
        if (
            recorded_before_ingest is not None
            and CampaignPhase(self.store.state()["phase"]) != self._expected_phase()
        ):
            status = self.status()
            return {
                "ingestion": {"accepted": 0, "infra_invalid_seen": 0},
                "parent_adopted": 0,
                "enqueue": {"enqueued": 0, "blocked": [], "queue": self.queue.counts()},
                "decision": recorded_before_ingest.as_dict(),
                "terminal_decision": recorded_before_ingest.as_dict(),
                "status": status,
                "updated_at": _utc_now(),
            }
        parent_adopted = self._adopt_frozen_parent_evidence(plan)
        ingestion = self.ingest()
        decision = self.finalize_if_complete()
        terminal_decision = self._recorded_decision()
        enqueue = (
            {"enqueued": 0, "blocked": [], "queue": self.queue.counts()}
            if terminal_decision is not None
            else self.enqueue_missing()
        )
        return {
            "ingestion": ingestion,
            "parent_adopted": parent_adopted,
            "enqueue": enqueue,
            "decision": decision.as_dict() if decision else None,
            "terminal_decision": (
                terminal_decision.as_dict() if terminal_decision else None
            ),
            "status": self.status(),
            "updated_at": _utc_now(),
        }


class CandidateGateRunner(PairedGateRunner):
    """Backward-compatible same-seed gate runner."""

    def __init__(
        self,
        *,
        campaign_root: str | Path,
        queue_root: str | Path,
        worker_hosts: tuple[str, ...],
        candidate_sha256: str | None = None,
    ) -> None:
        super().__init__(
            campaign_root=campaign_root,
            queue_root=queue_root,
            worker_hosts=worker_hosts,
            gate_kind="same_seed",
            candidate_sha256=candidate_sha256,
        )


__all__ = [
    "CandidateGateRunner",
    "PairedGateRunner",
    "PairedGateKind",
    "StaleCandidateError",
]
