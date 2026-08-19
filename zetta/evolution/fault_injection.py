# Copyright (c) 2026 Zetta Contributors
"""Deterministic campaign-stage crash injection and idempotent recovery.

The harness deliberately lives outside the campaign implementation.  It wraps
one already-existing lifecycle operation, records an immutable terminal
checkpoint after the operation, and injects a one-shot crash immediately
before or after that checkpoint.  Recovery never edits either ledger: an exact
terminal checkpoint is verified and skipped, while any partial or conflicting
record fails closed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from zetta.evolution.jsonio import (
    AppendOnlyLedger,
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    read_json,
)
from zetta.evolution.models import CampaignPhase
from zetta.evolution.store import CampaignStore


class LifecycleStage(str, Enum):
    """Fault-injectable stages in one campaign generation."""

    ROLLOUT = "ROLLOUT"
    CLUSTER = "CLUSTER"
    DIAGNOSE = "DIAGNOSE"
    PROPOSE = "PROPOSE"
    SAME_SEED_GATE = "SAME_SEED_GATE"
    REGRESSION_GATE = "REGRESSION_GATE"
    HELDOUT_GATE = "HELDOUT_GATE"
    PROMOTE = "PROMOTE"


class FaultPoint(str, Enum):
    """Supported write-boundary crash positions."""

    BEFORE_WRITE = "before_write"
    AFTER_WRITE = "after_write"


STAGE_ORDER = tuple(LifecycleStage)

_PHASE_CONTRACTS: dict[
    LifecycleStage, tuple[set[CampaignPhase], set[CampaignPhase]]
] = {
    LifecycleStage.ROLLOUT: (
        {CampaignPhase.ROLLOUT},
        {CampaignPhase.ROLLOUT},
    ),
    LifecycleStage.CLUSTER: (
        {CampaignPhase.ROLLOUT, CampaignPhase.CLUSTER},
        {CampaignPhase.CLUSTER},
    ),
    LifecycleStage.DIAGNOSE: (
        {CampaignPhase.CLUSTER, CampaignPhase.DIAGNOSE},
        {CampaignPhase.PROPOSE},
    ),
    LifecycleStage.PROPOSE: (
        {CampaignPhase.PROPOSE},
        {CampaignPhase.SAME_SEED_GATE},
    ),
    LifecycleStage.SAME_SEED_GATE: (
        {CampaignPhase.SAME_SEED_GATE},
        {CampaignPhase.REGRESSION_GATE, CampaignPhase.PROPOSE},
    ),
    LifecycleStage.REGRESSION_GATE: (
        {CampaignPhase.REGRESSION_GATE},
        {CampaignPhase.HELDOUT_GATE, CampaignPhase.PROPOSE},
    ),
    LifecycleStage.HELDOUT_GATE: (
        {CampaignPhase.HELDOUT_GATE},
        {
            CampaignPhase.HELDOUT_GATE,
            CampaignPhase.PROMOTE,
            CampaignPhase.PROPOSE,
        },
    ),
    LifecycleStage.PROMOTE: (
        {CampaignPhase.PROMOTE},
        {CampaignPhase.COMPLETE, CampaignPhase.ROLLOUT},
    ),
}

_TERMINAL_FIELDS = {
    "schema_version",
    "terminal_id",
    "plan_sha256",
    "manifest_sha256",
    "generation",
    "stage",
    "parent_bundle_sha256",
    "phase_before",
    "phase_after",
    "output",
    "output_sha256",
    "completed_at",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class FaultInjectionPlan:
    """Immutable crash schedule bound to one manifest and parent bundle."""

    plan_id: str
    manifest_sha256: str
    generation: int
    parent_bundle_sha256: str | None
    points: dict[str, tuple[str, ...]]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported fault plan schema")
        if not self.plan_id.strip():
            raise ValueError("fault plan requires plan_id")
        normalized_stages = set(self.points)
        known_stages = {stage.value for stage in STAGE_ORDER}
        if not normalized_stages.issubset(known_stages):
            raise ValueError(
                f"fault plan contains unknown stages: "
                f"{sorted(normalized_stages - known_stages)}"
            )
        known_points = {point.value for point in FaultPoint}
        for stage, points in self.points.items():
            if len(points) != len(set(points)):
                raise ValueError(f"fault points must be unique for {stage}")
            unknown = set(points) - known_points
            if unknown:
                raise ValueError(
                    f"fault plan contains unknown points for {stage}: {sorted(unknown)}"
                )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FaultInjectionPlan":
        payload = dict(value)
        payload["points"] = {
            str(stage): tuple(str(point) for point in points)
            for stage, points in dict(payload.get("points", {})).items()
        }
        return cls(**payload)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.as_dict())


@dataclass(frozen=True, slots=True)
class StageRunResult:
    """Result of a newly committed or recovered stage."""

    stage: LifecycleStage
    recovered: bool
    terminal_id: str
    output: dict[str, Any]


class InjectedCampaignCrash(RuntimeError):
    """Raised after a planned one-shot crash point has been durably recorded."""

    def __init__(self, stage: LifecycleStage, point: FaultPoint) -> None:
        self.stage = stage
        self.point = point
        super().__init__(f"injected campaign crash: {stage.value}:{point.value}")


def create_fault_plan(
    campaign_root: str | Path,
    *,
    plan_id: str,
    points: Mapping[LifecycleStage | str, Sequence[FaultPoint | str]],
) -> FaultInjectionPlan:
    """Freeze a fault plan under an initialized campaign root."""

    store = CampaignStore(campaign_root)
    manifest = store.manifest()
    state = store.state()
    normalized: dict[str, tuple[str, ...]] = {}
    for raw_stage, raw_points in points.items():
        stage = (
            raw_stage
            if isinstance(raw_stage, LifecycleStage)
            else LifecycleStage(str(raw_stage).upper())
        )
        normalized[stage.value] = tuple(
            point.value
            if isinstance(point, FaultPoint)
            else FaultPoint(str(point)).value
            for point in raw_points
        )
    plan = FaultInjectionPlan(
        plan_id=plan_id,
        manifest_sha256=manifest.sha256,
        generation=manifest.generation,
        parent_bundle_sha256=state.get("current_bundle_sha256"),
        points=normalized,
    )
    atomic_write_json(
        Path(campaign_root) / "fault_injection" / "plan.json",
        plan.as_dict(),
        overwrite=False,
    )
    return plan


class CampaignFaultHarness:
    """Append-only crash/recovery wrapper for campaign lifecycle operations."""

    def __init__(self, campaign_root: str | Path) -> None:
        self.store = CampaignStore(campaign_root)
        self.root = self.store.root / "fault_injection"
        self.plan_path = self.root / "plan.json"
        self.injections = AppendOnlyLedger(
            self.root / "injections.jsonl", key="injection_id"
        )
        self.terminals = AppendOnlyLedger(
            self.root / "stage_terminals.jsonl", key="terminal_id"
        )

    def plan(self) -> FaultInjectionPlan:
        if not self.plan_path.is_file():
            raise ValueError("fault injection plan has not been frozen")
        plan = FaultInjectionPlan.from_dict(read_json(self.plan_path))
        manifest = self.store.manifest()
        if plan.manifest_sha256 != manifest.sha256:
            raise ValueError("fault plan is bound to a different manifest")
        if plan.generation != manifest.generation:
            raise ValueError("fault plan generation does not match campaign")
        return plan

    def run_stage(
        self,
        stage: LifecycleStage | str,
        operation: Callable[[], Any],
    ) -> StageRunResult:
        """Run one lifecycle operation or recover its exact terminal checkpoint."""

        selected = (
            stage
            if isinstance(stage, LifecycleStage)
            else LifecycleStage(str(stage).upper())
        )
        plan = self.plan()
        terminal_rows = self._terminal_rows(plan)
        self._validate_order(terminal_rows)
        existing = terminal_rows.get(selected)
        if existing is not None:
            self._verify_terminal(plan, existing, terminal_rows)
            return StageRunResult(
                stage=selected,
                recovered=True,
                terminal_id=str(existing["terminal_id"]),
                output=dict(existing["output"]),
            )

        self._require_prior_terminals(selected, terminal_rows)
        self._verify_parent_binding(plan, terminal_rows)
        before = CampaignPhase(self.store.state()["phase"])
        allowed_before, allowed_after = _PHASE_CONTRACTS[selected]
        if before not in allowed_before:
            raise ValueError(
                f"{selected.value} cannot start from campaign phase {before.value}"
            )
        self._inject_once(plan, selected, FaultPoint.BEFORE_WRITE)
        operation()
        after = CampaignPhase(self.store.state()["phase"])
        if after not in allowed_after:
            raise ValueError(
                f"{selected.value} ended in unexpected campaign phase {after.value}"
            )
        output = self._capture_output(selected, plan)
        terminal = {
            "schema_version": 1,
            "terminal_id": f"{plan.sha256}:{selected.value}",
            "plan_sha256": plan.sha256,
            "manifest_sha256": plan.manifest_sha256,
            "generation": plan.generation,
            "stage": selected.value,
            "parent_bundle_sha256": plan.parent_bundle_sha256,
            "phase_before": before.value,
            "phase_after": after.value,
            "output": output,
            "output_sha256": canonical_sha256(output),
            "completed_at": _utc_now(),
        }
        self.terminals.append(terminal)
        self._inject_once(plan, selected, FaultPoint.AFTER_WRITE)
        return StageRunResult(
            stage=selected,
            recovered=False,
            terminal_id=str(terminal["terminal_id"]),
            output=output,
        )

    def verify(self) -> dict[str, Any]:
        """Validate the complete append-only recovery surface and return status."""

        plan = self.plan()
        rows = self._terminal_rows(plan)
        self._validate_order(rows)
        self._verify_parent_binding(plan, rows)
        for row in rows.values():
            self._verify_terminal(plan, row, rows)
        injections = self._injection_rows(plan)
        return {
            "plan_id": plan.plan_id,
            "plan_sha256": plan.sha256,
            "manifest_sha256": plan.manifest_sha256,
            "completed_stages": [stage.value for stage in STAGE_ORDER if stage in rows],
            "injections": [
                {
                    "stage": row["stage"],
                    "point": row["point"],
                }
                for row in injections
            ],
            "campaign_phase": self.store.state()["phase"],
        }

    def _inject_once(
        self,
        plan: FaultInjectionPlan,
        stage: LifecycleStage,
        point: FaultPoint,
    ) -> None:
        configured = set(plan.points.get(stage.value, ()))
        if point.value not in configured:
            return
        injection_id = f"{plan.sha256}:{stage.value}:{point.value}"
        if any(
            row.get("injection_id") == injection_id
            for row in self._injection_rows(plan)
        ):
            return
        self.injections.append(
            {
                "schema_version": 1,
                "injection_id": injection_id,
                "plan_sha256": plan.sha256,
                "manifest_sha256": plan.manifest_sha256,
                "stage": stage.value,
                "point": point.value,
                "injected_at": _utc_now(),
            }
        )
        raise InjectedCampaignCrash(stage, point)

    def _injection_rows(self, plan: FaultInjectionPlan) -> list[dict[str, Any]]:
        rows = self.injections.records()
        for row in rows:
            if row.get("plan_sha256") != plan.sha256:
                raise ValueError("injection ledger contains a foreign fault plan")
            try:
                LifecycleStage(str(row["stage"]))
                FaultPoint(str(row["point"]))
            except (KeyError, ValueError) as exc:
                raise ValueError("injection ledger contains an invalid record") from exc
        return rows

    def _terminal_rows(
        self, plan: FaultInjectionPlan
    ) -> dict[LifecycleStage, dict[str, Any]]:
        result: dict[LifecycleStage, dict[str, Any]] = {}
        for row in self.terminals.records():
            if set(row) != _TERMINAL_FIELDS:
                missing = sorted(_TERMINAL_FIELDS - set(row))
                extra = sorted(set(row) - _TERMINAL_FIELDS)
                raise ValueError(
                    f"incomplete terminal record; missing={missing}, extra={extra}"
                )
            if row.get("schema_version") != 1:
                raise ValueError("unsupported terminal record schema")
            if row.get("plan_sha256") != plan.sha256:
                raise ValueError("terminal record is bound to a foreign fault plan")
            stage = LifecycleStage(str(row["stage"]))
            if stage in result:
                raise ValueError(f"duplicate terminal stage: {stage.value}")
            if row.get("terminal_id") != f"{plan.sha256}:{stage.value}":
                raise ValueError("terminal record identity mismatch")
            output = row.get("output")
            if not isinstance(output, dict) or not output:
                raise ValueError("terminal record requires a complete output object")
            if row.get("output_sha256") != canonical_sha256(output):
                raise ValueError("terminal output digest mismatch")
            result[stage] = row
        return result

    def _validate_order(
        self, terminals: Mapping[LifecycleStage, Mapping[str, Any]]
    ) -> None:
        seen_gap = False
        for stage in STAGE_ORDER:
            if stage not in terminals:
                seen_gap = True
            elif seen_gap:
                raise ValueError(
                    f"terminal record for {stage.value} follows an incomplete stage"
                )

    def _require_prior_terminals(
        self,
        stage: LifecycleStage,
        terminals: Mapping[LifecycleStage, Mapping[str, Any]],
    ) -> None:
        index = STAGE_ORDER.index(stage)
        missing = [
            prior.value for prior in STAGE_ORDER[:index] if prior not in terminals
        ]
        if missing:
            raise ValueError(f"prior campaign stages are incomplete: {missing}")

    def _verify_parent_binding(
        self,
        plan: FaultInjectionPlan,
        terminals: Mapping[LifecycleStage, Mapping[str, Any]],
    ) -> None:
        state = self.store.state()
        if LifecycleStage.PROMOTE not in terminals:
            if state.get("current_bundle_sha256") != plan.parent_bundle_sha256:
                raise ValueError("stale parent bundle detected during recovery")
            candidate_sha = state.get("candidate_sha256")
            if candidate_sha:
                candidate_path = (
                    self.store.root / "candidates" / candidate_sha / "bundle.json"
                )
                candidate = read_json(candidate_path)
                if candidate.get("parent_sha256") != plan.parent_bundle_sha256:
                    raise ValueError("candidate parent is stale")
            return
        output = terminals[LifecycleStage.PROMOTE]["output"]
        expected = output.get("candidate_sha256")
        if (
            not isinstance(expected, str)
            or state.get("current_bundle_sha256") != expected
        ):
            raise ValueError("promoted bundle no longer matches terminal record")

    def _verify_terminal(
        self,
        plan: FaultInjectionPlan,
        terminal: Mapping[str, Any],
        terminals: Mapping[LifecycleStage, Mapping[str, Any]],
    ) -> None:
        if terminal.get("manifest_sha256") != plan.manifest_sha256:
            raise ValueError("terminal manifest digest mismatch")
        if terminal.get("generation") != plan.generation:
            raise ValueError("terminal generation mismatch")
        if terminal.get("parent_bundle_sha256") != plan.parent_bundle_sha256:
            raise ValueError("terminal parent binding mismatch")
        stage = LifecycleStage(str(terminal["stage"]))
        before = CampaignPhase(str(terminal["phase_before"]))
        after = CampaignPhase(str(terminal["phase_after"]))
        allowed_before, allowed_after = _PHASE_CONTRACTS[stage]
        if before not in allowed_before or after not in allowed_after:
            raise ValueError("terminal phase contract mismatch")
        self._verify_output(stage, dict(terminal["output"]))
        completed = [value for value in STAGE_ORDER if value in terminals]
        if completed and completed[-1] == stage:
            current = CampaignPhase(self.store.state()["phase"])
            if current != after:
                raise ValueError("latest terminal does not match campaign phase")

    @staticmethod
    def _record_refs(
        rows: Sequence[Mapping[str, Any]], identity: str
    ) -> list[dict[str, str]]:
        return [
            {
                "id": str(row[identity]),
                "sha256": canonical_sha256(dict(row)),
            }
            for row in rows
        ]

    def _capture_output(
        self, stage: LifecycleStage, plan: FaultInjectionPlan
    ) -> dict[str, Any]:
        if stage == LifecycleStage.ROLLOUT:
            rows = self.store.episodes.records()
            if len(rows) != self.store.manifest().expected_rollouts:
                raise ValueError(
                    "rollout terminal requires every preregistered valid episode"
                )
            return {
                "episodes": self._record_refs(rows, "logical_id"),
                "episode_artifacts": [
                    {
                        "logical_id": str(row["logical_id"]),
                        "path": str(
                            (
                                Path("episodes")
                                / str(row["logical_id"])
                                / "record.json"
                            ).as_posix()
                        ),
                        "sha256": file_sha256(
                            self.store.root
                            / "episodes"
                            / str(row["logical_id"])
                            / "record.json"
                        ),
                    }
                    for row in rows
                ],
                "valid_count": len(rows),
            }
        if stage == LifecycleStage.CLUSTER:
            path = self.store.root / "analysis" / "failure_clusters.json"
            if not path.is_file():
                raise ValueError("cluster terminal requires failure_clusters.json")
            return {
                "artifact": str(path.relative_to(self.store.root)),
                "artifact_sha256": file_sha256(path),
            }
        if stage == LifecycleStage.DIAGNOSE:
            rows = self.store.diagnoses.records()
            if not rows:
                raise ValueError("diagnose terminal requires an accepted diagnosis")
            return {"diagnoses": self._record_refs(rows, "diagnosis_id")}
        if stage == LifecycleStage.PROPOSE:
            candidate_sha = self.store.state().get("candidate_sha256")
            if not isinstance(candidate_sha, str):
                raise ValueError("propose terminal requires a current candidate")
            candidate_path = (
                self.store.root / "candidates" / candidate_sha / "bundle.json"
            )
            candidate = read_json(candidate_path)
            if candidate.get("parent_sha256") != plan.parent_bundle_sha256:
                raise ValueError("candidate parent is stale")
            return {
                "candidate_sha256": candidate_sha,
                "candidate_artifact_sha256": file_sha256(candidate_path),
            }
        if stage in {
            LifecycleStage.SAME_SEED_GATE,
            LifecycleStage.REGRESSION_GATE,
            LifecycleStage.HELDOUT_GATE,
        }:
            expected_kinds = {
                LifecycleStage.SAME_SEED_GATE: {"same_seed"},
                LifecycleStage.REGRESSION_GATE: {"regression"},
                LifecycleStage.HELDOUT_GATE: {
                    "heldout_10",
                    "heldout_20",
                    "heldout_50",
                },
            }[stage]
            rows = [
                row
                for row in self.store.gates.records()
                if row.get("kind") in expected_kinds
            ]
            if not rows:
                raise ValueError(f"{stage.value} terminal requires a gate decision")
            return {"gates": self._record_refs(rows, "decision_id")}
        rows = self.store.promotions.records()
        if not rows:
            raise ValueError("promote terminal requires an append-only promotion")
        promotion = rows[-1]
        candidate_sha = str(promotion["candidate_sha256"])
        bundle_path = (
            self.store.root
            / "bundles"
            / f"generation-{self.store.manifest().generation:04d}.json"
        )
        if not bundle_path.is_file():
            raise ValueError(
                "promote terminal requires the immutable generation bundle"
            )
        return {
            "promotion": self._record_refs((promotion,), "promotion_id")[0],
            "candidate_sha256": candidate_sha,
            "bundle_artifact_sha256": file_sha256(bundle_path),
        }

    def _verify_output(self, stage: LifecycleStage, output: dict[str, Any]) -> None:
        if stage == LifecycleStage.ROLLOUT:
            self._verify_refs(
                output.get("episodes"), self.store.episodes.records(), "logical_id"
            )
            artifacts = output.get("episode_artifacts")
            if not isinstance(artifacts, list) or not artifacts:
                raise ValueError("rollout terminal episode artifacts are incomplete")
            seen: set[str] = set()
            for artifact in artifacts:
                if not isinstance(artifact, dict) or set(artifact) != {
                    "logical_id",
                    "path",
                    "sha256",
                }:
                    raise ValueError("rollout terminal episode artifact is malformed")
                logical_id = artifact.get("logical_id")
                relative = artifact.get("path")
                digest = artifact.get("sha256")
                if (
                    not isinstance(logical_id, str)
                    or logical_id in seen
                    or not isinstance(relative, str)
                    or not isinstance(digest, str)
                ):
                    raise ValueError("rollout terminal episode artifact is malformed")
                path = self.store.root / relative
                if not path.is_file() or file_sha256(path) != digest:
                    raise ValueError(
                        "rollout terminal episode artifact is missing or changed"
                    )
                seen.add(logical_id)
            if output.get("valid_count") != self.store.manifest().expected_rollouts:
                raise ValueError("rollout terminal valid count mismatch")
            return
        if stage == LifecycleStage.CLUSTER:
            self._verify_artifact(output)
            return
        if stage == LifecycleStage.DIAGNOSE:
            self._verify_refs(
                output.get("diagnoses"),
                self.store.diagnoses.records(),
                "diagnosis_id",
            )
            return
        if stage == LifecycleStage.PROPOSE:
            candidate_sha = output.get("candidate_sha256")
            path = self.store.root / "candidates" / str(candidate_sha) / "bundle.json"
            if not path.is_file() or file_sha256(path) != output.get(
                "candidate_artifact_sha256"
            ):
                raise ValueError("candidate terminal artifact mismatch")
            candidate = read_json(path)
            if candidate.get("parent_sha256") != self.plan().parent_bundle_sha256:
                raise ValueError("candidate terminal has a stale parent")
            return
        if stage in {
            LifecycleStage.SAME_SEED_GATE,
            LifecycleStage.REGRESSION_GATE,
            LifecycleStage.HELDOUT_GATE,
        }:
            self._verify_refs(
                output.get("gates"), self.store.gates.records(), "decision_id"
            )
            return
        promotion = output.get("promotion")
        self._verify_refs(
            [promotion] if isinstance(promotion, dict) else promotion,
            self.store.promotions.records(),
            "promotion_id",
        )
        bundle_path = (
            self.store.root
            / "bundles"
            / f"generation-{self.store.manifest().generation:04d}.json"
        )
        if file_sha256(bundle_path) != output.get("bundle_artifact_sha256"):
            raise ValueError("promotion bundle artifact mismatch")

    def _verify_artifact(self, output: Mapping[str, Any]) -> None:
        relative = output.get("artifact")
        digest = output.get("artifact_sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise ValueError("terminal artifact reference is incomplete")
        path = self.store.root / relative
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError("terminal artifact is missing or changed")

    @staticmethod
    def _verify_refs(
        expected: Any,
        current_rows: Sequence[Mapping[str, Any]],
        identity: str,
    ) -> None:
        if not isinstance(expected, list) or not expected:
            raise ValueError("terminal ledger references are incomplete")
        current = {
            str(row[identity]): canonical_sha256(dict(row)) for row in current_rows
        }
        seen: set[str] = set()
        for reference in expected:
            if not isinstance(reference, dict) or set(reference) != {"id", "sha256"}:
                raise ValueError("terminal ledger reference is malformed")
            key = reference.get("id")
            digest = reference.get("sha256")
            if not isinstance(key, str) or key in seen or current.get(key) != digest:
                raise ValueError("terminal ledger reference is missing or changed")
            seen.add(key)


def _parse_fault(value: str) -> tuple[LifecycleStage, FaultPoint]:
    try:
        stage, point = value.split(":", maxsplit=1)
        return LifecycleStage(stage.upper()), FaultPoint(point.lower())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "fault must be STAGE:before_write or STAGE:after_write"
        ) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create-plan")
    create.add_argument("--campaign-root", required=True)
    create.add_argument("--plan-id", required=True)
    create.add_argument("--fault", action="append", type=_parse_fault, default=[])
    create.add_argument("--all-points", action="store_true")
    run = subparsers.add_parser("run-stage")
    run.add_argument("--campaign-root", required=True)
    run.add_argument(
        "--stage", required=True, choices=[stage.value for stage in STAGE_ORDER]
    )
    run.add_argument("operation", nargs=argparse.REMAINDER)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--campaign-root", required=True)
    status = subparsers.add_parser("status")
    status.add_argument("--campaign-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the minimal fault-injection CLI."""

    args = _parser().parse_args(argv)
    try:
        if args.command == "create-plan":
            values: dict[LifecycleStage, list[FaultPoint]] = {}
            if args.all_points:
                values = {stage: list(FaultPoint) for stage in STAGE_ORDER}
            for stage, point in args.fault:
                values.setdefault(stage, []).append(point)
            plan = create_fault_plan(
                args.campaign_root,
                plan_id=args.plan_id,
                points=values,
            )
            result: dict[str, Any] = {
                "plan_id": plan.plan_id,
                "plan_sha256": plan.sha256,
            }
        else:
            harness = CampaignFaultHarness(args.campaign_root)
            if args.command == "run-stage":
                operation = list(args.operation)
                if operation and operation[0] == "--":
                    operation = operation[1:]
                if not operation:
                    raise ValueError("run-stage requires an operation command")

                def run_operation() -> None:
                    subprocess.run(operation, check=True)

                stage_result = harness.run_stage(args.stage, run_operation)
                result = {
                    "stage": stage_result.stage.value,
                    "recovered": stage_result.recovered,
                    "terminal_id": stage_result.terminal_id,
                    "output": stage_result.output,
                }
            else:
                result = harness.verify()
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except InjectedCampaignCrash as exc:
        print(
            json.dumps(
                {
                    "injected": True,
                    "stage": exc.stage.value,
                    "point": exc.point.value,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 86


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
