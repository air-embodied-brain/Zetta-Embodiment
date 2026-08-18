# Copyright (c) 2026 RPent Contributors
"""Versioned data contracts for rollout evolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal

from rpent.evolution.jsonio import canonical_sha256

SHA256_LENGTH = 64

BaselineMode = Literal["strict_pure_vla", "active_bundle"]


@dataclass(frozen=True, slots=True)
class SafetyLayerConfig:
    """Frozen, non-learning safety boundary shared by every paired arm.

    This intentionally describes only interface and simulator integrity checks.
    Task progress, stagnation, terminal handoff, and task-specific collision
    heuristics belong to a learned candidate bundle and are never hard safety.
    """

    action_contract: str = "robocasa_named_action_v1"
    control_limits: str = "normalized_input_and_post_scale_clip_v1"
    simulation_health: str = "finite_observation_state_v1"
    joint_limit_shield: str = "not_implemented"
    contact_policy: str = "environment_native_only"

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(not isinstance(value, str) or not value for value in values.values()):
            raise ValueError("every safety-layer field must be a non-empty string")

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


class CampaignPhase(str, Enum):
    ROLLOUT = "rollout"
    CLUSTER = "cluster"
    DIAGNOSE = "diagnose"
    PROPOSE = "propose"
    SAME_SEED_GATE = "same_seed_gate"
    REGRESSION_GATE = "regression_gate"
    HELDOUT_GATE = "heldout_gate"
    PROMOTE = "promote"
    COMPLETE = "complete"


def _require_sha(name: str, value: str | None, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or len(value) != SHA256_LENGTH:
        raise ValueError(f"{name} must be a 64-character sha256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{name} must be hexadecimal") from exc


def _require_unique(name: str, values: tuple[int, ...], expected: int) -> None:
    if len(values) != expected:
        raise ValueError(f"{name} must contain exactly {expected} entries")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")


@dataclass(frozen=True, slots=True)
class CampaignManifest:
    campaign_id: str
    environment: str
    task: str
    generation: int
    code_commit: str
    prompt_sha256: str
    model: str
    tool_catalog_sha256: str
    rollout_seeds: tuple[int, ...]
    heldout_seeds: tuple[int, ...]
    policy_rng_by_seed: dict[str, int]
    parent_bundle_sha256: str | None = None
    baseline_mode: BaselineMode | None = None
    active_bundle_sha256: str | None = None
    safety_layer: SafetyLayerConfig = field(default_factory=SafetyLayerConfig)
    expected_rollouts: int = 50
    expected_heldout: int = 50
    initial_logical_slots: int = 8
    maximum_logical_slots: int = 12
    continuous_logical_slots: int = 8
    maximum_api_concurrency: int = 8
    episode_timeout_s: int = 900
    no_progress_timeout_s: int = 240
    target_valid_episodes_per_hour: float = 25.0
    max_infrastructure_attempts: int = 2
    reasoning_effort: str | None = None
    runtime: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1
    # This process-local compatibility marker is deliberately excluded from
    # serialized manifests.  It lets readers infer the new execution protocol
    # for pre-repair artifacts without changing their immutable digest.
    protocol_explicit: bool = field(default=True, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported campaign manifest schema")
        if not self.campaign_id or not self.environment or not self.task:
            raise ValueError("campaign_id, environment, and task are required")
        if self.generation < 0:
            raise ValueError("generation must be non-negative")
        if len(self.code_commit) != 40:
            raise ValueError("code_commit must be a full 40-character git sha")
        _require_sha("prompt_sha256", self.prompt_sha256)
        _require_sha("tool_catalog_sha256", self.tool_catalog_sha256)
        _require_sha("parent_bundle_sha256", self.parent_bundle_sha256, optional=True)
        if self.baseline_mode is None:
            object.__setattr__(
                self,
                "baseline_mode",
                "active_bundle" if self.parent_bundle_sha256 else "strict_pure_vla",
            )
        if (
            self.active_bundle_sha256 is None
            and self.baseline_mode == "active_bundle"
            and self.parent_bundle_sha256 is not None
        ):
            object.__setattr__(self, "active_bundle_sha256", self.parent_bundle_sha256)
        _require_sha("active_bundle_sha256", self.active_bundle_sha256, optional=True)
        if self.baseline_mode == "strict_pure_vla":
            if self.active_bundle_sha256 is not None:
                raise ValueError("strict pure-VLA baseline cannot activate a bundle")
            if self.generation != 0 or self.parent_bundle_sha256 is not None:
                raise ValueError("strict pure-VLA baseline is only valid for generation 0")
        elif self.baseline_mode == "active_bundle":
            if self.active_bundle_sha256 is None:
                raise ValueError("active-bundle mode requires active_bundle_sha256")
            if self.active_bundle_sha256 != self.parent_bundle_sha256:
                raise ValueError("active bundle must be the frozen campaign parent")
        else:  # pragma: no cover - Literal protects typed callers
            raise ValueError("unsupported baseline_mode")
        _require_unique("rollout_seeds", self.rollout_seeds, self.expected_rollouts)
        _require_unique("heldout_seeds", self.heldout_seeds, self.expected_heldout)
        if set(self.rollout_seeds) & set(self.heldout_seeds):
            raise ValueError("rollout and heldout seeds must be disjoint")
        required_rng = {
            str(seed) for seed in (*self.rollout_seeds, *self.heldout_seeds)
        }
        if set(self.policy_rng_by_seed) != required_rng:
            raise ValueError(
                "policy_rng_by_seed must exactly cover every preregistered seed"
            )
        if not (
            1
            <= self.continuous_logical_slots
            <= self.initial_logical_slots
            <= self.maximum_logical_slots
        ):
            raise ValueError("logical slot limits are inconsistent")
        if self.episode_timeout_s <= self.no_progress_timeout_s:
            raise ValueError("episode timeout must exceed no-progress timeout")
        if self.maximum_api_concurrency < 1:
            raise ValueError("maximum_api_concurrency must be positive")
        if self.protocol_explicit and self.maximum_api_concurrency > 20:
            raise ValueError("maximum_api_concurrency must not exceed 20")
        if self.max_infrastructure_attempts < 1:
            raise ValueError("max_infrastructure_attempts must be positive")
        if self.reasoning_effort not in {None, "low", "medium", "high", "xhigh"}:
            raise ValueError("unsupported campaign reasoning_effort")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CampaignManifest":
        value = dict(payload)
        # Never trust or persist the process-local compatibility marker.
        value.pop("protocol_explicit", None)
        value["rollout_seeds"] = tuple(int(v) for v in value["rollout_seeds"])
        value["heldout_seeds"] = tuple(int(v) for v in value["heldout_seeds"])
        value["policy_rng_by_seed"] = {
            str(k): int(v) for k, v in value["policy_rng_by_seed"].items()
        }
        safety = value.get("safety_layer")
        if isinstance(safety, dict):
            value["safety_layer"] = SafetyLayerConfig(**safety)
        # Read-only compatibility for manifests written before the explicit
        # Gen0/parent protocol was introduced. Newly prepared manifests always
        # materialize both fields.
        if "baseline_mode" not in value:
            parent = value.get("parent_bundle_sha256")
            value["baseline_mode"] = "active_bundle" if parent else "strict_pure_vla"
            value["active_bundle_sha256"] = parent
            value["protocol_explicit"] = False
        return cls(**value)

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        explicit = bool(value.pop("protocol_explicit"))
        if not explicit:
            # Preserve the exact canonical payload (and therefore SHA256) of
            # manifests created before these fields existed.  The inferred
            # values remain available to runtime code through attributes and
            # agent_view(), but are never written back over old provenance.
            value.pop("baseline_mode", None)
            value.pop("active_bundle_sha256", None)
            value.pop("safety_layer", None)
        return value

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.as_dict())

    def agent_view(self) -> dict[str, Any]:
        """Return seed-blind metadata safe for Stage1/Stage2 prompts."""

        return {
            "campaign_id": self.campaign_id,
            "environment": self.environment,
            "task": self.task,
            "generation": self.generation,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "episode_count": len(self.rollout_seeds),
            "parent_bundle_sha256": self.parent_bundle_sha256,
            "baseline_mode": self.baseline_mode,
            "active_bundle_sha256": self.active_bundle_sha256,
            "safety_layer": self.safety_layer.as_dict(),
        }


EpisodeStatus = Literal["valid", "infra_invalid"]


@dataclass(frozen=True, slots=True)
class FailureSegment:
    segment_id: str
    episode_id: str
    failure_class: str
    stage: str
    tool: str | None
    summary: str
    # ``None`` means that the trajectory is known to have failed, but the
    # immutable evidence does not localize the first divergence.  It must not
    # be encoded as step zero: step zero is a real observation and using it
    # poisons shadow-replay lead-time/recall statistics.
    earliest_divergence_step: int | None
    start_step: int
    end_step: int
    severity: float = 1.0
    state_signature: tuple[float, ...] = ()
    embedding: tuple[float, ...] = ()
    artifact_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.segment_id or not self.episode_id or not self.failure_class:
            raise ValueError("failure segment identifiers are required")
        if self.earliest_divergence_step is None:
            if not 0 <= self.start_step <= self.end_step:
                raise ValueError("unknown-divergence segment step interval is inconsistent")
        elif not (0 <= self.start_step <= self.earliest_divergence_step <= self.end_step):
            raise ValueError("failure segment step interval is inconsistent")
        if not 0 <= self.severity <= 1:
            raise ValueError("severity must be in [0, 1]")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    episode_id: str
    logical_id: str
    generation: int
    seed: int
    policy_rng: int
    bundle_sha256: str | None
    status: EpisodeStatus
    success: bool | None
    started_at: str
    finished_at: str
    elapsed_s: float
    artifact_index: dict[str, Any]
    safety_events: tuple[str, ...] = ()
    failure_segment: FailureSegment | None = None
    failure_segments: tuple[FailureSegment, ...] = ()
    invalid_reason: str | None = None
    attempt_index: int = 0
    schema_version: int = 1

    def __post_init__(self) -> None:
        _require_sha("bundle_sha256", self.bundle_sha256, optional=True)
        if self.status == "valid" and not isinstance(self.success, bool):
            raise ValueError("valid episode requires a boolean success value")
        if self.status == "infra_invalid" and self.success is not None:
            raise ValueError("infra-invalid episode cannot carry a success value")
        if self.status == "infra_invalid" and not self.invalid_reason:
            raise ValueError("infra-invalid episode requires invalid_reason")
        if self.success is True and (
            self.failure_segment is not None or self.failure_segments
        ):
            raise ValueError("successful episode cannot carry failure segments")
        if self.status == "infra_invalid" and (
            self.failure_segment is not None or self.failure_segments
        ):
            raise ValueError("infra-invalid episode cannot carry failure segments")
        if self.failure_segment is not None and (
            self.failure_segment.episode_id != self.episode_id
        ):
            raise ValueError("primary failure segment belongs to another episode")
        if any(
            segment.episode_id != self.episode_id for segment in self.failure_segments
        ):
            raise ValueError("failure segment belongs to another episode")
        segment_ids = [segment.segment_id for segment in self.failure_segments]
        if len(segment_ids) != len(set(segment_ids)):
            raise ValueError("failure segment identifiers must be unique per episode")
        if (
            self.failure_segment is not None
            and self.failure_segments
            and self.failure_segment != self.failure_segments[0]
        ):
            raise ValueError("primary failure segment must be the first full segment")
        if self.elapsed_s < 0:
            raise ValueError("elapsed_s must be non-negative")
        if self.attempt_index < 0:
            raise ValueError("attempt_index must be non-negative")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def attempt_id(self) -> str:
        return f"{self.logical_id}-a{self.attempt_index:03d}"

    @property
    def all_failure_segments(self) -> tuple[FailureSegment, ...]:
        """Return complete evidence while preserving legacy single-segment rows."""

        if self.failure_segments:
            return self.failure_segments
        if self.failure_segment is not None:
            return (self.failure_segment,)
        return ()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EpisodeRecord":
        value = dict(payload)

        def parse_segment(segment: Any) -> FailureSegment | None:
            if segment is None or isinstance(segment, FailureSegment):
                return segment
            if not isinstance(segment, dict):
                raise ValueError("failure segment must be an object")
            segment_value = dict(segment)
            for key in ("state_signature", "embedding", "artifact_paths"):
                segment_value[key] = tuple(segment_value.get(key, ()))
            return FailureSegment(**segment_value)

        value["failure_segment"] = parse_segment(value.get("failure_segment"))
        value["failure_segments"] = tuple(
            parsed
            for parsed in (
                parse_segment(segment) for segment in value.get("failure_segments", ())
            )
            if parsed is not None
        )
        value["safety_events"] = tuple(value.get("safety_events", ()))
        return cls(**value)


@dataclass(frozen=True, slots=True)
class TrajectoryIndex:
    """Seed-blind, content-addressed index handed to diagnosis agents."""

    episode_id: str
    logical_id: str
    success: bool
    action_count: int
    event_count: int
    artifact_paths: dict[str, str]
    artifact_sha256: dict[str, str]
    video_paths: tuple[str, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.episode_id or not self.logical_id:
            raise ValueError("trajectory index requires episode and logical IDs")
        if self.action_count < 0 or self.event_count < 0:
            raise ValueError("trajectory counts must be non-negative")
        if set(self.artifact_paths) != set(self.artifact_sha256):
            raise ValueError("every indexed artifact must have a digest")
        for name, digest in self.artifact_sha256.items():
            _require_sha(f"artifact_sha256[{name}]", digest)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.as_dict())


@dataclass(frozen=True, slots=True)
class FailureCluster:
    cluster_id: str
    hard_key: tuple[str, str, str]
    member_segment_ids: tuple[str, ...]
    episode_ids: tuple[str, ...]
    representative_segment_ids: tuple[str, ...]
    medoid_segment_id: str
    summary: str
    prevalence: float
    mean_severity: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


OwnerLayer = Literal[
    "tool",
    "perception",
    "critic",
    "planner",
    "vla",
    "recovery",
    "task",
    "infrastructure",
    "unknown",
]


@dataclass(frozen=True, slots=True)
class CausalDiagnosis:
    diagnosis_id: str
    cluster_id: str
    outcome: str
    immediate_trigger: str
    root_cause: str
    contributing_causes: tuple[str, ...]
    competing_hypotheses: tuple[str, ...]
    owner_layer: OwnerLayer
    affected_component: str
    earliest_divergence: str
    supporting_evidence_ids: tuple[str, ...]
    counterevidence_ids: tuple[str, ...]
    falsifier: str
    distinguishing_check: str
    required_validation: str
    confidence: float
    visual_evidence: tuple[dict[str, Any], ...] = ()
    # Immutable task identity supplied by the campaign manifest.  Keeping the
    # contract on the diagnosis binds Stage2 to the same objective that
    # Stage1 was asked to diagnose; the empty default preserves old digests.
    task_contract: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        required_text = {
            "diagnosis_id": self.diagnosis_id,
            "cluster_id": self.cluster_id,
            "outcome": self.outcome,
            "immediate_trigger": self.immediate_trigger,
            "root_cause": self.root_cause,
            "affected_component": self.affected_component,
            "earliest_divergence": self.earliest_divergence,
            "falsifier": self.falsifier,
            "distinguishing_check": self.distinguishing_check,
            "required_validation": self.required_validation,
        }
        missing = [name for name, value in required_text.items() if not value.strip()]
        if missing:
            raise ValueError(f"diagnosis text fields are required: {sorted(missing)}")
        if not 0 <= self.confidence <= 1:
            raise ValueError("diagnosis confidence must be in [0, 1]")
        for item in self.visual_evidence:
            if not isinstance(item, dict) or not isinstance(
                item.get("content_id"), str
            ):
                raise ValueError("visual evidence requires an opaque content_id")
            if not isinstance(item.get("access_record_id"), str):
                raise ValueError("visual evidence requires an immutable access_record_id")
        if not self.supporting_evidence_ids:
            raise ValueError("diagnosis requires supporting evidence")
        hypotheses = {value.strip().casefold() for value in self.competing_hypotheses}
        if len(hypotheses) < 2:
            raise ValueError("diagnosis requires at least two competing hypotheses")
        if self.task_contract:
            if not isinstance(self.task_contract, dict):
                raise ValueError("task_contract must be an object")
            for key in ("suite", "task", "language"):
                if not isinstance(self.task_contract.get(key), str) or not self.task_contract[key].strip():
                    raise ValueError(f"task_contract.{key} is required")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if not self.visual_evidence:
            # Empty visual_evidence is the semantic default for historical
            # diagnoses.  Omitting it preserves their previously committed
            # digest while new multimodal diagnoses bind the field normally.
            value.pop("visual_evidence", None)
        if not self.task_contract:
            value.pop("task_contract", None)
        return value

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.as_dict())


CriticOperator = Literal["lt", "le", "gt", "ge", "eq", "ne", "stagnant"]


@dataclass(frozen=True, slots=True)
class CriticPredicate:
    """Machine-executable activation guard for a temporal critic rule."""

    feature: str
    operator: Literal["lt", "le", "gt", "ge", "eq", "ne"]
    threshold: float | int | str | bool

    def __post_init__(self) -> None:
        if not self.feature.strip():
            raise ValueError("critic predicate feature is required")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CriticRule:
    rule_id: str
    title: str
    feature: str
    operator: CriticOperator
    threshold: float | int | str | bool
    dwell_steps: int
    cooldown_steps: int
    proposal: str
    evidence_ids: tuple[str, ...]
    safety_only: bool = False
    activation_conditions: tuple[CriticPredicate, ...] = ()

    def __post_init__(self) -> None:
        if not self.feature.strip():
            raise ValueError("critic feature is required")
        if self.dwell_steps < 1 or self.cooldown_steps < 0:
            raise ValueError("critic dwell/cooldown values are invalid")
        if not self.evidence_ids:
            raise ValueError("critic rule requires evidence")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RecoveryStep:
    tool: str
    parameters: dict[str, Any]
    stop_when: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RecoveryRule:
    recovery_id: str
    title: str
    trigger_rule_ids: tuple[str, ...]
    precondition: str
    steps: tuple[RecoveryStep, ...]
    safety_constraints: tuple[str, ...]
    stop_condition: str
    fallback: str
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.trigger_rule_ids:
            raise ValueError("recovery rule requires at least one critic trigger")
        if not self.steps:
            raise ValueError("recovery rule requires at least one actor step")
        if not self.evidence_ids:
            raise ValueError("recovery rule requires evidence")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CandidateBundle:
    candidate_id: str
    generation: int
    parent_sha256: str | None
    diagnosis_sha256: str
    causal_hypothesis: str
    mechanism_change: str
    validation_plan: str
    critic_rules: tuple[CriticRule, ...]
    recovery_rules: tuple[RecoveryRule, ...]
    tool_plugin: dict[str, Any] | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        _require_sha("parent_sha256", self.parent_sha256, optional=True)
        _require_sha("diagnosis_sha256", self.diagnosis_sha256)
        if not self.causal_hypothesis.strip():
            raise ValueError("candidate must bind exactly one causal hypothesis")
        if not self.mechanism_change.strip() or not self.validation_plan.strip():
            raise ValueError(
                "candidate requires one mechanism change and validation plan"
            )
        if not self.critic_rules and not self.recovery_rules and not self.tool_plugin:
            raise ValueError("candidate bundle must change one auditable behavior")
        if self.tool_plugin and (self.critic_rules or self.recovery_rules):
            raise ValueError(
                "an isolated tool-plugin candidate cannot mix critic/recovery changes"
            )
        rule_ids = [rule.rule_id for rule in self.critic_rules]
        recovery_ids = [rule.recovery_id for rule in self.recovery_rules]
        if len(set(rule_ids)) != len(rule_ids) or len(set(recovery_ids)) != len(
            recovery_ids
        ):
            raise ValueError("candidate rule identifiers must be unique")
        unknown_triggers = {
            trigger
            for recovery in self.recovery_rules
            for trigger in recovery.trigger_rule_ids
            if trigger not in set(rule_ids)
        }
        if unknown_triggers:
            raise ValueError(
                f"recovery rules reference unknown critic rules: {sorted(unknown_triggers)}"
            )
        covered_rules = {
            trigger
            for recovery in self.recovery_rules
            for trigger in recovery.trigger_rule_ids
        }
        uncovered_rules = set(rule_ids) - covered_rules
        if uncovered_rules:
            raise ValueError(
                "every critic rejection must have a frozen executable recovery "
                f"path: {sorted(uncovered_rules)}"
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.as_dict())

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CandidateBundle":
        critics = tuple(
            CriticRule(
                **{
                    **item,
                    "evidence_ids": tuple(item.get("evidence_ids", ())),
                    "activation_conditions": tuple(
                        condition
                        if isinstance(condition, CriticPredicate)
                        else CriticPredicate(**condition)
                        for condition in item.get("activation_conditions", ())
                    ),
                }
            )
            for item in payload.get("critic_rules", ())
        )
        recoveries = []
        for item in payload.get("recovery_rules", ()):
            recoveries.append(
                RecoveryRule(
                    recovery_id=item["recovery_id"],
                    title=item["title"],
                    trigger_rule_ids=tuple(item.get("trigger_rule_ids", ())),
                    precondition=item["precondition"],
                    steps=tuple(RecoveryStep(**step) for step in item["steps"]),
                    safety_constraints=tuple(item.get("safety_constraints", ())),
                    stop_condition=item["stop_condition"],
                    fallback=item["fallback"],
                    evidence_ids=tuple(item.get("evidence_ids", ())),
                )
            )
        return cls(
            candidate_id=payload["candidate_id"],
            generation=int(payload["generation"]),
            parent_sha256=payload.get("parent_sha256"),
            diagnosis_sha256=payload["diagnosis_sha256"],
            causal_hypothesis=payload["causal_hypothesis"],
            mechanism_change=payload["mechanism_change"],
            validation_plan=payload["validation_plan"],
            critic_rules=critics,
            recovery_rules=tuple(recoveries),
            tool_plugin=payload.get("tool_plugin"),
            schema_version=int(payload.get("schema_version", 1)),
        )


GateKind = Literal[
    "same_seed",
    "regression",
    "heldout_10",
    "heldout_20",
    "heldout_50",
]


@dataclass(frozen=True, slots=True)
class GateDecision:
    decision_id: str
    candidate_sha256: str
    parent_sha256: str | None
    kind: GateKind
    passed: bool
    conclusive: bool
    candidate_successes: int
    parent_successes: int
    paired_count: int
    candidate_wins: int
    parent_wins: int
    p_value: float | None
    alpha: float | None
    candidate_safety_events: int
    parent_safety_events: int
    rationale: str

    def __post_init__(self) -> None:
        _require_sha("candidate_sha256", self.candidate_sha256)
        _require_sha("parent_sha256", self.parent_sha256, optional=True)
        if self.paired_count < 1:
            raise ValueError("gate requires at least one paired episode")
        if self.passed and not self.conclusive:
            raise ValueError("an inconclusive gate cannot pass")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
