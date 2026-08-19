# Copyright (c) 2026 Zetta Contributors
"""Read-only Critic replay over immutable parent trajectories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from zetta.evolution.critic import TemporalCritic
from zetta.evolution.jsonio import canonical_sha256, file_sha256
from zetta.evolution.models import CandidateBundle, EpisodeRecord


def _scalar_names(value: Any, *, prefix: str = "") -> set[str]:
    if isinstance(value, dict):
        result: set[str] = set()
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            result.update(_scalar_names(item, prefix=name))
        return result
    return {prefix} if prefix and isinstance(value, (bool, int, float)) else set()


def _states(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("state"), dict):
                raise ValueError(f"invalid state row {path.name}:{line_number}")
            rows.append(row)
    return rows


def evaluate_shadow_replay(
    *,
    candidate: CandidateBundle,
    parent: CandidateBundle | None,
    target_records: tuple[tuple[EpisodeRecord, Path], ...],
    success_controls: tuple[tuple[EpisodeRecord, Path], ...],
) -> dict[str, Any]:
    """Measure whether only the candidate delta detects historical failures."""

    parent_ids = {rule.rule_id for rule in parent.critic_rules} if parent else set()
    delta_rules = tuple(
        rule for rule in candidate.critic_rules if rule.rule_id not in parent_ids
    )
    if len(delta_rules) != 1:
        raise ValueError("shadow replay requires exactly one candidate-delta critic")
    expected_parent = candidate.parent_sha256
    records = [*target_records, *success_controls]
    if not target_records:
        raise ValueError("shadow replay requires target failure trajectories")
    feature_names: set[str] = set()
    outcomes = []
    for role, entries in (("target_failure", target_records), ("success_control", success_controls)):
        for record, states_path in entries:
            if record.status != "valid":
                raise ValueError("infra-invalid trajectories cannot enter shadow replay")
            if record.bundle_sha256 != expected_parent:
                raise ValueError("shadow replay trajectory belongs to another parent")
            rows = _states(states_path)
            critic = TemporalCritic(delta_rules)
            first_trigger = None
            trigger_ids: list[str] = []
            unavailable_prefix_steps: list[int] = []
            evaluated_state = False
            replay_evaluable = True
            replay_inconclusive_reason: str | None = None
            for row in rows:
                state = row["state"]
                feature_names.update(_scalar_names(state))
                step = int(row.get("step_index", 0))
                try:
                    proposals = critic.evaluate(state, step_index=step)
                except KeyError:
                    if evaluated_state:
                        # A partially instrumented legacy trace can expose a
                        # feature on one row and omit it later.  Do not infer
                        # a detector failure or abort the campaign; retain the
                        # immutable rows and require online same-seed evidence.
                        replay_evaluable = False
                        replay_inconclusive_reason = (
                            "required_critic_features_changed_during_replay"
                        )
                        unavailable_prefix_steps.append(step)
                        break
                    unavailable_prefix_steps.append(step)
                    continue
                evaluated_state = True
                if proposals and first_trigger is None:
                    first_trigger = step
                trigger_ids.extend(str(item["rule_id"]) for item in proposals)
            if not evaluated_state:
                # Legacy audit rows may predate a Critic-only feature (for
                # example command.gripper or privileged.*).  They remain
                # immutable evidence, but cannot support a retrospective
                # detector claim.  Fail closed and require online same-seed
                # evidence instead of crashing proposal stage.
                replay_evaluable = False
                replay_inconclusive_reason = "required_critic_features_never_logged"
            known_divergences = [
                segment.earliest_divergence_step
                for segment in record.all_failure_segments
                if segment.earliest_divergence_step is not None
            ]
            divergence = min(known_divergences) if known_divergences else None
            before_divergence = (
                first_trigger is not None
                and divergence is not None
                and first_trigger <= divergence
            )
            outcomes.append(
                {
                    "episode_id": record.episode_id,
                    "role": role,
                    "trajectory_sha256": file_sha256(states_path),
                    "parent_bundle_sha256": record.bundle_sha256,
                    "first_trigger_step": first_trigger,
                    "triggered": first_trigger is not None,
                    "triggered_before_or_at_divergence": before_divergence,
                    "trigger_rule_ids": sorted(set(trigger_ids)),
                    "earliest_divergence_step": divergence,
                    "lead_time_steps": (
                        divergence - first_trigger
                        if divergence is not None and first_trigger is not None
                        else None
                    ),
                    "unavailable_feature_prefix_steps": unavailable_prefix_steps,
                    "replay_evaluable": replay_evaluable,
                    "replay_inconclusive_reason": replay_inconclusive_reason,
                }
            )
    target = [row for row in outcomes if row["role"] == "target_failure"]
    controls = [row for row in outcomes if row["role"] == "success_control"]
    known_target = [row for row in target if row["earliest_divergence_step"] is not None]
    detected = sum(bool(row["triggered_before_or_at_divergence"]) for row in known_target)
    false_positives = sum(bool(row["triggered"]) for row in controls)
    report = {
        "schema_version": 1,
        "candidate_sha256": candidate.sha256,
        "parent_bundle_sha256": expected_parent,
        "delta_rule_ids": [rule.rule_id for rule in delta_rules],
        "feature_schema_sha256": canonical_sha256(sorted(feature_names)),
        "target_count": len(target),
        "target_detected": detected,
        "target_recall": detected / len(known_target) if known_target else None,
        "target_known_divergence_count": len(known_target),
        "target_unknown_divergence_count": len(target) - len(known_target),
        "replay_unevaluable_count": sum(
            not bool(row["replay_evaluable"]) for row in outcomes
        ),
        "target_triggered_anywhere": sum(bool(row["triggered"]) for row in target),
        "success_control_count": len(controls),
        "success_control_false_positives": false_positives,
        "success_control_false_positive_rate": (
            false_positives / len(controls) if controls else None
        ),
        "preflight_conclusive": (
            bool(controls)
            and len(known_target) == len(target)
            and all(bool(row["replay_evaluable"]) for row in outcomes)
        ),
        "passed_detection_preflight": (
            bool(controls)
            and len(known_target) == len(target)
            and detected == len(target)
            and false_positives == 0
        ),
        "preflight_disposition": (
            "inconclusive_unknown_divergence"
            if len(known_target) != len(target)
            else "conclusive_passed"
            if bool(controls) and detected == len(target) and false_positives == 0
            else "conclusive_rejected"
        ),
        "outcomes": outcomes,
        "limitations": (
            "Read-only replay validates detection only. It cannot establish recovery "
            "causality because new actions produce a different future."
        ),
    }
    report["report_sha256"] = canonical_sha256(report)
    return report
