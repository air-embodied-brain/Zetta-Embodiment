# Copyright (c) 2026 RPent Contributors
"""Strict, content-addressed indexing of rollout trajectories.

This module deliberately accepts only completed, authoritative episode results.  It
never turns a partially written JSONL file into learning evidence and never exposes
environment or policy RNG seeds in the diagnosis-facing summary.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rpent.evolution.jsonio import canonical_sha256, file_sha256
from rpent.evolution.models import EpisodeRecord, FailureSegment, TrajectoryIndex

_STEP_KEYS = ("step_index", "step_idx", "action_index", "action_idx", "step")
_STAGE_KEYS = ("stage", "phase", "task_stage")
_TOOL_KEYS = ("tool", "tool_name", "selected_tool", "name")
_FORBIDDEN_AGENT_KEY = re.compile(r"(?:^|_)(?:seed|rng)(?:_|$)", re.IGNORECASE)
_SAFE_IDENTIFIER = re.compile(r"[^a-zA-Z0-9_.:/-]+")


class TrajectoryFormatError(ValueError):
    """Raised when a trajectory artifact is incomplete or structurally invalid."""


@dataclass(frozen=True, slots=True)
class TrajectoryArtifacts:
    """Paths which make up one immutable episode trajectory."""

    chunks: str | Path
    actions: str | Path
    states: str | Path
    tools: str | Path
    videos: tuple[str | Path, ...] = ()

    def named_paths(self) -> dict[str, Path]:
        result = {
            "chunks": Path(self.chunks),
            "actions": Path(self.actions),
            "states": Path(self.states),
            "tools": Path(self.tools),
        }
        for index, path in enumerate(self.videos):
            result[f"video-{index:02d}"] = Path(path)
        return result


@dataclass(frozen=True, slots=True)
class TrajectoryAnalysis:
    """Complete result of indexing one episode.

    Infrastructure-invalid attempts intentionally return ``index=None`` and no
    segments, so callers cannot accidentally include them in clustering.
    """

    index: TrajectoryIndex | None
    segments: tuple[FailureSegment, ...]


@dataclass(frozen=True, slots=True)
class _Event:
    source: str
    step: int
    payload: dict[str, Any]
    ordinal: int


@dataclass(frozen=True, slots=True)
class _Signal:
    priority: int
    failure_class: str
    stage: str
    tool: str | None
    step: int | None
    summary: str
    severity: float


def _strict_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a complete JSONL artifact or fail without returning partial rows."""

    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise TrajectoryFormatError(f"cannot read trajectory artifact {path}") from exc
    if payload and not payload.endswith(b"\n"):
        raise TrajectoryFormatError(f"truncated JSONL artifact: {path}")
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise TrajectoryFormatError(f"invalid UTF-8 JSONL artifact: {path}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise TrajectoryFormatError(f"blank JSONL record: {path}:{line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TrajectoryFormatError(
                f"malformed JSONL record: {path}:{line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise TrajectoryFormatError(
                f"JSONL record must be an object: {path}:{line_number}"
            )
        rows.append(value)
    return rows


def _result_dict(result: EpisodeRecord | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(result, EpisodeRecord):
        return result.as_dict()
    if not isinstance(result, Mapping):
        raise TypeError("result must be an EpisodeRecord or mapping")
    return dict(result)


def _validated_outcome(result: dict[str, Any]) -> tuple[str, bool | None]:
    status = result.get("status")
    if status not in {"valid", "infra_invalid"}:
        raise TrajectoryFormatError("authoritative result must declare a valid status")
    success = result.get("success")
    if status == "infra_invalid":
        if success is not None:
            raise TrajectoryFormatError("infra-invalid result cannot carry success")
        return status, None
    if not isinstance(success, bool):
        raise TrajectoryFormatError("valid result must carry boolean success")
    for key in ("authoritative_success", "libero_terminated"):
        if key in result and bool(result[key]) != success:
            raise TrajectoryFormatError(f"result success conflicts with {key}")
    return status, success


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, float) and value.is_integer() and value >= 0:
        return int(value)
    return None


def _step(payload: Mapping[str, Any], default: int) -> int:
    for key in _STEP_KEYS:
        value = _integer(payload.get(key))
        if value is not None:
            return value
    return default


def _identifier(value: Any, fallback: str) -> str:
    if not isinstance(value, str) or not value.strip():
        return fallback
    cleaned = _SAFE_IDENTIFIER.sub("-", value.strip()).strip("-")
    if not cleaned or _FORBIDDEN_AGENT_KEY.search(cleaned):
        return fallback
    return cleaned[:96]


def _field(payload: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _stage(payload: Mapping[str, Any], fallback: str = "closed_loop_execution") -> str:
    return _identifier(_field(payload, _STAGE_KEYS), fallback)


def _tool(payload: Mapping[str, Any]) -> str | None:
    value = _field(payload, _TOOL_KEYS)
    if value is None:
        return None
    return _identifier(value, "unknown-tool")


def _chunk_horizon(row: Mapping[str, Any]) -> int:
    environment = row.get("environment")
    candidates: tuple[Any, ...] = (
        row.get("executed_horizon"),
        environment.get("executed_horizon")
        if isinstance(environment, Mapping)
        else None,
    )
    for value in candidates:
        parsed = _integer(value)
        if parsed is not None:
            return parsed
    return 0


def _make_events(
    chunks: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    states: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> tuple[list[_Event], int]:
    events: list[_Event] = []
    ordinal = 0
    chunk_cursor = 0
    for row in chunks:
        events.append(_Event("chunk", _step(row, chunk_cursor), row, ordinal))
        ordinal += 1
        chunk_cursor += _chunk_horizon(row)
    for source, rows in (("action", actions), ("state", states), ("tool", tools)):
        for index, row in enumerate(rows):
            events.append(_Event(source, _step(row, index), row, ordinal))
            ordinal += 1
    events.sort(key=lambda event: (event.step, event.ordinal))
    explicit_action_end = max(
        (event.step + 1 for event in events if event.source == "action"), default=0
    )
    result_action_count = max(chunk_cursor, explicit_action_end, len(actions))
    return events, result_action_count


def _nested_mappings(value: Any) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        result.append(value)
        for nested in value.values():
            result.extend(_nested_mappings(nested))
    elif isinstance(value, list | tuple):
        for nested in value:
            result.extend(_nested_mappings(nested))
    return result


def _event_type(payload: Mapping[str, Any]) -> str:
    return " ".join(
        str(payload.get(key, "")).lower()
        for key in ("type", "event", "kind", "status", "outcome")
    )


def _critic_reject(event: _Event) -> _Signal | None:
    for payload in _nested_mappings(event.payload):
        event_type = _event_type(payload)
        proposals = payload.get("critic_proposals")
        explicit = (
            payload.get("critic_rejected") is True
            or payload.get("rejected") is True
            and "critic" in event_type
            or "critic_reject" in event_type
            or isinstance(proposals, list | tuple)
            and bool(proposals)
        )
        if explicit:
            step = _step(payload, event.step)
            tool = _tool(payload) or ("critic" if event.source != "tool" else None)
            return _Signal(
                0,
                "critic_reject",
                _stage(payload),
                tool,
                step,
                "Critic first rejected the proposed action at this point.",
                1.0,
            )
    return None


def _tool_error(event: _Event) -> _Signal | None:
    if event.source != "tool":
        return None
    for payload in _nested_mappings(event.payload):
        event_type = _event_type(payload)
        if "role1_contract_failure" in event_type:
            continue
        failed = (
            isinstance(payload.get("error"), str | Mapping)
            or payload.get("ok") is False
            or payload.get("success") is False
            or payload.get("failed") is True
            or any(
                token in event_type
                for token in ("tool_error", "failure", "failed", "exception")
            )
        )
        if failed:
            tool = _tool(payload) or _tool(event.payload) or "unknown-tool"
            return _Signal(
                1,
                "tool_error",
                _stage(payload),
                tool,
                _step(payload, event.step),
                f"Tool {tool} reported a structured execution error.",
                0.95,
            )
    return None


def _role1_contract_failure(event: _Event) -> _Signal | None:
    if event.source != "tool":
        return None
    for payload in _nested_mappings(event.payload):
        if "role1_contract_failure" not in _event_type(payload):
            continue
        return _Signal(
            1,
            "role1_contract_failure",
            _stage(payload),
            "role1",
            _step(payload, event.step),
            "Role1 returned an audited decision without an executable alternative.",
            1.0,
        )
    return None


def _explicit_no_progress(event: _Event) -> _Signal | None:
    for payload in _nested_mappings(event.payload):
        event_type = _event_type(payload)
        if (
            payload.get("no_progress") is True
            or payload.get("stagnant") is True
            or "no_progress" in event_type
            or "stagnant" in event_type
        ):
            return _Signal(
                2,
                "no_progress",
                _stage(payload),
                _tool(payload),
                _step(payload, event.step),
                "A structured event first reported no task progress.",
                0.85,
            )
    return None


def _progress_value(payload: Mapping[str, Any]) -> float | None:
    for key in ("task_progress", "progress", "completion"):
        value = payload.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            number = float(value)
            if math.isfinite(number):
                return number
    # Privileged simulator state is permitted for the rollout-learning critic.
    # Goal residuals are inverted so larger values consistently mean progress.
    for key, value in payload.items():
        normalized = str(key).lower().replace("-", "_")
        if (
            normalized.endswith(
                ("residual_to_success", "distance_to_goal", "goal_distance")
            )
            and isinstance(value, int | float)
            and not isinstance(value, bool)
        ):
            number = float(value)
            if math.isfinite(number):
                return -number
    for value in payload.values():
        if isinstance(value, Mapping):
            nested = _progress_value(value)
            if nested is not None:
                return nested
    return None


def _window_no_progress(events: list[_Event], window: int) -> _Signal | None:
    observations = [
        (event, value)
        for event in events
        if event.source == "state"
        and (value := _progress_value(event.payload)) is not None
    ]
    if len(observations) < window:
        return None

    def verified_engagement(payload: Mapping[str, Any]) -> bool:
        for key, value in payload.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized.endswith("target_contact") and value is True:
                return True
            if isinstance(value, Mapping) and verified_engagement(value):
                return True
        return False

    baseline = observations[0][1]
    for start in range(len(observations) - window + 1):
        sample = observations[start : start + window]
        values = [value for _, value in sample]
        tolerance = max(1e-4, abs(values[0]) * 1e-3)
        prior_values = [value for _, value in observations[: start + 1]]
        prior_progress = max(abs(value - baseline) for value in prior_values) > tolerance
        engaged = any(verified_engagement(event.payload) for event, _ in sample)
        if max(values) - min(values) <= tolerance and (prior_progress or engaged):
            event = sample[0][0]
            return _Signal(
                2,
                "no_progress",
                _stage(event.payload),
                _tool(event.payload),
                event.step,
                f"Task progress remained unchanged for {window} observations.",
                0.8,
            )
    return None


def _early_termination(event: _Event) -> _Signal | None:
    for payload in _nested_mappings(event.payload):
        event_type = _event_type(payload)
        explicit = (
            payload.get("early_termination") is True
            or payload.get("role1_terminated") is True
            or payload.get("terminate") is True
            or "early_termination" in event_type
            or "role1_terminated" in event_type
        )
        if explicit and payload.get("authoritative_success") is not True:
            return _Signal(
                3,
                "early_termination",
                _stage(payload),
                _tool(payload),
                _step(payload, event.step),
                "The actor terminated before authoritative task success.",
                0.9,
            )
    return None


def _safety_event(event: _Event) -> _Signal | None:
    for payload in _nested_mappings(event.payload):
        event_type = _event_type(payload)
        explicit = (
            payload.get("out_of_bounds") is True
            or payload.get("safety_violation") is True
            or any(
                token in event_type
                for token in ("out_of_bounds", "safety_violation")
            )
        )
        if explicit:
            return _Signal(
                0,
                "safety_event",
                _stage(payload),
                _tool(payload),
                _step(payload, event.step),
                "A structured safety event occurred before task completion.",
                1.0,
            )
    return None


def _text_embedding(text: str, *, dimensions: int = 32) -> tuple[float, ...]:
    """Return a deterministic local text embedding without an external model."""

    vector = [0.0] * dimensions
    tokens = re.findall(r"[a-z0-9_.:-]+", text.casefold())
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:2], "big") % dimensions
        vector[bucket] += 1.0 if digest[2] & 1 else -1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return tuple(vector)
    return tuple(round(value / norm, 8) for value in vector)


def _expected_actions(result: Mapping[str, Any]) -> int | None:
    for key in ("expected_actions", "max_actions", "policy_steps", "action_horizon"):
        value = _integer(result.get(key))
        if value is not None:
            return value
    artifact_index = result.get("artifact_index")
    if isinstance(artifact_index, Mapping):
        for key in (
            "expected_actions",
            "max_actions",
            "policy_steps",
            "action_horizon",
        ):
            value = _integer(artifact_index.get(key))
            if value is not None:
                return value
    return None


def _fallback_incomplete(result: Mapping[str, Any], action_count: int) -> _Signal:
    expected = _expected_actions(result)
    if expected is not None and action_count < expected:
        summary = "The valid episode ended before its authoritative action horizon."
    else:
        summary = (
            "The task remained incomplete at the authoritative horizon; no earlier "
            "structured divergence signal was recorded."
        )
    return _Signal(
        4,
        "horizon_incomplete",
        "closed_loop_execution",
        None,
        None,
        summary,
        0.6,
    )


def _state_signature(
    events: list[_Event], start_step: int, end_step: int
) -> tuple[float, ...]:
    values: list[float] = []
    for event in events:
        if event.source != "state" or not start_step <= event.step <= end_step:
            continue
        for key, value in sorted(event.payload.items()):
            if _FORBIDDEN_AGENT_KEY.search(str(key)):
                continue
            if isinstance(value, int | float) and not isinstance(value, bool):
                number = float(value)
                if math.isfinite(number):
                    values.append(round(number, 6))
            if len(values) >= 16:
                return tuple(values)
    return tuple(values)


def _content_references(digests: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(f"{name}@sha256:{digest}" for name, digest in sorted(digests.items()))


def _segments(
    *,
    result: Mapping[str, Any],
    events: list[_Event],
    action_count: int,
    digests: Mapping[str, str],
    context_before: int,
    context_after: int,
    no_progress_window: int,
) -> tuple[FailureSegment, ...]:
    if result["success"] is True:
        return ()
    signals: list[_Signal] = []
    detectors = (
        _safety_event,
        _critic_reject,
        _role1_contract_failure,
        _tool_error,
        _explicit_no_progress,
        _early_termination,
    )
    for detector in detectors:
        matches = [
            signal for event in events if (signal := detector(event)) is not None
        ]
        if matches:
            signals.append(
                min(matches, key=lambda item: (item.step, item.stage, item.tool or ""))
            )
    if not any(signal.failure_class == "no_progress" for signal in signals):
        stagnant = _window_no_progress(events, no_progress_window)
        if stagnant is not None:
            signals.append(stagnant)
    # The generic horizon outcome is useful only when no structured causal clue
    # exists. Adding it to every failed trajectory would create an artificial
    # 100%-prevalence cluster and hide the actionable failure modes.
    if not signals:
        signals.append(_fallback_incomplete(result, action_count))

    # One deterministic segment per failure class/tool/stage.  The priority only
    # determines ordering; an episode may retain several causal clues.
    unique: dict[tuple[str, str, str], _Signal] = {}
    for signal in signals:
        key = (signal.failure_class, signal.stage, signal.tool or "")
        existing = unique.get(key)
        signal_order = (
            signal.step if signal.step is not None else action_count,
            signal.priority,
        )
        existing_order = (
            existing.step if existing is not None and existing.step is not None else action_count,
            existing.priority if existing is not None else signal.priority,
        )
        if existing is None or signal_order < existing_order:
            unique[key] = signal

    episode_id = str(result["episode_id"])
    content_refs = _content_references(digests)
    output: list[FailureSegment] = []
    for signal in sorted(
        unique.values(),
        key=lambda item: (
            item.priority,
            item.step if item.step is not None else action_count,
            item.stage,
            item.tool or "",
        ),
    ):
        signal_step = signal.step if signal.step is not None else action_count
        start = max(0, signal_step - context_before) if signal.step is not None else 0
        observed_end = max(
            action_count, max((event.step for event in events), default=0)
        )
        end = (
            max(signal_step, min(observed_end, signal_step + context_after))
            if signal.step is not None
            else observed_end
        )
        identity = canonical_sha256(
            {
                "episode_id": episode_id,
                "failure_class": signal.failure_class,
                "stage": signal.stage,
                "tool": signal.tool,
                "earliest_divergence_step": signal.step,
                "start_step": start,
                "end_step": end,
                "artifact_sha256": dict(sorted(digests.items())),
            }
        )
        output.append(
            FailureSegment(
                segment_id=f"segment-{identity[:24]}",
                episode_id=episode_id,
                failure_class=signal.failure_class,
                stage=signal.stage,
                tool=signal.tool,
                summary=signal.summary,
                earliest_divergence_step=signal.step,
                start_step=start,
                end_step=end,
                severity=signal.severity,
                state_signature=_state_signature(events, start, end),
                embedding=_text_embedding(
                    " ".join(
                        (
                            signal.failure_class,
                            signal.stage,
                            signal.tool or "",
                            signal.summary,
                        )
                    )
                ),
                artifact_paths=content_refs,
            )
        )
    return tuple(output)


def index_episode_trajectory(
    *,
    result: EpisodeRecord | Mapping[str, Any],
    artifacts: TrajectoryArtifacts,
    context_before: int = 8,
    context_after: int = 8,
    no_progress_window: int = 8,
) -> TrajectoryAnalysis:
    """Build a stable index and structured failure segments for one episode."""

    if context_before < 0 or context_after < 0:
        raise ValueError("trajectory context widths must be non-negative")
    if no_progress_window < 2:
        raise ValueError("no_progress_window must be at least two")
    outcome = _result_dict(result)
    status, success = _validated_outcome(outcome)
    if status == "infra_invalid":
        return TrajectoryAnalysis(index=None, segments=())
    for key in ("episode_id", "logical_id"):
        if not isinstance(outcome.get(key), str) or not outcome[key]:
            raise TrajectoryFormatError(f"authoritative result requires {key}")

    named_paths = artifacts.named_paths()
    rows = {
        name: _strict_jsonl(named_paths[name])
        for name in ("chunks", "actions", "states", "tools")
    }
    for name, path in named_paths.items():
        if name.startswith("video-") and (
            not path.is_file() or path.stat().st_size == 0
        ):
            raise TrajectoryFormatError(f"missing or empty video artifact: {path}")
    digests = {name: file_sha256(path) for name, path in sorted(named_paths.items())}
    events, action_count = _make_events(
        rows["chunks"], rows["actions"], rows["states"], rows["tools"]
    )
    index = TrajectoryIndex(
        episode_id=outcome["episode_id"],
        logical_id=outcome["logical_id"],
        success=bool(success),
        action_count=action_count,
        event_count=sum(len(value) for value in rows.values()),
        artifact_paths={
            name: path.as_posix() for name, path in sorted(named_paths.items())
        },
        artifact_sha256=digests,
        video_paths=tuple(Path(path).as_posix() for path in artifacts.videos),
    )
    return TrajectoryAnalysis(
        index=index,
        segments=_segments(
            result=outcome,
            events=events,
            action_count=action_count,
            digests=digests,
            context_before=context_before,
            context_after=context_after,
            no_progress_window=no_progress_window,
        ),
    )


def build_trajectory_index(
    *, result: EpisodeRecord | Mapping[str, Any], artifacts: TrajectoryArtifacts
) -> TrajectoryIndex | None:
    """Compatibility wrapper returning only the content-addressed index."""

    return index_episode_trajectory(result=result, artifacts=artifacts).index


def extract_failure_segments(
    *,
    result: EpisodeRecord | Mapping[str, Any],
    artifacts: TrajectoryArtifacts,
    context_before: int = 8,
    context_after: int = 8,
    no_progress_window: int = 8,
) -> tuple[FailureSegment, ...]:
    """Compatibility wrapper returning only failure segments."""

    return index_episode_trajectory(
        result=result,
        artifacts=artifacts,
        context_before=context_before,
        context_after=context_after,
        no_progress_window=no_progress_window,
    ).segments


def trajectory_agent_summary(analysis: TrajectoryAnalysis) -> dict[str, Any]:
    """Return a path-, identifier-, seed-, and RNG-blind diagnosis summary."""

    if analysis.index is None:
        return {"included": False, "reason": "infrastructure_invalid"}
    summary = {
        "included": True,
        "success": analysis.index.success,
        "action_count": analysis.index.action_count,
        "event_count": analysis.index.event_count,
        "artifact_content_ids": [
            f"sha256:{digest}"
            for digest in sorted(analysis.index.artifact_sha256.values())
        ],
        "failure_segments": [
            {
                "segment_content_id": f"sha256:{canonical_sha256(segment.as_dict())}",
                "failure_class": segment.failure_class,
                "stage": segment.stage,
                "tool": segment.tool,
                "summary": segment.summary,
                "earliest_divergence_step": segment.earliest_divergence_step,
                "start_step": segment.start_step,
                "end_step": segment.end_step,
            }
            for segment in analysis.segments
        ],
    }
    if any(_FORBIDDEN_AGENT_KEY.search(key) for key in _walk_keys(summary)):
        raise AssertionError("agent-facing trajectory summary leaked seed/RNG metadata")
    return summary


def _walk_keys(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        return [str(key) for key in value for key in (key, *_walk_keys(value[key]))]
    if isinstance(value, list | tuple):
        return [key for item in value for key in _walk_keys(item)]
    return []
