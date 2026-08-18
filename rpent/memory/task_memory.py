"""Versioned task-scoped memory treated as an episode-level parameter block.

The mutable store is owned by the experiment orchestrator.  A rollout receives
only one immutable snapshot rendered into its initial prompt; the in-episode
planner never receives an update API or the mutable store path.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MEMORY_SCHEMA_VERSION = 1
MAX_MEMORY_BYTES = 24_000
MAX_EVIDENCE_ROWS = 64

PARAMETER_KEYS = {
    "strategy_summary",
    "decision_rules",
    "tool_parameters",
    "prompt_ladder",
    "failure_modes",
    "success_checks",
    "open_questions",
}

_ABSOLUTE_PATH = re.compile(r"(?:/data\d*/|[A-Za-z]:[\\/])")
_SEED_COORDINATE = re.compile(
    r"(?:world_?xyz|absolute_?xyz|pixel(?:s|_rc)?|row\s*[,=:]|col\s*[,=:])",
    re.IGNORECASE,
)
_FORBIDDEN_KEY_TOKENS = {"xyz", "world_xyz", "pixel", "pixels", "row", "col"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    """Return stable UTF-8 JSON used by the hash chain."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _empty_parameter_block() -> dict[str, Any]:
    return {
        "strategy_summary": (
            "No learned task-specific strategy yet. Use current-scene perception, "
            "the static RPent guidance, and conservative tool contracts."
        ),
        "decision_rules": [],
        "tool_parameters": [],
        "prompt_ladder": [],
        "failure_modes": [],
        "success_checks": [],
        "open_questions": [
            "Which seed-invariant strategy and parameters consistently improve success?"
        ],
    }


def build_initial_memory(*, task_key: str, suite: str, task: int) -> dict[str, Any]:
    """Create revision zero without encoding seed-specific task knowledge."""
    block = {
        "schema_version": MEMORY_SCHEMA_VERSION,
        "task_key": task_key,
        "suite": suite,
        "task": int(task),
        "revision": 0,
        "parent_sha256": None,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "updated_from_episode": None,
        "parameter_block": _empty_parameter_block(),
        "evidence": [],
        "constraints": {
            "update_boundary": "after_episode_only",
            "episode_mutability": "frozen",
            "seed_specific_coordinates_forbidden": True,
            "ground_truth_pose_or_bddl_forbidden": True,
            "max_bytes": MAX_MEMORY_BYTES,
        },
    }
    validate_task_memory(block, expected_task_key=task_key)
    return block


def _require_string_list(value: Any, *, name: str, limit: int) -> None:
    if not isinstance(value, list) or len(value) > limit:
        raise ValueError(f"{name} must be an array with at most {limit} items")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{name} items must be non-empty strings")


def _validate_tool_parameters(value: Any) -> None:
    if not isinstance(value, list) or len(value) > 16:
        raise ValueError("tool_parameters must be an array with at most 16 items")
    required = {"tool", "when", "parameters", "why", "confidence"}
    for row in value:
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError(
                "each tool_parameters row requires exactly " + ", ".join(sorted(required))
            )
        if not isinstance(row["tool"], str) or not row["tool"].strip():
            raise ValueError("tool_parameters.tool must be a non-empty string")
        if not isinstance(row["when"], str) or not row["when"].strip():
            raise ValueError("tool_parameters.when must be a non-empty string")
        if not isinstance(row["why"], str) or not row["why"].strip():
            raise ValueError("tool_parameters.why must be a non-empty string")
        if not isinstance(row["parameters"], dict):
            raise ValueError("tool_parameters.parameters must be an object")
        confidence = row["confidence"]
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            raise ValueError("tool_parameters.confidence must be in [0,1]")


def _validate_failure_modes(value: Any) -> None:
    if not isinstance(value, list) or len(value) > 16:
        raise ValueError("failure_modes must be an array with at most 16 items")
    required = {"symptom", "likely_cause", "response", "evidence_ids"}
    for row in value:
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError(
                "each failure_modes row requires exactly " + ", ".join(sorted(required))
            )
        for key in ("symptom", "likely_cause", "response"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f"failure_modes.{key} must be a non-empty string")
        _require_string_list(row["evidence_ids"], name="failure_modes.evidence_ids", limit=16)


def _walk_forbidden(value: Any, *, path: str = "parameter_block") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in _FORBIDDEN_KEY_TOKENS:
                raise ValueError(f"seed-specific coordinate key is forbidden: {path}.{key}")
            _walk_forbidden(child, path=f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _walk_forbidden(child, path=f"{path}[{index}]")
        return
    if isinstance(value, str):
        if _ABSOLUTE_PATH.search(value):
            raise ValueError(f"absolute artifact path is forbidden in memory: {path}")
        if _SEED_COORDINATE.search(value):
            raise ValueError(f"seed-specific coordinate language is forbidden: {path}")
        if "bddl" in value.lower():
            raise ValueError(f"BDDL-derived memory is forbidden: {path}")


def validate_parameter_block(value: Any) -> dict[str, Any]:
    """Validate and return a defensive copy of a model-authored parameter block."""
    if not isinstance(value, dict) or set(value) != PARAMETER_KEYS:
        raise ValueError(
            "parameter_block requires exactly: " + ", ".join(sorted(PARAMETER_KEYS))
        )
    if not isinstance(value["strategy_summary"], str) or not value[
        "strategy_summary"
    ].strip():
        raise ValueError("strategy_summary must be a non-empty string")
    if len(value["strategy_summary"]) > 2400:
        raise ValueError("strategy_summary is too long")
    _require_string_list(value["decision_rules"], name="decision_rules", limit=16)
    _validate_tool_parameters(value["tool_parameters"])
    _require_string_list(value["prompt_ladder"], name="prompt_ladder", limit=12)
    _validate_failure_modes(value["failure_modes"])
    _require_string_list(value["success_checks"], name="success_checks", limit=12)
    _require_string_list(value["open_questions"], name="open_questions", limit=12)
    _walk_forbidden(value)
    copied = deepcopy(value)
    if len(canonical_json(copied).encode("utf-8")) > MAX_MEMORY_BYTES:
        raise ValueError("parameter_block exceeds the memory byte limit")
    return copied


def validate_task_memory(
    value: Any,
    *,
    expected_task_key: str | None = None,
    expected_revision: int | None = None,
) -> dict[str, Any]:
    """Validate one complete canonical memory revision."""
    if not isinstance(value, dict):
        raise ValueError("task memory must be an object")
    if value.get("schema_version") != MEMORY_SCHEMA_VERSION:
        raise ValueError(f"task memory schema_version must be {MEMORY_SCHEMA_VERSION}")
    if not isinstance(value.get("task_key"), str) or not value["task_key"].strip():
        raise ValueError("task_key must be a non-empty string")
    if expected_task_key is not None and value["task_key"] != expected_task_key:
        raise ValueError(
            f"task_key mismatch: expected {expected_task_key!r}, got {value['task_key']!r}"
        )
    if not isinstance(value.get("suite"), str) or not value["suite"].strip():
        raise ValueError("suite must be a non-empty string")
    if not isinstance(value.get("task"), int) or value["task"] < 0:
        raise ValueError("task must be a non-negative integer")
    if not isinstance(value.get("revision"), int) or value["revision"] < 0:
        raise ValueError("revision must be a non-negative integer")
    if expected_revision is not None and value["revision"] != expected_revision:
        raise ValueError(
            f"revision mismatch: expected {expected_revision}, got {value['revision']}"
        )
    validate_parameter_block(value.get("parameter_block"))
    evidence = value.get("evidence")
    if not isinstance(evidence, list) or len(evidence) > MAX_EVIDENCE_ROWS:
        raise ValueError(f"evidence must contain at most {MAX_EVIDENCE_ROWS} rows")
    constraints = value.get("constraints")
    if not isinstance(constraints, dict) or constraints.get("update_boundary") != "after_episode_only":
        raise ValueError("memory constraints must enforce after_episode_only")
    if len(canonical_json(value).encode("utf-8")) > MAX_MEMORY_BYTES * 2:
        raise ValueError("task memory revision is too large")
    return deepcopy(value)


def render_episode_memory(value: dict[str, Any]) -> str:
    """Render one immutable snapshot for prompt injection."""
    block = validate_task_memory(value)
    digest = canonical_sha256(block)
    parameters = json.dumps(
        block["parameter_block"], ensure_ascii=False, sort_keys=True, indent=2
    )
    return f"""TASK-SCOPED EPISODE MEMORY (IMMUTABLE PARAMETER SNAPSHOT)

task_key: {block['task_key']}
revision: {block['revision']}
sha256: {digest}

This parameter block was frozen before the episode. Use it as fallible prior
knowledge, re-perceive the current seed, and prefer current observations when
they disagree. During this episode you MUST NOT create, edit, summarize, or
otherwise update task/tool/skill memory. Natural reasoning in this planner
conversation is allowed. The external orchestrator may rewrite memory only
after the episode has fully ended.

```json
{parameters}
```
"""


class TaskMemoryStore:
    """Append-only hash-chained revisions for one task memory block."""

    def __init__(self, root: str | Path, *, task_key: str, suite: str, task: int):
        self.root = Path(root).expanduser().resolve()
        self.task_key = task_key
        self.suite = suite
        self.task = int(task)
        self.task_dir = self.root / task_key.replace("/", "__")
        self.versions_dir = self.task_dir / "versions"
        self.current_path = self.task_dir / "current.json"
        self.events_path = self.task_dir / "updates.jsonl"

    def initialize(self) -> dict[str, Any]:
        if self.current_path.exists():
            return self.load()
        block = build_initial_memory(
            task_key=self.task_key,
            suite=self.suite,
            task=self.task,
        )
        self._write_revision(block, event=None)
        return block

    def load(self) -> dict[str, Any]:
        value = json.loads(self.current_path.read_text(encoding="utf-8"))
        return validate_task_memory(value, expected_task_key=self.task_key)

    def snapshot(self, destination: str | Path) -> dict[str, Any]:
        block = self.load()
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(path, block)
        return block

    def update(
        self,
        *,
        parameter_block: dict[str, Any],
        episode: dict[str, Any],
        updater: dict[str, Any],
    ) -> dict[str, Any]:
        previous = self.load()
        parameters = validate_parameter_block(parameter_block)
        evidence = [*previous.get("evidence", []), deepcopy(episode)][-MAX_EVIDENCE_ROWS:]
        block = {
            **previous,
            "revision": int(previous["revision"]) + 1,
            "parent_sha256": canonical_sha256(previous),
            "updated_at": _utc_now(),
            "updated_from_episode": deepcopy(episode),
            "parameter_block": parameters,
            "evidence": evidence,
        }
        validate_task_memory(
            block,
            expected_task_key=self.task_key,
            expected_revision=int(previous["revision"]) + 1,
        )
        event = {
            "event": "memory_updated",
            "timestamp": _utc_now(),
            "task_key": self.task_key,
            "revision": block["revision"],
            "parent_sha256": block["parent_sha256"],
            "memory_sha256": canonical_sha256(block),
            "episode": deepcopy(episode),
            "updater": deepcopy(updater),
        }
        self._write_revision(block, event=event)
        return block

    def _write_revision(
        self, block: dict[str, Any], *, event: dict[str, Any] | None
    ) -> None:
        validated = validate_task_memory(block, expected_task_key=self.task_key)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        version_path = self.versions_dir / f"rev-{validated['revision']:04d}.json"
        if version_path.exists():
            existing = json.loads(version_path.read_text(encoding="utf-8"))
            if canonical_sha256(existing) != canonical_sha256(validated):
                raise FileExistsError(f"refusing to overwrite memory revision: {version_path}")
        else:
            _atomic_write_json(version_path, validated)
        _atomic_write_json(self.current_path, validated)
        if event is not None:
            self.task_dir.mkdir(parents=True, exist_ok=True)
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(canonical_json(event) + "\n")
                handle.flush()
                os.fsync(handle.fileno())


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
