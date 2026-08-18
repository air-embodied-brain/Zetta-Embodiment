# Copyright (c) 2026 RPent Contributors
"""Recoverable offline Diagnoser/Evolver/gate lifecycle for one generation.

The Stage1/Stage2 names remain on disk solely for schema and recovery
compatibility. They are campaign-control agents; online episode authority
belongs only to Role1.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import secrets
from pathlib import Path
from typing import Any

from rpent.evolution.gating import _candidate_intervened, evaluate_paired_gate
from rpent.evolution.jsonio import (
    AppendOnlyLedger,
    atomic_write_json,
    canonical_sha256,
    directory_lock,
    file_sha256,
    read_json,
)
from rpent.evolution.models import (
    CampaignPhase,
    CandidateBundle,
    CausalDiagnosis,
    EpisodeRecord,
    FailureCluster,
    GateDecision,
)
from rpent.evolution.shadow_replay import evaluate_shadow_replay
from rpent.evolution.stages import CodexStageAgent
from rpent.evolution.store import CampaignStore

_RESOLVER_SCHEMA_VERSION = 1
_CONTENT_ID_PATTERN = re.compile(r"artifact-[0-9a-f]{64}\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_ARTIFACT_SUFFIXES = {
    ".avi",
    ".bin",
    ".bmp",
    ".json",
    ".jsonl",
    ".log",
    ".md",
    ".mkv",
    ".mov",
    ".mp4",
    ".npy",
    ".npz",
    ".png",
    ".pt",
    ".txt",
    ".webm",
    ".webp",
}
_SENSITIVE_TEXT = re.compile(
    r"(?i)(?:policy[\s_-]*rng|future[\s_-]*schedule|environment[\s_-]*seed|seed|rng)"
    r"(?:[\s_:=#-]*[a-z0-9./\\:-]+)?"
)
_PATH_TEXT = re.compile(r"(?:[A-Za-z]:\\\S+|(?:/[^\s/]+){2,})")


def _authoritative_task_contract(store: CampaignStore) -> dict[str, Any] | None:
    """Load immutable task identity from manifest runtime or a sidecar."""
    manifest = store.manifest()
    runtime = manifest.runtime
    value = runtime.get("task_contract") if isinstance(runtime, dict) else None
    sidecar = store.root / "task-contract.json"
    sidecar_value = read_json(sidecar) if sidecar.is_file() else None
    if value is not None and sidecar_value is not None and value != sidecar_value:
        raise ValueError("manifest and sidecar task contracts differ")
    if value is None:
        value = sidecar_value
    if value is None:
        if manifest.environment == "libero_pro":
            raise ValueError(
                "formal LIBERO campaign requires an authoritative task_contract"
            )
        return None
    if not isinstance(value, dict):
        raise ValueError("task_contract must be an object")
    contract = dict(value)
    for key in ("suite", "task", "language"):
        if not isinstance(contract.get(key), str) or not contract[key].strip():
            raise ValueError(f"task_contract.{key} is required")
        contract[key] = contract[key].strip()
    if contract["task"] != manifest.task:
        raise ValueError("task_contract.task does not match campaign manifest")
    normalized_language = " ".join(contract["language"].casefold().split())
    if contract.get("normalized_language", normalized_language) != normalized_language:
        raise ValueError("task_contract.normalized_language does not match language")
    expected_language_sha256 = canonical_sha256({"language": normalized_language})
    if contract.get("language_sha256", expected_language_sha256) != expected_language_sha256:
        raise ValueError("task_contract.language_sha256 does not match language")
    return contract

_DIAGNOSTIC_TELEMETRY_FEATURES = (
    "episode.step_index",
    "episode.reward",
    "episode.terminated",
    "episode.truncated",
    "robot.eef.x",
    "robot.eef.y",
    "robot.eef.z",
    "robot.eef.motion_m",
    "robot.eef.delta_available",
    "robot.eef.delta.x",
    "robot.eef.delta.y",
    "robot.eef.delta.z",
    "robot.gripper.opening",
    "command.available",
    "command.translation.x",
    "command.translation.y",
    "command.translation.z",
    "command.translation.norm",
    "command.rotation.norm",
    "command.rotation.x",
    "command.rotation.y",
    "command.rotation.z",
    "command.gripper",
    "command.realization.direction_available",
    "command.realization.direction_cosine",
    "command.realization.eef_motion_m",
    "command.realization.stalled",
    "privileged.available",
    "privileged.task.semantic_available",
    "privileged.task.success",
    "privileged.task.goal.predicate_count",
    "privileged.task.goal.evaluable_count",
    "privileged.task.goal.satisfied_count",
    "privileged.task.goal.progress_available",
    "privileged.task.goal.progress",
    "privileged.task.primary_relation",
    "privileged.task.primary_relation_satisfied",
    "privileged.task.manipulated_object.distance_to_eef_m",
    "privileged.task.manipulated_object.distance_to_target_m",
    "privileged.task.manipulated_object.gripper_contact",
    "privileged.task.manipulated_object.robot_contact",
    "privileged.task.manipulated_object.grasped",
    "privileged.task.manipulated_object.mechanical_engagement",
    "privileged.task.manipulated_object.coupled",
    "privileged.task.manipulated_object.ever_grasped",
    "privileged.task.manipulated_object.retained",
    "privileged.task.manipulated_object.released_now",
    "privileged.task.manipulated_object.ever_released",
    "privileged.task.manipulated_object.in_target",
    "privileged.task.manipulated_object.target_progress_available",
    "privileged.task.manipulated_object.target_distance_delta_m",
    "privileged.task.manipulated_object.target_progress_m",
    "privileged.task.stage.index",
    "privileged.task.stage.name",
    "privileged.task.target.distance_to_eef_m",
    "privileged.task.target.gripper_contact",
    "privileged.task.target.robot_contact",
    "privileged.task.target.is_nearest_entity",
    "privileged.task.target.distance_rank",
    "privileged.task.nearest_entity.distance_m",
    "privileged.contact.robot.count",
    "privileged.contact.gripper.count",
    "privileged.contact.force_available",
    "privileged.contact.max_normal_force_n",
    "privileged.joint.count",
)
_DIAGNOSTIC_TELEMETRY_PREFIXES = ("privileged.task.goal.predicate.",)
_DIAGNOSTIC_JOINT_PREFIX = "privileged.joint."
_DIAGNOSTIC_TRANSITION_FEATURES = (
    "command.realization.stalled",
    "privileged.task.success",
    "privileged.task.primary_relation_satisfied",
    "privileged.task.manipulated_object.gripper_contact",
    "privileged.task.manipulated_object.grasped",
    "privileged.task.manipulated_object.retained",
    "privileged.task.manipulated_object.released_now",
    "privileged.task.manipulated_object.in_target",
    "privileged.task.stage.name",
)
_DIAGNOSTIC_TELEMETRY_STRIDE = 5


def _resolver_path(store: CampaignStore) -> Path:
    return store.root / ".harness-private" / "artifact-resolver.json"


def _new_resolver(store: CampaignStore) -> dict[str, Any]:
    return {
        "schema_version": _RESOLVER_SCHEMA_VERSION,
        "manifest_sha256": store.manifest().sha256,
        "id_key_hex": secrets.token_hex(32),
        "entries": {},
        "aliases": {
            "episode_id": {},
            "logical_id": {},
            "segment_id": {},
        },
    }


def _load_resolver(store: CampaignStore) -> dict[str, Any]:
    path = _resolver_path(store)
    resolver = read_json(path) if path.exists() else _new_resolver(store)
    if resolver.get("schema_version") != _RESOLVER_SCHEMA_VERSION:
        raise ValueError("unsupported artifact resolver schema")
    if resolver.get("manifest_sha256") != store.manifest().sha256:
        raise ValueError("artifact resolver belongs to a different campaign manifest")
    key_hex = resolver.get("id_key_hex")
    if not isinstance(key_hex, str) or len(key_hex) != 64:
        raise ValueError("artifact resolver is missing its campaign-scoped ID key")
    try:
        bytes.fromhex(key_hex)
    except ValueError as exc:
        raise ValueError("artifact resolver ID key is malformed") from exc
    if not isinstance(resolver.get("entries"), dict):
        raise ValueError("artifact resolver entries are malformed")
    aliases = resolver.get("aliases")
    if not isinstance(aliases, dict):
        raise ValueError("artifact resolver aliases are malformed")
    for namespace in ("episode_id", "logical_id", "segment_id"):
        if not isinstance(aliases.get(namespace), dict):
            raise ValueError(
                f"artifact resolver alias namespace is malformed: {namespace}"
            )
    return resolver


def _persist_resolver(store: CampaignStore, resolver: dict[str, Any]) -> None:
    path = _resolver_path(store)
    atomic_write_json(path, resolver, overwrite=True)
    try:
        path.chmod(0o600)
    except OSError:
        # Some shared filesystems do not implement POSIX modes. The resolver is
        # still kept outside every Stage-agent output directory and is never
        # serialized into an agent payload.
        pass


def _keyed_digest(*, resolver: dict[str, Any], domain: bytes, digest: str) -> str:
    key = bytes.fromhex(str(resolver["id_key_hex"]))
    return hmac.new(key, domain + bytes.fromhex(digest), hashlib.sha256).hexdigest()


def _content_id(*, resolver: dict[str, Any], digest: str) -> str:
    return "artifact-" + _keyed_digest(
        resolver=resolver,
        domain=b"rpent-agent-content-id\0",
        digest=digest,
    )


def _agent_hash(*, resolver: dict[str, Any], digest: str) -> str:
    return "hmac-sha256:" + _keyed_digest(
        resolver=resolver,
        domain=b"rpent-agent-artifact-hash\0",
        digest=digest,
    )


def _artifact_type(value: Any, *, path: Path | None, role: str) -> str:
    if role in {
        "episode_record",
        "failure_segment",
        "candidate_gate_episode",
        "parent_gate_episode",
    }:
        return role
    if path is not None:
        suffix = path.suffix.lower()
        if suffix in {".mp4", ".webm", ".avi", ".mov", ".mkv"}:
            return "video"
        if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            return "image"
        if suffix in {".json", ".jsonl"}:
            return "structured_data"
        if suffix in {".txt", ".log", ".md"}:
            return "text"
        return "binary"
    if isinstance(value, str):
        return "text"
    return "structured_data"


def _existing_artifact_path(
    store: CampaignStore, row: dict[str, Any], value: Any
) -> Path | None:
    candidates = _lexical_artifact_paths(store, row, value)
    if not candidates:
        return None
    campaign_root = store.root.resolve()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_relative_to(campaign_root) and resolved.is_file():
            return resolved
    return None


def _lexical_artifact_paths(
    store: CampaignStore, row: dict[str, Any], value: Any
) -> tuple[Path, ...]:
    """Return normalized in-campaign locators without touching shared storage."""

    if not isinstance(value, str) or not value or "\n" in value or len(value) > 4096:
        return ()
    raw = Path(value)
    if (
        not raw.is_absolute()
        and "/" not in value
        and "\\" not in value
        and raw.suffix.lower() not in _ARTIFACT_SUFFIXES
    ):
        return ()
    candidates = [raw] if raw.is_absolute() else []
    candidates.extend(
        (
            store.root / raw,
            store.root / "episodes" / str(row["logical_id"]) / raw,
            store.root
            / "attempts"
            / str(row["logical_id"])
            / f"attempt-{int(row.get('attempt_index', 0)):03d}"
            / raw,
        )
    )
    campaign_root = Path(os.path.abspath(store.root))
    normalized: list[Path] = []
    for candidate in candidates:
        absolute = Path(os.path.abspath(candidate))
        if absolute.is_relative_to(campaign_root) and absolute not in normalized:
            normalized.append(absolute)
    return tuple(normalized)


def _source_digest(
    store: CampaignStore,
    row: dict[str, Any],
    value: Any,
    *,
    role: str,
    raw_key: str | None = None,
    frozen_digests_by_path: dict[str, str] | None = None,
) -> tuple[str, str, dict[str, Any]]:
    frozen_digest = None
    path = None
    if frozen_digests_by_path:
        for candidate in _lexical_artifact_paths(store, row, value):
            frozen_digest = frozen_digests_by_path.get(str(candidate))
            if frozen_digest is not None:
                path = candidate
                break
    if path is None:
        path = _existing_artifact_path(store, row, value)
    if path is not None:
        frozen_digest = frozen_digest or (frozen_digests_by_path or {}).get(
            str(path)
        )
        if frozen_digest is not None and not _SHA256_PATTERN.fullmatch(
            frozen_digest
        ):
            raise ValueError("episode artifact index contains a malformed digest")
        # Trajectory and visual builders hash every immutable artifact before
        # the EpisodeRecord is accepted.  Reuse that append-only provenance
        # here instead of re-reading multi-GB camera videos over shared storage.
        # resolve_agent_artifact() still re-hashes the particular image/video
        # that an Agent actually requests, so evidence consumption remains
        # fail closed without imposing an O(all videos) Stage1 startup cost.
        digest = frozen_digest or file_sha256(path)
        source = {
            "kind": "file",
            "path": str(path),
            "role": role,
            "raw_key": raw_key,
            "episode_id": row["episode_id"],
            "logical_id": row["logical_id"],
            "digest_authority": (
                "accepted_episode_record"
                if frozen_digest is not None
                else "stage_index_file_hash"
            ),
        }
    else:
        digest = canonical_sha256(value)
        source = {
            "kind": "inline",
            "value": value,
            "role": role,
            "raw_key": raw_key,
            "episode_id": row["episode_id"],
            "logical_id": row["logical_id"],
        }
    return digest, _artifact_type(value, path=path, role=role), source


def _frozen_artifact_digests_by_path(
    store: CampaignStore,
    row: dict[str, Any],
    artifact_index: dict[str, Any],
) -> dict[str, str]:
    """Recover rollout-time artifact hashes from one accepted EpisodeRecord."""

    frozen: dict[str, str] = {}

    def register(container: Any, *, paths_key: str) -> None:
        if not isinstance(container, dict):
            return
        paths = container.get(paths_key)
        digests = container.get("artifact_sha256")
        # Historical/development records may predate frozen artifact hashes;
        # retain the verified file-hash fallback for those rows.
        if digests is None or paths is None:
            return
        if not isinstance(paths, dict) or not isinstance(digests, dict):
            raise ValueError("episode artifact provenance is incomplete")
        if set(paths) != set(digests):
            raise ValueError("episode artifact path/hash keys do not match")
        for name, locator in paths.items():
            digest = digests[name]
            if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest):
                raise ValueError("episode artifact provenance has a malformed digest")
            paths = _lexical_artifact_paths(store, row, locator)
            if not paths:
                raise ValueError("accepted episode artifact locator is unsafe")
            for path in paths:
                key = str(path)
                existing = frozen.get(key)
                if existing is not None and existing != digest:
                    raise ValueError(
                        "accepted episode contains conflicting artifact hashes"
                    )
                frozen[key] = digest

    register(artifact_index.get("trajectory_index"), paths_key="artifact_paths")
    register(artifact_index.get("visual_evidence"), paths_key="artifacts")
    return frozen


def _fixed_summary(artifact_type: str, *, success: bool | None = None) -> str:
    if artifact_type in {
        "episode_record",
        "candidate_gate_episode",
        "parent_gate_episode",
    }:
        outcome = "successful" if success else "unsuccessful"
        prefix = {
            "episode_record": "rollout",
            "candidate_gate_episode": "candidate gate arm",
            "parent_gate_episode": "parent gate arm",
        }[artifact_type]
        return f"valid {outcome} {prefix} evidence"
    summaries = {
        "failure_segment": "failure-segment evidence",
        "video": "video evidence",
        "image": "image evidence",
        "structured_data": "structured evidence",
        "text": "text evidence",
        "binary": "binary evidence",
    }
    return summaries[artifact_type]


def _indexed_summary(
    artifact_type: str, *, raw_key: str, success: bool | None
) -> str:
    outcome = "successful" if success else "unsuccessful"
    key = raw_key.casefold()
    if "privileged_state_summary" in key and artifact_type == "structured_data":
        return f"bounded privileged-state timeline for {outcome} diagnostic evidence"
    if "overview_contact_sheet" in key:
        return f"synchronized three-camera {outcome} episode overview image"
    if "divergence_contact_sheet" in key:
        return f"synchronized three-camera {outcome} divergence-window image"
    if "divergence_clip" in key:
        return f"synchronized three-camera {outcome} divergence-window video"
    if "visual" in key and artifact_type == "structured_data":
        return f"step/frame/camera alignment for {outcome} visual evidence"
    if "video" in key and artifact_type == "video":
        camera = raw_key.rsplit(".", 1)[-1].replace("robot0_", "")
        return f"{outcome} episode camera video: {camera}"
    if artifact_type == "structured_data" and (
        key == "states" or key.endswith(".states")
    ):
        return (
            f"{outcome} full per-step command, realized motion, and privileged "
            "Critic state trace"
        )
    if artifact_type == "structured_data" and (
        key == "actions" or key.endswith(".actions")
    ):
        return f"{outcome} full per-step requested action trace"
    if artifact_type == "structured_data" and (
        key == "chunks" or key.endswith(".chunks")
    ):
        return f"{outcome} VLA chunk execution trace"
    return _fixed_summary(artifact_type)


def _nested_artifact_leaves(value: Any, *, prefix: str) -> list[tuple[str, Any]]:
    result: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, item in sorted(value.items(), key=lambda row: str(row[0])):
            result.extend(
                _nested_artifact_leaves(item, prefix=f"{prefix}.{str(key)}")
            )
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            result.extend(
                _nested_artifact_leaves(item, prefix=f"{prefix}.{index:03d}")
            )
    else:
        result.append((prefix, value))
    return result


def _register_artifact(
    *,
    resolver: dict[str, Any],
    digest: str,
    artifact_type: str,
    summary: str,
    source: dict[str, Any],
) -> dict[str, str]:
    content_id = _content_id(resolver=resolver, digest=digest)
    agent_hash = _agent_hash(resolver=resolver, digest=digest)
    entries = resolver["entries"]
    existing = entries.get(content_id)
    if existing is None:
        existing = {
            "content_sha256": digest,
            "agent_hash": agent_hash,
            "types": [],
            "sources": [],
        }
        entries[content_id] = existing
    if (
        existing.get("content_sha256") != digest
        or existing.get("agent_hash") != agent_hash
    ):
        raise ValueError(f"artifact content-ID collision: {content_id}")
    if artifact_type not in existing["types"]:
        existing["types"].append(artifact_type)
        existing["types"].sort()
    source_fingerprint = canonical_sha256(source)
    known = {canonical_sha256(item) for item in existing["sources"]}
    if source_fingerprint not in known:
        existing["sources"].append(source)
        existing["sources"].sort(key=canonical_sha256)
    return {
        "type": artifact_type,
        "summary": summary,
        "hash": agent_hash,
        "content_id": content_id,
    }


def _set_alias(
    resolver: dict[str, Any], *, namespace: str, raw: str, content_id: str
) -> None:
    aliases = resolver["aliases"][namespace]
    existing = aliases.get(raw)
    if existing is not None and existing != content_id:
        raise ValueError(f"artifact resolver alias changed: {namespace}:{raw}")
    aliases[raw] = content_id


def _episode_evidence_items(
    store: CampaignStore,
    row: dict[str, Any],
    *,
    episode_role: str,
) -> list[tuple[str, str, str, dict[str, Any]]]:
    """Return every immutable evidence object belonging to one valid episode."""

    items: list[tuple[str, str, str, dict[str, Any]]] = []
    digest, artifact_type, source = _source_digest(
        store,
        row,
        row,
        role=episode_role,
    )
    items.append(
        (
            digest,
            artifact_type,
            _fixed_summary(artifact_type, success=bool(row.get("success"))),
            source,
        )
    )

    raw_segments = row.get("failure_segments")
    segments = (
        [segment for segment in raw_segments if isinstance(segment, dict)]
        if isinstance(raw_segments, (list, tuple)) and raw_segments
        else []
    )
    legacy_segment = row.get("failure_segment")
    if not segments and isinstance(legacy_segment, dict):
        segments = [legacy_segment]
    for segment in segments:
        segment_digest, segment_type, segment_source = _source_digest(
            store,
            row,
            segment,
            role="failure_segment",
        )
        items.append(
            (
                segment_digest,
                segment_type,
                _fixed_summary(segment_type),
                segment_source,
            )
        )

    artifact_index = row.get("artifact_index", {})
    if isinstance(artifact_index, dict):
        frozen_digests = _frozen_artifact_digests_by_path(
            store, row, artifact_index
        )
        for raw_key, value in sorted(
            artifact_index.items(), key=lambda item: str(item[0])
        ):
            artifact_digest, indexed_type, indexed_source = _source_digest(
                store,
                row,
                value,
                role="indexed_artifact",
                raw_key=str(raw_key),
                frozen_digests_by_path=frozen_digests,
            )
            items.append(
                (
                    artifact_digest,
                    indexed_type,
                    _fixed_summary(indexed_type),
                    indexed_source,
                )
            )
            for nested_key, nested_value in _nested_artifact_leaves(
                value, prefix=str(raw_key)
            ):
                nested_path = next(
                    (
                        candidate
                        for candidate in _lexical_artifact_paths(
                            store, row, nested_value
                        )
                        if str(candidate) in frozen_digests
                    ),
                    None,
                )
                if nested_path is None:
                    nested_path = _existing_artifact_path(
                        store, row, nested_value
                    )
                if nested_path is None:
                    continue
                nested_digest, nested_type, nested_source = _source_digest(
                    store,
                    row,
                    nested_value,
                    role="indexed_artifact",
                    raw_key=nested_key,
                    frozen_digests_by_path=frozen_digests,
                )
                items.append(
                    (
                        nested_digest,
                        nested_type,
                        _indexed_summary(
                            nested_type,
                            raw_key=nested_key,
                            success=bool(row.get("success")),
                        ),
                        nested_source,
                    )
                )
    return items


def _register_episode_evidence(
    *,
    store: CampaignStore,
    resolver: dict[str, Any],
    descriptors: dict[tuple[str, str], dict[str, str]],
    row: dict[str, Any],
    episode_role: str,
    register_aliases: bool,
) -> None:
    for digest, artifact_type, summary, source in _episode_evidence_items(
        store,
        row,
        episode_role=episode_role,
    ):
        descriptor = _register_artifact(
            resolver=resolver,
            digest=digest,
            artifact_type=artifact_type,
            summary=summary,
            source=source,
        )
        descriptors[(descriptor["content_id"], descriptor["type"])] = descriptor
        if not register_aliases:
            continue
        if artifact_type == episode_role:
            _set_alias(
                resolver,
                namespace="episode_id",
                raw=str(row["episode_id"]),
                content_id=descriptor["content_id"],
            )
            _set_alias(
                resolver,
                namespace="logical_id",
                raw=str(row["logical_id"]),
                content_id=descriptor["content_id"],
            )
        elif artifact_type == "failure_segment":
            segment = source.get("value")
            if isinstance(segment, dict) and segment.get("segment_id"):
                _set_alias(
                    resolver,
                    namespace="segment_id",
                    raw=str(segment["segment_id"]),
                    content_id=descriptor["content_id"],
                )


def _flatten_diagnostic_scalars(
    value: Any, *, prefix: str = ""
) -> dict[str, bool | int | float | str | None]:
    if isinstance(value, dict):
        flattened: dict[str, bool | int | float | str | None] = {}
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_diagnostic_scalars(item, prefix=name))
        return flattened
    if prefix and (value is None or isinstance(value, (bool, int, float, str))):
        return {prefix: value}
    return {}


def _diagnostic_telemetry_payload(
    store: CampaignStore,
    row: dict[str, Any],
    *,
    episode_alias: str,
) -> dict[str, Any] | None:
    """Build a compact, seed-blind Critic trace from immutable rollout state."""

    artifact_index = row.get("artifact_index")
    if not isinstance(artifact_index, dict):
        return None
    states_path = _existing_artifact_path(store, row, artifact_index.get("states"))
    if states_path is None:
        return None

    parsed_rows: list[dict[str, Any]] = []
    with states_path.open(encoding="utf-8") as stream:
        for ordinal, line in enumerate(stream):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid state artifact at {states_path.name}:{ordinal + 1}"
                ) from exc
            if not isinstance(payload, dict):
                continue
            state = payload.get("state")
            if not isinstance(state, dict):
                continue
            flattened = _flatten_diagnostic_scalars(state)
            features = {}
            for name, value in flattened.items():
                if name in _DIAGNOSTIC_TELEMETRY_FEATURES or name.startswith(
                    _DIAGNOSTIC_TELEMETRY_PREFIXES
                ):
                    features[name] = value
                    continue
                if name.startswith(_DIAGNOSTIC_JOINT_PREFIX):
                    joint_name = name[len(_DIAGNOSTIC_JOINT_PREFIX) :].split(
                        ".", 1
                    )[0]
                    if not joint_name.startswith(("robot0_", "gripper0_")):
                        features[name] = value
            if not features:
                continue
            raw_step = payload.get("step_index", payload.get("step", ordinal))
            step_index = (
                int(raw_step)
                if isinstance(raw_step, int) and not isinstance(raw_step, bool)
                else ordinal
            )
            parsed_rows.append(
                {
                    "step_index": step_index,
                    "ordinal": ordinal,
                    "features": features,
                }
            )
    if not parsed_rows:
        return None

    selected: list[dict[str, Any]] = []
    previous_transition: dict[str, Any] = {}
    for index, telemetry_row in enumerate(parsed_rows):
        features = telemetry_row["features"]
        current_transition = {
            name: value
            for name, value in features.items()
            if name in _DIAGNOSTIC_TRANSITION_FEATURES
            or isinstance(value, bool)
        }
        transitioned = index > 0 and any(
            previous_transition.get(name) != value
            for name, value in current_transition.items()
        )
        keep = (
            index == 0
            or index == len(parsed_rows) - 1
            or telemetry_row["ordinal"] % _DIAGNOSTIC_TELEMETRY_STRIDE == 0
            or transitioned
        )
        if keep:
            selected.append(
                {
                    "step_index": telemetry_row["step_index"],
                    "features": features,
                }
            )
        previous_transition.update(current_transition)

    feature_names = sorted(
        {
            name
            for telemetry_row in parsed_rows
            for name in telemetry_row["features"]
        }
    )
    sampled_step_indices = [row["step_index"] for row in selected]
    series = {
        name: [row["features"].get(name) for row in selected]
        for name in feature_names
    }
    return {
        "schema_version": 1,
        "telemetry_kind": "diagnostic_episode_trace",
        "episode": episode_alias,
        "outcome": "success" if row.get("success") is True else "failure",
        "sampling": {
            "stride_steps": _DIAGNOSTIC_TELEMETRY_STRIDE,
            "first_and_last_rows_retained": True,
            "boolean_and_stage_transitions_retained": True,
            "source_row_count": len(parsed_rows),
            "sampled_row_count": len(selected),
        },
        "sampled_step_indices": sampled_step_indices,
        "series_alignment": (
            "Every series array is positionally aligned with sampled_step_indices; "
            "null means the feature was unavailable at that sampled step."
        ),
        "series": series,
    }


def _register_diagnostic_telemetry(
    *,
    store: CampaignStore,
    resolver: dict[str, Any],
    descriptors: dict[tuple[str, str], dict[str, str]],
    row: dict[str, Any],
) -> dict[str, Any] | None:
    episode_alias = resolver["aliases"]["episode_id"].get(str(row["episode_id"]))
    if not isinstance(episode_alias, str):
        raise ValueError("diagnostic telemetry has no opaque episode alias")
    payload = _diagnostic_telemetry_payload(
        store,
        row,
        episode_alias=episode_alias,
    )
    if payload is None:
        return None
    outcome = str(payload["outcome"])
    summary = (
        f"{outcome} seed-blind sampled requested-action, realized-EEF, gripper, "
        "contact, grasp-retention, and task-progress telemetry"
    )
    source = {
        "kind": "inline",
        "value": payload,
        "role": "diagnostic_telemetry",
        "raw_key": "diagnostic_telemetry",
        "episode_id": row["episode_id"],
        "logical_id": row["logical_id"],
    }
    descriptor = _register_artifact(
        resolver=resolver,
        digest=canonical_sha256(payload),
        artifact_type="structured_data",
        summary=summary,
        source=source,
    )
    descriptors[(descriptor["content_id"], descriptor["type"])] = descriptor
    return {
        "episode": episode_alias,
        "outcome": outcome,
        "role": (
            "success_comparator_telemetry"
            if outcome == "success"
            else "failure_episode_telemetry"
        ),
        "content_id": descriptor["content_id"],
        "summary": summary,
        "source_row_count": payload["sampling"]["source_row_count"],
        "sampled_row_count": payload["sampling"]["sampled_row_count"],
        "feature_count": len(payload["series"]),
        "signal_groups": [
            "requested_action",
            "realized_eef_motion",
            "gripper",
            "contact",
            "grasp_retention",
            "task_goal_progress",
            "articulation_progress",
        ],
    }


def _completed_gate_episode_rows(
    store: CampaignStore,
) -> list[tuple[str, dict[str, Any], str]]:
    """Read only fully reproduced same-seed gates with immutable provenance."""

    decisions: dict[str, dict[str, Any]] = {}
    for row in store.gates.records():
        if row.get("kind") != "same_seed":
            continue
        candidate_sha256 = row.get("candidate_sha256")
        if not isinstance(candidate_sha256, str):
            raise ValueError("same-seed gate decision has no candidate digest")
        if candidate_sha256 in decisions:
            raise ValueError("candidate has multiple same-seed gate decisions")
        decisions[candidate_sha256] = row
    rows: list[tuple[str, dict[str, Any], str]] = []
    for candidate_row in store.candidate_ledger.records():
        candidate_sha256 = str(candidate_row["candidate_sha256"])
        decision_row = decisions.get(candidate_sha256)
        if decision_row is None:
            continue
        bundle_path = store.root / "candidates" / candidate_sha256 / "bundle.json"
        bundle = read_json(bundle_path)
        if canonical_sha256(bundle) != candidate_sha256:
            raise ValueError(
                "candidate artifact digest mismatch while indexing gate evidence"
            )
        parent_sha256 = bundle.get("parent_sha256")
        gate_root = store.root / "candidates" / candidate_sha256 / "gates" / "same_seed"
        plan_path = gate_root / "plan.json"
        if not plan_path.is_file():
            raise ValueError("decided same-seed gate is missing its immutable plan")
        plan = read_json(plan_path)
        manifest = store.manifest()
        bindings = {
            "kind": "same_seed",
            "manifest_sha256": manifest.sha256,
            "generation": manifest.generation,
            "candidate_sha256": candidate_sha256,
            "parent_sha256": parent_sha256,
        }
        mismatched = [key for key, value in bindings.items() if plan.get(key) != value]
        if mismatched:
            raise ValueError(f"decided gate plan binding changed: {sorted(mismatched)}")
        pairs = plan.get("pairs")
        if not isinstance(pairs, list) or not pairs:
            raise ValueError("decided same-seed gate has no registered pairs")
        expected: dict[str, dict[str, Any]] = {}
        expected_seeds: list[int] = []
        for pair_index, pair in enumerate(pairs):
            if not isinstance(pair, dict) or pair.get("pair_index") != pair_index:
                raise ValueError("decided gate pair indexes are not contiguous")
            seed = pair.get("seed")
            policy_rng = pair.get("policy_rng")
            if (
                not isinstance(seed, int)
                or manifest.policy_rng_by_seed.get(str(seed)) != policy_rng
            ):
                raise ValueError("decided gate pair violates preregistration")
            expected_seeds.append(seed)
            arms = pair.get("logical_ids")
            if not isinstance(arms, dict) or set(arms) != {"parent", "candidate"}:
                raise ValueError("decided gate pair has malformed logical arms")
            for arm, logical_id in arms.items():
                if not isinstance(logical_id, str) or logical_id in expected:
                    raise ValueError("decided gate logical arms are not unique")
                expected[logical_id] = {
                    "arm": arm,
                    "seed": seed,
                    "policy_rng": policy_rng,
                    "bundle_sha256": (
                        candidate_sha256 if arm == "candidate" else parent_sha256
                    ),
                }

        valid_ledger = AppendOnlyLedger(
            gate_root / "ledgers" / "valid.jsonl", key="logical_id"
        )
        valid_rows = valid_ledger.records()
        by_logical = {str(row.get("logical_id")): row for row in valid_rows}
        early_impossible = (
            decision_row.get("kind") == "same_seed"
            and decision_row.get("passed") is False
            and decision_row.get("conclusive") is True
            and str(decision_row.get("rationale", "")).startswith(
                "same-seed gate rejected early as mathematically impossible:"
            )
        )
        if len(by_logical) != len(valid_rows):
            raise ValueError("decided gate valid arms differ from immutable plan")
        if early_impossible:
            if not set(by_logical).issubset(expected):
                raise ValueError("decided gate valid arms differ from immutable plan")
            parent_ids = {
                logical_id
                for logical_id, row in expected.items()
                if row["arm"] == "parent"
            }
            if not parent_ids.issubset(by_logical):
                raise ValueError("early gate decision is missing a parent arm")
        elif set(by_logical) != set(expected):
            raise ValueError("decided gate valid arms differ from immutable plan")
        attempts = AppendOnlyLedger(
            gate_root / "ledgers" / "attempts.jsonl", key="attempt_id"
        ).records()
        attempts_by_id = {str(row.get("attempt_id")): row for row in attempts}
        candidate_records: list[EpisodeRecord] = []
        parent_records: list[EpisodeRecord] = []
        staged: list[tuple[str, dict[str, Any], str]] = []
        logical_ids = expected if not early_impossible else {
            logical_id: expected[logical_id] for logical_id in sorted(by_logical)
        }
        for logical_id, expected_row in logical_ids.items():
            row = by_logical[logical_id]
            record = EpisodeRecord.from_dict(row)
            if record.status != "valid":
                raise ValueError("decided gate valid ledger contains an invalid arm")
            required = {
                "seed": expected_row["seed"],
                "policy_rng": expected_row["policy_rng"],
                "bundle_sha256": expected_row["bundle_sha256"],
                "generation": manifest.generation,
            }
            changed = [
                key for key, value in required.items() if getattr(record, key) != value
            ]
            if changed:
                raise ValueError(
                    f"decided gate valid arm changed frozen fields: {sorted(changed)}"
                )
            canonical = gate_root / "episodes" / logical_id / "record.json"
            if not canonical.is_file() or canonical_sha256(
                read_json(canonical)
            ) != canonical_sha256(row):
                raise ValueError("decided gate canonical record is missing or changed")
            attempt = attempts_by_id.get(record.attempt_id)
            if attempt is None:
                raise ValueError("decided gate valid arm has no accepted attempt")
            attempt_payload = dict(attempt)
            attempt_payload.pop("attempt_id", None)
            if canonical_sha256(attempt_payload) != canonical_sha256(row):
                raise ValueError("decided gate accepted attempt differs from valid arm")
            if expected_row["arm"] == "candidate":
                candidate_records.append(record)
                role = "candidate_gate_episode"
            else:
                parent_records.append(record)
                role = "parent_gate_episode"
            staged.append((candidate_sha256, row, role))

        if early_impossible:
            pair_count = len(expected_seeds)
            effective_rate = effective_same_seed_gate_pass_rate(
                store,
                candidate_sha256=candidate_sha256,
                plan=plan,
            )
            required_successes = math.ceil(pair_count * effective_rate)
            candidate_successes = sum(
                bool(record.success) for record in candidate_records
            )
            parent_successes = sum(bool(record.success) for record in parent_records)
            remaining = pair_count - len(candidate_records)
            upper_bound = candidate_successes + remaining
            if upper_bound >= required_successes and upper_bound > parent_successes:
                raise ValueError("early gate rejection is not mathematically impossible")
            candidate_by_seed = {record.seed: record for record in candidate_records}
            parent_by_seed = {record.seed: record for record in parent_records}
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
            observed_ids = tuple(sorted(record.logical_id for record in candidate_records))
            expected_candidate_ids = {
                logical_id
                for logical_id, row in expected.items()
                if row["arm"] == "candidate"
            }
            missing_ids = tuple(sorted(expected_candidate_ids - set(observed_ids)))
            rationale = (
                "same-seed gate rejected early as mathematically impossible: "
                f"observed candidate successes {candidate_successes}/{pair_count}, "
                f"remaining candidate arms {remaining}, upper bound "
                f"{upper_bound}/{pair_count}, required "
                f"{required_successes}/{pair_count}; parent successes observed "
                f"{parent_successes}/{pair_count}; valid candidate arms "
                f"{len(observed_ids)}, missing arms {len(missing_ids)}"
            )
            decision_payload = {
                "kind": "same_seed",
                "candidate_sha256": candidate_sha256,
                "parent_sha256": parent_sha256,
                "plan_sha256": canonical_sha256(plan),
                "candidate_successes": candidate_successes,
                "parent_successes": parent_successes,
                "paired_count": pair_count,
                "candidate_wins": candidate_wins,
                "parent_wins": parent_wins,
                "observed_candidate_ids": observed_ids,
                "missing_candidate_ids": missing_ids,
                "effective_same_seed_pass_rate": effective_rate,
            }
            reproduced = GateDecision(
                decision_id=f"gate-{canonical_sha256(decision_payload)[:20]}",
                candidate_sha256=candidate_sha256,
                parent_sha256=parent_sha256,
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
                candidate_safety_events=sum(
                    len(record.safety_events) for record in candidate_records
                ),
                parent_safety_events=sum(
                    len(record.safety_events) for record in parent_records
                ),
                rationale=rationale,
            )
        else:
            reproduced = evaluate_paired_gate(
                kind="same_seed",
                candidate_sha256=candidate_sha256,
                parent_sha256=parent_sha256,
                candidate_records=candidate_records,
                parent_records=parent_records,
                expected_seeds=tuple(expected_seeds),
                same_seed_pass_rate=effective_same_seed_gate_pass_rate(
                    store,
                    candidate_sha256=candidate_sha256,
                    plan=plan,
                ),
            )
        if reproduced != GateDecision(**decision_row):
            raise ValueError(
                "recorded gate decision differs from immutable arm evidence"
            )
        rows.extend(staged)
    return rows


def _frozen_same_seed_pass_rate(plan: dict[str, Any]) -> float:
    """Recover the exact threshold used by the immutable gate decision."""

    version = int(plan.get("schema_version", 1))
    if version < 2:
        return 1.0
    value = plan.get("same_seed_pass_rate")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0 < float(value) <= 1
    ):
        raise ValueError("decided gate plan has an invalid same-seed pass rate")
    return float(value)


def _visual_relation_role(raw_key: str) -> str | None:
    key = raw_key.casefold()
    if "visual_evidence.artifacts.privileged_state_summary" in key:
        return "privileged_state_summary"
    if "visual_evidence.artifacts.overview_contact_sheet" in key:
        return "episode_overview"
    if "visual_evidence.artifacts.divergence_contact_sheet" in key:
        return "divergence_window"
    if "visual_evidence.artifacts.event_contact_sheet" in key:
        return "event_window"
    if "visual_evidence.artifacts.divergence_clip" in key:
        return "divergence_clip"
    if "visual_evidence.artifacts.manifest" in key:
        return "frame_alignment"
    return None


def _agent_visual_relationships(
    *, rows: list[dict[str, Any]], resolver: dict[str, Any]
) -> list[dict[str, Any]]:
    """Expose seed-blind episode/segment-to-visual ownership edges."""

    aliases = resolver["aliases"]
    visual_by_episode: dict[str, dict[tuple[str, str], dict[str, str]]] = {}
    for content_id, entry in resolver["entries"].items():
        for source in entry.get("sources", ()):
            raw_key = source.get("raw_key")
            episode_id = source.get("episode_id")
            if not isinstance(raw_key, str) or not isinstance(episode_id, str):
                continue
            relation_role = _visual_relation_role(raw_key)
            if relation_role is None:
                continue
            artifact_type = (
                "video"
                if "video" in entry.get("types", ())
                else "image"
                if "image" in entry.get("types", ())
                else "structured_data"
            )
            visual_by_episode.setdefault(episode_id, {})[
                (content_id, relation_role)
            ] = {
                "content_id": content_id,
                "type": artifact_type,
                "role": relation_role,
            }

    relationships = []
    for row in rows:
        raw_episode_id = str(row["episode_id"])
        visual = sorted(
            visual_by_episode.get(raw_episode_id, {}).values(),
            key=lambda item: (item["role"], item["content_id"]),
        )
        if not visual:
            continue
        episode_alias = aliases["episode_id"].get(raw_episode_id)
        if episode_alias is None:
            raise ValueError("visual relationship has no opaque episode alias")
        segments = []
        record = EpisodeRecord.from_dict(row)
        for segment in record.all_failure_segments:
            segment_alias = aliases["segment_id"].get(segment.segment_id)
            if segment_alias is None:
                raise ValueError("visual relationship has no opaque segment alias")
            segments.append(
                {
                    "segment": segment_alias,
                    "start_step": segment.start_step,
                    "earliest_divergence_step": segment.earliest_divergence_step,
                    "end_step": segment.end_step,
                }
            )
        relationships.append(
            {
                "episode": episode_alias,
                "outcome": "success" if row.get("success") is True else "failure",
                "segments": segments,
                "visual_evidence": visual,
            }
        )
    return sorted(relationships, key=lambda item: item["episode"])


def _agent_artifact_context(
    store: CampaignStore,
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    """Build the prompt-safe index and its private, recoverable resolver.

    Agent-visible descriptors deliberately contain only four fixed fields.
    Raw paths, filenames, logical IDs, seed-bearing values, policy RNG and
    schedules remain exclusively in ``.harness-private/artifact-resolver.json``.
    """

    path = _resolver_path(store)
    path.parent.mkdir(parents=True, exist_ok=True)
    with directory_lock(path.with_name(".artifact-resolver.lock")):
        resolver = _load_resolver(store)
        descriptors: dict[tuple[str, str], dict[str, str]] = {}
        for row in store.episodes.records():
            # CampaignStore never accepts infra-invalid rows into this ledger,
            # but keep the prompt boundary fail-safe if a legacy ledger exists.
            if row.get("status") != "valid":
                continue
            _register_episode_evidence(
                store=store,
                resolver=resolver,
                descriptors=descriptors,
                row=row,
                episode_role="episode_record",
                register_aliases=True,
            )
        for _, row, role in _completed_gate_episode_rows(store):
            _register_episode_evidence(
                store=store,
                resolver=resolver,
                descriptors=descriptors,
                row=row,
                episode_role=role,
                register_aliases=False,
            )
        rollout_rows = [
            row for row in store.episodes.records() if row.get("status") == "valid"
        ]
        diagnostic_telemetry = [
            telemetry
            for row in rollout_rows
            if (
                telemetry := _register_diagnostic_telemetry(
                    store=store,
                    resolver=resolver,
                    descriptors=descriptors,
                    row=row,
                )
            )
            is not None
        ]
        _persist_resolver(store, resolver)
        index = {
            "artifacts": sorted(
                descriptors.values(),
                key=lambda item: (item["content_id"], item["type"]),
            ),
            "relationships": _agent_visual_relationships(
                rows=rollout_rows,
                resolver=resolver,
            ),
            "diagnostic_telemetry": sorted(
                diagnostic_telemetry,
                key=lambda item: (item["outcome"], item["episode"]),
            ),
        }
        aliases = {
            namespace: dict(values) for namespace, values in resolver["aliases"].items()
        }
        return index, aliases


def _agent_artifact_index(store: CampaignStore) -> dict[str, Any]:
    return _agent_artifact_context(store)[0]


def _scalar_feature_names(value: Any, *, prefix: str = "") -> set[str]:
    if isinstance(value, dict):
        names: set[str] = set()
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            names.update(_scalar_feature_names(item, prefix=name))
        return names
    if value is None or isinstance(value, (bool, int, float, str)):
        return {prefix} if prefix and not _SENSITIVE_TEXT.search(prefix) else set()
    return set()


def _observed_critic_features(
    store: CampaignStore, *, require_command_rows: bool = False
) -> tuple[str, ...]:
    """Derive feature names Stage2 may bind from live state artifacts.

    Gen0 pure-policy LIBERO traces contain a full privileged reset row but do not
    configure the online Critic, so subsequent rows contain only robot and
    command telemetry.  A proposal for a new candidate must be grounded in the
    latter rows; keeping the default union preserves the legacy diagnostic
    catalog used by existing campaigns and tests.
    """

    features: set[str] = set()
    rows = list(store.episodes.records())
    rows.extend(row for _, row, _ in _completed_gate_episode_rows(store))
    for row in rows:
        if row.get("status") != "valid":
            continue
        index = row.get("artifact_index")
        if not isinstance(index, dict):
            continue
        path = _existing_artifact_path(store, row, index.get("states"))
        if path is None:
            continue
        rows_payload: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid state artifact at {path.name}:{line_number}"
                    ) from exc
                state = payload.get("state") if isinstance(payload, dict) else None
                if isinstance(state, dict):
                    rows_payload.append(state)
        command_rows = [
            state
            for state in rows_payload
            if state.get("command.available") is True
        ]
        selected_rows = command_rows if require_command_rows and command_rows else rows_payload
        for state in selected_rows:
            features.update(_scalar_feature_names(state))
    return tuple(sorted(features))


def _candidate_feature_contract(
    *,
    candidate: CandidateBundle,
    parent_bundle: CandidateBundle | None,
    trajectories: tuple[tuple[EpisodeRecord, Path], ...],
) -> dict[str, Any]:
    """Check that every candidate Critic rule is evaluable on replay states.

    A feature can exist somewhere in a trajectory while still being absent from
    every action decision state (the Gen0 LIBERO reset/action split is one such
    case).  Shadow replay cannot safely forward-fill that value, so require the
    rule feature and all activation predicates to co-occur in at least one state
    row for every trajectory used by the replay.
    """

    parent_ids = (
        {rule.rule_id for rule in parent_bundle.critic_rules}
        if parent_bundle
        else set()
    )
    delta_rules = tuple(
        rule for rule in candidate.critic_rules if rule.rule_id not in parent_ids
    )
    required_by_rule = {
        rule.rule_id: {
            rule.feature,
            *(condition.feature for condition in rule.activation_conditions),
        }
        for rule in delta_rules
    }
    trajectory_reports: list[dict[str, Any]] = []
    all_features: set[str] = set()
    for record, path in trajectories:
        row_features: list[set[str]] = []
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid state artifact at {path.name}:{line_number}"
                    ) from exc
                state = payload.get("state") if isinstance(payload, dict) else None
                if not isinstance(state, dict):
                    continue
                names = _scalar_feature_names(state)
                row_features.append(names)
                all_features.update(names)
        unavailable_rules: dict[str, list[str]] = {}
        for rule_id, required in required_by_rule.items():
            # Shadow replay may skip a reset-only prefix until the first state
            # where a rule is evaluable, but TemporalCritic then evaluates every
            # subsequent row.  Require that suffix to remain complete; merely
            # seeing each field once is insufficient and would let a sparse
            # privileged feature crash replay midway through a trajectory.
            first_evaluable = next(
                (
                    index
                    for index, names in enumerate(row_features)
                    if required <= names
                ),
                None,
            )
            missing_after_first = (
                set().union(
                    *(required - names for names in row_features[first_evaluable:])
                )
                if first_evaluable is not None
                else set(required)
            )
            if first_evaluable is None or missing_after_first:
                unavailable_rules[rule_id] = sorted(missing_after_first)
        # Keep the report seed-blind while distinguishing a typo from a sparse
        # feature that disappears after replay has started.
        trajectory_reports.append(
            {
                "trajectory_sha256": file_sha256(path),
                "required_rule_count": len(required_by_rule),
                "unavailable_rule_ids": sorted(unavailable_rules),
                "unavailable_feature_names": sorted(
                    {name for values in unavailable_rules.values() for name in values}
                ),
            }
        )
    unsupported = sorted(
        {
            name
            for required in required_by_rule.values()
            for name in required
            if name not in all_features
        }
    )
    unavailable = any(row["unavailable_rule_ids"] for row in trajectory_reports)
    return {
        "schema_version": 1,
        "delta_rule_ids": sorted(required_by_rule),
        "trajectory_count": len(trajectory_reports),
        "feature_schema_sha256": canonical_sha256(sorted(all_features)),
        "unsupported_feature_names": unsupported,
        "trajectory_reports": trajectory_reports,
        "eligible": bool(trajectories) and not unsupported and not unavailable,
    }


def _active_stage1_context(store: CampaignStore) -> dict[str, Any]:
    """Load the Diagnose context bound to the active cluster target.

    Formal campaigns place each Stage1 invocation under the immutable target
    digest.  The former flat path is retained only as a compatibility fallback
    for already-written development artifacts.
    """

    target_sha256 = store.state().get("active_cluster_target_sha256")
    candidates: list[Path] = []
    if isinstance(target_sha256, str) and target_sha256:
        candidates.append(
            store.root
            / "agents"
            / "diagnosis"
            / target_sha256
            / "stage1-diagnosis"
        )
    candidates.append(store.root / "agents" / "diagnosis" / "stage1-diagnosis")
    for stage_root in candidates:
        context_path = stage_root / "context.json"
        if not context_path.is_file():
            continue
        context = read_json(context_path)
        output_path = stage_root / "output.json"
        if output_path.is_file():
            output = read_json(output_path)
            expected = context.get("output_sha256")
            if expected is not None and expected != canonical_sha256(output):
                raise ValueError("Stage1 context/output digest mismatch")
            if isinstance(target_sha256, str) and target_sha256:
                accepted_path = (
                    store.root
                    / "analysis"
                    / "accepted_diagnoses"
                    / f"{target_sha256}.json"
                )
                if accepted_path.is_file() and canonical_sha256(
                    read_json(accepted_path)
                ) != canonical_sha256(output):
                    raise ValueError("Stage1 output differs from accepted diagnosis")
        return context
    return {}


def _latest_stage2_context(store: CampaignStore) -> dict[str, Any]:
    """Continue the latest committed Stage2 thread, falling back to Stage1."""

    candidate_rows = store.candidate_ledger.records()
    if candidate_rows:
        candidate_row = candidate_rows[-1]
        candidate_sha256 = str(candidate_row["candidate_sha256"])
        bundle_path = store.root / "candidates" / candidate_sha256 / "bundle.json"
        bundle_payload = read_json(bundle_path)
        if canonical_sha256(bundle_payload) != candidate_sha256:
            raise ValueError("latest Stage2 candidate bundle digest mismatch")
        bundle = CandidateBundle.from_dict(bundle_payload)

        # Contract-rejected proposals are intentionally absent from the
        # candidate ledger, so ledger length cannot identify the Stage2 agent
        # directory. Bind the context to the registered bundle projection.
        stage_roots = sorted(
            (
                path
                for path in (store.root / "agents").glob(
                    "candidate-*/stage2-proposal"
                )
                if path.is_dir()
            ),
            key=lambda path: path.parent.name,
            reverse=True,
        )
        stage_root: Path | None = None
        for candidate_stage_root in stage_roots:
            candidate_output_path = candidate_stage_root / "output.json"
            candidate_context_path = candidate_stage_root / "context.json"
            if not candidate_output_path.is_file() or not candidate_context_path.is_file():
                continue
            candidate_output = read_json(candidate_output_path)
            output_projection = {
                key: candidate_output.get(key)
                for key in (
                    "candidate_id",
                    "mechanism_change",
                    "validation_plan",
                    "critic_rules",
                    "recovery_rules",
                    "tool_plugin",
                )
            }
            bundle_projection = {
                "candidate_id": bundle.candidate_id,
                "mechanism_change": bundle.mechanism_change,
                "validation_plan": bundle.validation_plan,
                "critic_rules": [rule.as_dict() for rule in bundle.critic_rules],
                "recovery_rules": [rule.as_dict() for rule in bundle.recovery_rules],
                "tool_plugin": bundle.tool_plugin,
            }
            if canonical_sha256(output_projection) == canonical_sha256(
                bundle_projection
            ):
                stage_root = candidate_stage_root
                break
        if stage_root is None:
            raise ValueError("latest Stage2 output has no matching candidate bundle")

        path = stage_root / "context.json"
        context = read_json(path)
        output_path = stage_root / "output.json"
        output = read_json(output_path)
        if context.get("output_sha256") != canonical_sha256(output):
            raise ValueError("Stage2 context/output digest mismatch")
        output_projection = {
            key: output.get(key)
            for key in (
                "candidate_id",
                "mechanism_change",
                "validation_plan",
                "critic_rules",
                "recovery_rules",
                "tool_plugin",
            )
        }
        bundle_projection = {
            "candidate_id": bundle.candidate_id,
            "mechanism_change": bundle.mechanism_change,
            "validation_plan": bundle.validation_plan,
            "critic_rules": [rule.as_dict() for rule in bundle.critic_rules],
            "recovery_rules": [rule.as_dict() for rule in bundle.recovery_rules],
            "tool_plugin": bundle.tool_plugin,
        }
        if canonical_sha256(output_projection) != canonical_sha256(
            bundle_projection
        ):
            raise ValueError("Stage2 output does not match candidate ledger bundle")
        attempt_name = context.get("successful_attempt")
        if not isinstance(attempt_name, str) or not attempt_name.startswith(
            "attempt-"
        ):
            raise ValueError("Stage2 context has no successful attempt binding")
        attempt_root = stage_root / attempt_name
        invocation_path = attempt_root / "invocation.json"
        attempt_output_path = attempt_root / "output.json"
        if not invocation_path.is_file() or not attempt_output_path.is_file():
            raise ValueError("Stage2 successful attempt artifacts are incomplete")
        invocation = read_json(invocation_path)
        if canonical_sha256(read_json(attempt_output_path)) != canonical_sha256(
            output
        ):
            raise ValueError("Stage2 attempt output differs from committed output")
        if invocation.get("session_id") != context.get(
            "session_id"
        ) or invocation.get("provider_thread_id") != context.get(
            "provider_thread_id"
        ):
            raise ValueError("Stage2 context is not bound to its provider attempt")
        diagnosis_context = _active_stage1_context(store)
        if diagnosis_context:
            if context.get("session_id") != diagnosis_context.get("session_id"):
                raise ValueError(
                    "Stage2 context belongs to another logical session"
                )
        return context
    return _active_stage1_context(store)


def _gate_descriptors_for_candidate(
    store: CampaignStore,
    *,
    candidate_sha256: str,
    artifact_index: dict[str, Any],
) -> list[dict[str, str]]:
    resolver = _load_resolver(store)
    visible = {
        (str(row["content_id"]), str(row["type"])): row
        for row in artifact_index.get("artifacts", ())
        if isinstance(row, dict)
    }
    selected: dict[tuple[str, str], dict[str, str]] = {}
    for sha256, row, role in _completed_gate_episode_rows(store):
        if sha256 != candidate_sha256:
            continue
        for digest, artifact_type, _, _ in _episode_evidence_items(
            store,
            row,
            episode_role=role,
        ):
            key = (_content_id(resolver=resolver, digest=digest), artifact_type)
            descriptor = visible.get(key)
            if descriptor is None:
                raise ValueError(
                    "completed gate evidence is absent from artifact index"
                )
            selected[key] = descriptor
    grouped: dict[str, list[dict[str, str]]] = {}
    for descriptor in selected.values():
        grouped.setdefault(str(descriptor["type"]), []).append(descriptor)
    limits = {
        "candidate_gate_episode": 8,
        "parent_gate_episode": 8,
        "failure_segment": 8,
        "structured_data": 32,
        "image": 8,
        "video": 8,
        "text": 8,
    }
    bounded: list[dict[str, str]] = []
    for artifact_type, rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: str(row["content_id"]))
        bounded.extend(ordered[: limits.get(artifact_type, 4)])
    return sorted(bounded, key=lambda row: (row["content_id"], row["type"]))


def _bounded_gate_descriptors(
    evidence: list[dict[str, str]], *, per_descriptor_kind: int = 2
) -> list[dict[str, str]]:
    """Keep deterministic, diverse gate evidence without duplicating a full gate."""

    if per_descriptor_kind < 1:
        raise ValueError("per_descriptor_kind must be positive")
    counts: dict[str, int] = {}
    summaries: dict[str, set[str]] = {}
    selected: list[dict[str, str]] = []
    ordered = sorted(
        evidence,
        key=lambda item: (
            str(item.get("type", "")),
            str(item.get("summary", "")),
            str(item.get("content_id", "")),
        ),
    )
    for row in ordered:
        kind = str(row.get("type", ""))
        summary = str(row.get("summary", ""))
        if counts.get(kind, 0) >= per_descriptor_kind:
            continue
        seen = summaries.setdefault(kind, set())
        if summary in seen:
            continue
        seen.add(summary)
        counts[kind] = counts.get(kind, 0) + 1
        selected.append(row)
    # A descriptor type may have only one summary. Fill its remaining bounded
    # slots deterministically so paired artifacts remain inspectable without
    # granting every unique summary its own prompt budget.
    selected_ids = {str(row.get("content_id", "")) for row in selected}
    for row in ordered:
        kind = str(row.get("type", ""))
        if counts.get(kind, 0) >= per_descriptor_kind:
            continue
        content_id = str(row.get("content_id", ""))
        if content_id in selected_ids:
            continue
        selected_ids.add(content_id)
        counts[kind] = counts.get(kind, 0) + 1
        selected.append(row)
    return sorted(selected, key=lambda row: (row["content_id"], row["type"]))


def _development_evidence_summary(store: CampaignStore) -> list[dict[str, Any]]:
    """Return seed/path-free development calibration for Stage2 refinement."""

    development_root = store.root / "analysis" / "development-evidence"
    evidence: list[dict[str, Any]] = []
    if not development_root.is_dir():
        return evidence
    for evidence_path in sorted(development_root.glob("*.json")):
        try:
            payload = read_json(evidence_path)
        except (OSError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        result = payload.get("result")
        if not isinstance(result, dict):
            continue
        close = result.get("close")
        if not isinstance(close, dict):
            close = {}
        evidence.append(
            {
                "evidence_kind": "development_privileged_tool_smoke",
                "artifact_sha256": file_sha256(evidence_path),
                "task_id": payload.get("task"),
                "grasp_offset_xyz": payload.get("grasp_offset_xyz"),
                "manipulated_object": payload.get("manipulated_object"),
                "target": payload.get("target"),
                "status": result.get("status"),
                "official_success": bool(payload.get("official_success", False)),
                "contact_seen": bool(close.get("contact_seen", False)),
                "grasp_verified": bool(close.get("grasp_verified", False)),
                "retention_confirmation_steps": close.get(
                    "retention_confirmation_steps"
                ),
            }
        )
    return evidence


def _rejected_gate_refinement_context(
    store: CampaignStore,
    *,
    artifact_index: dict[str, Any],
) -> dict[str, Any] | None:
    """Build a seed-blind Stage2 refinement checkpoint for the active rejection."""

    state = store.state()
    candidate_sha256 = state.get("candidate_sha256")
    operator_rejection: dict[str, Any] | None = None
    if not isinstance(candidate_sha256, str) or not candidate_sha256:
        rejection_id = state.get("operator_rejection_id")
        matches = [
            row
            for row in _operator_candidate_rejections(store)
            if row.get("rejection_id") == rejection_id
        ]
        if len(matches) > 1:
            raise ValueError("operator rejection ID is ambiguous")
        if len(matches) == 1:
            operator_rejection = matches[0]
            candidate_sha256 = str(operator_rejection["candidate_sha256"])
        else:
            return None
    decisions = [
        row
        for row in store.gates.records()
        if row.get("candidate_sha256") == candidate_sha256
    ]
    if not decisions and operator_rejection is None:
        return None
    if decisions and bool(decisions[-1].get("passed")):
        return None
    bundle_path = store.root / "candidates" / candidate_sha256 / "bundle.json"
    bundle = read_json(bundle_path)
    if canonical_sha256(bundle) != candidate_sha256:
        raise ValueError("rejected candidate artifact digest mismatch")
    previous_candidate = {
        key: bundle.get(key)
        for key in (
            "candidate_id",
            "causal_hypothesis",
            "mechanism_change",
            "validation_plan",
            "critic_rules",
            "recovery_rules",
            "tool_plugin",
        )
    }
    if operator_rejection is not None:
        context = {
            "mode": "refine_operator_rejected_noop_candidate",
            "previous_candidate": previous_candidate,
            "preflight_rejection": {
                key: operator_rejection.get(key)
                for key in (
                    "rejection_id",
                    "rejection_kind",
                    "preflight_disposition",
                    "equivalent_to_candidate_sha256",
                    "normalized_recovery_steps",
                    "tool_parameter_defaults",
                    "reason",
                )
            },
            "gate_evidence": [],
            "required_change": (
                "Replace the behavior-equivalent explicit-default change with "
                "one materially different parameter change supported by the "
                "development smoke calibration."
            ),
        }
        development_evidence = _development_evidence_summary(store)
        if development_evidence:
            context["development_evidence"] = development_evidence
        return context
    decision = decisions[-1]
    evidence = _bounded_gate_descriptors(
        _gate_descriptors_for_candidate(
            store,
            candidate_sha256=candidate_sha256,
            artifact_index=artifact_index,
        ),
        per_descriptor_kind=2,
    )
    if not evidence:
        raise ValueError("rejected gate has no completed prompt-safe evidence")
    context = {
        "mode": "refine_rejected_candidate",
        "previous_candidate": previous_candidate,
        "paired_gate_result": {
            "stage": str(decision["kind"]).replace("same_seed", "target_failure_pair"),
            "passed": False,
            "conclusive": bool(decision.get("conclusive")),
            "candidate_successes": int(decision.get("candidate_successes", 0)),
            "parent_successes": int(decision.get("parent_successes", 0)),
            "paired_count": int(decision.get("paired_count", 0)),
            "candidate_wins": int(decision.get("candidate_wins", 0)),
            "parent_wins": int(decision.get("parent_wins", 0)),
            "candidate_safety_events": int(decision.get("candidate_safety_events", 0)),
            "parent_safety_events": int(decision.get("parent_safety_events", 0)),
        },
        "gate_evidence": evidence,
        "required_change": (
            "Use the paired live evidence to replace or materially refine the "
            "rejected mechanism while preserving one-causal-change scope."
        ),
    }
    # Keep causal attribution explicit and seed-blind. A natural candidate
    # success without an intervention is not evidence for the recovery.
    completed_rows = _completed_gate_episode_rows(store)
    current_candidate_rows = [
        EpisodeRecord.from_dict(row)
        for sha256, row, role in completed_rows
        if sha256 == candidate_sha256 and role == "candidate_gate_episode"
    ]
    candidate_interventions = sum(
        _candidate_intervened(record) for record in current_candidate_rows
    )
    successful_interventions = sum(
        bool(record.success) and _candidate_intervened(record)
        for record in current_candidate_rows
    )
    context["paired_gate_result"].update(
        {
            "candidate_interventions": candidate_interventions,
            "successful_candidate_interventions": successful_interventions,
        }
    )
    rejected_history = []
    history_details: list[dict[str, Any]] = []
    for prior in store.gates.records():
        if prior.get("passed") is not False or prior.get("kind") != "same_seed":
            continue
        prior_sha256 = prior.get("candidate_sha256")
        if not isinstance(prior_sha256, str):
            continue
        prior_rows = [
            EpisodeRecord.from_dict(row)
            for sha256, row, role in completed_rows
            if sha256 == prior_sha256 and role == "candidate_gate_episode"
        ]
        prior_bundle = read_json(
            store.root / "candidates" / prior_sha256 / "bundle.json"
        )
        prior_interventions = sum(_candidate_intervened(record) for record in prior_rows)
        prior_successful_interventions = sum(
            bool(record.success) and _candidate_intervened(record)
            for record in prior_rows
        )
        rejected_history.append(
            {
                "candidate_interventions": prior_interventions,
                "successful_candidate_interventions": prior_successful_interventions,
            }
        )
        history_details.append(
            {
                "candidate_id": prior_bundle.get("candidate_id"),
                "critic_rules": prior_bundle.get("critic_rules", ()),
                "recovery_rules": prior_bundle.get("recovery_rules", ()),
                "candidate_interventions": prior_interventions,
                "successful_candidate_interventions": prior_successful_interventions,
            }
        )
    context["rejected_gate_history"] = {
        "rejected_same_seed_candidate_count": len(rejected_history),
        "gates_with_zero_successful_candidate_interventions": sum(
            row["successful_candidate_interventions"] == 0
            for row in rejected_history
        ),
        "total_candidate_interventions": sum(
            row["candidate_interventions"] for row in rejected_history
        ),
        "total_successful_candidate_interventions": sum(
            row["successful_candidate_interventions"] for row in rejected_history
        ),
    }
    current_history = {
        "candidate_id": bundle.get("candidate_id"),
        "critic_rules": bundle.get("critic_rules", ()),
        "recovery_rules": bundle.get("recovery_rules", ()),
        "candidate_interventions": candidate_interventions,
        "successful_candidate_interventions": successful_interventions,
    }
    isolation = _select_causal_isolation_directive(
        current=current_history,
        history=history_details,
    )
    if isolation is not None:
        context["causal_isolation_directive"] = isolation
    # Development-only calibration stays outside heldout and gate ledgers, but
    # can guide the next bounded parameter test. Expose only a seed/path-free
    # summary so providers cannot use it as a hidden rollout schedule.
    development_root = store.root / "analysis" / "development-evidence"
    development_evidence: list[dict[str, Any]] = []
    if development_root.is_dir():
        for evidence_path in sorted(development_root.glob("*.json")):
            try:
                payload = read_json(evidence_path)
            except (OSError, ValueError):
                continue
            if not isinstance(payload, dict):
                continue
            result = payload.get("result")
            if not isinstance(result, dict):
                continue
            close = result.get("close")
            if not isinstance(close, dict):
                close = {}
            move_budgets = [
                phase.get("max_steps")
                for phase in result.values()
                if isinstance(phase, dict)
                and isinstance(phase.get("max_steps"), int)
            ]
            tool_parameters: dict[str, Any] = {}
            grasp_offset = payload.get("grasp_offset_xyz")
            if (
                isinstance(grasp_offset, list)
                and len(grasp_offset) == 3
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    for value in grasp_offset
                )
            ):
                tool_parameters["grasp_offset_xyz"] = [
                    float(value) for value in grasp_offset
                ]
            if move_budgets:
                tool_parameters["max_steps_per_move"] = max(move_budgets)
            retention_steps = close.get("required_retention_confirmation_steps")
            if isinstance(retention_steps, int) and not isinstance(
                retention_steps, bool
            ):
                tool_parameters["grasp_confirm_steps"] = retention_steps
            development_evidence.append(
                {
                    "evidence_kind": "development_privileged_tool_smoke",
                    "artifact_sha256": file_sha256(evidence_path),
                    "task_id": payload.get("task"),
                    "grasp_offset_xyz": payload.get("grasp_offset_xyz"),
                    "manipulated_object": payload.get("manipulated_object"),
                    "target": payload.get("target"),
                    "status": result.get("status"),
                    "official_success": bool(payload.get("official_success", False)),
                    "contact_seen": bool(close.get("contact_seen", False)),
                    "grasp_verified": bool(close.get("grasp_verified", False)),
                    "retention_confirmation_steps": close.get(
                        "retention_confirmation_steps"
                    ),
                    "tool": result.get("name"),
                    "tool_parameters": tool_parameters,
                }
            )
    if development_evidence:
        context["development_evidence"] = development_evidence
        calibration = _select_development_calibration_directive(
            history=history_details,
            development_evidence=development_evidence,
        )
        if calibration is not None:
            # Successful, seed-blind development calibration is stronger than
            # a prior underexposed recovery handoff. Keep the proven trigger,
            # but require Stage2 to materialize the tested tool parameters.
            context["causal_isolation_directive"] = calibration
    shadow_path = (
        store.root
        / "analysis"
        / "candidate-shadow-replay"
        / f"{candidate_sha256}.json"
    )
    precommit_path = shadow_path.with_suffix(".precommit.json")
    if shadow_path.is_file() or precommit_path.is_file():
        if not shadow_path.is_file() or not precommit_path.is_file():
            raise ValueError("rejected candidate shadow evidence is incomplete")
        shadow = read_json(shadow_path)
        precommit = read_json(precommit_path)
        if (
            shadow.get("candidate_sha256") != candidate_sha256
            or precommit.get("candidate_sha256") != candidate_sha256
            or precommit.get("shadow_report_sha256")
            != canonical_sha256(shadow)
        ):
            raise ValueError("rejected candidate shadow evidence changed")
        target_steps = sorted(
            int(row["first_trigger_step"])
            for row in shadow.get("outcomes", ())
            if isinstance(row, dict)
            and row.get("role") == "target_failure"
            and isinstance(row.get("first_trigger_step"), int)
        )
        context["previous_detector_replay"] = {
            "target_count": int(shadow.get("target_count", 0)),
            "target_triggered_anywhere": int(
                shadow.get("target_triggered_anywhere", 0)
            ),
            "success_control_count": int(shadow.get("success_control_count", 0)),
            "success_control_false_positives": int(
                shadow.get("success_control_false_positives", 0)
            ),
            "first_target_trigger_step_min": min(target_steps)
            if target_steps
            else None,
            "first_target_trigger_step_max": max(target_steps)
            if target_steps
            else None,
            "interpretation": (
                "Detection-only replay. Low target coverage or late first triggers "
                "must be corrected before adding more activation guards."
            ),
        }
    return context


def _select_causal_isolation_directive(
    *, current: dict[str, Any], history: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Bind refinement to a proven trigger and an underexposed recovery."""

    successful_prior = [
        row
        for row in history
        if int(row.get("successful_candidate_interventions", 0)) > 0
        and row.get("critic_rules")
    ]
    if not successful_prior:
        return None
    best_trigger = max(
        successful_prior,
        key=lambda row: (
            int(row.get("successful_candidate_interventions", 0)),
            int(row.get("candidate_interventions", 0)),
        ),
    )
    if int(current.get("successful_candidate_interventions", 0)) >= int(
        best_trigger.get("successful_candidate_interventions", 0)
    ):
        return None
    underexposed = [
        row
        for row in history
        if row is not best_trigger
        and int(row.get("candidate_interventions", 0)) > 0
        and int(row.get("successful_candidate_interventions", 0)) == 0
        and row.get("recovery_rules")
    ]
    if not underexposed:
        return None
    recovery_source = min(
        underexposed,
        key=lambda row: int(row.get("candidate_interventions", 0)),
    )
    recovery_steps = [
        step
        for rule in recovery_source.get("recovery_rules", ())
        if isinstance(rule, dict)
        for step in rule.get("steps", ())
        if isinstance(step, dict)
    ]
    if not recovery_steps:
        return None
    current_steps = [
        step
        for rule in current.get("recovery_rules", ())
        if isinstance(rule, dict)
        for step in rule.get("steps", ())
        if isinstance(step, dict)
    ]
    # Once the isolated recovery has itself been live-tested, repeating the
    # same byte-for-byte handoff cannot add causal information. Let the next
    # refinement select a competing audited recovery or parameter test.
    if current_steps == recovery_steps:
        return None
    return {
        "mode": "preserve_proven_trigger_retest_underexposed_recovery",
        "rationale": (
            "Preserve the earlier critic that produced attributed rescues and "
            "reuse the least-exposed rejected recovery to isolate handoff behavior."
        ),
        "preserve_critic_rules_byte_for_byte": list(
            best_trigger.get("critic_rules", ())
        ),
        "reuse_recovery_steps_byte_for_byte": recovery_steps,
        "trigger_successful_interventions": int(
            best_trigger.get("successful_candidate_interventions", 0)
        ),
        "recovery_prior_interventions": int(
            recovery_source.get("candidate_interventions", 0)
        ),
    }


def _select_development_calibration_directive(
    *,
    history: list[dict[str, Any]],
    development_evidence: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Bind Stage2 to replicated successful development-tool parameters."""

    successful = [
        row
        for row in development_evidence
        if bool(row.get("official_success"))
        and row.get("status") == "success"
        and isinstance(row.get("tool"), str)
        and isinstance(row.get("tool_parameters"), dict)
        and row["tool_parameters"]
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in successful:
        identity = canonical_sha256(
            {"tool": row["tool"], "parameters": row["tool_parameters"]}
        )
        grouped.setdefault(identity, []).append(row)
    replicated = [rows for rows in grouped.values() if len(rows) >= 2]
    if not replicated:
        return None
    calibrated = max(
        replicated,
        key=lambda rows: (len(rows), canonical_sha256(rows[0]["tool_parameters"])),
    )
    proven_triggers = [
        row
        for row in history
        if int(row.get("successful_candidate_interventions", 0)) > 0
        and row.get("critic_rules")
    ]
    if not proven_triggers:
        return None
    best_trigger = max(
        proven_triggers,
        key=lambda row: (
            int(row.get("successful_candidate_interventions", 0)),
            int(row.get("candidate_interventions", 0)),
        ),
    )
    exemplar = calibrated[0]
    return {
        "mode": "preserve_proven_trigger_bind_replicated_tool_calibration",
        "rationale": (
            "Preserve the strongest attributed trigger and require the next "
            "recovery to use the exact parameters replicated by successful "
            "non-heldout development smokes."
        ),
        "preserve_critic_rules_byte_for_byte": list(
            best_trigger.get("critic_rules", ())
        ),
        "required_recovery_tool": str(exemplar["tool"]),
        "required_recovery_parameters": dict(exemplar["tool_parameters"]),
        "development_success_support": len(calibrated),
        "trigger_successful_interventions": int(
            best_trigger.get("successful_candidate_interventions", 0)
        ),
    }


def _bounded_refinement_artifact_index(
    *,
    artifact_index: dict[str, Any],
    diagnosis: CausalDiagnosis,
    refinement_context: dict[str, Any] | None,
) -> dict[str, Any]:
    """Keep refinement prompts bounded without discarding stored evidence."""

    if refinement_context is None:
        return artifact_index
    required_ids = {
        *diagnosis.supporting_evidence_ids,
        *diagnosis.counterevidence_ids,
    }
    for row in refinement_context.get("gate_evidence", ()):
        if isinstance(row, dict) and isinstance(row.get("content_id"), str):
            required_ids.add(str(row["content_id"]))
    artifacts = [
        row
        for row in artifact_index.get("artifacts", ())
        if isinstance(row, dict) and row.get("content_id") in required_ids
    ]
    indexed_ids = {
        str(row["content_id"])
        for row in artifacts
        if isinstance(row.get("content_id"), str)
    }
    missing_gate = {
        str(row["content_id"])
        for row in refinement_context.get("gate_evidence", ())
        if isinstance(row, dict)
        and isinstance(row.get("content_id"), str)
        and row["content_id"] not in indexed_ids
    }
    if missing_gate:
        raise ValueError("bounded refinement index lost rejected-gate evidence")
    return {
        "artifacts": sorted(
            artifacts, key=lambda row: (str(row["content_id"]), str(row["type"]))
        ),
        "relationships": [],
        "selection": {
            "mode": "active_diagnosis_and_latest_rejected_gate",
            "source_artifact_count": len(artifact_index.get("artifacts", ())),
            "selected_artifact_count": len(artifacts),
        },
    }


def resolve_agent_artifact(
    campaign_root: str | Path, content_id: str
) -> dict[str, Any]:
    """Resolve one opaque reference for the Harness, never for prompt inclusion.

    The returned locator or inline value can be read by a Harness evidence
    handler after an agent requests ``content_id``. Every source is re-hashed,
    so stale or replaced files fail closed. Callers must summarize/redact the
    resolved evidence before placing any of it in a later agent message.
    """

    if not _CONTENT_ID_PATTERN.fullmatch(content_id):
        raise ValueError("malformed artifact content_id")
    store = CampaignStore(campaign_root)
    resolver = _load_resolver(store)
    entry = resolver["entries"].get(content_id)
    if not isinstance(entry, dict):
        raise KeyError(content_id)
    digest = entry.get("content_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("artifact resolver entry has a malformed digest")
    expected_id = _content_id(resolver=resolver, digest=digest)
    expected_hash = _agent_hash(resolver=resolver, digest=digest)
    if expected_id != content_id or entry.get("agent_hash") != expected_hash:
        raise ValueError("artifact resolver entry failed keyed-integrity validation")
    stale_sources = 0
    for source in entry.get("sources", ()):
        if source.get("kind") == "file":
            source_path = Path(str(source.get("path", "")))
            campaign_root = store.root.resolve()
            resolved = source_path.resolve()
            if (
                not resolved.is_relative_to(campaign_root)
                or not resolved.is_file()
                or file_sha256(resolved) != digest
            ):
                stale_sources += 1
                continue
            return {
                "content_id": content_id,
                "content_sha256": digest,
                "kind": "file",
                "path": str(resolved),
                "source": source,
            }
        if (
            source.get("kind") == "inline"
            and canonical_sha256(source.get("value")) == digest
        ):
            return {
                "content_id": content_id,
                "content_sha256": digest,
                "kind": "inline",
                "value": source.get("value"),
                "source": source,
            }
        stale_sources += 1
    raise ValueError(
        f"artifact resolver has no intact source for {content_id}; stale={stale_sources}"
    )


def _redact_display_text(value: str) -> str:
    redacted = _PATH_TEXT.sub("[redacted-locator]", value)
    redacted = _SENSITIVE_TEXT.sub("[redacted-sensitive-metadata]", redacted)
    return redacted[:1000]


def _agent_cluster(
    cluster: FailureCluster, aliases: dict[str, dict[str, str]]
) -> FailureCluster:
    segment_aliases = aliases["segment_id"]
    episode_aliases = aliases["episode_id"]

    def segment(raw: str) -> str:
        if raw not in segment_aliases:
            raise ValueError(f"cluster references an unindexed segment: {raw}")
        return segment_aliases[raw]

    def episode(raw: str) -> str:
        if raw not in episode_aliases:
            raise ValueError(f"cluster references an unindexed episode: {raw}")
        return episode_aliases[raw]

    return FailureCluster(
        cluster_id=cluster.cluster_id,
        hard_key=tuple(_redact_display_text(value) for value in cluster.hard_key),
        member_segment_ids=tuple(
            segment(value) for value in cluster.member_segment_ids
        ),
        episode_ids=tuple(episode(value) for value in cluster.episode_ids),
        representative_segment_ids=tuple(
            segment(value) for value in cluster.representative_segment_ids
        ),
        medoid_segment_id=segment(cluster.medoid_segment_id),
        summary=_redact_display_text(cluster.summary),
        prevalence=cluster.prevalence,
        mean_severity=cluster.mean_severity,
    )


def _cluster(value: dict[str, Any]) -> FailureCluster:
    payload = dict(value)
    for key in (
        "hard_key",
        "member_segment_ids",
        "episode_ids",
        "representative_segment_ids",
    ):
        payload[key] = tuple(payload[key])
    return FailureCluster(**payload)


def load_accepted_cluster_report(store: CampaignStore) -> dict[str, Any]:
    """Load the exact cluster partition accepted by Stage1 and verify lineage."""

    deterministic_path = store.root / "analysis" / "failure_clusters.json"
    deterministic = read_json(deterministic_path)
    manifest = store.manifest()
    if deterministic.get("manifest_sha256") != manifest.sha256:
        raise ValueError("failure-cluster report is bound to another manifest")
    multimodal_path = store.root / "analysis" / "failure_clusters.multimodal.json"
    if not multimodal_path.is_file():
        return deterministic
    multimodal = read_json(multimodal_path)
    if multimodal.get("manifest_sha256") != manifest.sha256:
        raise ValueError("multimodal cluster report is bound to another manifest")
    if multimodal.get("deterministic_source_sha256") != canonical_sha256(
        deterministic
    ):
        raise ValueError("multimodal cluster report has a stale deterministic source")
    return multimodal


def materialize_cluster_targets(
    store: CampaignStore, report: dict[str, Any]
) -> dict[str, Any]:
    """Rank primary/secondary targets by independent failed-episode support."""

    rows = [dict(row) for row in report.get("clusters", ())]
    rows.sort(
        key=lambda row: (
            -len({str(value) for value in row.get("episode_ids", ())}),
            -float(row.get("prevalence", 0.0)),
            -float(row.get("mean_severity", 0.0)),
            str(row.get("cluster_id", "")),
        )
    )
    report_sha256 = canonical_sha256(report)
    targets: list[dict[str, Any]] = []
    for rank, row in enumerate(rows[:2]):
        episode_ids = tuple(sorted({str(value) for value in row.get("episode_ids", ())}))
        member_segment_ids = tuple(
            sorted({str(value) for value in row.get("member_segment_ids", ())})
        )
        binding = {
            "manifest_sha256": store.manifest().sha256,
            "cluster_report_sha256": report_sha256,
            "cluster_id": str(row["cluster_id"]),
            "rank": rank,
            "episode_ids": episode_ids,
            "member_segment_ids": member_segment_ids,
        }
        targets.append(
            {
                **binding,
                "target_sha256": canonical_sha256(binding),
                "unique_failure_episode_count": len(episode_ids),
                "prevalence_by_episode": float(row.get("prevalence", 0.0)),
                "mean_severity": float(row.get("mean_severity", 0.0)),
            }
        )
    payload = {
        "schema_version": 1,
        "manifest_sha256": store.manifest().sha256,
        "cluster_report_sha256": report_sha256,
        "agent_dominant_cluster_id": report.get("dominant_cluster_id"),
        "ranking_authority": "harness_unique_failure_episode_count",
        "targets": targets,
    }
    path = store.root / "analysis" / "cluster_targets.json"
    if path.is_file():
        existing = read_json(path)
        if canonical_sha256(existing) != canonical_sha256(payload):
            raise ValueError("cluster target ranking changed during recovery")
        return existing
    atomic_write_json(path, payload, overwrite=False)
    return payload


def _evolution_policy(store: CampaignStore) -> dict[str, Any]:
    raw = store.manifest().runtime.get("evolution_policy", {})
    if not isinstance(raw, dict):
        raise ValueError("runtime.evolution_policy must be an object")
    false_positive_limit_key = "shadow_success_control_max_false_positive_rate"
    false_positive_limit_raw = raw.get(false_positive_limit_key, 0.0)
    if isinstance(false_positive_limit_raw, bool) or not isinstance(
        false_positive_limit_raw, (int, float)
    ):
        raise ValueError(
            f"{false_positive_limit_key} must be a finite number in [0, 1]"
        )
    false_positive_limit = float(false_positive_limit_raw)
    if not math.isfinite(false_positive_limit) or not 0 <= false_positive_limit <= 1:
        raise ValueError(
            f"{false_positive_limit_key} must be a finite number in [0, 1]"
        )
    provisional_confidence_key = "provisional_min_diagnosis_confidence"
    # Confidence ranks hypotheses but is not evidence from a live intervention.
    # Historical manifests therefore inherit no provisional confidence gate;
    # campaigns that need one must preregister it explicitly.
    provisional_confidence_raw = raw.get(provisional_confidence_key, 0.0)
    if isinstance(provisional_confidence_raw, bool) or not isinstance(
        provisional_confidence_raw, (int, float)
    ):
        raise ValueError(
            f"{provisional_confidence_key} must be a finite number in [0, 1]"
        )
    provisional_confidence = float(provisional_confidence_raw)
    if not math.isfinite(provisional_confidence) or not 0 <= provisional_confidence <= 1:
        raise ValueError(
            f"{provisional_confidence_key} must be a finite number in [0, 1]"
        )
    policy = {
        "same_seed_pass_rate": float(raw.get("same_seed_pass_rate", 0.5)),
        # Historical manifests had no per-gate iteration cap. Their effective
        # bound was the cluster candidate budget, so preserve that behavior
        # when reading an old manifest.
        "same_seed_max_rounds": int(
            raw.get(
                "same_seed_max_rounds",
                int(raw.get("max_candidate_rounds_per_cluster", 5)),
            )
        ),
        "same_seed_max_rounds_explicit": "same_seed_max_rounds" in raw,
        # ``test`` still runs and records the fixed held-out block, but its
        # result cannot become a training signal or reject a candidate.
        "heldout_mode": str(raw.get("heldout_mode", "validation")),
        "heldout_alpha": float(raw.get("heldout_alpha", 0.025)),
        "heldout_min_gain": int(raw.get("heldout_min_gain", 1)),
        "heldout_min_success_rate": float(
            raw.get("heldout_min_success_rate", 0.0)
        ),
        "heldout_require_significance": raw.get(
            "heldout_require_significance", True
        ),
        "heldout_max_rounds": int(raw.get("heldout_max_rounds", 1)),
        "heldout_max_rounds_explicit": "heldout_max_rounds" in raw,
        "max_candidate_rounds_per_cluster": int(
            raw.get("max_candidate_rounds_per_cluster", 5)
        ),
        "maximum_target_clusters": int(raw.get("maximum_target_clusters", 2)),
        "maximum_total_candidate_rounds": int(
            raw.get(
                "maximum_total_candidate_rounds",
                int(raw.get("max_candidate_rounds_per_cluster", 5))
                * int(raw.get("maximum_target_clusters", 2)),
            )
        ),
        "maximum_total_candidate_rounds_explicit": (
            "maximum_total_candidate_rounds" in raw
        ),
        "skip_regression_gate": bool(raw.get("skip_regression_gate", False)),
        "shadow_success_control_max_false_positive_rate": false_positive_limit,
        "shadow_success_control_false_positive_rate_explicit": (
            false_positive_limit_key in raw
        ),
        "diagnosis_max_artifact_reads": int(
            raw.get("diagnosis_max_artifact_reads", 18)
        ),
        "cluster_max_artifact_reads": int(raw.get("cluster_max_artifact_reads", 12)),
        "provisional_min_diagnosis_confidence": provisional_confidence,
        "defer_inconclusive_for_provisional": bool(
            raw.get("defer_inconclusive_for_provisional", False)
        ),
    }
    if not 0 < policy["same_seed_pass_rate"] <= 1:
        raise ValueError("same_seed_pass_rate must be in (0, 1]")
    if policy["same_seed_max_rounds"] < 1:
        raise ValueError("same_seed_max_rounds must be positive")
    if policy["heldout_mode"] not in {"test", "validation"}:
        raise ValueError("heldout_mode must be 'test' or 'validation'")
    if not 0 < policy["heldout_alpha"] < 1:
        raise ValueError("heldout_alpha must be in (0, 1)")
    if policy["heldout_min_gain"] < 0:
        raise ValueError("heldout_min_gain must be non-negative")
    if not 0 <= policy["heldout_min_success_rate"] <= 1:
        raise ValueError("heldout_min_success_rate must be in [0, 1]")
    if not isinstance(policy["heldout_require_significance"], bool):
        raise ValueError("heldout_require_significance must be boolean")
    if policy["heldout_max_rounds"] < 1:
        raise ValueError("heldout_max_rounds must be positive")
    if policy["max_candidate_rounds_per_cluster"] < 1:
        raise ValueError("max_candidate_rounds_per_cluster must be positive")
    if policy["maximum_target_clusters"] not in {1, 2}:
        raise ValueError("maximum_target_clusters must be one or two")
    if policy["maximum_total_candidate_rounds"] < 1:
        raise ValueError("maximum_total_candidate_rounds must be positive")
    if not 3 <= policy["diagnosis_max_artifact_reads"] <= 64:
        raise ValueError("diagnosis_max_artifact_reads must be in [3, 64]")
    if not 6 <= policy["cluster_max_artifact_reads"] <= 64:
        raise ValueError("cluster_max_artifact_reads must be in [6, 64]")
    return policy


def _shadow_live_gate_admission(
    store: CampaignStore, shadow_report: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate the success-control false-positive gate independently.

    Target divergence annotations can be incomplete. That must not make an
    observed success-control false-positive rate disappear behind the broader
    replay ``preflight_conclusive`` flag.
    """

    policy = _evolution_policy(store)
    limit = float(policy["shadow_success_control_max_false_positive_rate"])
    explicit = bool(policy["shadow_success_control_false_positive_rate_explicit"])
    authorization: dict[str, Any] | None = None
    candidate_sha256 = shadow_report.get("candidate_sha256")
    if isinstance(candidate_sha256, str) and _SHA256_PATTERN.fullmatch(
        candidate_sha256
    ):
        authorization_path = (
            store.root
            / "analysis"
            / "shadow-live-gate-authorizations"
            / f"{candidate_sha256}.json"
        )
        if authorization_path.is_file():
            authorization = read_json(authorization_path)
            if (
                authorization.get("manifest_sha256") != store.manifest().sha256
                or authorization.get("candidate_sha256") != candidate_sha256
                or authorization.get("authorization_kind")
                != "timeboxed_candidate_shadow_falsification"
            ):
                raise ValueError("shadow live-gate authorization binding changed")
            authorized_limit = authorization.get("max_false_positive_rate")
            if (
                isinstance(authorized_limit, bool)
                or not isinstance(authorized_limit, (int, float))
                or not math.isfinite(float(authorized_limit))
                or not 0 < float(authorized_limit) <= 1
            ):
                raise ValueError("shadow live-gate authorization rate is invalid")
            limit = float(authorized_limit)
            explicit = True
    control_count = shadow_report.get("success_control_count", 0)
    false_positives = shadow_report.get("success_control_false_positives", 0)
    if (
        isinstance(control_count, bool)
        or not isinstance(control_count, int)
        or control_count < 0
    ):
        raise ValueError("shadow success_control_count must be a non-negative integer")
    if (
        isinstance(false_positives, bool)
        or not isinstance(false_positives, int)
        or not 0 <= false_positives <= control_count
    ):
        raise ValueError(
            "shadow success_control_false_positives must be within the control count"
        )
    observed = false_positives / control_count if control_count else None
    recorded = shadow_report.get("success_control_false_positive_rate")
    if control_count:
        if (
            isinstance(recorded, bool)
            or not isinstance(recorded, (int, float))
            or not math.isfinite(float(recorded))
            or not math.isclose(float(recorded), observed, abs_tol=1e-12)
        ):
            raise ValueError(
                "shadow success-control false-positive rate is inconsistent"
            )
    elif recorded is not None:
        raise ValueError(
            "shadow success-control false-positive rate requires success controls"
        )
    exceeded = observed is not None and observed > limit
    relaxed_override = explicit and limit > 0
    return {
        "schema_version": 1,
        "criterion": "success_control_false_positive_rate",
        "configured_max_rate": limit,
        "observed_rate": observed,
        "success_control_count": control_count,
        "success_control_false_positives": false_positives,
        "threshold_source": (
            "candidate_timeboxed_falsification_authorization"
            if authorization is not None
            else
            "campaign_explicit_relaxed_override"
            if relaxed_override
            else "campaign_explicit_zero"
            if explicit
            else "default_zero"
        ),
        "relaxed_override": relaxed_override,
        "authorization_id": (
            authorization.get("authorization_id")
            if authorization is not None
            else None
        ),
        "eligible_for_live_gate": not exceeded,
        "disposition": (
            "rejected_false_positive_rate_exceeded"
            if exceeded
            else "not_evaluated_no_success_controls"
            if observed is None
            else "accepted_under_relaxed_override"
            if relaxed_override
            else "accepted_strict_zero_false_positives"
        ),
    }


def authorize_shadow_falsification(
    *,
    campaign_root: str | Path,
    candidate_output: str | Path,
    max_false_positive_rate: float,
    deadline: str,
    reason: str,
) -> dict[str, Any]:
    """Authorize one timeboxed live falsification without relaxing the campaign."""

    store = CampaignStore(campaign_root)
    state = store.state()
    if CampaignPhase(state["phase"]) != CampaignPhase.PROPOSE:
        raise ValueError("shadow falsification authorization requires propose phase")
    provisional = _load_provisional_authorization(store)
    if provisional is None:
        raise ValueError("shadow falsification requires provisional authorization")
    rate = float(max_false_positive_rate)
    if not math.isfinite(rate) or not 0 < rate <= 1:
        raise ValueError("max_false_positive_rate must be in (0, 1]")
    if not deadline.strip() or not reason.strip():
        raise ValueError("shadow falsification requires deadline and reason")
    output_path = Path(candidate_output).resolve()
    agents_root = (store.root / "agents").resolve()
    if not output_path.is_relative_to(agents_root) or not output_path.is_file():
        raise ValueError("candidate output is not an in-campaign Agent artifact")
    if output_path.name != "output.json" or not output_path.parent.name.startswith(
        "attempt-"
    ):
        raise ValueError("candidate output is not an immutable attempt output")
    from rpent.evolution.stages import _candidate_from_payload

    diagnoses = store.diagnoses.records()
    if not diagnoses:
        raise ValueError("shadow falsification requires a diagnosis")
    candidate = _candidate_from_payload(
        read_json(output_path),
        generation=store.manifest().generation,
        parent_sha256=state.get("current_bundle_sha256"),
        diagnosis=_diagnosis(diagnoses[-1]),
    )
    shadow_path = (
        store.root
        / "analysis"
        / "candidate-shadow-replay"
        / f"{candidate.sha256}.json"
    )
    if shadow_path.exists():
        # A supervisor may finish the immutable shadow replay before its
        # operator has a chance to review the false-positive rate.  Preserve
        # that report and allow a separate, explicitly bound authorization to
        # override only the zero-rate admission decision.
        precommit_path = shadow_path.with_suffix(".precommit.json")
        if not precommit_path.is_file():
            raise ValueError("existing shadow replay has no precommit binding")
        shadow = read_json(shadow_path)
        precommit = read_json(precommit_path)
        if (
            shadow.get("candidate_sha256") != candidate.sha256
            or precommit.get("candidate_sha256") != candidate.sha256
            or precommit.get("shadow_report_sha256")
            != canonical_sha256(shadow)
            or shadow.get("preflight_disposition")
            != "rejected_success_control_false_positive_rate"
        ):
            raise ValueError(
                "existing shadow replay is not an unadmitted false-positive rejection"
            )
    payload = {
        "schema_version": 1,
        "authorization_kind": "timeboxed_candidate_shadow_falsification",
        "manifest_sha256": store.manifest().sha256,
        "candidate_sha256": candidate.sha256,
        "candidate_output_sha256": file_sha256(output_path),
        "candidate_output": str(output_path.relative_to(store.root.resolve())),
        "provisional_authorization_id": provisional["authorization_id"],
        "max_false_positive_rate": rate,
        "deadline": deadline.strip(),
        "reason": reason.strip(),
        "default_campaign_policy_unchanged": True,
    }
    payload["authorization_id"] = "shadow-falsification-" + canonical_sha256(
        payload
    )[:24]
    path = (
        store.root
        / "analysis"
        / "shadow-live-gate-authorizations"
        / f"{candidate.sha256}.json"
    )
    atomic_write_json(path, payload, overwrite=False)
    return {"authorization": payload, "path": str(path)}


def _shadow_candidate_rejections(store: CampaignStore) -> tuple[dict[str, Any], ...]:
    root = store.root / "analysis" / "shadow-candidate-rejections"
    records = []
    for path in sorted(root.glob("*.json")):
        payload = read_json(path)
        candidate_sha256 = payload.get("candidate_sha256")
        if (
            not isinstance(candidate_sha256, str)
            or not _SHA256_PATTERN.fullmatch(candidate_sha256)
            or path.stem != candidate_sha256
            or payload.get("manifest_sha256") != store.manifest().sha256
            or payload.get("rejection_kind")
            not in {
                "immutable_shadow_preflight_rejection",
                "trajectory_feature_contract_rejection",
            }
        ):
            raise ValueError("shadow candidate rejection binding changed")
        if payload.get("rejection_kind") == "trajectory_feature_contract_rejection":
            if (
                not isinstance(payload.get("diagnosis_sha256"), str)
                or not isinstance(payload.get("cluster_id"), str)
                or payload.get("preflight_disposition")
                != "rejected_trajectory_feature_contract"
                or not isinstance(payload.get("feature_contract"), dict)
            ):
                raise ValueError("trajectory feature-contract rejection binding changed")
        records.append(payload)
    return tuple(records)


def _operator_candidate_rejections(
    store: CampaignStore,
) -> tuple[dict[str, Any], ...]:
    """Load append-only rejections for registered candidates stopped pre-gate."""

    root = store.root / "analysis" / "operator-candidate-rejections"
    registered = {
        str(row["candidate_sha256"]) for row in store.candidate_ledger.records()
    }
    gated = {
        str(row["candidate_sha256"])
        for row in store.gates.records()
        if isinstance(row.get("candidate_sha256"), str)
    }
    records = []
    for path in sorted(root.glob("*.json")):
        payload = read_json(path)
        candidate_sha256 = payload.get("candidate_sha256")
        if (
            not isinstance(candidate_sha256, str)
            or not _SHA256_PATTERN.fullmatch(candidate_sha256)
            or path.stem != candidate_sha256
            or candidate_sha256 not in registered
            or candidate_sha256 in gated
            or payload.get("manifest_sha256") != store.manifest().sha256
            or payload.get("rejection_kind")
            != "registered_candidate_noop_preflight_rejection"
            or payload.get("preflight_disposition")
            != "rejected_behavior_equivalent_to_registered_candidate"
            or not isinstance(payload.get("diagnosis_sha256"), str)
            or not isinstance(payload.get("cluster_id"), str)
            or not isinstance(payload.get("equivalent_to_candidate_sha256"), str)
            or not isinstance(payload.get("normalized_recovery_steps"), list)
        ):
            raise ValueError("operator candidate rejection binding changed")
        records.append(payload)
    return tuple(records)


def reject_registered_noop_candidate(
    *,
    campaign_root: str | Path,
    candidate_sha256: str,
    equivalent_to_candidate_sha256: str,
    tool_parameter_defaults: dict[str, dict[str, Any]],
    reason: str,
) -> dict[str, Any]:
    """Reject a registered but ungated candidate proven equal after defaults."""

    store = CampaignStore(campaign_root)
    state = store.state()
    if CampaignPhase(state["phase"]) != CampaignPhase.SAME_SEED_GATE:
        raise ValueError("registered no-op rejection requires same-seed phase")
    if state.get("candidate_sha256") != candidate_sha256:
        raise ValueError("registered no-op rejection does not match active candidate")
    if not reason.strip():
        raise ValueError("registered no-op rejection requires a reason")
    if any(
        row.get("candidate_sha256") == candidate_sha256
        for row in store.gates.records()
    ):
        raise ValueError("candidate with a gate decision cannot be rejected preflight")
    registered = {
        str(row["candidate_sha256"]) for row in store.candidate_ledger.records()
    }
    if candidate_sha256 not in registered:
        raise ValueError("active candidate is not registered")
    if equivalent_to_candidate_sha256 not in registered:
        raise ValueError("equivalent reference candidate is not registered")
    if equivalent_to_candidate_sha256 == candidate_sha256:
        raise ValueError("candidate cannot be its own equivalence reference")

    def load_bundle(sha256: str) -> CandidateBundle:
        path = store.root / "candidates" / sha256 / "bundle.json"
        payload = read_json(path)
        if canonical_sha256(payload) != sha256:
            raise ValueError("registered candidate bundle digest changed")
        return CandidateBundle.from_dict(payload)

    candidate = load_bundle(candidate_sha256)
    reference = load_bundle(equivalent_to_candidate_sha256)
    if candidate.diagnosis_sha256 != reference.diagnosis_sha256:
        raise ValueError("no-op reference belongs to another diagnosis")
    if candidate.critic_rules != reference.critic_rules:
        raise ValueError("no-op candidate changed its critic rules")

    def normalize(bundle: CandidateBundle) -> list[dict[str, Any]]:
        normalized = []
        for rule in bundle.recovery_rules:
            for step in rule.steps:
                defaults = tool_parameter_defaults.get(step.tool, {})
                if not isinstance(defaults, dict):
                    raise ValueError("tool parameter defaults must be nested objects")
                normalized.append(
                    {
                        "tool": step.tool,
                        "parameters": {**defaults, **step.parameters},
                    }
                )
        return normalized

    normalized_candidate = normalize(candidate)
    normalized_reference = normalize(reference)
    if normalized_candidate != normalized_reference:
        raise ValueError("candidate is not behavior-equivalent after tool defaults")
    cluster_id = _diagnosis_cluster_id(store, candidate.diagnosis_sha256)
    payload = {
        "schema_version": 1,
        "rejection_kind": "registered_candidate_noop_preflight_rejection",
        "manifest_sha256": store.manifest().sha256,
        "candidate_sha256": candidate_sha256,
        "candidate_id": candidate.candidate_id,
        "equivalent_to_candidate_sha256": equivalent_to_candidate_sha256,
        "diagnosis_sha256": candidate.diagnosis_sha256,
        "cluster_id": cluster_id,
        "preflight_disposition": (
            "rejected_behavior_equivalent_to_registered_candidate"
        ),
        "tool_parameter_defaults": tool_parameter_defaults,
        "normalized_recovery_steps": normalized_candidate,
        "reason": reason.strip(),
    }
    payload["rejection_id"] = "operator-rejection-" + canonical_sha256(payload)[:24]
    path = (
        store.root
        / "analysis"
        / "operator-candidate-rejections"
        / f"{candidate_sha256}.json"
    )
    if path.is_file():
        if canonical_sha256(read_json(path)) != canonical_sha256(payload):
            raise ValueError("operator candidate rejection already exists with new content")
    else:
        atomic_write_json(path, payload, overwrite=False)
    target, state_updates = _advance_after_candidate_rejection(
        store, candidate_sha256
    )
    state_updates["candidate_sha256"] = None
    state_updates["operator_rejection_id"] = payload["rejection_id"]
    store.transition(target, state_updates=state_updates)
    return {"rejection": payload, "path": str(path), "state": store.state()}


def _record_candidate_feature_contract_rejection(
    *,
    store: CampaignStore,
    candidate: CandidateBundle,
    candidate_output: Path,
    contract: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    """Persist one unregistered candidate's immutable feature-contract failure."""

    if not reason.strip():
        raise ValueError("candidate feature-contract rejection requires a reason")
    output_path = candidate_output.resolve()
    agents_root = (store.root / "agents").resolve()
    if (
        not output_path.is_relative_to(agents_root)
        or not output_path.is_file()
        or output_path.name != "output.json"
        or not output_path.parent.name.startswith("attempt-")
    ):
        raise ValueError("candidate output is not an immutable attempt output")
    diagnoses = store.diagnoses.records()
    if not diagnoses:
        raise ValueError("candidate feature-contract rejection requires a diagnosis")
    diagnosis = _diagnosis(diagnoses[-1])
    if candidate.diagnosis_sha256 != diagnosis.sha256:
        raise ValueError("candidate feature-contract diagnosis binding changed")
    if any(
        row.get("candidate_sha256") == candidate.sha256
        for row in store.candidate_ledger.records()
    ):
            raise ValueError(
                "registered candidate must not receive a feature-contract rejection"
            )
    payload = {
        "schema_version": 1,
        "rejection_kind": "trajectory_feature_contract_rejection",
        "manifest_sha256": store.manifest().sha256,
        "candidate_sha256": candidate.sha256,
        "candidate_id": candidate.candidate_id,
        "generation": candidate.generation,
        "parent_sha256": candidate.parent_sha256,
        "diagnosis_sha256": candidate.diagnosis_sha256,
        "cluster_id": diagnosis.cluster_id,
        "candidate_output_sha256": file_sha256(output_path),
        "candidate_output": str(output_path.relative_to(store.root.resolve())),
        "preflight_disposition": "rejected_trajectory_feature_contract",
        "feature_contract": contract,
        "reason": reason.strip(),
    }
    payload["rejection_id"] = "shadow-rejection-" + canonical_sha256(payload)[:24]
    path = (
        store.root
        / "analysis"
        / "shadow-candidate-rejections"
        / f"{candidate.sha256}.json"
    )
    if path.is_file():
        existing = read_json(path)
        if canonical_sha256(existing) != canonical_sha256(payload):
            raise ValueError(
                "candidate feature-contract rejection already exists with new content"
            )
        return {"rejection": existing, "path": str(path)}
    atomic_write_json(path, payload, overwrite=False)
    return {"rejection": payload, "path": str(path)}


def reject_shadow_candidate(
    *, campaign_root: str | Path, candidate_output: str | Path, reason: str
) -> dict[str, Any]:
    """Append-only rejection of an unregistered candidate's failed shadow test."""

    store = CampaignStore(campaign_root)
    state = store.state()
    if CampaignPhase(state["phase"]) != CampaignPhase.PROPOSE:
        raise ValueError("shadow candidate rejection requires propose phase")
    if not reason.strip():
        raise ValueError("shadow candidate rejection requires a reason")
    output_path = Path(candidate_output).resolve()
    agents_root = (store.root / "agents").resolve()
    if not output_path.is_relative_to(agents_root) or not output_path.is_file():
        raise ValueError("candidate output is not an in-campaign Agent artifact")
    if output_path.name != "output.json" or not output_path.parent.name.startswith(
        "attempt-"
    ):
        raise ValueError("candidate output is not an immutable attempt output")
    from rpent.evolution.stages import _candidate_from_payload

    diagnoses = store.diagnoses.records()
    if not diagnoses:
        raise ValueError("shadow candidate rejection requires a diagnosis")
    diagnosis = _diagnosis(diagnoses[-1])
    candidate = _candidate_from_payload(
        read_json(output_path),
        generation=store.manifest().generation,
        parent_sha256=state.get("current_bundle_sha256"),
        diagnosis=diagnosis,
    )
    if any(
        row.get("candidate_sha256") == candidate.sha256
        for row in store.candidate_ledger.records()
    ):
        raise ValueError("registered candidate must be rejected by an online gate")
    shadow_path = (
        store.root
        / "analysis"
        / "candidate-shadow-replay"
        / f"{candidate.sha256}.json"
    )
    precommit_path = shadow_path.with_suffix(".precommit.json")
    if not shadow_path.is_file() or not precommit_path.is_file():
        raise ValueError("shadow candidate rejection has no immutable replay")
    shadow = read_json(shadow_path)
    precommit = read_json(precommit_path)
    admission = shadow.get("live_gate_admission")
    if (
        shadow.get("candidate_sha256") != candidate.sha256
        or precommit.get("candidate_sha256") != candidate.sha256
        or precommit.get("shadow_report_sha256") != canonical_sha256(shadow)
        or not isinstance(admission, dict)
        or admission.get("eligible_for_live_gate") is not False
        or not str(shadow.get("preflight_disposition", "")).startswith("rejected_")
    ):
        raise ValueError("candidate shadow replay is not a rejected preflight")
    payload = {
        "schema_version": 1,
        "rejection_kind": "immutable_shadow_preflight_rejection",
        "manifest_sha256": store.manifest().sha256,
        "candidate_sha256": candidate.sha256,
        "diagnosis_sha256": candidate.diagnosis_sha256,
        "cluster_id": diagnosis.cluster_id,
        "candidate_output_sha256": file_sha256(output_path),
        "candidate_output": str(output_path.relative_to(store.root.resolve())),
        "shadow_report_sha256": canonical_sha256(shadow),
        "preflight_disposition": shadow["preflight_disposition"],
        "observed_false_positive_rate": admission.get("observed_rate"),
        "reason": reason.strip(),
    }
    payload["rejection_id"] = "shadow-rejection-" + canonical_sha256(payload)[:24]
    path = (
        store.root
        / "analysis"
        / "shadow-candidate-rejections"
        / f"{candidate.sha256}.json"
    )
    if path.is_file():
        existing = read_json(path)
        if canonical_sha256(existing) != canonical_sha256(payload):
            raise ValueError("shadow candidate rejection already exists with new content")
        return {"rejection": existing, "path": str(path)}
    atomic_write_json(path, payload, overwrite=False)
    return {"rejection": payload, "path": str(path)}


def _validate_shadow_live_gate_admission(
    store: CampaignStore, shadow_report: dict[str, Any]
) -> dict[str, Any]:
    expected = _shadow_live_gate_admission(store, shadow_report)
    recorded = shadow_report.get("live_gate_admission")
    if recorded is not None and recorded != expected:
        # The replay report is immutable and may have been committed under the
        # default zero false-positive policy.  A later authorization is an
        # append-only, candidate-bound exception; only the admission fields
        # may differ, while the measured rate and control counts must match.
        override_fields = {
            "authorization_id",
            "configured_max_rate",
            "disposition",
            "eligible_for_live_gate",
            "relaxed_override",
            "threshold_source",
        }
        recorded_core = {
            key: value
            for key, value in recorded.items()
            if key not in override_fields
        }
        expected_core = {
            key: value for key, value in expected.items() if key not in override_fields
        }
        authorized_replay_override = (
            expected["relaxed_override"]
            and expected["authorization_id"] is not None
            and recorded.get("eligible_for_live_gate") is False
            and recorded.get("disposition") == "rejected_false_positive_rate_exceeded"
            and recorded_core == expected_core
        )
        if not authorized_replay_override:
            raise ValueError("candidate shadow live-gate admission binding changed")
    if expected["relaxed_override"] and recorded is None:
        raise ValueError(
            "relaxed shadow false-positive override lacks an immutable audit marker"
        )
    if not expected["eligible_for_live_gate"]:
        raise ValueError(
            "candidate shadow preflight rejected: success-control false-positive "
            f"rate {expected['observed_rate']:.6g} exceeds configured maximum "
            f"{expected['configured_max_rate']:.6g}"
        )
    return expected


def _frozen_tool_names(tool_catalog: dict[str, Any]) -> set[str]:
    """Return the executable names frozen into either environment catalog.

    RoboCasa manifests include a task binding; LIBERO catalogs use the same
    top-level ``tools`` list without importing environment-specific code into
    the generic campaign lifecycle.
    """

    binding = tool_catalog.get("task_binding")
    if isinstance(binding, dict):
        names = binding.get("tool_names")
        if isinstance(names, list) and names:
            return {str(name) for name in names if str(name).strip()}
    tools = tool_catalog.get("tools")
    if not isinstance(tools, list) or not tools:
        raise ValueError("frozen tool catalog has no executable tools")
    names = {
        str(row.get("name"))
        for row in tools
        if isinstance(row, dict) and str(row.get("name", "")).strip()
    }
    if not names:
        raise ValueError("frozen tool catalog contains no named tools")
    return names


def _diagnosis_cluster_id(store: CampaignStore, diagnosis_sha256: str) -> str:
    matches = [
        row
        for row in store.diagnoses.records()
        if canonical_sha256(row) == diagnosis_sha256
    ]
    if len(matches) != 1:
        raise ValueError("candidate diagnosis is missing or ambiguous")
    return str(matches[0]["cluster_id"])


def _rejected_candidates_for_cluster(
    store: CampaignStore, cluster_id: str
) -> tuple[str, ...]:
    """Return unique candidates rejected by preflight or a conclusive gate.

    The five-round bound applies to candidate hypotheses, not merely to the
    first same-seed screen. A candidate that passes same-seed but later fails
    regression or heldout validation has still consumed one refinement round
    for this cluster and must not create an unbounded PROPOSE loop.
    """

    rejected: set[str] = set()
    for decision in store.gates.records():
        if decision.get("passed") is not False:
            continue
        if decision.get("kind") == "heldout_10" and not decision.get("conclusive"):
            continue
        candidate_sha256 = decision.get("candidate_sha256")
        if not isinstance(candidate_sha256, str):
            raise ValueError("gate decision has no candidate binding")
        bundle = read_json(
            store.root / "candidates" / candidate_sha256 / "bundle.json"
        )
        if _diagnosis_cluster_id(store, str(bundle["diagnosis_sha256"])) == cluster_id:
            rejected.add(candidate_sha256)
    for rejection in _shadow_candidate_rejections(store):
        if rejection.get("cluster_id") == cluster_id:
            rejected.add(str(rejection["candidate_sha256"]))
    for rejection in _operator_candidate_rejections(store):
        if rejection.get("cluster_id") == cluster_id:
            rejected.add(str(rejection["candidate_sha256"]))
    return tuple(sorted(rejected))


def _all_rejected_candidates(store: CampaignStore) -> set[str]:
    rejected = {
        str(row["candidate_sha256"])
        for row in store.gates.records()
        if row.get("passed") is False
        and not (row.get("kind") == "heldout_10" and not row.get("conclusive"))
    }
    rejected.update(
        str(row["candidate_sha256"]) for row in _shadow_candidate_rejections(store)
    )
    rejected.update(
        str(row["candidate_sha256"])
        for row in _operator_candidate_rejections(store)
    )
    return rejected


def _advance_after_candidate_rejection(
    store: CampaignStore, candidate_sha256: str
) -> tuple[CampaignPhase, dict[str, Any]]:
    """Choose refine, secondary-cluster, or bounded completion after rejection."""

    candidate = read_json(
        store.root / "candidates" / candidate_sha256 / "bundle.json"
    )
    cluster_id = _diagnosis_cluster_id(store, str(candidate["diagnosis_sha256"]))
    rejected = _rejected_candidates_for_cluster(store, cluster_id)
    if candidate_sha256 not in rejected:
        raise ValueError("candidate rejection is not recorded in the immutable ledger")
    policy = _evolution_policy(store)
    rejected_total = _all_rejected_candidates(store)
    if (
        policy["maximum_total_candidate_rounds_explicit"]
        and len(rejected_total) >= policy["maximum_total_candidate_rounds"]
    ):
        return CampaignPhase.COMPLETE, {
            "candidate_round": len(rejected),
            "optimization_outcome": "maximum_total_candidate_rounds_exhausted",
        }
    if len(rejected) < policy["max_candidate_rounds_per_cluster"]:
        return CampaignPhase.PROPOSE, {
            "candidate_round": len(rejected) + 1,
            "optimization_outcome": "refine_active_cluster",
        }

    cluster_report = load_accepted_cluster_report(store)
    targets = materialize_cluster_targets(store, cluster_report).get("targets", ())
    current_matches = [
        row for row in targets if row.get("cluster_id") == cluster_id
    ]
    if len(current_matches) != 1:
        raise ValueError("rejected candidate cluster has no target rank")
    next_rank = int(current_matches[0]["rank"]) + 1
    next_matches = [
        row
        for row in targets
        if int(row.get("rank", -1)) == next_rank
        and next_rank < policy["maximum_target_clusters"]
    ]
    if len(next_matches) == 1:
        return CampaignPhase.DIAGNOSE, {
            "candidate_sha256": None,
            "active_cluster_rank": next_rank,
            "active_cluster_id": next_matches[0]["cluster_id"],
            "active_cluster_target_sha256": next_matches[0]["target_sha256"],
            "candidate_round": 0,
            "optimization_outcome": "primary_exhausted_switch_secondary",
        }
    if next_matches:
        raise ValueError("secondary cluster target is ambiguous")
    return CampaignPhase.COMPLETE, {
        "candidate_sha256": None,
        "candidate_round": len(rejected),
        "optimization_outcome": "no_candidate_passed_primary_or_secondary",
    }


def _diagnosis(value: dict[str, Any]) -> CausalDiagnosis:
    payload = dict(value)
    for key in (
        "contributing_causes",
        "competing_hypotheses",
        "supporting_evidence_ids",
        "counterevidence_ids",
        "visual_evidence",
    ):
        payload[key] = tuple(payload.get(key, ()))
    return CausalDiagnosis(**payload)


def _diagnosis_is_inconclusive(diagnosis: CausalDiagnosis) -> bool:
    return diagnosis.root_cause.strip().casefold().startswith("inconclusive")


def _provisional_authorization_path(
    store: CampaignStore, authorization_id: str
) -> Path:
    return (
        store.root
        / "analysis"
        / "provisional-hypothesis-authorizations"
        / f"{authorization_id}.json"
    )


def _load_provisional_authorization(
    store: CampaignStore, diagnosis: CausalDiagnosis | None = None
) -> dict[str, Any] | None:
    state = store.state()
    authorization_id = state.get("provisional_authorization_id")
    if authorization_id is None:
        return None
    if not isinstance(authorization_id, str) or not authorization_id:
        raise ValueError("provisional authorization ID is malformed")
    path = _provisional_authorization_path(store, authorization_id)
    if not path.is_file():
        raise ValueError("provisional authorization artifact is missing")
    payload = read_json(path)
    if file_sha256(path) != state.get("provisional_authorization_sha256"):
        raise ValueError("provisional authorization digest changed")
    if payload.get("authorization_id") != authorization_id:
        raise ValueError("provisional authorization identity changed")
    if payload.get("manifest_sha256") != store.manifest().sha256:
        raise ValueError("provisional authorization targets another manifest")
    if payload.get("cluster_target_sha256") != state.get(
        "active_cluster_target_sha256"
    ):
        raise ValueError("provisional authorization cluster binding changed")
    if diagnosis is not None and payload.get("diagnosis_sha256") != diagnosis.sha256:
        raise ValueError("provisional authorization diagnosis binding changed")
    if float(payload.get("same_seed_pass_rate", 0.0)) != float(
        state.get("provisional_same_seed_pass_rate", 0.0)
    ):
        raise ValueError("provisional same-seed threshold changed")
    if bool(payload.get("skip_regression")) != bool(
        state.get("provisional_skip_regression")
    ):
        raise ValueError("provisional regression policy changed")
    return payload


def effective_same_seed_pass_rate(store: CampaignStore) -> float:
    """Return the manifest rate or an explicitly authorized timeboxed rate."""

    authorization = _load_provisional_authorization(store)
    if authorization is not None:
        return float(authorization["same_seed_pass_rate"])
    return float(
        store.manifest().runtime.get("evolution_policy", {}).get(
            "same_seed_pass_rate", 0.5
        )
    )


def _same_seed_threshold_authorization_path(
    store: CampaignStore, candidate_sha256: str
) -> Path:
    return (
        store.root
        / "analysis"
        / "same-seed-threshold-authorizations"
        / f"{candidate_sha256}.json"
    )


def _same_seed_gate_plan_path(store: CampaignStore, candidate_sha256: str) -> Path:
    return (
        store.root
        / "candidates"
        / candidate_sha256
        / "gates"
        / "same_seed"
        / "plan.json"
    )


def _assert_heldout_not_started(
    store: CampaignStore, candidate_sha256: str
) -> None:
    if any(
        row.get("candidate_sha256") == candidate_sha256
        and row.get("kind") in {"heldout_10", "heldout_20", "heldout_50"}
        for row in store.gates.records()
    ):
        raise ValueError("same-seed threshold override is too late: heldout started")
    candidate_root = store.root / "candidates" / candidate_sha256
    for gate_name in ("heldout", "heldout_20"):
        gate_root = candidate_root / "gates" / gate_name
        if gate_root.exists() and any(path.is_file() for path in gate_root.rglob("*")):
            raise ValueError("same-seed threshold override is too late: heldout started")


def load_same_seed_threshold_authorization(
    store: CampaignStore,
    *,
    candidate_sha256: str,
) -> dict[str, Any] | None:
    """Load and fully validate one candidate-bound threshold exception."""

    path = _same_seed_threshold_authorization_path(store, candidate_sha256)
    if not path.is_file():
        return None
    payload = read_json(path)
    authorization_id = payload.get("authorization_id")
    if not isinstance(authorization_id, str) or not authorization_id:
        raise ValueError("same-seed threshold authorization identity is malformed")
    identity_payload = dict(payload)
    identity_payload.pop("authorization_id", None)
    expected_id = "same-seed-threshold-" + canonical_sha256(identity_payload)[:24]
    if authorization_id != expected_id:
        raise ValueError("same-seed threshold authorization identity changed")

    ledger_rows = [
        row
        for row in store.same_seed_threshold_authorizations.records()
        if row.get("authorization_id") == authorization_id
    ]
    if len(ledger_rows) != 1:
        raise ValueError("same-seed threshold authorization ledger binding is missing")
    artifact_sha256 = file_sha256(path)
    ledger = ledger_rows[0]
    if (
        ledger.get("artifact_sha256") != artifact_sha256
        or ledger.get("candidate_sha256") != candidate_sha256
        or ledger.get("manifest_sha256") != store.manifest().sha256
        or ledger.get("generation") != store.manifest().generation
    ):
        raise ValueError("same-seed threshold authorization ledger binding changed")

    state = store.state()
    if state.get("candidate_sha256") == candidate_sha256:
        state_bindings = {
            "same_seed_threshold_authorization_id": authorization_id,
            "same_seed_threshold_authorization_sha256": artifact_sha256,
            "same_seed_threshold_authorization_candidate_sha256": candidate_sha256,
            "same_seed_threshold_authorization_plan_sha256": payload.get("plan_sha256"),
        }
        changed = [
            key for key, value in state_bindings.items() if state.get(key) != value
        ]
        if changed:
            raise ValueError(
                "same-seed threshold authorization state binding changed: "
                f"{sorted(changed)}"
            )

    manifest = store.manifest()
    if (
        payload.get("authorization_kind")
        != "candidate_bound_same_seed_threshold_override"
        or payload.get("manifest_sha256") != manifest.sha256
        or payload.get("generation") != manifest.generation
        or payload.get("candidate_sha256") != candidate_sha256
    ):
        raise ValueError("same-seed threshold authorization campaign binding changed")
    bundle_path = store.root / "candidates" / candidate_sha256 / "bundle.json"
    if not bundle_path.is_file() or canonical_sha256(read_json(bundle_path)) != candidate_sha256:
        raise ValueError("same-seed threshold authorization candidate bundle is stale")
    if payload.get("candidate_bundle_sha256") != candidate_sha256:
        raise ValueError("same-seed threshold authorization candidate binding changed")

    plan_path = _same_seed_gate_plan_path(store, candidate_sha256)
    if not plan_path.is_file():
        raise ValueError("same-seed threshold authorization gate plan is missing")
    plan = read_json(plan_path)
    if canonical_sha256(plan) != payload.get("plan_sha256"):
        raise ValueError("same-seed threshold authorization gate plan is stale")
    if (
        plan.get("manifest_sha256") != manifest.sha256
        or plan.get("generation") != manifest.generation
        or plan.get("candidate_sha256") != candidate_sha256
    ):
        raise ValueError("same-seed threshold authorization plan binding changed")
    pairs = plan.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != payload.get("paired_count"):
        raise ValueError("same-seed threshold authorization pair count changed")
    old_rate = plan.get("same_seed_pass_rate")
    if (
        isinstance(old_rate, bool)
        or not isinstance(old_rate, (int, float))
        or float(old_rate) != float(payload.get("old_same_seed_pass_rate", 0.0))
    ):
        raise ValueError("same-seed threshold authorization old threshold changed")
    new_rate = payload.get("new_same_seed_pass_rate")
    minimum_successes = payload.get("minimum_same_seed_successes")
    if (
        isinstance(new_rate, bool)
        or not isinstance(new_rate, (int, float))
        or isinstance(minimum_successes, bool)
        or not isinstance(minimum_successes, int)
        or not 1 <= minimum_successes <= len(pairs)
        or not math.isclose(
            float(new_rate), minimum_successes / len(pairs), abs_tol=1e-12
        )
        or not 0 < float(new_rate) < float(old_rate)
    ):
        raise ValueError("same-seed threshold authorization is not a valid reduction")
    for field in ("reason", "deadline", "author"):
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            raise ValueError(f"same-seed threshold authorization lacks {field}")
    return payload


def effective_same_seed_gate_pass_rate(
    store: CampaignStore,
    *,
    candidate_sha256: str,
    plan: dict[str, Any],
) -> float:
    """Resolve an immutable plan threshold plus a candidate-bound reduction."""

    version = int(plan.get("schema_version", 1))
    if version < 2:
        planned_rate = 1.0
    else:
        planned_rate = plan.get("same_seed_pass_rate")
        if (
            isinstance(planned_rate, bool)
            or not isinstance(planned_rate, (int, float))
            or not 0 < float(planned_rate) <= 1
        ):
            raise ValueError("same-seed gate plan has an invalid pass rate")
        planned_rate = float(planned_rate)
    authorization = load_same_seed_threshold_authorization(
        store, candidate_sha256=candidate_sha256
    )
    if authorization is None:
        return planned_rate
    if not math.isclose(
        float(authorization["old_same_seed_pass_rate"]),
        planned_rate,
        abs_tol=1e-12,
    ):
        raise ValueError("same-seed threshold authorization targets a stale threshold")
    return float(authorization["new_same_seed_pass_rate"])


def authorize_same_seed_threshold_override(
    *,
    campaign_root: str | Path,
    minimum_same_seed_successes: int,
    skip_regression: bool,
    reason: str,
    deadline: str,
    author: str,
) -> dict[str, Any]:
    """Append one bounded threshold reduction without rewriting a gate plan."""

    store = CampaignStore(campaign_root)
    authorization_dir = store.root / "analysis" / "same-seed-threshold-authorizations"
    authorization_dir.mkdir(parents=True, exist_ok=True)
    with directory_lock(authorization_dir / ".authorize.lock"):
        state = store.state()
        if CampaignPhase(state["phase"]) != CampaignPhase.SAME_SEED_GATE:
            raise ValueError("same-seed threshold override requires same_seed_gate phase")
        candidate_sha256 = state.get("candidate_sha256")
        if not isinstance(candidate_sha256, str) or not _SHA256_PATTERN.fullmatch(
            candidate_sha256
        ):
            raise ValueError("same-seed threshold override requires an active candidate")
        if any(
            row.get("candidate_sha256") == candidate_sha256
            and row.get("kind") == "same_seed"
            for row in store.gates.records()
        ):
            raise ValueError("same-seed threshold override is too late: decision exists")
        _assert_heldout_not_started(store, candidate_sha256)
        if (
            isinstance(minimum_same_seed_successes, bool)
            or not isinstance(minimum_same_seed_successes, int)
        ):
            raise ValueError("minimum same-seed successes must be an integer")
        normalized = {
            "reason": reason.strip(),
            "deadline": deadline.strip(),
            "author": author.strip(),
        }
        missing = [field for field, value in normalized.items() if not value]
        if missing:
            raise ValueError(
                "same-seed threshold override requires " + ", ".join(sorted(missing))
            )

        manifest = store.manifest()
        if state.get("generation") != manifest.generation:
            raise ValueError("same-seed threshold override generation is stale")
        bundle_path = store.root / "candidates" / candidate_sha256 / "bundle.json"
        if not bundle_path.is_file() or canonical_sha256(read_json(bundle_path)) != candidate_sha256:
            raise ValueError("same-seed threshold override candidate bundle is stale")
        plan_path = _same_seed_gate_plan_path(store, candidate_sha256)
        if not plan_path.is_file():
            raise ValueError("same-seed threshold override requires a frozen gate plan")
        plan = read_json(plan_path)
        required_bindings = {
            "kind": "same_seed",
            "manifest_sha256": manifest.sha256,
            "generation": manifest.generation,
            "candidate_sha256": candidate_sha256,
        }
        changed = [
            key for key, value in required_bindings.items() if plan.get(key) != value
        ]
        if changed:
            raise ValueError(
                f"same-seed threshold override plan binding changed: {sorted(changed)}"
            )
        pairs = plan.get("pairs")
        if not isinstance(pairs, list) or not pairs:
            raise ValueError("same-seed threshold override plan has no pairs")
        pair_count = len(pairs)
        if not 1 <= minimum_same_seed_successes <= pair_count:
            raise ValueError("minimum same-seed successes must be within the gate")
        old_rate = plan.get("same_seed_pass_rate")
        if (
            int(plan.get("schema_version", 1)) < 2
            or isinstance(old_rate, bool)
            or not isinstance(old_rate, (int, float))
            or not 0 < float(old_rate) <= 1
        ):
            raise ValueError("same-seed threshold override requires a valid v2 plan")
        new_rate = minimum_same_seed_successes / pair_count
        if not new_rate < float(old_rate):
            raise ValueError("same-seed threshold override can only lower the threshold")

        source_ids = {str(pair.get("source_episode_id")) for pair in pairs}
        source_rows = [
            row
            for row in store.episodes.records()
            if str(row.get("episode_id")) in source_ids
        ]
        if len(source_rows) != pair_count or any(
            row.get("status") != "valid" or row.get("success") is not False
            for row in source_rows
        ):
            raise ValueError(
                "same-seed threshold reduction requires all source parents to be "
                "canonical failures"
            )

        payload = {
            "schema_version": 1,
            "authorization_kind": "candidate_bound_same_seed_threshold_override",
            "manifest_sha256": manifest.sha256,
            "generation": manifest.generation,
            "candidate_sha256": candidate_sha256,
            "candidate_bundle_sha256": candidate_sha256,
            "parent_sha256": plan.get("parent_sha256"),
            "diagnosis_sha256": read_json(bundle_path).get("diagnosis_sha256"),
            "plan_sha256": canonical_sha256(plan),
            "paired_count": pair_count,
            "source_parent_successes": 0,
            "old_same_seed_pass_rate": float(old_rate),
            "minimum_same_seed_successes": minimum_same_seed_successes,
            "new_same_seed_pass_rate": new_rate,
            "skip_regression": bool(skip_regression),
            **normalized,
            "gate_plan_unchanged": True,
            "gate_ledger_unchanged": True,
            "heldout_unchanged": True,
        }
        authorization_id = "same-seed-threshold-" + canonical_sha256(payload)[:24]
        payload["authorization_id"] = authorization_id
        path = _same_seed_threshold_authorization_path(store, candidate_sha256)
        existing = read_json(path) if path.is_file() else None
        if existing is not None and canonical_sha256(existing) != canonical_sha256(payload):
            raise ValueError(
                "candidate already has a different same-seed threshold authorization"
            )
        artifact_sha256 = atomic_write_json(path, payload, overwrite=False)
        store.same_seed_threshold_authorizations.append(
            {
                "authorization_id": authorization_id,
                "artifact_sha256": artifact_sha256,
                "manifest_sha256": manifest.sha256,
                "generation": manifest.generation,
                "candidate_sha256": candidate_sha256,
                "plan_sha256": payload["plan_sha256"],
            }
        )
        expected_state = {
            "same_seed_threshold_authorization_id": authorization_id,
            "same_seed_threshold_authorization_sha256": artifact_sha256,
            "same_seed_threshold_authorization_candidate_sha256": candidate_sha256,
            "same_seed_threshold_authorization_plan_sha256": payload["plan_sha256"],
            "same_seed_threshold_authorization_skip_regression": bool(
                skip_regression
            ),
        }
        pointer_changed = any(state.get(key) != value for key, value in expected_state.items())
        if pointer_changed:
            store.update_state(**expected_state)
        authorization = load_same_seed_threshold_authorization(
            store, candidate_sha256=candidate_sha256
        )
        return {
            "authorization": authorization,
            "authorization_sha256": artifact_sha256,
            "path": str(path),
            "state": store.state(),
        }


def authorize_provisional_hypothesis(
    *,
    campaign_root: str | Path,
    minimum_same_seed_successes: int = 1,
    skip_regression: bool = False,
    deadline: str,
) -> dict[str, Any]:
    """Auditably test a strong leading hypothesis without relabeling diagnosis.

    The diagnosis remains inconclusive. This authorization only permits Stage2
    to create one falsifiable candidate and records relaxed, timeboxed gates.
    """

    store = CampaignStore(campaign_root)
    state = store.state()
    if CampaignPhase(state["phase"]) != CampaignPhase.COMPLETE:
        raise ValueError("provisional authorization requires complete phase")
    if state.get("optimization_outcome") != "no_actionable_cluster_diagnosis":
        raise ValueError("campaign has no terminal inconclusive diagnosis")
    rows = store.diagnoses.records()
    if not rows:
        raise ValueError("provisional authorization requires a diagnosis")
    diagnosis = _diagnosis(rows[-1])
    if not _diagnosis_is_inconclusive(diagnosis):
        raise ValueError("provisional authorization is only for inconclusive diagnosis")
    if diagnosis.cluster_id != state.get("active_cluster_id"):
        raise ValueError("latest diagnosis is not bound to the active cluster")
    policy = _evolution_policy(store)
    minimum_confidence = float(policy["provisional_min_diagnosis_confidence"])
    if minimum_confidence > 0.0 and diagnosis.confidence < minimum_confidence:
        raise ValueError(
            "provisional diagnosis confidence is below the manifest policy "
            f"threshold {minimum_confidence}"
        )
    if len(diagnosis.visual_evidence) < 3:
        raise ValueError("provisional diagnosis lacks multimodal support")
    if len(diagnosis.competing_hypotheses) < 2 or not diagnosis.falsifier.strip():
        raise ValueError("provisional diagnosis is not falsifiable")
    report = load_accepted_cluster_report(store)
    targets = materialize_cluster_targets(store, report)
    matches = [
        row
        for row in targets.get("targets", ())
        if row.get("cluster_id") == diagnosis.cluster_id
        and row.get("target_sha256") == state.get("active_cluster_target_sha256")
    ]
    if len(matches) != 1:
        raise ValueError("provisional diagnosis has no unique frozen cluster target")
    target_count = int(matches[0].get("unique_failure_episode_count", 0))
    target_episode_ids = set(matches[0].get("episode_ids", ()))
    valid_failure_ids = {
        str(row.get("episode_id"))
        for row in store.episodes.records()
        if row.get("status") == "valid" and row.get("success") is False
    }
    if target_episode_ids - valid_failure_ids:
        raise ValueError("provisional target contains a non-canonical failure episode")
    if len(target_episode_ids) != target_count:
        raise ValueError("provisional target failure count is inconsistent")
    if not 1 <= minimum_same_seed_successes <= target_count:
        raise ValueError("minimum same-seed successes exceed the target cluster")
    rate = minimum_same_seed_successes / target_count
    payload = {
        "schema_version": 1,
        "authorization_kind": "timeboxed_provisional_hypothesis_test",
        "manifest_sha256": store.manifest().sha256,
        "diagnosis_sha256": diagnosis.sha256,
        "diagnosis_id": diagnosis.diagnosis_id,
        "cluster_id": diagnosis.cluster_id,
        "cluster_report_sha256": canonical_sha256(report),
        "cluster_target_sha256": matches[0]["target_sha256"],
        "target_failure_count": target_count,
        "minimum_same_seed_successes": minimum_same_seed_successes,
        "same_seed_pass_rate": rate,
        "skip_regression": bool(skip_regression),
        "heldout_label": "fixed_20_seed_validation_sr_not_unbiased_test",
        "deadline": deadline,
        "episodes_ledger_sha256": file_sha256(store.episodes.path),
        "diagnoses_ledger_sha256": file_sha256(store.diagnoses.path),
        "existing_candidate_count": len(store.candidate_ledger.records()),
        "existing_gate_count": len(store.gates.records()),
        "diagnosis_remains_inconclusive": True,
        "minimum_diagnosis_confidence": minimum_confidence,
        "observed_diagnosis_confidence": diagnosis.confidence,
        "diagnosis_confidence": diagnosis.confidence,
        "confidence_is_not_an_authorization_gate": minimum_confidence == 0.0,
        "hypothesis_status": "leading_experiment_target_not_confirmed_root_cause",
    }
    authorization_id = "provisional-" + canonical_sha256(payload)[:24]
    payload["authorization_id"] = authorization_id
    path = _provisional_authorization_path(store, authorization_id)
    atomic_write_json(path, payload, overwrite=False)
    authorization_sha256 = file_sha256(path)
    updated = store.reopen_completed_provisional_hypothesis(
        authorization_id=authorization_id,
        authorization_sha256=authorization_sha256,
        diagnosis_sha256=diagnosis.sha256,
        same_seed_pass_rate=rate,
        skip_regression=skip_regression,
    )
    return {
        "authorization": payload,
        "authorization_sha256": authorization_sha256,
        "state": updated,
    }


def _route_inconclusive_diagnosis(
    *,
    store: CampaignStore,
    diagnosis: CausalDiagnosis,
    targets: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    target_rank: int,
) -> dict[str, Any]:
    """Skip an unresolved cluster without allowing an unfalsifiable candidate."""

    policy = _evolution_policy(store)
    if policy["defer_inconclusive_for_provisional"]:
        store.transition(
            CampaignPhase.COMPLETE,
            state_updates={
                "candidate_sha256": None,
                "optimization_outcome": "no_actionable_cluster_diagnosis",
            },
        )
        return {
            **diagnosis.as_dict(),
            "optimization_outcome": "no_actionable_cluster_diagnosis",
            "candidate_created": False,
            "provisional_hypothesis_eligible": True,
            "secondary_cluster_deferred": target_rank + 1 < len(targets),
        }
    next_rank = target_rank + 1
    matches = [
        row
        for row in targets
        if int(row.get("rank", -1)) == next_rank
        and next_rank < policy["maximum_target_clusters"]
    ]
    report = {
        **diagnosis.as_dict(),
        "optimization_outcome": "inconclusive_cluster_skipped",
        "candidate_created": False,
    }
    if len(matches) == 1:
        selected = matches[0]
        store.update_state(
            active_cluster_rank=next_rank,
            active_cluster_id=selected["cluster_id"],
            active_cluster_target_sha256=selected["target_sha256"],
            optimization_outcome="inconclusive_cluster_skipped",
        )
        return {
            **report,
            "next_cluster_rank": next_rank,
            "next_cluster_id": selected["cluster_id"],
            "next_cluster_target_sha256": selected["target_sha256"],
        }
    if matches:
        raise ValueError("next cluster after inconclusive diagnosis is ambiguous")
    store.transition(
        CampaignPhase.COMPLETE,
        state_updates={
            "candidate_sha256": None,
            "optimization_outcome": "no_actionable_cluster_diagnosis",
        },
    )
    return {**report, "optimization_outcome": "no_actionable_cluster_diagnosis"}


def _materialize_multimodal_cluster_review(
    *,
    store: CampaignStore,
    deterministic_report: dict[str, Any],
    artifact_index: dict[str, Any],
    aliases: dict[str, dict[str, str]],
    model: str,
) -> dict[str, Any]:
    """Run/recover the visual Cluster Agent and deterministically materialize it."""

    manifest = store.manifest()
    output_path = store.root / "analysis" / "failure_clusters.multimodal.json"
    if output_path.is_file():
        return read_json(output_path)
    raw_clusters = tuple(_cluster(row) for row in deterministic_report.get("clusters", ()))
    agent_clusters = tuple(_agent_cluster(cluster, aliases) for cluster in raw_clusters)
    agent = CodexStageAgent(
        output_root=store.root / "agents" / "cluster",
        model=model,
        reasoning_effort=manifest.reasoning_effort,
        artifact_reader=lambda content_id: resolve_agent_artifact(
            store.root, content_id
        ),
        environment_name=manifest.environment,
        max_artifact_reads=int(
            _evolution_policy(store)["cluster_max_artifact_reads"]
        ),
    )
    review = agent.review_clusters(
        clusters=agent_clusters,
        artifact_index=artifact_index,
        task_contract=_authoritative_task_contract(store),
    )
    raw_by_alias = {value: key for key, value in aliases["segment_id"].items()}
    segment_by_id = {
        segment.segment_id: segment
        for row in store.episodes.records()
        for segment in EpisodeRecord.from_dict(row).all_failure_segments
    }
    episode_by_segment = {
        segment.segment_id: record.episode_id
        for row in store.episodes.records()
        for record in (EpisodeRecord.from_dict(row),)
        for segment in record.all_failure_segments
    }
    clusters = []
    for index, group in enumerate(review["groups"]):
        members = tuple(raw_by_alias[str(value)] for value in group["member_segment_ids"])
        representatives = tuple(
            raw_by_alias[str(value)] for value in group["representative_segment_ids"]
        )
        episodes = tuple(sorted({episode_by_segment[value] for value in members}))
        cluster = FailureCluster(
            cluster_id=f"visual-cluster-{canonical_sha256(sorted(members))[:16]}",
            hard_key=("multimodal_review", "generation_batch", f"group_{index:03d}"),
            member_segment_ids=members,
            episode_ids=episodes,
            representative_segment_ids=representatives,
            medoid_segment_id=representatives[0],
            summary=str(group["summary"]),
            prevalence=len(episodes)
            / max(1, int(deterministic_report["failures_with_segments"])),
            mean_severity=sum(segment_by_id[value].severity for value in members)
            / len(members),
        )
        clusters.append(cluster)
    dominant = clusters[int(review["dominant_group_index"])]
    report = {
        **deterministic_report,
        "schema_version": 2,
        "deterministic_source_sha256": canonical_sha256(deterministic_report),
        "cluster_method": "deterministic_segments_then_multimodal_agent_review",
        "clusters": [cluster.as_dict() for cluster in clusters],
        "dominant_cluster_id": dominant.cluster_id,
        "visual_review": review,
    }
    atomic_write_json(output_path, report, overwrite=False)
    return report


def run_diagnosis_stage(
    *,
    campaign_root: str | Path,
    tool_catalog: dict[str, Any],
    model: str | None = None,
) -> dict[str, Any]:
    store = CampaignStore(campaign_root)
    manifest = store.manifest()
    state = store.state()
    phase = CampaignPhase(state["phase"])
    if phase not in {CampaignPhase.CLUSTER, CampaignPhase.DIAGNOSE}:
        raise ValueError("diagnosis can only start from cluster/diagnose phase")
    deterministic_report = read_json(
        store.root / "analysis" / "failure_clusters.json"
    )
    clusters = deterministic_report.get("clusters", ())
    valid_episodes = int(deterministic_report.get("valid_episodes", 0))
    successes = int(deterministic_report.get("successes", 0))
    failures = valid_episodes - successes
    if not clusters:
        if failures:
            raise ValueError(
                "failed generation has no failure clusters; every valid failure "
                "must retain at least one trajectory segment"
            )
        completion = {
            "schema_version": 1,
            "manifest_sha256": manifest.sha256,
            "generation": manifest.generation,
            "valid_episodes": valid_episodes,
            "successes": successes,
            "optimization_outcome": "no_failures_to_optimize",
        }
        completion_path = store.root / "analysis" / "no-failure-completion.json"
        if completion_path.is_file():
            if canonical_sha256(read_json(completion_path)) != canonical_sha256(
                completion
            ):
                raise ValueError("no-failure completion changed during recovery")
        else:
            atomic_write_json(completion_path, completion, overwrite=False)
        if phase != CampaignPhase.COMPLETE:
            store.transition(
                CampaignPhase.COMPLETE,
                state_updates={"optimization_outcome": "no_failures_to_optimize"},
            )
        return completion
    if phase == CampaignPhase.CLUSTER:
        store.transition(CampaignPhase.DIAGNOSE)
    artifact_index, aliases = _agent_artifact_context(store)
    has_visual = any(
        isinstance(row, dict) and row.get("type") in {"image", "video"}
        for row in artifact_index.get("artifacts", ())
    )
    if manifest.protocol_explicit and not has_visual:
        raise ValueError(
            "formal failure clustering requires synchronized visual artifacts; "
            "deterministic-only clustering is development/audit-only"
        )
    report = (
        _materialize_multimodal_cluster_review(
            store=store,
            deterministic_report=deterministic_report,
            artifact_index=artifact_index,
            aliases=aliases,
            model=model or manifest.model,
        )
        if has_visual
        else deterministic_report
    )
    targets = materialize_cluster_targets(store, report)
    target_rank = int(store.state().get("active_cluster_rank", 0))
    target_rows = [
        row for row in targets.get("targets", ()) if int(row.get("rank", -1)) == target_rank
    ]
    if len(target_rows) != 1:
        raise ValueError("requested primary/secondary failure cluster is unavailable")
    target = target_rows[0]
    dominant = target["cluster_id"]
    cluster_rows = [
        row for row in report.get("clusters", ()) if row["cluster_id"] == dominant
    ]
    if len(cluster_rows) != 1:
        raise ValueError("exactly one dominant failure cluster is required")
    cluster_context_path = (
        store.root
        / "agents"
        / "cluster"
        / "multimodal-cluster-review"
        / "context.json"
    )
    inherited_evidence_access_logs: tuple[tuple[str | Path, str | None], ...] = ()
    if store.candidate_ledger.records():
        context = _latest_stage2_context(store)
    elif cluster_context_path.is_file():
        context = read_json(cluster_context_path)
        successful_attempt = context.get("successful_attempt")
        cluster_attempt_root = (
            cluster_context_path.parent / str(successful_attempt)
            if isinstance(successful_attempt, str)
            else None
        )
        inherited_access_path = (
            cluster_attempt_root / "evidence-access.jsonl"
            if cluster_attempt_root is not None
            else None
        )
        inherited_invocation_path = (
            cluster_attempt_root / "invocation.json"
            if cluster_attempt_root is not None
            else None
        )
        if (
            inherited_access_path is not None
            and inherited_access_path.is_file()
            and inherited_invocation_path is not None
            and inherited_invocation_path.is_file()
        ):
            inherited_invocation = read_json(inherited_invocation_path)
            inherited_evidence_access_logs = (
                (
                    inherited_access_path,
                    inherited_invocation.get("visual_access_log_sha256"),
                ),
            )
    else:
        context = {}
    provider_thread_id = context.get("provider_thread_id")
    if not isinstance(provider_thread_id, str) or not provider_thread_id.strip():
        provider_thread_id = None
    session_id = context.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        session_id = None
    target_sha256 = str(target["target_sha256"])
    store.update_state(
        active_cluster_rank=target_rank,
        active_cluster_id=dominant,
        active_cluster_target_sha256=target_sha256,
    )
    agent = CodexStageAgent(
        output_root=store.root / "agents" / "diagnosis" / target_sha256,
        model=model or manifest.model,
        reasoning_effort=manifest.reasoning_effort,
        artifact_reader=lambda content_id: resolve_agent_artifact(
            store.root, content_id
        ),
        environment_name=manifest.environment,
        session_id=session_id,
        thread_id=provider_thread_id,
        reconstructed=bool(context.get("reconstructed", False))
        or (bool(context) and provider_thread_id is None),
        max_artifact_reads=int(
            _evolution_policy(store)["diagnosis_max_artifact_reads"]
        ),
        inherited_evidence_access_logs=inherited_evidence_access_logs,
    )
    diagnosis = agent.diagnose(
        cluster=_agent_cluster(_cluster(cluster_rows[0]), aliases),
        artifact_index=artifact_index,
        tool_catalog=tool_catalog,
        task_contract=_authoritative_task_contract(store),
    )
    store.register_diagnosis(diagnosis)
    if _diagnosis_is_inconclusive(diagnosis):
        inconclusive_path = (
            store.root
            / "analysis"
            / "inconclusive_diagnoses"
            / f"{target_sha256}.json"
        )
        atomic_write_json(
            inconclusive_path,
            diagnosis.as_dict(),
            overwrite=False,
        )
        return _route_inconclusive_diagnosis(
            store=store,
            diagnosis=diagnosis,
            targets=targets.get("targets", ()),
            target_rank=target_rank,
        )
    accepted_path = store.root / "analysis" / "accepted_diagnoses" / f"{target_sha256}.json"
    atomic_write_json(
        accepted_path,
        diagnosis.as_dict(),
        overwrite=False,
    )
    compatibility_path = store.root / "analysis" / "accepted_diagnosis.json"
    if not compatibility_path.exists():
        atomic_write_json(compatibility_path, diagnosis.as_dict(), overwrite=False)
    store.transition(CampaignPhase.PROPOSE)
    return diagnosis.as_dict()


def _recover_unadvanced_candidate(store: CampaignStore) -> dict[str, Any] | None:
    rows = store.candidate_ledger.records()
    if not rows:
        return None
    candidate_sha256 = str(rows[-1]["candidate_sha256"])
    if any(
        row.get("candidate_sha256") == candidate_sha256
        for row in _operator_candidate_rejections(store)
    ):
        return None
    decisions = [
        row
        for row in store.gates.records()
        if row.get("candidate_sha256") == candidate_sha256
    ]
    if decisions:
        return None
    shadow_report: dict[str, Any] | None = None
    if store.manifest().protocol_explicit:
        shadow_root = store.root / "analysis" / "candidate-shadow-replay"
        shadow_path = shadow_root / f"{candidate_sha256}.json"
        precommit_path = shadow_root / f"{candidate_sha256}.precommit.json"
        if not shadow_path.is_file() or not precommit_path.is_file():
            raise ValueError("candidate registration has no complete shadow preflight")
        shadow_report = read_json(shadow_path)
        precommit = read_json(precommit_path)
        if (
            precommit.get("candidate_sha256") != candidate_sha256
            or precommit.get("shadow_report_sha256")
            != canonical_sha256(shadow_report)
        ):
            raise ValueError("candidate shadow precommit binding changed")
        _validate_shadow_live_gate_admission(store, shadow_report)
    store.recover_registered_candidate(candidate_sha256)
    store.transition(CampaignPhase.SAME_SEED_GATE)
    bundle = read_json(store.root / "candidates" / candidate_sha256 / "bundle.json")
    return {
        "candidate_sha256": candidate_sha256,
        "bundle": bundle,
        "shadow_replay": shadow_report,
        "recovered_registration": True,
    }


def run_proposal_stage(
    *,
    campaign_root: str | Path,
    tool_catalog: dict[str, Any],
    model: str | None = None,
) -> dict[str, Any]:
    store = CampaignStore(campaign_root)
    lock = store.root / ".harness-private" / ".proposal-stage.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with directory_lock(lock, timeout_s=1800.0):
        return _run_proposal_stage_locked(
            campaign_root=campaign_root,
            tool_catalog=tool_catalog,
            model=model,
        )


def _run_proposal_stage_locked(
    *,
    campaign_root: str | Path,
    tool_catalog: dict[str, Any],
    model: str | None = None,
) -> dict[str, Any]:
    store = CampaignStore(campaign_root)
    manifest = store.manifest()
    state = store.state()
    if CampaignPhase(state["phase"]) != CampaignPhase.PROPOSE:
        raise ValueError("proposal requires propose phase")
    recovered = _recover_unadvanced_candidate(store)
    if recovered is not None:
        return recovered
    state = store.state()
    rows = store.diagnoses.records()
    if not rows:
        raise ValueError("proposal requires an accepted diagnosis")
    diagnosis = _diagnosis(rows[-1])
    provisional = _load_provisional_authorization(store, diagnosis)
    if _diagnosis_is_inconclusive(diagnosis) and provisional is None:
        raise ValueError("proposal cannot use an inconclusive diagnosis")
    policy = _evolution_policy(store)
    rejected_rounds = _rejected_candidates_for_cluster(store, diagnosis.cluster_id)
    rejected_total = _all_rejected_candidates(store)
    if (
        policy["maximum_total_candidate_rounds_explicit"]
        and len(rejected_total) >= policy["maximum_total_candidate_rounds"]
    ):
        raise ValueError("total candidate round limit is exhausted")
    if len(rejected_rounds) >= policy["max_candidate_rounds_per_cluster"]:
        raise ValueError("candidate round limit is exhausted for the active cluster")
    artifact_index = _agent_artifact_index(store)
    refinement_context = _rejected_gate_refinement_context(
        store,
        artifact_index=artifact_index,
    )
    # A provisional hypothesis is reconstructed from immutable evidence. The
    # Diagnoser thread may already be context-exhausted and its uncertainty is
    # intentionally preserved rather than hidden behind a claimed resume.
    # Rejected-candidate refinement is always a fresh logical session; skip
    # validating the previous Stage2 transcript because its session binding is
    # intentionally stale after the failed gate.
    resume_context = (
        {}
        if provisional is not None or refinement_context is not None
        else _latest_stage2_context(store)
    )
    provider_thread_id = resume_context.get("provider_thread_id")
    if not isinstance(provider_thread_id, str) or not provider_thread_id.strip():
        provider_thread_id = None
    session_id = resume_context.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        session_id = None
    # Rejected-candidate refinement must start a fresh provider thread. The
    # previous Stage2 transcript can exceed the provider's 1 MiB request limit
    # even when the bounded refinement payload is small. Immutable evidence and
    # the prior candidate remain in the explicit payload, so no audit context is
    # lost by dropping the oversized conversational history.
    if refinement_context is not None:
        provider_thread_id = None
        session_id = None
        reconstructed = True
    else:
        reconstructed = bool(resume_context.get("reconstructed", False))
    agent_output_root = (
        store.root
        / "agents"
        / f"candidate-{len(store.candidate_ledger.records()) + len(_shadow_candidate_rejections(store)):03d}"
    )
    agent = CodexStageAgent(
        output_root=agent_output_root,
        model=model or manifest.model,
        reasoning_effort=manifest.reasoning_effort,
        artifact_reader=lambda content_id: resolve_agent_artifact(
            store.root, content_id
        ),
        session_id=session_id,
        thread_id=provider_thread_id,
        reconstructed=reconstructed or provider_thread_id is None,
        environment_name=manifest.environment,
    )
    artifact_index = _bounded_refinement_artifact_index(
        artifact_index=artifact_index,
        diagnosis=diagnosis,
        refinement_context=refinement_context,
    )
    # Existing committed Stage2 output must be rehydrated with the broad union
    # so a contract rejection can be recorded from the immutable attempt.  A
    # fresh proposal receives only features observed on action rows, preventing
    # another reset-only privileged predicate from being emitted.
    stage2_root = agent_output_root / "stage2-proposal"
    committed_stage2_output = stage2_root / "output.json"
    observed_features = _observed_critic_features(
        store, require_command_rows=not committed_stage2_output.is_file()
    )
    parent_bundle: CandidateBundle | None = None
    parent_sha256 = state.get("current_bundle_sha256")
    if parent_sha256 is not None:
        from rpent.evolution.campaign import resolve_bundle_file

        parent_bundle = CandidateBundle.from_dict(
            read_json(resolve_bundle_file(store, str(parent_sha256)))
        )
    candidate = agent.propose(
        generation=store.manifest().generation,
        parent_sha256=parent_sha256,
        parent_bundle=parent_bundle,
        diagnosis=diagnosis,
        tool_catalog=tool_catalog,
        artifact_index=artifact_index,
        available_critic_features=observed_features,
        refinement_context=refinement_context,
        provisional_hypothesis=provisional,
    )
    allowed = _frozen_tool_names(tool_catalog)
    selected = {step.tool for rule in candidate.recovery_rules for step in rule.steps}
    unknown = selected - allowed
    if unknown:
        raise ValueError(
            f"candidate uses tools outside the frozen task binding: {sorted(unknown)}"
        )
    shadow_path = (
        store.root / "analysis" / "candidate-shadow-replay" / f"{candidate.sha256}.json"
    )
    if shadow_path.is_file():
        # Resume a candidate whose immutable replay was rejected by the default
        # zero-FP policy after an operator has added a timeboxed override.
        precommit_path = shadow_path.with_suffix(".precommit.json")
        if not precommit_path.is_file():
            raise ValueError("candidate shadow replay has no precommit binding")
        shadow_report = read_json(shadow_path)
        precommit = read_json(precommit_path)
        if (
            shadow_report.get("candidate_sha256") != candidate.sha256
            or precommit.get("candidate_sha256") != candidate.sha256
            or precommit.get("parent_bundle_sha256") != candidate.parent_sha256
            or precommit.get("shadow_report_sha256") != canonical_sha256(shadow_report)
        ):
            raise ValueError("candidate shadow precommit binding changed")
        _validate_shadow_live_gate_admission(store, shadow_report)
        sha256 = store.register_candidate(candidate)
        store.transition(CampaignPhase.SAME_SEED_GATE)
        return {
            "candidate_sha256": sha256,
            "bundle": candidate.as_dict(),
            "shadow_replay": shadow_report,
            "authorized_shadow_replay": True,
        }
    cluster_report = load_accepted_cluster_report(store)
    cluster_rows = [
        row
        for row in cluster_report.get("clusters", ())
        if row.get("cluster_id") == diagnosis.cluster_id
    ]
    if len(cluster_rows) == 1:
        target_episode_ids = set(cluster_rows[0].get("episode_ids", ()))
        shadow_target_mode = "accepted_cluster"
    else:
        # Compatibility for a legacy, already-committed Diagnoser row whose
        # textual cluster label predates deterministic cluster IDs. New formal
        # campaigns always take the exact accepted-cluster path above.
        target_episode_ids = {
            str(row.get("episode_id"))
            for row in store.episodes.records()
            if row.get("status") == "valid" and row.get("success") is False
        }
        shadow_target_mode = "legacy_all_failures_fallback"
    target_records: list[tuple[EpisodeRecord, Path]] = []
    success_controls: list[tuple[EpisodeRecord, Path]] = []
    for row in store.episodes.records():
        record = EpisodeRecord.from_dict(row)
        index = row.get("artifact_index")
        states_path = (
            _existing_artifact_path(store, row, index.get("states"))
            if isinstance(index, dict)
            else None
        )
        if states_path is None:
            continue
        if record.episode_id in target_episode_ids:
            target_records.append((record, states_path))
        elif record.success:
            success_controls.append((record, states_path))
    bound_target_ids = {record.episode_id for record, _ in target_records}
    if bound_target_ids != target_episode_ids:
        missing = sorted(target_episode_ids - bound_target_ids)
        raise ValueError(
            "shadow replay is missing immutable state trajectories for target "
            f"episodes: {missing}"
        )
    replay_trajectories = (*target_records, *success_controls)
    feature_contract = _candidate_feature_contract(
        candidate=candidate,
        parent_bundle=parent_bundle,
        trajectories=replay_trajectories,
    )
    if not feature_contract["eligible"]:
        output_path = stage2_root / "output.json"
        context = read_json(stage2_root / "context.json")
        attempt_name = context.get("successful_attempt")
        if isinstance(attempt_name, str) and attempt_name.startswith("attempt-"):
            attempt_output = stage2_root / attempt_name / "output.json"
            if attempt_output.is_file():
                output_path = attempt_output
        rejection = _record_candidate_feature_contract_rejection(
            store=store,
            candidate=candidate,
            candidate_output=output_path,
            contract=feature_contract,
            reason=(
                "candidate critic rule cannot be evaluated on every immutable "
                "shadow trajectory at one state row; generate a rule using only "
                "features co-observed on action rows"
            ),
        )
        return {
            "candidate_sha256": candidate.sha256,
            "bundle": candidate.as_dict(),
            "candidate_rejected": True,
            "rejection": rejection,
            "next_candidate_round": len(
                _rejected_candidates_for_cluster(store, diagnosis.cluster_id)
            )
            + 1,
        }
    shadow_report = evaluate_shadow_replay(
        candidate=candidate,
        parent=parent_bundle,
        target_records=tuple(target_records),
        success_controls=tuple(success_controls),
    )
    shadow_report["target_selection"] = shadow_target_mode
    shadow_report["live_gate_admission"] = _shadow_live_gate_admission(
        store, shadow_report
    )
    detection_rejected = bool(shadow_report["preflight_conclusive"]) and int(
        shadow_report["target_detected"]
    ) != int(shadow_report["target_count"])
    admission = shadow_report["live_gate_admission"]
    if detection_rejected:
        disposition = "rejected_insufficient_target_recall"
    elif not admission["eligible_for_live_gate"]:
        disposition = "rejected_success_control_false_positive_rate"
    elif not shadow_report["preflight_conclusive"]:
        disposition = (
            "inconclusive_detection_with_relaxed_false_positive_override"
            if admission["relaxed_override"]
            else "inconclusive_without_success_control_online_gate_required"
        )
    elif admission["relaxed_override"]:
        disposition = "detection_supported_under_relaxed_false_positive_override"
    else:
        disposition = "detection_supported_proceed_to_online_gate"
    shadow_report["preflight_disposition"] = disposition
    shadow_report.pop("report_sha256", None)
    shadow_report["report_sha256"] = canonical_sha256(shadow_report)
    if shadow_path.is_file():
        if canonical_sha256(read_json(shadow_path)) != canonical_sha256(shadow_report):
            raise ValueError("candidate shadow report changed during recovery")
    else:
        atomic_write_json(shadow_path, shadow_report, overwrite=False)
    precommit = {
        "schema_version": 1,
        "candidate_sha256": candidate.sha256,
        "parent_bundle_sha256": candidate.parent_sha256,
        "shadow_report_sha256": canonical_sha256(shadow_report),
        "target_trajectory_sha256": sorted(
            row["trajectory_sha256"]
            for row in shadow_report["outcomes"]
            if row["role"] == "target_failure"
        ),
    }
    precommit_path = shadow_path.with_suffix(".precommit.json")
    if precommit_path.is_file():
        if canonical_sha256(read_json(precommit_path)) != canonical_sha256(precommit):
            raise ValueError("candidate shadow precommit changed during recovery")
    else:
        atomic_write_json(precommit_path, precommit, overwrite=False)
    if detection_rejected:
        raise ValueError(
            "candidate shadow preflight rejected: conclusive detector has "
            "insufficient target recall"
        )
    _validate_shadow_live_gate_admission(store, shadow_report)
    sha256 = store.register_candidate(candidate)
    store.transition(CampaignPhase.SAME_SEED_GATE)
    return {
        "candidate_sha256": sha256,
        "bundle": candidate.as_dict(),
        "shadow_replay": shadow_report,
    }


def record_gate_and_advance(
    *, campaign_root: str | Path, decision: GateDecision
) -> dict[str, Any]:
    store = CampaignStore(campaign_root)
    store.record_gate(decision)
    policy = _evolution_policy(store)
    state = store.state()
    if decision.kind == "same_seed":
        if decision.passed:
            authorization = _load_provisional_authorization(store)
            if authorization is not None and authorization["skip_regression"]:
                return store.transition_timeboxed_same_seed_to_heldout(
                    authorization_id=str(authorization["authorization_id"]),
                    authorization_sha256=str(
                        store.state()["provisional_authorization_sha256"]
                    ),
                )
            threshold_authorization = load_same_seed_threshold_authorization(
                store, candidate_sha256=decision.candidate_sha256
            )
            if (
                threshold_authorization is not None
                and bool(threshold_authorization.get("skip_regression"))
            ):
                return store.transition(
                    CampaignPhase.HELDOUT_GATE,
                    state_updates={
                        "optimization_outcome": (
                            "same_seed_passed_threshold_override_regression_skipped"
                        ),
                        "same_seed_threshold_authorization_id": threshold_authorization[
                            "authorization_id"
                        ],
                        "same_seed_threshold_authorization_sha256": file_sha256(
                            _same_seed_threshold_authorization_path(
                                store, decision.candidate_sha256
                            )
                        ),
                        "same_seed_threshold_authorization_candidate_sha256": decision.candidate_sha256,
                        "same_seed_threshold_authorization_plan_sha256": threshold_authorization[
                            "plan_sha256"
                        ],
                        "same_seed_threshold_authorization_skip_regression": True,
                    },
                )
            target = (
                CampaignPhase.HELDOUT_GATE
                if policy["skip_regression_gate"]
                else CampaignPhase.REGRESSION_GATE
            )
            state_updates: dict[str, Any] = {
                "optimization_outcome": (
                    "same_seed_passed_regression_skipped"
                    if policy["skip_regression_gate"]
                    else "same_seed_candidate_accepted"
                ),
            }
        else:
            rounds = int(state.get("same_seed_gate_rounds", 0)) + 1
            if (
                policy["same_seed_max_rounds_explicit"]
                and rounds >= policy["same_seed_max_rounds"]
            ):
                target = CampaignPhase.COMPLETE
                state_updates = {
                    "same_seed_gate_rounds": rounds,
                    "optimization_outcome": "same_seed_gate_iteration_budget_exhausted",
                }
            else:
                target, state_updates = _advance_after_candidate_rejection(
                    store, decision.candidate_sha256
                )
                state_updates["same_seed_gate_rounds"] = rounds
    elif decision.kind == "regression":
        if decision.passed:
            state_updates = {}
            target = CampaignPhase.HELDOUT_GATE
        else:
            target, state_updates = _advance_after_candidate_rejection(
                store, decision.candidate_sha256
            )
    else:
        heldout_rounds = int(state.get("heldout_gate_rounds", 0)) + 1
        state_updates = {"heldout_gate_rounds": heldout_rounds}
        if (
            policy["heldout_mode"] == "test"
            and decision.kind == "heldout_10"
            and not decision.conclusive
        ):
            # The legacy 10->50 held-out protocol still needs its second
            # measurement stage in report-only mode.
            target = CampaignPhase.HELDOUT_GATE
        elif policy["heldout_mode"] == "test":
            # The fixed block is an unbiased report-only measurement. It is
            # mandatory to execute, but never feeds candidate selection or
            # promotion in test mode.
            target = CampaignPhase.PROMOTE
            state_updates["optimization_outcome"] = "heldout_test_recorded"
        elif decision.passed:
            target = CampaignPhase.PROMOTE
        elif decision.kind == "heldout_10" and not decision.conclusive:
            target = CampaignPhase.HELDOUT_GATE
        elif (
            policy["heldout_max_rounds_explicit"]
            and heldout_rounds >= policy["heldout_max_rounds"]
        ):
            target = CampaignPhase.COMPLETE
            state_updates["optimization_outcome"] = (
                "heldout_validation_iteration_budget_exhausted"
            )
        else:
            target, rejection_updates = _advance_after_candidate_rejection(
                store, decision.candidate_sha256
            )
            state_updates.update(rejection_updates)
    if target != CampaignPhase(store.state()["phase"]):
        store.transition(target, state_updates=state_updates)
    elif state_updates:
        store.update_state(**state_updates)
    return store.state()


def promote_and_complete(*, campaign_root: str | Path) -> dict[str, Any]:
    store = CampaignStore(campaign_root)
    state = store.state()
    candidate = state.get("candidate_sha256")
    if not candidate:
        raise ValueError("no candidate is ready for promotion")
    promotion = store.promote(candidate)
    store.transition(CampaignPhase.COMPLETE)
    return promotion
