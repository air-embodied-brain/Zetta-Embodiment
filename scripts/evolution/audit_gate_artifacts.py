#!/usr/bin/env python3
# Copyright (c) 2026 RPent Contributors
"""Audit paired gate artifacts and build a human-readable video catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _arm(logical_id: str) -> str:
    arm = logical_id.rsplit("-", 1)[-1]
    if arm not in {"parent", "candidate"}:
        raise ValueError(f"cannot infer paired arm from logical_id={logical_id!r}")
    return arm


def _is_private_geometry_field(key: object) -> bool:
    """Match the Role1 runtime's absolute-geometry privacy predicate."""

    normalized = str(key).strip().lower().replace("-", "_")
    segments = normalized.replace(".", "_").split("_")
    return bool(
        {"position", "orientation", "pose", "xyz", "quat", "quaternion", "pos"}
        & set(segments)
    ) or "target_offset" in normalized


def _private_geometry_paths(value: Any, *, path: str = "") -> list[str]:
    result: list[str] = []
    if isinstance(value, dict):
        for raw_key, item in value.items():
            key = str(raw_key)
            child = f"{path}.{key}" if path else key
            if _is_private_geometry_field(key):
                result.append(child)
            if (
                key in {"feature", "required_feature"}
                and isinstance(item, str)
                and _is_private_geometry_field(item)
            ):
                result.append(f"{child}={item}")
            result.extend(_private_geometry_paths(item, path=child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.extend(_private_geometry_paths(item, path=f"{path}[{index}]"))
    return result


def _record_attempt(record: dict[str, Any], *, source: Path) -> dict[str, Any]:
    artifact_index = record.get("artifact_index", {})
    trajectory = artifact_index.get("trajectory_index", {})
    artifact_paths = trajectory.get("artifact_paths", {})
    video_path = artifact_paths.get("video-00")
    attempt_dir = Path(video_path).parent.parent if video_path else source.parent
    index_path = attempt_dir / "videos" / "VIDEO_INDEX.json"
    if not index_path.is_file():
        visual = artifact_index.get("visual_evidence", {})
        visual_artifacts = visual.get("artifacts", {}) if isinstance(visual, dict) else {}
        candidate_index = visual_artifacts.get("video_index")
        if candidate_index:
            index_path = Path(candidate_index)
    return {
        "arm": _arm(str(record.get("logical_id", ""))),
        "attempt_dir": attempt_dir,
        "record_path": source,
        "record": record,
        "video_index_path": index_path,
        "video_index": _read_json(index_path) if index_path.is_file() else None,
    }


def discover_attempts(
    campaign_root: Path, *, records_ledger: Path | None = None
) -> list[dict[str, Any]]:
    if records_ledger is not None:
        attempts: list[dict[str, Any]] = []
        for line in records_ledger.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            logical_id = str(record.get("logical_id", ""))
            if logical_id.endswith(("-parent", "-candidate")):
                attempts.append(_record_attempt(record, source=records_ledger))
        return attempts
    attempts: list[dict[str, Any]] = []
    for record_path in sorted(campaign_root.rglob("episode_record.json")):
        record = _read_json(record_path)
        logical_id = str(record.get("logical_id", ""))
        if not logical_id.endswith(("-parent", "-candidate")):
            continue
        index_path = record_path.parent / "videos" / "VIDEO_INDEX.json"
        attempts.append(
            {
                "arm": _arm(logical_id),
                "attempt_dir": record_path.parent,
                "record_path": record_path,
                "record": record,
                "video_index_path": index_path,
                "video_index": _read_json(index_path) if index_path.is_file() else None,
            }
        )
    return attempts


def _initial_identity(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("artifact_index", {}).get("initial_observation_identity")
    return value if isinstance(value, dict) else {}


def audit_pairs(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    duplicate_arms: list[dict[str, Any]] = []
    for attempt in attempts:
        seed = int(attempt["record"]["seed"])
        arm = str(attempt["arm"])
        if arm in grouped[seed]:
            duplicate_arms.append({"seed": seed, "arm": arm})
        grouped[seed][arm] = attempt

    rows: list[dict[str, Any]] = []
    for seed, arms in sorted(grouped.items()):
        parent = arms.get("parent")
        candidate = arms.get("candidate")
        row: dict[str, Any] = {
            "seed": seed,
            "arms_present": sorted(arms),
            "complete": parent is not None and candidate is not None,
        }
        if parent is None or candidate is None:
            rows.append(row)
            continue
        parent_record = parent["record"]
        candidate_record = candidate["record"]
        parent_identity = _initial_identity(parent_record)
        candidate_identity = _initial_identity(candidate_record)
        row.update(
            {
                "policy_rng": candidate_record.get("policy_rng"),
                "policy_rng_match": candidate_record.get("policy_rng")
                == parent_record.get("policy_rng"),
                "status": {
                    "parent": parent_record.get("status"),
                    "candidate": candidate_record.get("status"),
                },
                "success": {
                    "parent": parent_record.get("success"),
                    "candidate": candidate_record.get("success"),
                },
                "bundle_sha256": {
                    "parent": parent_record.get("bundle_sha256"),
                    "candidate": candidate_record.get("bundle_sha256"),
                },
                "state_sha256_match": parent_identity.get("state_sha256")
                == candidate_identity.get("state_sha256")
                and bool(parent_identity.get("state_sha256")),
                "camera_sha256_match": parent_identity.get("camera_sha256")
                == candidate_identity.get("camera_sha256")
                and bool(parent_identity.get("camera_sha256")),
                "safety_events": {
                    "parent": len(parent_record.get("safety_events", [])),
                    "candidate": len(candidate_record.get("safety_events", [])),
                },
                "candidate_intervention": bool(
                    candidate_record.get("artifact_index", {}).get(
                        "candidate_intervention", False
                    )
                ),
                "attempt_dirs": {
                    "parent": str(parent["attempt_dir"]),
                    "candidate": str(candidate["attempt_dir"]),
                },
            }
        )
        row["valid"] = all(
            (
                row["policy_rng_match"],
                row["state_sha256_match"],
                row["camera_sha256_match"],
                row["status"]["parent"] == "valid",
                row["status"]["candidate"] == "valid",
            )
        )
        rows.append(row)

    return {
        "pair_count": len(rows),
        "complete_pairs": sum(bool(row["complete"]) for row in rows),
        "valid_pairs": sum(bool(row.get("valid")) for row in rows),
        "duplicate_arms": duplicate_arms,
        "state_mismatches": [
            row["seed"] for row in rows if row["complete"] and not row["state_sha256_match"]
        ],
        "camera_mismatches": [
            row["seed"] for row in rows if row["complete"] and not row["camera_sha256_match"]
        ],
        "policy_rng_mismatches": [
            row["seed"] for row in rows if row["complete"] and not row["policy_rng_match"]
        ],
        "safety_event_count": sum(
            sum(row.get("safety_events", {}).values()) for row in rows
        ),
        "pairs": rows,
    }


def audit_role1_privacy(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    """Verify private geometry never entered Role1-visible persisted payloads."""

    input_files = 0
    actor_audit_files = 0
    violations: list[dict[str, Any]] = []
    for attempt in attempts:
        role1_root = Path(attempt["attempt_dir"]) / "role1"
        for path in sorted(role1_root.glob("invocations/*/input.json")):
            input_files += 1
            payload = _read_json(path)
            hits = _private_geometry_paths(payload.get("user_payload", {}))
            if hits:
                violations.append(
                    {"kind": "input", "path": str(path), "private_paths": hits}
                )
        for path in sorted(role1_root.glob("actor/*.json")):
            actor_audit_files += 1
            payload = _read_json(path)
            hits: list[str] = []
            for field in ("event", "result"):
                hits.extend(
                    _private_geometry_paths(payload.get(field, {}), path=field)
                )
            if hits:
                violations.append(
                    {
                        "kind": "actor_audit",
                        "path": str(path),
                        "private_paths": hits,
                    }
                )
    return {
        "predicate": "libero_role1_private_geometry_v1",
        "input_files": input_files,
        "actor_audit_files": actor_audit_files,
        "violations": violations,
    }


def indexed_videos(attempt: dict[str, Any]) -> list[dict[str, Any]]:
    index = attempt.get("video_index")
    if not isinstance(index, dict):
        return []
    videos: list[dict[str, Any]] = []
    for row in index.get("videos", []):
        if isinstance(row, dict) and row.get("path"):
            videos.append({"kind": str(row.get("camera", "unknown")), "path": Path(row["path"])})
    divergence = index.get("related_artifacts", {}).get("divergence_clip")
    if divergence:
        videos.append({"kind": "divergence", "path": Path(divergence)})
    return videos


def probe_video(
    path: Path, *, ffprobe: str = "ffprobe", ffmpeg: str = "ffmpeg", samples: int = 5
) -> dict[str, Any]:
    probe = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,avg_frame_rate,nb_frames:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if probe.returncode != 0:
        return {"path": str(path), "decodable": False, "error": probe.stderr.strip()}
    metadata = json.loads(probe.stdout)
    streams = metadata.get("streams", [])
    if not streams:
        return {"path": str(path), "decodable": False, "error": "no video stream"}
    stream = streams[0]
    duration = float(metadata.get("format", {}).get("duration") or 0.0)
    fps = max(float(samples) / max(duration, 0.001), 0.01)
    decode = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            f"fps={fps:.9f},scale=64:64:flags=area,format=gray",
            "-frames:v",
            str(samples),
            "-f",
            "rawvideo",
            "-",
        ],
        check=False,
        capture_output=True,
        timeout=60,
    )
    frame_size = 64 * 64
    frames = [
        decode.stdout[offset : offset + frame_size]
        for offset in range(0, len(decode.stdout), frame_size)
        if len(decode.stdout[offset : offset + frame_size]) == frame_size
    ]
    means = [sum(frame) / frame_size for frame in frames]
    spatial_stdev = []
    for frame, mean in zip(frames, means, strict=True):
        spatial_stdev.append(
            math.sqrt(sum((value - mean) ** 2 for value in frame) / frame_size)
        )
    frame_changes = [
        sum(abs(first - second) for first, second in zip(left, right, strict=True))
        / frame_size
        for left, right in zip(frames, frames[1:])
    ]
    decodable = decode.returncode == 0 and bool(frames)
    nonblack = bool(frames) and max(means) > 2.0 and max(spatial_stdev) > 2.0
    nonconstant = len(frames) == 1 or max(frame_changes, default=0.0) > 0.25
    return {
        "path": str(path),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "codec": stream.get("codec_name"),
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "duration_s": duration,
        "declared_frames": stream.get("nb_frames"),
        "sampled_frames": len(frames),
        "sample_mean_luma": means,
        "sample_spatial_stdev": spatial_stdev,
        "maximum_sample_change": max(frame_changes, default=0.0),
        "sample_sha256": hashlib.sha256(b"".join(frames)).hexdigest(),
        "decodable": decodable,
        "nonblack": nonblack,
        "nonconstant": nonconstant,
        "readable": decodable
        and nonblack
        and nonconstant
        and int(stream.get("width") or 0) >= 64
        and int(stream.get("height") or 0) >= 64,
        "decode_error": decode.stderr.decode("utf-8", errors="replace").strip(),
    }


def audit_videos(
    attempts: list[dict[str, Any]], *, ffprobe: str, ffmpeg: str
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    camera_samples: dict[tuple[int, str], dict[str, str]] = defaultdict(dict)
    for attempt in attempts:
        record = attempt["record"]
        for video in indexed_videos(attempt):
            row = {
                "seed": int(record["seed"]),
                "policy_rng": int(record["policy_rng"]),
                "arm": attempt["arm"],
                "kind": video["kind"],
                **probe_video(video["path"], ffprobe=ffprobe, ffmpeg=ffmpeg),
            }
            # A no-divergence diagnostic clip can be intentionally static.
            # Preserve ``nonconstant`` for inspection, but do not reject an
            # otherwise valid clip solely because the rollout never diverged.
            if (
                row["kind"] == "divergence"
                and row.get("decodable")
                and row.get("nonblack")
                and int(row.get("width") or 0) >= 64
                and int(row.get("height") or 0) >= 64
            ):
                row["readable"] = True
            rows.append(row)
            if row["kind"] in {"agentview", "wrist"}:
                camera_samples[(row["seed"], row["arm"])][row["kind"]] = row.get(
                    "sample_sha256", ""
                )
    indistinct_views = [
        {"seed": seed, "arm": arm}
        for (seed, arm), values in sorted(camera_samples.items())
        if values.get("agentview") and values.get("agentview") == values.get("wrist")
    ]
    return {
        "video_count": len(rows),
        "decodable": sum(bool(row.get("decodable")) for row in rows),
        "nonblack": sum(bool(row.get("nonblack")) for row in rows),
        "nonconstant": sum(bool(row.get("nonconstant")) for row in rows),
        "readable": sum(bool(row.get("readable")) for row in rows),
        "indistinct_agentview_wrist": indistinct_views,
        "videos": rows,
    }


def _catalog_link(link: Path, target: Path, *, mode: str) -> None:
    if mode == "manifest":
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    relative = os.path.relpath(target, link.parent)
    if link.is_symlink():
        if os.readlink(link) != relative:
            raise ValueError(f"catalog link already targets a different artifact: {link}")
        return
    if link.exists():
        raise ValueError(f"catalog entry already exists and is not a symlink: {link}")
    link.symlink_to(relative)


def materialize_catalog(
    campaign_root: Path,
    attempts: list[dict[str, Any]],
    pair_audit: dict[str, Any],
    *,
    mode: str,
    catalog_root: Path | None = None,
) -> dict[str, Any]:
    root = catalog_root or (campaign_root / "video-catalog")
    root.mkdir(parents=True, exist_ok=True)
    provenance: dict[str, Any] = {}
    manifest_path = campaign_root / "manifest.json"
    if manifest_path.is_file():
        manifest = _read_json(manifest_path)
        provenance.update(
            {
                "campaign_id": manifest.get("campaign_id"),
                "source_code_commit": manifest.get("code_commit"),
                "manifest_file_sha256": hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest(),
            }
        )
    if attempts:
        records_path = Path(attempts[0]["record_path"])
        gate_dir = records_path.parent.parent
        plan_path = gate_dir / "plan.json"
        if plan_path.is_file():
            provenance["gate_kind"] = gate_dir.name
            provenance["gate_plan_file_sha256"] = hashlib.sha256(
                plan_path.read_bytes()
            ).hexdigest()
    if provenance:
        _atomic_json(root / "PROVENANCE.json", provenance)
    entries: list[dict[str, Any]] = []
    for attempt in sorted(
        attempts, key=lambda value: (int(value["record"]["seed"]), value["arm"])
    ):
        record = attempt["record"]
        seed = int(record["seed"])
        policy_rng = int(record["policy_rng"])
        arm = str(attempt["arm"])
        arm_root = root / f"seed-{seed:04d}__policy-{policy_rng}" / arm
        arm_root.mkdir(parents=True, exist_ok=True)
        linked: dict[str, str] = {}
        for video in indexed_videos(attempt):
            link = arm_root / f"{video['kind']}.mp4"
            _catalog_link(link, video["path"], mode=mode)
            linked[str(video["kind"])] = os.path.relpath(video["path"], arm_root)
        entry = {
            "seed": seed,
            "policy_rng": policy_rng,
            "arm": arm,
            "bundle_sha256": record.get("bundle_sha256"),
            "status": record.get("status"),
            "success": record.get("success"),
            "candidate_intervention": bool(
                record.get("artifact_index", {}).get("candidate_intervention", False)
            ),
            "source_attempt": str(attempt["attempt_dir"]),
            "videos": linked,
        }
        _atomic_json(arm_root / "INDEX.json", entry)
        entries.append(entry)

    lines = [
        "# Paired gate video catalog",
        "",
        "Each seed directory contains parent and candidate views. Files are relative",
        "symbolic links; original rollout artifacts are never moved or copied.",
        "",
    ]
    if provenance:
        lines.extend(
            [
                "## Immutable provenance",
                "",
                f"- Campaign: `{provenance.get('campaign_id', '')}`",
                f"- Source code commit: `{provenance.get('source_code_commit', '')}`",
                f"- Manifest file SHA-256: `{provenance.get('manifest_file_sha256', '')}`",
                f"- Gate kind: `{provenance.get('gate_kind', '')}`",
                f"- Gate plan file SHA-256: `{provenance.get('gate_plan_file_sha256', '')}`",
                "",
            ]
        )
    lines.extend(
        [
            "| Seed | Policy RNG | Parent | Candidate | Intervention |",
            "| ---: | ---: | :---: | :---: | :---: |",
        ]
    )
    for pair in pair_audit["pairs"]:
        if not pair.get("complete"):
            continue
        lines.append(
            f"| {pair['seed']} | {pair['policy_rng']} | "
            f"{'success' if pair['success']['parent'] else 'failure'} | "
            f"{'success' if pair['success']['candidate'] else 'failure'} | "
            f"{'yes' if pair['candidate_intervention'] else 'no'} |"
        )
    (root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _atomic_json(
        root / "CATALOG.json",
        {"schema_version": 1, "provenance": provenance, "entries": entries},
    )
    return {
        "root": str(root),
        "entries": len(entries),
        "mode": mode,
        "provenance": provenance,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--records-ledger",
        type=Path,
        help="Audit exactly the paired records in one gate valid.jsonl ledger.",
    )
    parser.add_argument("--catalog-mode", choices=("symlink", "manifest"), default="symlink")
    parser.add_argument(
        "--catalog-root",
        type=Path,
        help="Write this gate's human-readable catalog to an isolated root.",
    )
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()

    campaign_root = args.campaign_root.resolve()
    attempts = discover_attempts(campaign_root, records_ledger=args.records_ledger)
    pairs = audit_pairs(attempts)
    role1_privacy = audit_role1_privacy(attempts)
    videos = audit_videos(attempts, ffprobe=args.ffprobe, ffmpeg=args.ffmpeg)
    catalog = materialize_catalog(
        campaign_root,
        attempts,
        pairs,
        mode=args.catalog_mode,
        catalog_root=args.catalog_root.resolve() if args.catalog_root else None,
    )
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "campaign_root": str(campaign_root),
        "attempt_count": len(attempts),
        "pairs": pairs,
        "role1_privacy": role1_privacy,
        "video_audit": videos,
        "catalog": catalog,
    }
    output = args.output or campaign_root / "audits" / "paired-gate-artifacts.json"
    _atomic_json(output, report)
    passed = (
        pairs["pair_count"] == pairs["complete_pairs"] == pairs["valid_pairs"]
        and not pairs["duplicate_arms"]
        and not role1_privacy["violations"]
        and videos["video_count"] == videos["readable"]
        and not videos["indistinct_agentview_wrist"]
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "passed": passed,
                "pairs": pairs["valid_pairs"],
                "videos": videos["video_count"],
                "readable_videos": videos["readable"],
                "role1_input_files": role1_privacy["input_files"],
                "privacy_violations": len(role1_privacy["violations"]),
                "catalog": catalog,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
