# Copyright (c) 2026 Zetta Contributors
from __future__ import annotations

import json
from pathlib import Path

from scripts.evolution import audit_gate_artifacts as audit_module
from scripts.evolution.audit_gate_artifacts import (
    audit_pairs,
    audit_role1_privacy,
    audit_videos,
    discover_attempts,
    materialize_catalog,
)


def _attempt(root: Path, *, seed: int, arm: str, success: bool) -> dict[str, object]:
    attempt = root / "candidates" / ("a" * 64) / "gates" / "heldout_20" / "attempts" / f"pair-{seed:03d}-{arm}" / "attempt-000"
    videos = attempt / "videos"
    videos.mkdir(parents=True)
    record = {
        "logical_id": f"pair-{seed:03d}-{arm}",
        "seed": seed,
        "policy_rng": seed + 100,
        "status": "valid",
        "success": success,
        "bundle_sha256": "a" * 64 if arm == "candidate" else None,
        "safety_events": [],
        "artifact_index": {
            "candidate_intervention": arm == "candidate" and success,
            "trajectory_index": {
                "artifact_paths": {
                    "video-00": str(videos / "episode_agentview.mp4")
                }
            },
            "initial_observation_identity": {
                "state_sha256": f"state-{seed}",
                "camera_sha256": {"main": f"main-{seed}", "wrist": f"wrist-{seed}"},
            },
        },
    }
    (attempt / "episode_record.json").write_text(json.dumps(record), encoding="utf-8")
    index = {
        "videos": [
            {"camera": "agentview", "path": str(videos / "episode_agentview.mp4")},
            {"camera": "wrist", "path": str(videos / "episode_wrist.mp4")},
        ],
        "related_artifacts": {},
    }
    (videos / "VIDEO_INDEX.json").write_text(json.dumps(index), encoding="utf-8")
    return record


def test_pair_audit_and_human_readable_catalog(tmp_path: Path) -> None:
    _attempt(tmp_path, seed=3, arm="parent", success=False)
    _attempt(tmp_path, seed=3, arm="candidate", success=True)

    attempts = discover_attempts(tmp_path)
    audit = audit_pairs(attempts)
    catalog = materialize_catalog(tmp_path, attempts, audit, mode="manifest")

    assert audit["valid_pairs"] == 1
    assert audit["pairs"][0]["candidate_intervention"] is True
    assert catalog["entries"] == 2
    readme = (tmp_path / "video-catalog" / "README.md").read_text(encoding="utf-8")
    assert "| 3 | 103 | failure | success | yes |" in readme
    assert (tmp_path / "video-catalog" / "seed-0003__policy-103" / "parent" / "INDEX.json").is_file()


def test_pair_audit_reports_reset_and_camera_mismatch(tmp_path: Path) -> None:
    _attempt(tmp_path, seed=7, arm="parent", success=False)
    _attempt(tmp_path, seed=7, arm="candidate", success=True)
    candidate = next(tmp_path.rglob("pair-007-candidate/attempt-000/episode_record.json"))
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    identity = payload["artifact_index"]["initial_observation_identity"]
    identity["state_sha256"] = "different"
    identity["camera_sha256"] = {"main": "different"}
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    audit = audit_pairs(discover_attempts(tmp_path))

    assert audit["valid_pairs"] == 0
    assert audit["state_mismatches"] == [7]
    assert audit["camera_mismatches"] == [7]


def test_catalog_root_isolated_per_gate(tmp_path: Path) -> None:
    _attempt(tmp_path, seed=21, arm="parent", success=False)
    _attempt(tmp_path, seed=21, arm="candidate", success=True)
    attempts = discover_attempts(tmp_path)
    audit = audit_pairs(attempts)
    catalog_root = tmp_path / "video-catalogs" / "heldout-20"
    catalog = materialize_catalog(
        tmp_path,
        attempts,
        audit,
        mode="manifest",
        catalog_root=catalog_root,
    )

    assert catalog["root"] == str(catalog_root)
    assert (catalog_root / "README.md").is_file()
    assert (catalog_root / "seed-0021__policy-121" / "candidate" / "INDEX.json").is_file()
    assert not (tmp_path / "video-catalog" / "seed-0021__policy-121").exists()


def test_static_divergence_clip_remains_readable(monkeypatch) -> None:
    attempt = {
        "arm": "candidate",
        "record": {"seed": 3, "policy_rng": 103},
    }
    monkeypatch.setattr(
        audit_module,
        "indexed_videos",
        lambda _: [{"kind": "divergence", "path": Path("static.mp4")}],
    )
    monkeypatch.setattr(
        audit_module,
        "probe_video",
        lambda path, **_: {
            "path": str(path),
            "width": 512,
            "height": 288,
            "decodable": True,
            "nonblack": True,
            "nonconstant": False,
            "readable": False,
        },
    )

    audit = audit_videos([attempt], ffprobe="ffprobe", ffmpeg="ffmpeg")

    assert audit["video_count"] == 1
    assert audit["nonconstant"] == 0
    assert audit["readable"] == 1


def test_role1_privacy_audit_matches_runtime_geometry_filter(tmp_path: Path) -> None:
    attempt_dir = tmp_path / "attempt"
    invocation = attempt_dir / "role1" / "invocations" / "invocation-1"
    actor = attempt_dir / "role1" / "actor"
    invocation.mkdir(parents=True)
    actor.mkdir(parents=True)
    input_path = invocation / "input.json"
    input_path.write_text(
        json.dumps(
            {
                "user_payload": {
                    "critic_observations": {
                        "fields": {"privileged.task.success": False}
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (actor / "event.json").write_text(
        json.dumps({"event": {}, "result": {"grasp_verified": True}}),
        encoding="utf-8",
    )
    attempt = {"attempt_dir": attempt_dir}

    clean = audit_role1_privacy([attempt])

    assert clean["input_files"] == 1
    assert clean["actor_audit_files"] == 1
    assert clean["violations"] == []

    input_path.write_text(
        json.dumps({"user_payload": {"object_pose": [0.1, 0.2, 0.3]}}),
        encoding="utf-8",
    )
    exposed = audit_role1_privacy([attempt])
    assert exposed["violations"][0]["private_paths"] == ["object_pose"]


def test_gate_ledger_scopes_catalog_to_one_candidate_round(tmp_path: Path) -> None:
    records = [
        _attempt(tmp_path, seed=11, arm="parent", success=False),
        _attempt(tmp_path, seed=11, arm="candidate", success=True),
    ]
    ledger = tmp_path / "candidate" / "gates" / "same_seed" / "ledgers" / "valid.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"campaign_id": "demo", "code_commit": "deadbeef"}),
        encoding="utf-8",
    )
    (ledger.parent.parent / "plan.json").write_text("{\"plan\": true}", encoding="utf-8")

    attempts = discover_attempts(tmp_path, records_ledger=ledger)
    audit = audit_pairs(attempts)
    catalog = materialize_catalog(tmp_path, attempts, audit, mode="manifest")

    assert audit["valid_pairs"] == 1
    assert catalog["entries"] == 2
    assert (
        tmp_path
        / "video-catalog"
        / "seed-0011__policy-111"
        / "candidate"
        / "INDEX.json"
    ).is_file()
    provenance = json.loads(
        (tmp_path / "video-catalog" / "PROVENANCE.json").read_text(encoding="utf-8")
    )
    assert provenance["source_code_commit"] == "deadbeef"
    catalog_payload = json.loads(
        (tmp_path / "video-catalog" / "CATALOG.json").read_text(encoding="utf-8")
    )
    assert catalog_payload["provenance"] == provenance
    candidate = next(
        entry for entry in catalog_payload["entries"] if entry["arm"] == "candidate"
    )
    assert candidate["bundle_sha256"] == "a" * 64
