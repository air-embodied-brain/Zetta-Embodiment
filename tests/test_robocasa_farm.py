# Copyright (c) 2026 RPent Contributors
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from rpent.evolution.jsonio import atomic_write_json
from scripts.evolution.serve_robocasa_farm import (
    _bound_session_is_orphaned,
    _health_restart_due,
    _lease_is_live,
    _public_health,
    env_server_command,
)


def test_direct_bound_session_without_observed_lease_is_not_orphaned() -> None:
    assert not _bound_session_is_orphaned(
        phase="EPISODE_ACTIVE",
        lease_observed=False,
        lease_live=False,
    )


def test_expired_observed_worker_lease_marks_bound_session_orphaned() -> None:
    assert _bound_session_is_orphaned(
        phase="EPISODE_ACTIVE",
        lease_observed=True,
        lease_live=False,
    )


def test_live_worker_lease_and_free_slot_are_not_orphaned() -> None:
    assert not _bound_session_is_orphaned(
        phase="EPISODE_ACTIVE",
        lease_observed=True,
        lease_live=True,
    )
    assert not _bound_session_is_orphaned(
        phase="FREE",
        lease_observed=True,
        lease_live=False,
    )


def test_health_restart_grace_never_kills_a_live_leased_reset() -> None:
    assert not _health_restart_due(
        first_failure_at=10.0,
        now=1000.0,
        grace_s=300.0,
        lease_live=True,
    )
    assert not _health_restart_due(
        first_failure_at=10.0,
        now=309.9,
        grace_s=300.0,
        lease_live=False,
    )
    assert _health_restart_due(
        first_failure_at=10.0,
        now=310.0,
        grace_s=300.0,
        lease_live=False,
    )


def test_farm_passes_gpu_gate_and_bounded_http_contract_to_every_slot(
    tmp_path: Path,
) -> None:
    command = env_server_command(
        python="python",
        port=18807,
        camera_size=128,
        max_steps=1000,
        root=tmp_path,
        gpu="7",
        gpu_operation_slots=2,
        maximum_inflight_requests=3,
    )
    joined = " ".join(command)
    assert "--operation-gate-root" in command
    assert str(tmp_path / "operation-gates") in command
    assert "--operation-gate-gpu 7" in joined
    assert "--operation-gate-slots 2" in joined
    assert "--maximum-inflight-requests 3" in joined
    assert "--cold-reset-lock" in command


def test_farm_only_trusts_complete_unexpired_broker_lease(tmp_path: Path) -> None:
    lease = tmp_path / "lease.json"
    assert not _lease_is_live(lease)
    atomic_write_json(
        lease,
        {
            "lease_id": "lease-a",
            "expires_at": (
                datetime.now(timezone.utc) + timedelta(minutes=1)
            ).isoformat(),
        },
    )
    assert _lease_is_live(lease)
    payload = {
        "lease_id": "lease-a",
        "expires_at": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
    }
    atomic_write_json(lease, payload, overwrite=True)
    assert not _lease_is_live(lease)


def test_ready_manifest_health_never_persists_write_capability() -> None:
    public = _public_health(
        {
            "status": "healthy",
            "persistent": True,
            "renderer": {"ready": True},
            "gpu_visible": "2",
            "egl_device": "2",
            "write_protocol": {
                "binding_token": "must-not-leak",
                "generation": 4,
                "phase": "FREE",
                "lost": False,
                "next_operation_seq": 7,
                "cached_results": 12,
            },
        }
    )
    assert public["write_protocol"] == {
        "generation": 4,
        "phase": "FREE",
        "lost": False,
    }
    assert "must-not-leak" not in str(public)
