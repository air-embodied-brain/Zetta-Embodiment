# Copyright (c) 2026 RPent Contributors
from __future__ import annotations

import uuid
from typing import Any

import pytest

from robots.robocasa.operation_protocol import (
    IdempotentWriteRegistry,
    payload_sha256,
)


def _payload(
    registry: IdempotentWriteRegistry,
    *,
    request_id: str,
    episode_id: str,
    sequence: int,
    value: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"value": value}
    payload["_operation"] = {
        "request_id": request_id,
        "session_id": "session-" + "a" * 16,
        "binding_token": registry.binding_token,
        "episode_id": episode_id,
        "operation_seq": sequence,
        "payload_sha256": payload_sha256(payload),
    }
    return payload


def test_duplicate_same_request_returns_cached_result_without_reexecution() -> None:
    registry = IdempotentWriteRegistry()
    episode = "episode-" + uuid.uuid4().hex
    request = "request-" + uuid.uuid4().hex
    payload = _payload(
        registry, request_id=request, episode_id=episode, sequence=0, value=1
    )
    calls = 0

    def operation() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"calls": calls}

    first = registry.execute("/reset", payload, operation, side_effect_marker=lambda: 0)
    second = registry.execute(
        "/reset", payload, operation, side_effect_marker=lambda: 0
    )
    assert calls == 1
    assert second == first
    assert first.payload["_operation"]["outcome"] == "COMMITTED"


def test_same_request_with_changed_payload_fails_closed() -> None:
    registry = IdempotentWriteRegistry()
    episode = "episode-" + uuid.uuid4().hex
    request = "request-" + uuid.uuid4().hex
    first = _payload(
        registry, request_id=request, episode_id=episode, sequence=0, value=1
    )
    registry.execute("/reset", first, lambda: {}, side_effect_marker=lambda: 0)
    conflict = _payload(
        registry, request_id=request, episode_id=episode, sequence=0, value=2
    )
    with pytest.raises(ValueError, match="conflicting"):
        registry.execute("/reset", conflict, lambda: {}, side_effect_marker=lambda: 0)


def test_stale_binding_and_out_of_order_sequence_are_rejected() -> None:
    registry = IdempotentWriteRegistry()
    episode = "episode-" + uuid.uuid4().hex
    reset = _payload(
        registry,
        request_id="request-" + uuid.uuid4().hex,
        episode_id=episode,
        sequence=0,
        value=1,
    )
    reset["_operation"]["binding_token"] = "binding-" + "f" * 16
    with pytest.raises(ValueError, match="stale"):
        registry.execute("/reset", reset, lambda: {}, side_effect_marker=lambda: 0)

    reset["_operation"]["binding_token"] = registry.binding_token
    registry.execute("/reset", reset, lambda: {}, side_effect_marker=lambda: 0)
    step = _payload(
        registry,
        request_id="request-" + uuid.uuid4().hex,
        episode_id=episode,
        sequence=3,
        value=2,
    )
    with pytest.raises(ValueError, match="out of order"):
        registry.execute(
            "/execute_chunk", step, lambda: {}, side_effect_marker=lambda: 0
        )


def test_side_effecting_exception_is_cached_and_binding_becomes_lost() -> None:
    registry = IdempotentWriteRegistry()
    episode = "episode-" + uuid.uuid4().hex
    reset = _payload(
        registry,
        request_id="request-" + uuid.uuid4().hex,
        episode_id=episode,
        sequence=0,
        value=1,
    )
    registry.execute("/reset", reset, lambda: {}, side_effect_marker=lambda: 0)
    step = _payload(
        registry,
        request_id="request-" + uuid.uuid4().hex,
        episode_id=episode,
        sequence=1,
        value=2,
    )
    marker = [0]

    def fail_after_write() -> dict[str, Any]:
        marker[0] = 1
        raise RuntimeError("synthetic worker loss")

    first = registry.execute(
        "/execute_chunk", step, fail_after_write, side_effect_marker=lambda: marker[0]
    )
    second = registry.execute(
        "/execute_chunk", step, fail_after_write, side_effect_marker=lambda: marker[0]
    )
    assert second == first
    assert first.payload["_operation"]["outcome"] == "OUTCOME_UNKNOWN"
    assert first.payload["_operation"]["side_effect_applied"] is True
    assert registry.state["lost"] is True

    following = _payload(
        registry,
        request_id="request-" + uuid.uuid4().hex,
        episode_id=episode,
        sequence=1,
        value=3,
    )
    with pytest.raises(ValueError, match="outcome is unknown"):
        registry.execute(
            "/execute_chunk",
            following,
            lambda: {},
            side_effect_marker=lambda: marker[0],
        )


def test_payload_digest_detects_semantic_mutation() -> None:
    registry = IdempotentWriteRegistry()
    payload = _payload(
        registry,
        request_id="request-" + uuid.uuid4().hex,
        episode_id="episode-" + uuid.uuid4().hex,
        sequence=0,
        value=1,
    )
    payload["value"] = 2
    with pytest.raises(ValueError, match="digest mismatch"):
        registry.execute("/reset", payload, lambda: {}, side_effect_marker=lambda: 0)


def test_release_rotates_generation_and_allows_a_new_session() -> None:
    registry = IdempotentWriteRegistry()
    first_episode = "episode-" + uuid.uuid4().hex
    reset = _payload(
        registry,
        request_id="request-" + uuid.uuid4().hex,
        episode_id=first_episode,
        sequence=0,
        value=1,
    )
    registry.execute("/reset", reset, lambda: {}, side_effect_marker=lambda: 0)
    release = _payload(
        registry,
        request_id="request-" + uuid.uuid4().hex,
        episode_id=first_episode,
        sequence=1,
        value=2,
    )
    old_token = registry.binding_token
    terminal = registry.execute(
        "/release",
        release,
        lambda: {"finalized": True},
        side_effect_marker=lambda: 0,
        release_binding=True,
    )
    assert terminal.payload["binding_released"] is True
    assert registry.binding_token != old_token
    assert registry.state["generation"] == 1
    assert registry.state["phase"] == "FREE"

    second_episode = "episode-" + uuid.uuid4().hex
    second = _payload(
        registry,
        request_id="request-" + uuid.uuid4().hex,
        episode_id=second_episode,
        sequence=0,
        value=3,
    )
    second["_operation"]["session_id"] = "session-" + "b" * 16
    registry.execute("/reset", second, lambda: {}, side_effect_marker=lambda: 0)
    assert registry.state["phase"] == "EPISODE_ACTIVE"


def test_old_binding_cannot_write_after_release_but_release_retry_is_cached() -> None:
    registry = IdempotentWriteRegistry()
    episode = "episode-" + uuid.uuid4().hex
    reset = _payload(
        registry,
        request_id="request-" + uuid.uuid4().hex,
        episode_id=episode,
        sequence=0,
        value=1,
    )
    registry.execute("/reset", reset, lambda: {}, side_effect_marker=lambda: 0)
    release = _payload(
        registry,
        request_id="request-" + uuid.uuid4().hex,
        episode_id=episode,
        sequence=1,
        value=2,
    )
    first = registry.execute(
        "/release",
        release,
        lambda: {"finalized": True},
        side_effect_marker=lambda: 0,
        release_binding=True,
    )
    duplicate = registry.execute(
        "/release",
        release,
        lambda: pytest.fail("duplicate release must not execute"),
        side_effect_marker=lambda: 0,
        release_binding=True,
    )
    assert duplicate == first
    stale = {
        "value": 4,
        "_operation": {
            **release["_operation"],
            "request_id": "request-" + uuid.uuid4().hex,
        },
    }
    stale["_operation"]["payload_sha256"] = payload_sha256(stale)
    with pytest.raises(ValueError, match="stale"):
        registry.execute(
            "/execute_chunk", stale, lambda: {}, side_effect_marker=lambda: 0
        )
