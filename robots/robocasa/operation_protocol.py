# Copyright (c) 2026 RPent Contributors
"""Idempotent write envelope for a persistent RoboCasa environment.

Transport timeout is not cancellation: a physical action may already have
executed.  Every write therefore carries a process-generation binding, episode
identity, monotonic operation sequence and payload digest.  Duplicate requests
with the same digest return the cached terminal result; conflicting or stale
requests fail closed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any

_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:-]{8,128}$")


def payload_sha256(payload: dict[str, Any]) -> str:
    """Digest only the semantic operation payload, excluding its envelope."""

    semantic = {key: value for key, value in payload.items() if key != "_operation"}
    encoded = json.dumps(
        semantic, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class WriteEnvelope:
    request_id: str
    session_id: str
    binding_token: str
    episode_id: str
    operation_seq: int
    payload_sha256: str

    @classmethod
    def parse(cls, payload: dict[str, Any]) -> WriteEnvelope:
        raw = payload.get("_operation")
        if not isinstance(raw, dict):
            raise ValueError("write request requires an _operation envelope")
        values = {
            name: str(raw.get(name, ""))
            for name in (
                "request_id",
                "session_id",
                "binding_token",
                "episode_id",
                "payload_sha256",
            )
        }
        for name in ("request_id", "session_id", "binding_token", "episode_id"):
            if not _IDENTIFIER.fullmatch(values[name]):
                raise ValueError(f"invalid operation {name}")
        if not re.fullmatch(r"[0-9a-f]{64}", values["payload_sha256"]):
            raise ValueError("invalid operation payload_sha256")
        sequence = raw.get("operation_seq")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ValueError("operation_seq must be a non-negative integer")
        envelope = cls(operation_seq=sequence, **values)
        if envelope.payload_sha256 != payload_sha256(payload):
            raise ValueError("operation payload digest mismatch")
        return envelope

    def public_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "binding_token": self.binding_token,
            "episode_id": self.episode_id,
            "operation_seq": self.operation_seq,
            "payload_sha256": self.payload_sha256,
        }


@dataclass(frozen=True, slots=True)
class WriteResult:
    status: HTTPStatus
    payload: dict[str, Any]


class IdempotentWriteRegistry:
    """Process-generation write registry with bounded terminal-result cache."""

    def __init__(self, *, maximum_cached_results: int = 256) -> None:
        if maximum_cached_results < 1:
            raise ValueError("maximum_cached_results must be positive")
        self.binding_token = uuid.uuid4().hex
        self.maximum_cached_results = maximum_cached_results
        self._lock = threading.RLock()
        self._generation = 0
        self._session_id: str | None = None
        self._episode_id: str | None = None
        self._next_sequence = 0
        self._lost = False
        self._cache: OrderedDict[str, tuple[str, str, WriteResult]] = OrderedDict()

    @property
    def state(self) -> dict[str, Any]:
        with self._lock:
            return {
                "binding_token": self.binding_token,
                "generation": self._generation,
                "phase": (
                    "LOST"
                    if self._lost
                    else "EPISODE_ACTIVE"
                    if self._episode_id is not None
                    else "BOUND"
                    if self._session_id is not None
                    else "FREE"
                ),
                "session_bound": self._session_id is not None,
                "episode_bound": self._episode_id is not None,
                "next_operation_seq": self._next_sequence,
                "lost": self._lost,
                "cached_results": len(self._cache),
            }

    def _terminal(
        self,
        envelope: WriteEnvelope,
        *,
        status: HTTPStatus,
        result: dict[str, Any],
        outcome: str,
        side_effect_applied: bool,
    ) -> WriteResult:
        body = dict(result)
        body["_operation"] = {
            **envelope.public_dict(),
            "outcome": outcome,
            "side_effect_applied": side_effect_applied,
        }
        return WriteResult(status=status, payload=body)

    def _cache_result(
        self, endpoint: str, envelope: WriteEnvelope, result: WriteResult
    ) -> None:
        self._cache[envelope.request_id] = (
            endpoint,
            envelope.payload_sha256,
            copy.deepcopy(result),
        )
        self._cache.move_to_end(envelope.request_id)
        while len(self._cache) > self.maximum_cached_results:
            self._cache.popitem(last=False)

    def execute(
        self,
        endpoint: str,
        payload: dict[str, Any],
        operation: Callable[[], dict[str, Any]],
        *,
        side_effect_marker: Callable[[], Any],
        release_binding: bool = False,
    ) -> WriteResult:
        """Execute one write exactly once for this live process generation."""

        envelope = WriteEnvelope.parse(payload)
        with self._lock:
            cached = self._cache.get(envelope.request_id)
            if cached is not None:
                cached_endpoint, cached_digest, cached_result = cached
                if (
                    cached_endpoint != endpoint
                    or cached_digest != envelope.payload_sha256
                ):
                    raise ValueError("request_id was reused with conflicting content")
                return copy.deepcopy(cached_result)
            if envelope.binding_token != self.binding_token:
                raise ValueError("stale environment binding token")
            if self._lost:
                raise ValueError("environment outcome is unknown; start a new attempt")
            if self._session_id is None:
                self._session_id = envelope.session_id
            elif envelope.session_id != self._session_id:
                raise ValueError("environment is bound to another session")

            is_reset = endpoint == "/reset"
            if release_binding and is_reset:
                raise ValueError("reset cannot release an environment binding")
            if is_reset:
                if envelope.operation_seq != 0:
                    raise ValueError("reset must use operation_seq zero")
                if self._episode_id == envelope.episode_id:
                    raise ValueError(
                        "episode reset already committed with another request"
                    )
            else:
                if self._episode_id != envelope.episode_id:
                    raise ValueError(
                        "operation episode_id does not match active episode"
                    )
                if envelope.operation_seq != self._next_sequence:
                    raise ValueError("operation sequence is stale or out of order")

            before = side_effect_marker()
            try:
                value = operation()
            except Exception as exc:
                # A reset is conservatively treated as having possible side
                # effects once dispatched.  A chunk is known to have written
                # when its marker changes.  Either case poisons this binding:
                # replaying could duplicate physical actions.
                side_effect = is_reset or side_effect_marker() != before
                terminal = self._terminal(
                    envelope,
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                    result={"error": type(exc).__name__, "detail": str(exc)},
                    outcome="OUTCOME_UNKNOWN" if side_effect else "REJECTED",
                    side_effect_applied=side_effect,
                )
                self._cache_result(endpoint, envelope, terminal)
                if side_effect:
                    self._lost = True
                return terminal

            if is_reset:
                self._episode_id = envelope.episode_id
                self._next_sequence = 1
            else:
                self._next_sequence += 1
            if release_binding:
                released_generation = self._generation
                self._generation += 1
                self._session_id = None
                self._episode_id = None
                self._next_sequence = 0
                self._lost = False
                self.binding_token = uuid.uuid4().hex
                value = {
                    **value,
                    "binding_released": True,
                    "released_generation": released_generation,
                }
            terminal = self._terminal(
                envelope,
                status=HTTPStatus.OK,
                result=value,
                outcome="COMMITTED",
                side_effect_applied=True,
            )
            self._cache_result(endpoint, envelope, terminal)
            return terminal
