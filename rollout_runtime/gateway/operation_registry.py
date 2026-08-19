"""``request_id`` idempotency table, operation state, and cancellation registration.

Invariants:

- Same ``request_id`` + same request -> return the existing state or cached result;
- Same ``request_id`` + different request -> ``IDEMPOTENCY_CONFLICT``;
- Terminal results enter a TTL cache and are purged after expiry.

Only guarantees effectively-once within the process lifetime; it does not
promise exactly-once under failure scenarios.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Mapping
from typing import Any

from rollout_runtime.api import codec
from rollout_runtime.api.enums import (
    TERMINAL_OPERATION_STATES,
    EnvOperation,
    ErrorCode,
    OperationState,
)
from rollout_runtime.api.errors import (
    InvalidTransition,
    RuntimeApiError,
    RuntimeErrorInfo,
    make_error,
)
from rollout_runtime.api.ids import RequestId, SessionId
from rollout_runtime.api.messages import CancelOutcome, OperationStatus

__all__ = [
    "ALLOWED_OPERATION_TRANSITIONS",
    "DEFAULT_RESULT_TTL_SECONDS",
    "OperationRecord",
    "OperationRegistry",
    "request_digest",
]

DEFAULT_RESULT_TTL_SECONDS = 300.0
"""Result cache duration for terminal operations."""

REQUEST_DIGEST_DOMAIN = "rollout_runtime/request/v1"
"""Domain-separation prefix for the request digest."""

ALLOWED_OPERATION_TRANSITIONS: Mapping[OperationState, frozenset[OperationState]] = {
    OperationState.ACCEPTED: frozenset(
        {
            OperationState.QUEUED,
            OperationState.RUNNING,
            OperationState.SUCCEEDED,
            OperationState.FAILED,
            OperationState.CANCELLED,
            OperationState.OUTCOME_UNKNOWN,
        }
    ),
    OperationState.QUEUED: frozenset(
        {
            OperationState.RUNNING,
            OperationState.SUCCEEDED,
            OperationState.FAILED,
            OperationState.CANCELLED,
            OperationState.OUTCOME_UNKNOWN,
        }
    ),
    OperationState.RUNNING: frozenset(
        {
            OperationState.SUCCEEDED,
            OperationState.FAILED,
            OperationState.CANCELLED,
            OperationState.OUTCOME_UNKNOWN,
        }
    ),
    OperationState.SUCCEEDED: frozenset(),
    OperationState.FAILED: frozenset(),
    OperationState.CANCELLED: frozenset(),
    OperationState.OUTCOME_UNKNOWN: frozenset(),
}
"""operation state machine; terminal states cannot be transitioned again."""


def request_digest(
    operation: EnvOperation, session_id: SessionId | None, payload: Any
) -> str:
    """Compute a request fingerprint, used for idempotency conflict detection.

    Args:
        operation: Operation type.
        session_id: Target session.
        payload: Operation parameters (a protocol object or a native structure).

    Returns:
        64-character hex sha256 digest.
    """
    return codec.digest(
        {
            "operation": operation.name,
            "session_id": session_id,
            "payload": payload,
        },
        prefix=REQUEST_DIGEST_DOMAIN,
    )


@dataclasses.dataclass
class OperationRecord:
    """A registration entry for a single operation.

    Attributes:
        request_id: Request identifier.
        session_id: Target session.
        operation: Operation type.
        digest: Request fingerprint.
        state: Current state.
        created_at: Registration time.
        updated_at: Most recent update time.
        value: Success result (cached at terminal state).
        error: Failure reason.
        side_effect_applied: Whether the environment side effect has occurred.
        cancel_requested: Whether cancellation has been registered.
        worker_rank: The rank it was dispatched to.
        terminal_at: Time it entered a terminal state.
    """

    request_id: RequestId
    session_id: SessionId | None
    operation: EnvOperation
    digest: str
    state: OperationState = OperationState.ACCEPTED
    created_at: float = 0.0
    updated_at: float = 0.0
    value: Any = None
    error: RuntimeErrorInfo | None = None
    side_effect_applied: bool = False
    cancel_requested: bool = False
    worker_rank: int | None = None
    terminal_at: float | None = None

    @property
    def is_terminal(self) -> bool:
        """Whether it has already entered a terminal state.

        Returns:
            True if terminal.
        """
        return self.state in TERMINAL_OPERATION_STATES

    def status(self) -> OperationStatus:
        """Generate the external status.

        Returns:
            ``OperationStatus``.
        """
        return OperationStatus(
            request_id=self.request_id,
            session_id=self.session_id,
            operation=self.operation,
            state=self.state,
            created_at=self.created_at,
            updated_at=self.updated_at,
            side_effect_applied=self.side_effect_applied,
            error=self.error,
        )


class OperationRegistry:
    """Operation idempotency table and short-lived result cache."""

    def __init__(
        self,
        *,
        time_source: Callable[[], float] = time.time,
        result_ttl_seconds: float = DEFAULT_RESULT_TTL_SECONDS,
    ) -> None:
        """Initialize.

        Args:
            time_source: Time source, injectable for testing.
            result_ttl_seconds: Terminal result cache duration.
        """
        self._now = time_source
        self._ttl = result_ttl_seconds
        self._records: dict[RequestId, OperationRecord] = {}

    def __len__(self) -> int:
        """Return the number of registration entries.

        Returns:
            Current number of retained records.
        """
        return len(self._records)

    def find(self, request_id: RequestId) -> OperationRecord | None:
        """Look up a record by id.

        Args:
            request_id: Request identifier.

        Returns:
            The record, or ``None``.
        """
        return self._records.get(request_id)

    def get(self, request_id: RequestId) -> OperationRecord:
        """Look up a record by id, raising if not found.

        Args:
            request_id: Request identifier.

        Returns:
            The record.

        Raises:
            RuntimeApiError: The record does not exist (``INVALID_ARGUMENT``).
        """
        record = self._records.get(request_id)
        if record is None:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"unknown request_id: {request_id}",
                    request_id=request_id,
                )
            )
        return record

    def begin(
        self,
        request_id: RequestId,
        *,
        session_id: SessionId | None,
        operation: EnvOperation,
        payload: Any = None,
        digest: str | None = None,
    ) -> tuple[OperationRecord, bool]:
        """Register an operation, or hit the idempotency cache.

        Args:
            request_id: Request identifier.
            session_id: Target session.
            operation: Operation type.
            payload: Operation parameters, used to compute the fingerprint.
            digest: Explicitly given fingerprint (skips computation).

        Returns:
            ``(record, whether newly created)``. If the existing record is
            terminal, the caller should return its cached result directly.

        Raises:
            RuntimeApiError: Same ``request_id`` maps to a different request
                (``IDEMPOTENCY_CONFLICT``).
        """
        fingerprint = digest or request_digest(operation, session_id, payload)
        existing = self._records.get(request_id)
        if existing is not None:
            if existing.digest != fingerprint:
                raise RuntimeApiError(
                    make_error(
                        ErrorCode.IDEMPOTENCY_CONFLICT,
                        f"request_id {request_id} was already used for a different request",
                        request_id=request_id,
                        recorded_operation=existing.operation.name,
                        recorded_session_id=existing.session_id,
                    )
                )
            return existing, False

        now = self._now()
        record = OperationRecord(
            request_id=request_id,
            session_id=session_id,
            operation=operation,
            digest=fingerprint,
            state=OperationState.ACCEPTED,
            created_at=now,
            updated_at=now,
        )
        self._records[request_id] = record
        return record, True

    def _transition(
        self, request_id: RequestId, to_state: OperationState
    ) -> OperationRecord:
        record = self.get(request_id)
        if to_state not in ALLOWED_OPERATION_TRANSITIONS[record.state]:
            raise InvalidTransition(str(request_id), record.state, to_state)
        record.state = to_state
        record.updated_at = self._now()
        if record.is_terminal:
            record.terminal_at = record.updated_at
        return record

    def mark_queued(self, request_id: RequestId) -> OperationRecord:
        """Mark as queued.

        Args:
            request_id: Request identifier.

        Returns:
            The updated record.
        """
        return self._transition(request_id, OperationState.QUEUED)

    def mark_running(
        self, request_id: RequestId, *, worker_rank: int | None = None
    ) -> OperationRecord:
        """Mark as running.

        Args:
            request_id: Request identifier.
            worker_rank: The rank it was dispatched to.

        Returns:
            The updated record.
        """
        record = self._transition(request_id, OperationState.RUNNING)
        if worker_rank is not None:
            record.worker_rank = worker_rank
        return record

    def succeed(
        self, request_id: RequestId, value: Any, *, side_effect_applied: bool = True
    ) -> OperationRecord:
        """Mark as succeeded and cache the result.

        Args:
            request_id: Request identifier.
            value: Success result.
            side_effect_applied: Whether the environment side effect has occurred.

        Returns:
            The updated record.
        """
        record = self._transition(request_id, OperationState.SUCCEEDED)
        record.value = value
        record.side_effect_applied = side_effect_applied
        return record

    def fail(
        self,
        request_id: RequestId,
        error: RuntimeErrorInfo,
        *,
        side_effect_applied: bool | None = None,
    ) -> OperationRecord:
        """Mark as failed and cache the error.

        Args:
            request_id: Request identifier.
            error: Failure reason.
            side_effect_applied: Override the side-effect flag; ``None`` uses
                the value carried by ``error``.

        Returns:
            The updated record.
        """
        record = self._transition(request_id, OperationState.FAILED)
        record.error = error
        record.side_effect_applied = (
            error.side_effect_applied
            if side_effect_applied is None
            else side_effect_applied
        )
        return record

    def cancel(
        self,
        request_id: RequestId,
        *,
        side_effect_applied: bool = False,
        message: str = "cancelled",
    ) -> OperationRecord:
        """Mark as cancelled.

        Args:
            request_id: Request identifier.
            side_effect_applied: Whether the environment side effect has occurred.
            message: Description.

        Returns:
            The updated record.
        """
        record = self._transition(request_id, OperationState.CANCELLED)
        record.error = make_error(
            ErrorCode.CANCELLED, message, side_effect_applied=side_effect_applied
        )
        record.side_effect_applied = side_effect_applied
        return record

    def mark_outcome_unknown(
        self, request_id: RequestId, message: str = "worker lost"
    ) -> OperationRecord:
        """Mark the outcome as unknown (the worker was lost; must not be replayed automatically).

        Args:
            request_id: Request identifier.
            message: Description.

        Returns:
            The updated record.
        """
        record = self._transition(request_id, OperationState.OUTCOME_UNKNOWN)
        record.error = make_error(ErrorCode.WORKER_LOST, message)
        return record

    def request_cancel(self, request_id: RequestId) -> CancelOutcome:
        """Register a cancellation request (the four states described in the architecture).

        - Not yet dispatched (``ACCEPTED`` / ``QUEUED``) -> cancel directly,
          ``side_effect_applied=false``;
        - Already dispatched (``RUNNING``) -> only register intent; the
          EnvWorker decides whether it can stop before the env step;
        - Already terminal -> return the existing result as-is.

        Args:
            request_id: Request identifier.

        Returns:
            Cancellation outcome.
        """
        record = self.get(request_id)
        if record.is_terminal:
            return CancelOutcome(
                request_id=request_id,
                state=record.state,
                side_effect_applied=record.side_effect_applied,
                message="operation already finished",
            )
        record.cancel_requested = True
        if record.state in (OperationState.ACCEPTED, OperationState.QUEUED):
            self.cancel(request_id, message="cancelled before dispatch")
            return CancelOutcome(
                request_id=request_id,
                state=OperationState.CANCELLED,
                side_effect_applied=False,
                message="cancelled before dispatch",
            )
        return CancelOutcome(
            request_id=request_id,
            state=record.state,
            side_effect_applied=record.side_effect_applied,
            message="cancel forwarded to env worker (best effort)",
        )

    def status(self, request_id: RequestId) -> OperationStatus:
        """Query the operation status.

        Args:
            request_id: Request identifier.

        Returns:
            ``OperationStatus``.
        """
        return self.get(request_id).status()

    def purge(self, now: float | None = None) -> list[RequestId]:
        """Purge terminal records that exceeded the TTL.

        Args:
            now: Current time; ``None`` uses the injected time source.

        Returns:
            List of purged request_ids.
        """
        moment = self._now() if now is None else now
        removed = [
            request_id
            for request_id, record in self._records.items()
            if record.is_terminal
            and moment - (record.terminal_at or record.updated_at) >= self._ttl
        ]
        for request_id in removed:
            self._records.pop(request_id, None)
        return removed
