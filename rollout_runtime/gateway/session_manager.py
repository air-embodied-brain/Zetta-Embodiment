"""Session lifecycle, leases, and idempotency index.

The Gateway is the sole writer of session logical state; the EnvWorker holds
environment execution state. The two do not share field ownership; this
module only touches the former.

State transition table:

```text
CREATING --binding committed--> READY --close|lease expired--> CLOSING --ack--> CLOSED
CREATING --binding failed-----> FAILED
READY    --binding lost-------> LOST
FAILED / LOST ---------------> CLOSED     # cleanup after caller close or error-retention expiry
```

``READY`` only means environment resources are bound, **not that reset has
happened**: before reset, ``observe`` / ``action_step`` / ``policy_step``
return ``SESSION_NOT_READY``. v1 does not recover ``LOST`` (a newly created
environment cannot resume physical state); the caller must close and create
a new one.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from collections.abc import Callable, Iterator, Mapping
from typing import Any

from rollout_runtime.api.enums import ErrorCode, SessionState
from rollout_runtime.api.errors import (
    InvalidTransition,
    RuntimeApiError,
    RuntimeErrorInfo,
    make_error,
)
from rollout_runtime.api.ids import (
    BindingToken,
    EpisodeId,
    OperationSeq,
    RequestId,
    SessionId,
    new_session_id,
)
from rollout_runtime.api.messages import (
    CreateSessionRequest,
    EnvSpecMsg,
    SessionHandle,
    SessionStatus,
    WorkerSummary,
)

__all__ = [
    "ALLOWED_SESSION_TRANSITIONS",
    "DEFAULT_ERROR_RETENTION_SECONDS",
    "SessionManager",
    "SessionRecord",
]

DEFAULT_ERROR_RETENTION_SECONDS = 300.0
"""How long ``FAILED`` / ``LOST`` records remain queryable before being cleared."""

ALLOWED_SESSION_TRANSITIONS: Mapping[SessionState, frozenset[SessionState]] = {
    SessionState.CREATING: frozenset({SessionState.READY, SessionState.FAILED}),
    SessionState.READY: frozenset({SessionState.CLOSING, SessionState.LOST}),
    SessionState.CLOSING: frozenset({SessionState.CLOSED}),
    SessionState.FAILED: frozenset({SessionState.CLOSED}),
    SessionState.LOST: frozenset({SessionState.CLOSED}),
    SessionState.CLOSED: frozenset(),
}
"""The only set of legal transition edges; anything else is ``InvalidTransition``.

Note there is **no** ``CREATING -> CLOSING``: a close request during
creation gets ``SESSION_NOT_READY``, and the creation-failure path uniformly
goes through ``CREATING -> FAILED``.
"""


@dataclasses.dataclass
class SessionRecord:
    """A Gateway-side session record.

    Attributes:
        session_id: Session identifier.
        application_id: Owning application.
        client_session_key: Caller-supplied idempotency key.
        env_spec: Environment specification.
        env_spec_digest: Environment specification digest.
        default_policy_id: Default policy.
        state: Logical lifecycle.
        lease_expiration: Lease expiration time.
        created_at: Creation time.
        updated_at: Most recent state-change time.
        metadata: Pass-through fields.
        gateway_epoch: Gateway epoch.
        worker_rank: The sticky-bound EnvWorker rank.
        binding_token: Binding identifier returned by the EnvWorker.
        episode_id: Current episode; ``None`` means not yet reset.
        next_operation_seq: Next operation sequence number to allocate.
        active_operation: The currently executing mutating operation.
        worker_summary: Non-authoritative worker summary.
        error: Error for ``FAILED`` / ``LOST``.
        reap_pending: Lease already expired but an operation is still
            running; close once it finishes.
        terminal_at: Time it entered ``FAILED`` / ``LOST`` / ``CLOSED``.
        lock: Per-session lock, serializing mutating operations.
    """

    session_id: SessionId
    application_id: str
    client_session_key: str
    env_spec: EnvSpecMsg
    env_spec_digest: str
    default_policy_id: str | None
    state: SessionState
    lease_expiration: float
    created_at: float
    updated_at: float
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    gateway_epoch: int = 1
    worker_rank: int | None = None
    binding_token: BindingToken | None = None
    episode_id: EpisodeId | None = None
    next_operation_seq: int = 1
    active_operation: RequestId | None = None
    worker_summary: WorkerSummary | None = None
    error: RuntimeErrorInfo | None = None
    reap_pending: bool = False
    terminal_at: float | None = None
    lock: asyncio.Lock = dataclasses.field(
        default_factory=asyncio.Lock, repr=False, compare=False
    )

    def handle(self) -> SessionHandle:
        """Generate the external handle.

        Returns:
            A ``SessionHandle`` without rank / slot / binding_token.
        """
        return SessionHandle(
            session_id=self.session_id,
            application_id=self.application_id,
            env_spec_digest=self.env_spec_digest,
            default_policy_id=self.default_policy_id,
            lease_expiration=self.lease_expiration,
            gateway_epoch=self.gateway_epoch,
        )

    def status(self) -> SessionStatus:
        """Generate a status snapshot.

        Returns:
            ``SessionStatus``; ``worker_summary`` is a non-authoritative cache.
        """
        return SessionStatus(
            session_id=self.session_id,
            state=self.state,
            lease_expiration=self.lease_expiration,
            episode_id=self.episode_id,
            active_operation=self.active_operation,
            next_operation_seq=OperationSeq(self.next_operation_seq),
            worker_summary=self.worker_summary,
            error=self.error,
        )


class SessionManager:
    """Session records, idempotency index, lifecycle state machine, and leases."""

    def __init__(
        self,
        *,
        gateway_epoch: int = 1,
        time_source: Callable[[], float] = time.time,
        error_retention_seconds: float = DEFAULT_ERROR_RETENTION_SECONDS,
        default_lease_seconds: float = 300.0,
    ) -> None:
        """Initialize.

        Args:
            gateway_epoch: Gateway instance epoch, written into ``SessionHandle``.
            time_source: Time source, injectable for testing.
            error_retention_seconds: ``FAILED`` / ``LOST`` record retention duration.
            default_lease_seconds: Default value used when a request does
                not give ``lease_seconds``.
        """
        self._gateway_epoch = gateway_epoch
        self._now = time_source
        self._error_retention_seconds = error_retention_seconds
        self._default_lease_seconds = default_lease_seconds
        self._sessions: dict[SessionId, SessionRecord] = {}
        self._by_client_key: dict[tuple[str, str], SessionId] = {}

    # ---------------------------------------------------------------- Queries

    def __len__(self) -> int:
        """Return the number of records.

        Returns:
            Number of currently retained session records (including
            terminal ones not yet purged).
        """
        return len(self._sessions)

    def __iter__(self) -> Iterator[SessionRecord]:
        """Iterate over all records.

        Returns:
            Record iterator.
        """
        return iter(list(self._sessions.values()))

    def get(self, session_id: SessionId) -> SessionRecord:
        """Look up a record by id.

        Args:
            session_id: Session identifier.

        Returns:
            The session record.

        Raises:
            RuntimeApiError: The record does not exist (``UNKNOWN_SESSION``).
        """
        record = self._sessions.get(session_id)
        if record is None:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.UNKNOWN_SESSION,
                    f"unknown session: {session_id}",
                    session_id=session_id,
                )
            )
        return record

    def find(self, session_id: SessionId) -> SessionRecord | None:
        """Look up a record by id, returning ``None`` if not found.

        Args:
            session_id: Session identifier.

        Returns:
            The record, or ``None``.
        """
        return self._sessions.get(session_id)

    def status(self, session_id: SessionId) -> SessionStatus:
        """Return a status snapshot.

        Args:
            session_id: Session identifier.

        Returns:
            ``SessionStatus``.
        """
        return self.get(session_id).status()

    def require_ready(self, session_id: SessionId) -> SessionRecord:
        """Require the session to be in ``READY``.

        Args:
            session_id: Session identifier.

        Returns:
            The session record.

        Raises:
            RuntimeApiError: State is not ``READY`` (``SESSION_NOT_READY``),
                or the record does not exist.
        """
        record = self.get(session_id)
        if record.state is not SessionState.READY:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.SESSION_NOT_READY,
                    f"session {session_id} is {record.state.name}",
                    session_id=session_id,
                    state=record.state.name,
                    error=record.error.message if record.error else "",
                )
            )
        return record

    def require_episode(self, session_id: SessionId) -> SessionRecord:
        """Require the session to have already been reset.

        Args:
            session_id: Session identifier.

        Returns:
            The session record.

        Raises:
            RuntimeApiError: Not yet reset (``SESSION_NOT_READY``).
        """
        record = self.require_ready(session_id)
        if record.episode_id is None:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.SESSION_NOT_READY,
                    f"session {session_id} has no episode; call reset first",
                    session_id=session_id,
                )
            )
        return record

    def sessions_on_rank(self, worker_rank: int) -> list[SessionRecord]:
        """List active sessions bound to a given rank.

        Args:
            worker_rank: EnvWorker rank.

        Returns:
            List of records in ``CREATING`` / ``READY`` state.
        """
        return [
            record
            for record in self._sessions.values()
            if record.worker_rank == worker_rank
            and record.state in (SessionState.CREATING, SessionState.READY)
        ]

    def active_session_count(self, application_id: str) -> int:
        """Count active sessions for an application.

        Args:
            application_id: Application identifier.

        Returns:
            Number of records in ``CREATING`` / ``READY`` state.
        """
        return sum(
            1
            for record in self._sessions.values()
            if record.application_id == application_id
            and record.state in (SessionState.CREATING, SessionState.READY)
        )

    # ---------------------------------------------------------------- Creation

    def create(
        self, request: CreateSessionRequest, *, session_id: SessionId | None = None
    ) -> tuple[SessionRecord, bool]:
        """Register a ``CREATING`` record, or hit ``client_session_key`` idempotency.

        Args:
            request: Creation request.
            session_id: Explicit session id (for testing); ``None`` means auto-generate.

        Returns:
            ``(record, whether newly created)``; on idempotency hit, returns
            the existing record with the second item ``False``.

        Raises:
            RuntimeApiError: ``lease_seconds`` is invalid (``INVALID_ARGUMENT``).
        """
        if request.lease_seconds <= 0:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"lease_seconds must be positive, got {request.lease_seconds}",
                )
            )
        key = (request.application_id, request.client_session_key)
        existing_id = self._by_client_key.get(key)
        if existing_id is not None:
            existing = self._sessions.get(existing_id)
            if existing is not None and existing.state not in (
                SessionState.CLOSED,
                SessionState.CLOSING,
            ):
                return existing, False
            self._by_client_key.pop(key, None)

        now = self._now()
        lease_seconds = request.lease_seconds or self._default_lease_seconds
        record = SessionRecord(
            session_id=session_id or new_session_id(),
            application_id=request.application_id,
            client_session_key=request.client_session_key,
            env_spec=request.env_spec,
            env_spec_digest=request.env_spec.digest(),
            default_policy_id=request.default_policy_id,
            state=SessionState.CREATING,
            lease_expiration=now + lease_seconds,
            created_at=now,
            updated_at=now,
            metadata=dict(request.metadata),
            gateway_epoch=self._gateway_epoch,
        )
        self._sessions[record.session_id] = record
        if request.client_session_key:
            self._by_client_key[key] = record.session_id
        return record, True

    # ------------------------------------------------------------ State transitions

    def transition(
        self,
        session_id: SessionId,
        to_state: SessionState,
        *,
        error: RuntimeErrorInfo | None = None,
    ) -> SessionRecord:
        """Execute one lifecycle transition.

        Args:
            session_id: Session identifier.
            to_state: Target state.
            error: Error to retain when entering ``FAILED`` / ``LOST``.

        Returns:
            The record after transition.

        Raises:
            InvalidTransition: The target state is not in the set of legal edges.
        """
        record = self.get(session_id)
        if to_state not in ALLOWED_SESSION_TRANSITIONS[record.state]:
            raise InvalidTransition(str(session_id), record.state, to_state)
        record.state = to_state
        record.updated_at = self._now()
        if error is not None:
            record.error = error
        if to_state in (SessionState.FAILED, SessionState.LOST, SessionState.CLOSED):
            record.terminal_at = record.updated_at
        return record

    def commit_binding(
        self,
        session_id: SessionId,
        *,
        worker_rank: int,
        binding_token: BindingToken,
        lease_expiration: float | None = None,
    ) -> SessionRecord:
        """Commit the EnvWorker binding and transition to ``READY``.

        Args:
            session_id: Session identifier.
            worker_rank: The bound EnvWorker rank.
            binding_token: The opaque identifier returned by the EnvWorker.
            lease_expiration: Override the lease expiration time.

        Returns:
            The record in ``READY`` state.
        """
        record = self.get(session_id)
        record.worker_rank = worker_rank
        record.binding_token = binding_token
        if lease_expiration is not None:
            record.lease_expiration = lease_expiration
        return self.transition(session_id, SessionState.READY)

    def fail(self, session_id: SessionId, error: RuntimeErrorInfo) -> SessionRecord:
        """Creation failure: ``CREATING -> FAILED``.

        Args:
            session_id: Session identifier.
            error: Failure reason.

        Returns:
            The record in ``FAILED`` state.
        """
        return self.transition(session_id, SessionState.FAILED, error=error)

    def mark_lost(
        self, session_id: SessionId, error: RuntimeErrorInfo | None = None
    ) -> SessionRecord:
        """Worker binding lost: ``READY -> LOST``.

        v1 does not recover ``LOST``: ``ALLOWED_SESSION_TRANSITIONS`` has no
        ``LOST -> READY`` edge, so any recovery attempt would raise
        ``InvalidTransition``.

        Args:
            session_id: Session identifier.
            error: Reason for the loss.

        Returns:
            The record in ``LOST`` state.
        """
        return self.transition(
            session_id,
            SessionState.LOST,
            error=error
            or make_error(
                ErrorCode.WORKER_LOST,
                f"env worker binding lost for session {session_id}",
                session_id=session_id,
            ),
        )

    def begin_close(self, session_id: SessionId) -> SessionRecord:
        """Begin closing: ``READY -> CLOSING``.

        Args:
            session_id: Session identifier.

        Returns:
            The record in ``CLOSING`` state.
        """
        return self.transition(session_id, SessionState.CLOSING)

    def finish_close(self, session_id: SessionId) -> SessionRecord:
        """Finish closing: ``CLOSING`` / ``FAILED`` / ``LOST`` -> ``CLOSED``.

        Args:
            session_id: Session identifier.

        Returns:
            The record in ``CLOSED`` state.
        """
        return self.transition(session_id, SessionState.CLOSED)

    # ------------------------------------------------------------ Leases and sequence numbers

    def renew(
        self,
        session_id: SessionId,
        lease_seconds: float,
        *,
        application_id: str | None = None,
    ) -> SessionRecord:
        """Renew the lease.

        Args:
            session_id: Session identifier.
            lease_seconds: New lease duration.
            application_id: Verify caller ownership; ``None`` skips the check.

        Returns:
            The record after renewal.

        Raises:
            RuntimeApiError: Invalid lease duration, ownership mismatch, or
                state is not ``READY``.
        """
        if lease_seconds <= 0:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"lease_seconds must be positive, got {lease_seconds}",
                )
            )
        record = self.require_ready(session_id)
        if application_id is not None and record.application_id != application_id:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT,
                    f"session {session_id} does not belong to {application_id!r}",
                    session_id=session_id,
                )
            )
        record.lease_expiration = self._now() + lease_seconds
        record.updated_at = self._now()
        return record

    def allocate_operation_seq(self, record: SessionRecord) -> OperationSeq:
        """Allocate the next ``operation_seq``.

        Must be called while holding the per-session lock, to ensure a
        deterministic mutation order for the same session.

        Args:
            record: Session record.

        Returns:
            The new operation sequence number.
        """
        seq = OperationSeq(record.next_operation_seq)
        record.next_operation_seq += 1
        return seq

    def set_episode(self, session_id: SessionId, episode_id: EpisodeId) -> None:
        """Record a new ``episode_id`` returned by the worker.

        Args:
            session_id: Session identifier.
            episode_id: The new episode.
        """
        record = self.get(session_id)
        record.episode_id = episode_id
        record.updated_at = self._now()

    def set_worker_summary(
        self, session_id: SessionId, summary: WorkerSummary | None
    ) -> None:
        """Cache the worker summary (non-authoritative).

        Args:
            session_id: Session identifier.
            summary: Worker summary.
        """
        record = self._sessions.get(session_id)
        if record is not None and summary is not None:
            record.worker_summary = summary

    def begin_operation(self, record: SessionRecord, request_id: RequestId) -> None:
        """Mark the session as executing a mutating operation.

        Args:
            record: Session record.
            request_id: Operation identifier.

        Raises:
            RuntimeApiError: An operation is already running (``SESSION_NOT_READY``).
        """
        if record.active_operation is not None:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.SESSION_NOT_READY,
                    f"session {record.session_id} already has an active operation",
                    session_id=record.session_id,
                    active_operation=record.active_operation,
                )
            )
        record.active_operation = request_id

    def end_operation(self, record: SessionRecord, request_id: RequestId) -> None:
        """Clear the running-operation marker.

        Args:
            record: Session record.
            request_id: Operation identifier; ignored if it doesn't match the
                current marker.
        """
        if record.active_operation == request_id:
            record.active_operation = None

    def expired_sessions(self, now: float | None = None) -> list[SessionRecord]:
        """List ``READY`` sessions whose lease has expired.

        Args:
            now: Current time; ``None`` uses the injected time source.

        Returns:
            List of expired records.
        """
        moment = self._now() if now is None else now
        return [
            record
            for record in self._sessions.values()
            if record.state is SessionState.READY and record.lease_expiration <= moment
        ]

    def purge_terminal(self, now: float | None = None) -> list[SessionId]:
        """Purge terminal records.

        ``CLOSED`` is purged immediately; ``FAILED`` / ``LOST`` are retained
        for ``error_retention_seconds`` so the caller can query them before
        purging.

        Args:
            now: Current time; ``None`` uses the injected time source.

        Returns:
            List of purged session ids.
        """
        moment = self._now() if now is None else now
        removed: list[SessionId] = []
        for session_id, record in list(self._sessions.items()):
            if record.state is SessionState.CLOSED:
                removed.append(session_id)
            elif record.state in (SessionState.FAILED, SessionState.LOST):
                terminal_at = record.terminal_at or record.updated_at
                if moment - terminal_at >= self._error_retention_seconds:
                    removed.append(session_id)
        for session_id in removed:
            record = self._sessions.pop(session_id)
            key = (record.application_id, record.client_session_key)
            if self._by_client_key.get(key) == session_id:
                self._by_client_key.pop(key, None)
        return removed
