"""Authentication, quotas, deadlines, and backpressure.

Invariant: **a full queue is explicitly rejected with ``QUEUE_FULL``, never
queued**. Backpressure propagates from the data plane toward the entry point
level by level, rather than accumulating unbounded inside the Gateway.

v1 has no dedicated ``UNAUTHENTICATED`` error code (``ErrorCode`` is already
finalized); authentication failures fall under ``INVALID_ARGUMENT``, marked
with ``reason="authentication"`` in the ``detail``. The served form will
parse ``application_id`` from the token instead of trusting the client's
self-reported value.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Mapping
from typing import Any

from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.errors import RuntimeApiError, make_error
from rollout_runtime.api.ids import SessionId

__all__ = ["AdmissionConfig", "AdmissionController", "AdmissionSnapshot"]


@dataclasses.dataclass(kw_only=True)
class AdmissionConfig:
    """Admission configuration.

    Attributes:
        require_auth: Whether to enforce token verification.
        tokens: Mapping of ``application_id`` to the expected token.
        max_sessions_per_application: Concurrent session limit per application.
        max_inflight_operations_per_application: In-flight operation limit per application.
        max_inflight_operations_per_session: In-flight operation limit per session.
            Serialization of mutating operations is handled by the per-session
            lock; this limit exists to leave concurrency headroom for read-only
            operations (``observe`` / ``extension_call``) while still
            preventing a single session from flooding the queue.
        max_total_inflight_operations: Global in-flight operation limit (backpressure watermark).
    """

    require_auth: bool = False
    tokens: dict[str, str] = dataclasses.field(default_factory=dict)
    max_sessions_per_application: int = 64
    max_inflight_operations_per_application: int = 128
    max_inflight_operations_per_session: int = 8
    max_total_inflight_operations: int = 512


@dataclasses.dataclass(frozen=True, kw_only=True)
class AdmissionSnapshot:
    """Admission count snapshot, for metrics export and test assertions.

    Attributes:
        sessions_per_application: Number of admitted sessions per application.
        inflight_per_application: Number of in-flight operations per application.
        inflight_per_session: Number of in-flight operations per session.
        total_inflight: Global number of in-flight operations.
        rejected: Rejection count per error code.
    """

    sessions_per_application: dict[str, int]
    inflight_per_application: dict[str, int]
    inflight_per_session: dict[str, int]
    total_inflight: int
    rejected: dict[str, int]


class AdmissionController:
    """Authentication, quota, deadline, and backpressure decisions."""

    def __init__(
        self,
        config: AdmissionConfig | None = None,
        *,
        time_source: Callable[[], float] = time.time,
    ) -> None:
        """Initialize.

        Args:
            config: Admission configuration; ``None`` uses defaults.
            time_source: Time source, injectable for testing.
        """
        self._config = config or AdmissionConfig()
        self._now = time_source
        self._sessions: dict[str, int] = {}
        self._inflight_app: dict[str, int] = {}
        self._inflight_session: dict[SessionId, int] = {}
        self._total_inflight = 0
        self._rejected: dict[str, int] = {}

    @property
    def config(self) -> AdmissionConfig:
        """Return the configuration.

        Returns:
            Admission configuration.
        """
        return self._config

    def _reject(
        self,
        code: ErrorCode,
        message: str,
        **detail: Any,
    ) -> RuntimeApiError:
        self._rejected[code.name] = self._rejected.get(code.name, 0) + 1
        return RuntimeApiError(make_error(code, message, **detail))

    # ------------------------------------------------------------------ Authentication

    def authenticate(self, application_id: str, token: str | None) -> None:
        """Verify application identity.

        Args:
            application_id: Application identifier.
            token: Token supplied by the caller.

        Raises:
            RuntimeApiError: ``application_id`` is empty or the token does
                not match (``INVALID_ARGUMENT``).
        """
        if not application_id:
            raise self._reject(
                ErrorCode.INVALID_ARGUMENT,
                "application_id must not be empty",
                reason="authentication",
            )
        if not self._config.require_auth:
            return
        expected = self._config.tokens.get(application_id)
        if not expected or token != expected:
            raise self._reject(
                ErrorCode.INVALID_ARGUMENT,
                f"authentication failed for application {application_id!r}",
                reason="authentication",
                application_id=application_id,
            )

    # -------------------------------------------------------------- Session quota

    def admit_session(self, application_id: str, token: str | None = None) -> None:
        """Admit a new session and occupy its quota.

        Args:
            application_id: Application identifier.
            token: Authentication token.

        Raises:
            RuntimeApiError: Authentication failed, or the concurrent session
                limit was exceeded (``QUOTA_EXCEEDED``).
        """
        self.authenticate(application_id, token)
        current = self._sessions.get(application_id, 0)
        if current >= self._config.max_sessions_per_application:
            raise self._reject(
                ErrorCode.QUOTA_EXCEEDED,
                f"application {application_id!r} reached its session limit "
                f"({self._config.max_sessions_per_application})",
                application_id=application_id,
                limit=self._config.max_sessions_per_application,
                current=current,
            )
        self._sessions[application_id] = current + 1

    def release_session(self, application_id: str) -> None:
        """Release the session quota.

        Args:
            application_id: Application identifier.
        """
        current = self._sessions.get(application_id, 0)
        if current <= 1:
            self._sessions.pop(application_id, None)
        else:
            self._sessions[application_id] = current - 1

    # ------------------------------------------------------------ Operation admission

    def admit_operation(
        self,
        application_id: str,
        session_id: SessionId,
        *,
        deadline: float | None = None,
        queue_depth: int = 0,
        queue_capacity: int | None = None,
    ) -> None:
        """Admit an operation and occupy its in-flight quota.

        Args:
            application_id: Application identifier.
            session_id: Target session.
            deadline: Absolute unix timestamp; already elapsed is rejected immediately.
            queue_depth: Current depth of the downstream queue (reported by
                transport / worker).
            queue_capacity: Downstream queue capacity; ``None`` means no queue
                check is performed.

        Raises:
            RuntimeApiError: The deadline already elapsed
                (``DEADLINE_EXCEEDED``), the queue is full (``QUEUE_FULL``),
                or a quota was exceeded (``QUOTA_EXCEEDED``).
        """
        if deadline is not None and deadline <= self._now():
            raise self._reject(
                ErrorCode.DEADLINE_EXCEEDED,
                f"deadline already elapsed for session {session_id}",
                session_id=session_id,
                deadline=deadline,
            )
        if queue_capacity is not None and queue_depth >= queue_capacity:
            raise self._reject(
                ErrorCode.QUEUE_FULL,
                f"downstream queue is full ({queue_depth}/{queue_capacity})",
                session_id=session_id,
                queue_depth=queue_depth,
                queue_capacity=queue_capacity,
            )
        if self._total_inflight >= self._config.max_total_inflight_operations:
            raise self._reject(
                ErrorCode.QUEUE_FULL,
                "gateway reached its global in-flight limit "
                f"({self._config.max_total_inflight_operations})",
                limit=self._config.max_total_inflight_operations,
            )
        app_inflight = self._inflight_app.get(application_id, 0)
        if app_inflight >= self._config.max_inflight_operations_per_application:
            raise self._reject(
                ErrorCode.QUOTA_EXCEEDED,
                f"application {application_id!r} reached its in-flight limit "
                f"({self._config.max_inflight_operations_per_application})",
                application_id=application_id,
                limit=self._config.max_inflight_operations_per_application,
                current=app_inflight,
            )
        session_inflight = self._inflight_session.get(session_id, 0)
        if session_inflight >= self._config.max_inflight_operations_per_session:
            raise self._reject(
                ErrorCode.QUOTA_EXCEEDED,
                f"session {session_id} reached its in-flight limit "
                f"({self._config.max_inflight_operations_per_session})",
                session_id=session_id,
                limit=self._config.max_inflight_operations_per_session,
                current=session_inflight,
            )
        self._inflight_app[application_id] = app_inflight + 1
        self._inflight_session[session_id] = session_inflight + 1
        self._total_inflight += 1

    def complete_operation(self, application_id: str, session_id: SessionId) -> None:
        """Release the in-flight quota.

        Args:
            application_id: Application identifier.
            session_id: Target session.
        """
        app_inflight = self._inflight_app.get(application_id, 0)
        if app_inflight <= 1:
            self._inflight_app.pop(application_id, None)
        else:
            self._inflight_app[application_id] = app_inflight - 1
        session_inflight = self._inflight_session.get(session_id, 0)
        if session_inflight <= 1:
            self._inflight_session.pop(session_id, None)
        else:
            self._inflight_session[session_id] = session_inflight - 1
        self._total_inflight = max(0, self._total_inflight - 1)

    def forget_session(self, session_id: SessionId) -> None:
        """Clear the in-flight count for a session (called when the session closes).

        Args:
            session_id: Target session.
        """
        remaining = self._inflight_session.pop(session_id, 0)
        self._total_inflight = max(0, self._total_inflight - remaining)

    # ------------------------------------------------------------------ Observation

    def snapshot(self) -> AdmissionSnapshot:
        """Return the count snapshot.

        Returns:
            Current quota usage and rejection counts.
        """
        return AdmissionSnapshot(
            sessions_per_application=dict(self._sessions),
            inflight_per_application=dict(self._inflight_app),
            inflight_per_session={
                str(key): value for key, value in self._inflight_session.items()
            },
            total_inflight=self._total_inflight,
            rejected=dict(self._rejected),
        )

    def rejected_counts(self) -> Mapping[str, int]:
        """Return rejection counts grouped by error code.

        Returns:
            Mapping of error code name to count.
        """
        return dict(self._rejected)
