"""Protocol enums.

All enums are transmitted on the wire by ``name`` (see ``api.codec``), so
renaming a member is a breaking change while adding a member is not.
"""

from __future__ import annotations

import enum

__all__ = [
    "CONTROL_PLANE_OPERATIONS",
    "EnvOperation",
    "ErrorCode",
    "MUTATING_OPERATIONS",
    "OperationState",
    "Priority",
    "SessionState",
    "TERMINAL_OPERATION_STATES",
    "TERMINAL_SESSION_STATES",
]


class SessionState(enum.Enum):
    """The Gateway-side logical session lifecycle."""

    CREATING = "creating"
    READY = "ready"
    CLOSING = "closing"
    CLOSED = "closed"
    FAILED = "failed"
    LOST = "lost"


class OperationState(enum.Enum):
    """The queryable state of an external operation (see
    ``get_request_status``)."""

    ACCEPTED = "accepted"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    OUTCOME_UNKNOWN = "outcome_unknown"


class EnvOperation(enum.Enum):
    """Operation types an EnvWorker can execute.

    The first seven go over the command channel, the last five over the
    high-priority control channel.
    """

    RESET = "reset"
    OBSERVE = "observe"
    ACTION_STEP = "action_step"
    POLICY_STEP = "policy_step"
    POLICY_INFER = "policy_infer"
    RUN_EPISODE = "run_episode"
    EXTENSION_CALL = "extension_call"
    CREATE_BINDING = "create_binding"
    RELEASE_BINDING = "release_binding"
    RENEW_LEASE = "renew_lease"
    CANCEL = "cancel"
    HEARTBEAT = "heartbeat"


class Priority(enum.Enum):
    """Scheduling priority; smaller ``value`` means higher priority (used by
    the batch scheduler for ordering)."""

    INTERACTIVE = 0
    BATCH = 1
    BACKGROUND = 2

    @property
    def sort_key(self) -> int:
        """Return the sort key.

        Returns:
            An integer; smaller means higher priority.
        """
        return int(self.value)


class ErrorCode(enum.Enum):
    """The normalized external error surface."""

    # Caller errors
    UNKNOWN_SESSION = "unknown_session"
    SESSION_NOT_READY = "session_not_ready"
    EPISODE_TERMINATED = "episode_terminated"
    STALE_BINDING = "stale_binding"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    INVALID_ARGUMENT = "invalid_argument"
    UNSUPPORTED_EXTENSION = "unsupported_extension"
    UNSUPPORTED_ENV_SPEC = "unsupported_env_spec"
    # Capacity / timing
    QUEUE_FULL = "queue_full"
    QUOTA_EXCEEDED = "quota_exceeded"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    CANCELLED = "cancelled"
    # System
    WORKER_LOST = "worker_lost"
    ENV_FAILURE = "env_failure"
    POLICY_FAILURE = "policy_failure"
    INTERNAL = "internal"


MUTATING_OPERATIONS = frozenset(
    {
        EnvOperation.RESET,
        EnvOperation.ACTION_STEP,
        EnvOperation.POLICY_STEP,
        EnvOperation.RUN_EPISODE,
    }
)
"""Operations that require an assigned ``operation_seq`` and are serialized
within the per-session lock.

``EXTENSION_CALL`` is not included: all five LIBERO privileged methods are
read-only. ``POLICY_INFER`` is also not included: it only reads the cached
observation and runs one model forward pass without touching environment
state (semantically equivalent to ``OBSERVE`` plus one inference call).
"""

CONTROL_PLANE_OPERATIONS = frozenset(
    {
        EnvOperation.CREATE_BINDING,
        EnvOperation.RELEASE_BINDING,
        EnvOperation.RENEW_LEASE,
        EnvOperation.CANCEL,
        EnvOperation.HEARTBEAT,
    }
)
"""Operations that go over the independent high-priority control channel
(the last five ``EnvOperation`` members).

They are **not** entered into the ``OperationRegistry``: there is no
``request_id`` idempotency semantics, nor any notion of "late result
finalization". The Gateway therefore needs to guard against one thing:
a late control-plane response (e.g. a heartbeat that arrives after a
liveness-check timeout) must not be counted into ``late_results_total``,
otherwise that metric would be completely dominated by heartbeats during
rank jitter.
"""

TERMINAL_SESSION_STATES = frozenset({SessionState.CLOSED})
"""Session states that no longer allow any transition."""

TERMINAL_OPERATION_STATES = frozenset(
    {
        OperationState.SUCCEEDED,
        OperationState.FAILED,
        OperationState.CANCELLED,
        OperationState.OUTCOME_UNKNOWN,
    }
)
"""Terminal states of an operation; once reached, the result enters the TTL
cache."""
