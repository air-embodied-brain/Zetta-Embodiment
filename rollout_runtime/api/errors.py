"""Error normalization.

Worker public methods must be total functions: they catch ``BaseException``
internally, normalize it into ``RuntimeErrorInfo`` via ``normalize_exception``,
and return it with the ``ResultEnvelope``, never letting an exception escape
across the WorkerGroup boundary (otherwise rlinf's
``WorkerGroupFuncResult._wait_for_results`` will ``os.kill(pid, SIGUSR1)`` and
kill the entire job).
"""

from __future__ import annotations

import asyncio
import dataclasses
import traceback
from typing import Any

from rollout_runtime.api import codec
from rollout_runtime.api.enums import ErrorCode

__all__ = [
    "RETRYABLE_ERROR_CODES",
    "InvalidTransition",
    "RuntimeApiError",
    "RuntimeErrorInfo",
    "is_resource_exhausted",
    "make_error",
    "normalize_exception",
]

RETRYABLE_ERROR_CODES = frozenset(
    {
        ErrorCode.QUEUE_FULL,
        ErrorCode.QUOTA_EXCEEDED,
        ErrorCode.RESOURCE_EXHAUSTED,
        ErrorCode.DEADLINE_EXCEEDED,
    }
)
"""Error codes that are safe to retry.

``WORKER_LOST`` / ``ENV_FAILURE`` / ``POLICY_FAILURE`` are never retryable:
whether the environment side effect already occurred cannot be determined,
so automatic replay is explicitly disallowed.
"""

_TRACEBACK_LIMIT_CHARS = 2000


@dataclasses.dataclass(frozen=True, kw_only=True)
class RuntimeErrorInfo:
    """The normalized error payload.

    Attributes:
        code: The external error code.
        message: A human-readable message; format stability is not
            guaranteed.
        retryable: Whether the caller can safely retry.
        side_effect_applied: Whether the environment side effect has already
            occurred (the cancellation semantics).
        detail: Structured supplementary information (exception type,
            traceback excerpt, quota numbers, etc.).
    """

    code: ErrorCode
    message: str = ""
    retryable: bool = False
    side_effect_applied: bool = False
    detail: dict[str, Any] = dataclasses.field(default_factory=dict)


def make_error(
    code: ErrorCode,
    message: str = "",
    *,
    retryable: bool | None = None,
    side_effect_applied: bool = False,
    **detail: Any,
) -> RuntimeErrorInfo:
    """Construct a ``RuntimeErrorInfo``, deriving ``retryable`` by default
    from the error code.

    Args:
        code: The external error code.
        message: A human-readable message.
        retryable: Explicitly override the retryable flag; ``None`` means
            derive it from ``RETRYABLE_ERROR_CODES``.
        side_effect_applied: Whether the environment side effect has already
            occurred.
        **detail: Structured fields to place into ``detail``.

    Returns:
        The normalized error payload.
    """
    return RuntimeErrorInfo(
        code=code,
        message=message,
        retryable=code in RETRYABLE_ERROR_CODES if retryable is None else retryable,
        side_effect_applied=side_effect_applied,
        detail=dict(detail),
    )


def is_resource_exhausted(info: RuntimeErrorInfo) -> bool:
    """Determine whether the error is a structured resource-exhaustion error.

    Args:
        info: The normalized error payload.

    Returns:
        ``True`` if the error code is ``RESOURCE_EXHAUSTED``, or the
        structured detail declares OOM / GPU memory.
    """
    if info.code is ErrorCode.RESOURCE_EXHAUSTED:
        return True
    reason = str(info.detail.get("reason", "")).lower()
    resource = str(info.detail.get("resource", "")).lower()
    return reason in {"oom", "out_of_memory", "resource_exhausted"} or resource in {
        "gpu_memory",
        "cuda_memory",
        "memory",
    }


class RuntimeApiError(Exception):
    """A runtime exception carrying a ``RuntimeErrorInfo``.

    Only used within the Gateway and at the caller boundary; at the worker
    boundary, errors are always returned, never raised.
    """

    def __init__(self, info: RuntimeErrorInfo) -> None:
        """Initialize the exception.

        Args:
            info: The normalized error payload.
        """
        super().__init__(f"{info.code.name}: {info.message}")
        self.info = info

    @classmethod
    def of(
        cls,
        code: ErrorCode,
        message: str = "",
        **detail: Any,
    ) -> RuntimeApiError:
        """Construct an exception directly from an error code.

        Args:
            code: The external error code.
            message: A human-readable message.
            **detail: Structured fields to place into ``detail``.

        Returns:
            The constructed exception instance.
        """
        return cls(make_error(code, message, **detail))


class InvalidTransition(RuntimeApiError):
    """An illegal lifecycle transition (session state machine guard)."""

    def __init__(self, entity: str, from_state: Any, to_state: Any) -> None:
        """Initialize the exception.

        Args:
            entity: The entity identifier (session_id or request_id).
            from_state: The current state.
            to_state: The target state.
        """
        from_name = getattr(from_state, "name", str(from_state))
        to_name = getattr(to_state, "name", str(to_state))
        super().__init__(
            make_error(
                ErrorCode.SESSION_NOT_READY,
                f"illegal transition {from_name} -> {to_name} for {entity}",
                entity=entity,
                from_state=from_name,
                to_state=to_name,
            )
        )
        self.entity = entity
        self.from_state = from_state
        self.to_state = to_state


def normalize_exception(
    exc: BaseException,
    *,
    default_code: ErrorCode = ErrorCode.INTERNAL,
    side_effect_applied: bool = False,
    include_traceback: bool = True,
) -> RuntimeErrorInfo:
    """Normalize an arbitrary exception into a ``RuntimeErrorInfo``.

    Args:
        exc: The caught exception.
        default_code: The error code to use when the exception type is not
            recognized.
        side_effect_applied: Whether the environment side effect has already
            occurred (determined by the call site).
        include_traceback: Whether to write a traceback excerpt into
            ``detail``.

    Returns:
        The normalized error payload.
    """
    if isinstance(exc, RuntimeApiError):
        info = exc.info
        if side_effect_applied and not info.side_effect_applied:
            return dataclasses.replace(info, side_effect_applied=True)
        return info

    if isinstance(exc, asyncio.CancelledError):
        code = ErrorCode.CANCELLED
    elif isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        code = ErrorCode.DEADLINE_EXCEEDED
    elif isinstance(exc, (TypeError, ValueError, KeyError)):
        code = ErrorCode.INVALID_ARGUMENT
    elif isinstance(exc, NotImplementedError):
        code = ErrorCode.UNSUPPORTED_EXTENSION
    else:
        code = default_code

    detail: dict[str, Any] = {"exception": type(exc).__name__}
    if include_traceback:
        text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        detail["traceback"] = text[-_TRACEBACK_LIMIT_CHARS:]

    return RuntimeErrorInfo(
        code=code,
        message=str(exc) or type(exc).__name__,
        retryable=code in RETRYABLE_ERROR_CODES,
        side_effect_applied=side_effect_applied,
        detail=detail,
    )


codec.register_messages(globals())
