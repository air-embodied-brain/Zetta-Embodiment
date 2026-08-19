"""Per-item result type for batch APIs.

``Result[T] = Ok(value) | Err(error)``. Batch entry points allow partial
success internally and provide no cross-session transactions; callers must
check each item individually.
"""

from __future__ import annotations

import dataclasses
from typing import Generic, TypeAlias, TypeVar

from rollout_runtime.api import codec
from rollout_runtime.api.errors import RuntimeApiError, RuntimeErrorInfo

__all__ = ["Err", "Ok", "Result", "err", "ok", "unwrap"]

T = TypeVar("T")


@dataclasses.dataclass(frozen=True)
class Ok(Generic[T]):
    """A successful result.

    Attributes:
        value: The success payload.
    """

    value: T

    @property
    def is_ok(self) -> bool:
        """Whether this is a success.

        Returns:
            Always ``True``.
        """
        return True


@dataclasses.dataclass(frozen=True)
class Err:
    """A failed result.

    Attributes:
        error: The normalized error payload.
    """

    error: RuntimeErrorInfo

    @property
    def is_ok(self) -> bool:
        """Whether this is a success.

        Returns:
            Always ``False``.
        """
        return False


Result: TypeAlias = Ok[T] | Err


def ok(value: T) -> Ok[T]:
    """Wrap a success result.

    Args:
        value: The success payload.

    Returns:
        An ``Ok`` instance.
    """
    return Ok(value)


def err(error: RuntimeErrorInfo) -> Err:
    """Wrap a failure result.

    Args:
        error: The normalized error payload.

    Returns:
        An ``Err`` instance.
    """
    return Err(error)


def unwrap(result: Result[T]) -> T:
    """Extract the success value, raising ``RuntimeApiError`` on failure.

    Args:
        result: One item from a batch result.

    Returns:
        The success payload.

    Raises:
        RuntimeApiError: ``result`` is an ``Err``.
    """
    if isinstance(result, Err):
        raise RuntimeApiError(result.error)
    return result.value


codec.register_messages(globals())
