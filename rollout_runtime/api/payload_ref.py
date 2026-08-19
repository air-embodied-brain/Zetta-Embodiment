"""Structure definitions for payload references.

Only the structure lives here; numpy encoding/decoding and the size budget
are in ``core.payload``, because the ``api`` layer only allows stdlib. The
other benefit of this split is that the Gateway can tally and forward
payloads without ever importing numpy.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any, TypeAlias

from rollout_runtime.api import codec

__all__ = ["InlineBytes", "ObjectRefId", "PayloadCodec", "PayloadRef", "payload_nbytes"]


class PayloadCodec(enum.Enum):
    """Byte encoding for inline payloads."""

    RAW = "raw"
    """Raw contiguous bytes, restored to an array with ``shape`` / ``dtype``."""

    PNG = "png"
    """PNG-compressed uint8 HWC image."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class InlineBytes:
    """A payload transmitted inline with the message.

    Attributes:
        codec: The byte encoding.
        shape: The logical shape.
        dtype: The numpy dtype name (e.g. ``"uint8"`` / ``"float32"``).
        data: The encoded bytes.
    """

    codec: PayloadCodec
    shape: tuple[int, ...]
    dtype: str
    data: bytes


@dataclasses.dataclass(frozen=True, kw_only=True)
class ObjectRefId:
    """A payload reference into an external object store.

    v1 never produces this form (payloads are always inlined and counted);
    the structure is kept so that switching to the Ray object store or shared
    memory later, based on profiling, does not require a protocol change.

    Attributes:
        id: The identifier in the object store.
        shape: The logical shape.
        dtype: The numpy dtype name.
        meta: Backend-specific metadata.
    """

    id: str
    shape: tuple[int, ...]
    dtype: str
    meta: dict[str, Any] = dataclasses.field(default_factory=dict)


PayloadRef: TypeAlias = InlineBytes | ObjectRefId


def payload_nbytes(ref: PayloadRef | None) -> int:
    """Return the number of bytes a payload occupies on the wire.

    Args:
        ref: The payload reference; ``None`` counts as 0.

    Returns:
        The byte count; ``ObjectRefId`` counts as 0 (the real bytes are not
        in the message).
    """
    if isinstance(ref, InlineBytes):
        return len(ref.data)
    return 0


codec.register_messages(globals())
