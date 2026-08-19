"""msgpack byte-level encoding/decoding.

This is the only module in the ``api`` layer allowed to import a third-party
library (see the ``API_THIRD_PARTY_ALLOWLIST`` layering guard in
``tests/runtime/test_layering.py``): serialization uses msgpack, while the rest
of the api modules stay pure stdlib so control-plane logic and digest
computation don't depend on any third-party library.
"""

from __future__ import annotations

from typing import Any

import msgpack

from rollout_runtime.api import codec

__all__ = ["decode_bytes", "encode_bytes"]


def encode_bytes(obj: Any) -> bytes:
    """Serialize a protocol object into msgpack bytes.

    Args:
        obj: A protocol dataclass, enum, or native container.

    Returns:
        The msgpack byte string.
    """
    return msgpack.packb(codec.encode(obj), use_bin_type=True)


def decode_bytes(data: bytes, hint: Any = None) -> Any:
    """Deserialize a protocol object from msgpack bytes.

    Args:
        data: The output of ``encode_bytes``.
        hint: The target type annotation; ``None`` uses the ``"@"`` tag-driven
            dynamic decode path.

    Returns:
        The decoded object.
    """
    unpacked = msgpack.unpackb(data, raw=False, strict_map_key=False)
    return codec.decode(unpacked, hint)
