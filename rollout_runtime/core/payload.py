"""Payload encoding/decoding and the size budget.

Rules:

- ``<= 256 KiB`` is inline; images use PNG (uint8 HWC), arrays use raw +
  shape/dtype.
- ``> 256 KiB`` is still inlined in v1, but accumulates a warning counter;
  whether to switch to the Ray object store / shared memory is decided
  later based on profiling.
- The per-request payload budget has a hard cap of 8 MiB; exceeding it
  returns ``INVALID_ARGUMENT`` to avoid overloading the Channel.

PNG encoding/decoding is a self-contained stdlib (``zlib``) + numpy
implementation, without pulling in a third-party imaging library. Two
reasons for this: first, ``core`` is only allowed to depend on stdlib +
numpy; second, the encoded bytes are deterministic (fixed compression
level, fixed filter 0), which is required for legacy-parity hash
comparisons of images, and must not be affected by an imaging library's
version.
"""

from __future__ import annotations

import dataclasses
import struct
import zlib

import numpy as np

from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.errors import RuntimeApiError, make_error
from rollout_runtime.api.payload_ref import (
    InlineBytes,
    ObjectRefId,
    PayloadCodec,
    PayloadRef,
    payload_nbytes,
)

__all__ = [
    "INLINE_THRESHOLD_BYTES",
    "PNG_COMPRESS_LEVEL",
    "REQUEST_PAYLOAD_LIMIT_BYTES",
    "PayloadStats",
    "check_payload_budget",
    "decode_array",
    "decode_image",
    "decode_payload",
    "encode_array",
    "encode_image",
    "encode_payload",
    "stats",
]

INLINE_THRESHOLD_BYTES = 256 * 1024
"""Beyond this size, the payload is still inlined, but recorded as
oversize."""

REQUEST_PAYLOAD_LIMIT_BYTES = 8 * 1024 * 1024
"""The hard cap for a single request's payload; exceeding it returns
``INVALID_ARGUMENT`` directly."""

PNG_COMPRESS_LEVEL = 6
"""Fixed compression level, guaranteeing deterministic bytes for the
same array."""

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_CHANNELS_TO_COLOR_TYPE = {1: 0, 2: 4, 3: 2, 4: 6}
_COLOR_TYPE_TO_CHANNELS = {0: 1, 4: 2, 2: 3, 6: 4}


@dataclasses.dataclass
class PayloadStats:
    """In-process payload counters (exported by ``gateway.metrics``).

    Attributes:
        encoded_count: The number of encode calls.
        encoded_bytes: The total bytes produced by encoding.
        decoded_count: The number of decode calls.
        decoded_bytes: The total bytes consumed by decoding.
        oversize_count: The number of times the inline threshold was
            exceeded.
        oversize_bytes: The total bytes exceeding the inline threshold.
    """

    encoded_count: int = 0
    encoded_bytes: int = 0
    decoded_count: int = 0
    decoded_bytes: int = 0
    oversize_count: int = 0
    oversize_bytes: int = 0

    def reset(self) -> None:
        """Reset all counters to zero (for tests)."""
        self.encoded_count = 0
        self.encoded_bytes = 0
        self.decoded_count = 0
        self.decoded_bytes = 0
        self.oversize_count = 0
        self.oversize_bytes = 0


_STATS = PayloadStats()


def stats() -> PayloadStats:
    """Return the in-process payload counters.

    Returns:
        The mutable global counter instance.
    """
    return _STATS


def _record_encoded(nbytes: int) -> None:
    _STATS.encoded_count += 1
    _STATS.encoded_bytes += nbytes
    if nbytes > INLINE_THRESHOLD_BYTES:
        _STATS.oversize_count += 1
        _STATS.oversize_bytes += nbytes


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def _png_encode(array: np.ndarray) -> bytes:
    height, width, channels = array.shape
    color_type = _CHANNELS_TO_COLOR_TYPE.get(channels)
    if color_type is None:
        raise RuntimeApiError(
            make_error(
                ErrorCode.INVALID_ARGUMENT,
                f"unsupported channel count for png payload: {channels}",
            )
        )
    scanlines = np.zeros((height, 1 + width * channels), dtype=np.uint8)
    scanlines[:, 1:] = array.reshape(height, width * channels)
    header = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    return b"".join(
        (
            _PNG_SIGNATURE,
            _png_chunk(b"IHDR", header),
            _png_chunk(b"IDAT", zlib.compress(scanlines.tobytes(), PNG_COMPRESS_LEVEL)),
            _png_chunk(b"IEND", b""),
        )
    )


def _paeth(left: int, up: int, up_left: int) -> int:
    estimate = left + up - up_left
    d_left = abs(estimate - left)
    d_up = abs(estimate - up)
    d_up_left = abs(estimate - up_left)
    if d_left <= d_up and d_left <= d_up_left:
        return left
    if d_up <= d_up_left:
        return up
    return up_left


def _unfilter(scanlines: np.ndarray, filters: np.ndarray, channels: int) -> np.ndarray:
    """Reverse PNG per-scanline filtering.

    Args:
        scanlines: A ``[height, width * channels]`` uint8 array.
        filters: The filter type for each row.
        channels: The number of bytes per pixel.

    Returns:
        The restored ``[height, width * channels]`` array.

    Raises:
        RuntimeApiError: An unknown filter type was encountered.
    """
    if bool(np.all(filters == 0)):
        return scanlines
    out = scanlines.astype(np.int32, copy=True)
    height, stride = out.shape
    for row in range(height):
        kind = int(filters[row])
        if kind == 0:
            continue
        current = out[row]
        previous = out[row - 1] if row > 0 else np.zeros(stride, dtype=np.int32)
        if kind == 1:
            for index in range(channels, stride):
                current[index] = (current[index] + current[index - channels]) & 0xFF
        elif kind == 2:
            out[row] = (current + previous) & 0xFF
        elif kind == 3:
            for index in range(stride):
                left = current[index - channels] if index >= channels else 0
                current[index] = (
                    current[index] + ((left + previous[index]) >> 1)
                ) & 0xFF
        elif kind == 4:
            for index in range(stride):
                left = current[index - channels] if index >= channels else 0
                up_left = previous[index - channels] if index >= channels else 0
                current[index] = (
                    current[index] + _paeth(left, int(previous[index]), int(up_left))
                ) & 0xFF
        else:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.INVALID_ARGUMENT, f"unknown png filter type: {kind}"
                )
            )
    return out.astype(np.uint8)


def _png_decode(data: bytes) -> np.ndarray:
    if not data.startswith(_PNG_SIGNATURE):
        raise RuntimeApiError(
            make_error(ErrorCode.INVALID_ARGUMENT, "payload is not a png stream")
        )
    offset = len(_PNG_SIGNATURE)
    width = height = channels = 0
    idat = bytearray()
    while offset + 8 <= len(data):
        (length,) = struct.unpack_from(">I", data, offset)
        tag = data[offset + 4 : offset + 8]
        body = data[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if tag == b"IHDR":
            width, height, depth, color_type = struct.unpack_from(">IIBB", body, 0)
            interlace = body[12]
            if depth != 8 or interlace != 0:
                raise RuntimeApiError(
                    make_error(
                        ErrorCode.INVALID_ARGUMENT,
                        "only 8-bit non-interlaced png payloads are supported",
                    )
                )
            channels = _COLOR_TYPE_TO_CHANNELS[color_type]
        elif tag == b"IDAT":
            idat += body
        elif tag == b"IEND":
            break
    raw = zlib.decompress(bytes(idat))
    stride = 1 + width * channels
    buffer = np.frombuffer(raw, dtype=np.uint8).reshape(height, stride)
    pixels = _unfilter(buffer[:, 1:], buffer[:, 0], channels)
    return pixels.reshape(height, width, channels)


def encode_array(array: np.ndarray) -> InlineBytes:
    """Encode an array as a raw inline payload.

    Args:
        array: An array of any dtype (automatically made C-contiguous).

    Returns:
        An inline payload with ``codec=RAW``.
    """
    contiguous = np.ascontiguousarray(array)
    data = contiguous.tobytes()
    _record_encoded(len(data))
    return InlineBytes(
        codec=PayloadCodec.RAW,
        shape=tuple(int(dim) for dim in contiguous.shape),
        dtype=str(contiguous.dtype),
        data=data,
    )


def encode_image(array: np.ndarray) -> InlineBytes:
    """Encode a uint8 HWC image as a PNG inline payload.

    Args:
        array: A ``[H, W]`` or ``[H, W, C]`` uint8 array.

    Returns:
        An inline payload with ``codec=PNG``.

    Raises:
        RuntimeApiError: The dtype is not uint8, or the dimensions are
            invalid.
    """
    if array.dtype != np.uint8:
        raise RuntimeApiError(
            make_error(
                ErrorCode.INVALID_ARGUMENT,
                f"png payload requires uint8, got {array.dtype}",
            )
        )
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3:
        raise RuntimeApiError(
            make_error(
                ErrorCode.INVALID_ARGUMENT,
                f"png payload requires HWC layout, got shape {array.shape}",
            )
        )
    contiguous = np.ascontiguousarray(array)
    data = _png_encode(contiguous)
    _record_encoded(len(data))
    return InlineBytes(
        codec=PayloadCodec.PNG,
        shape=tuple(int(dim) for dim in contiguous.shape),
        dtype="uint8",
        data=data,
    )


def encode_payload(array: np.ndarray) -> InlineBytes:
    """Automatically choose an encoding based on the array.

    uint8 2D/3D arrays (with 1/2/3/4 channels) use PNG; everything else
    uses raw.

    Args:
        array: The array to encode.

    Returns:
        The inline payload.
    """
    if array.dtype == np.uint8 and array.ndim in (2, 3):
        channels = 1 if array.ndim == 2 else array.shape[2]
        if channels in _CHANNELS_TO_COLOR_TYPE:
            return encode_image(array)
    return encode_array(array)


def decode_array(ref: PayloadRef) -> np.ndarray:
    """Decode a raw inline payload.

    Args:
        ref: The payload reference.

    Returns:
        The restored array.

    Raises:
        RuntimeApiError: The payload is not a raw inline payload, or the
            byte count doesn't match shape/dtype.
    """
    if not isinstance(ref, InlineBytes) or ref.codec is not PayloadCodec.RAW:
        raise RuntimeApiError(
            make_error(ErrorCode.INVALID_ARGUMENT, "expected a raw inline payload")
        )
    dtype = np.dtype(ref.dtype)
    expected = int(np.prod(ref.shape)) * dtype.itemsize if ref.shape else dtype.itemsize
    if len(ref.data) != expected:
        raise RuntimeApiError(
            make_error(
                ErrorCode.INVALID_ARGUMENT,
                f"raw payload size mismatch: got {len(ref.data)}, expected {expected}",
            )
        )
    _STATS.decoded_count += 1
    _STATS.decoded_bytes += len(ref.data)
    return np.frombuffer(ref.data, dtype=dtype).reshape(ref.shape).copy()


def decode_image(ref: PayloadRef) -> np.ndarray:
    """Decode a PNG inline payload.

    Args:
        ref: The payload reference.

    Returns:
        A ``[H, W, C]`` uint8 array.

    Raises:
        RuntimeApiError: The payload is not a PNG inline payload, or the
            decoded shape doesn't match the declared shape.
    """
    if not isinstance(ref, InlineBytes) or ref.codec is not PayloadCodec.PNG:
        raise RuntimeApiError(
            make_error(ErrorCode.INVALID_ARGUMENT, "expected a png inline payload")
        )
    array = _png_decode(ref.data)
    if tuple(ref.shape) != array.shape:
        raise RuntimeApiError(
            make_error(
                ErrorCode.INVALID_ARGUMENT,
                f"png payload shape mismatch: header {array.shape}, ref {ref.shape}",
            )
        )
    _STATS.decoded_count += 1
    _STATS.decoded_bytes += len(ref.data)
    return array


def decode_payload(ref: PayloadRef) -> np.ndarray:
    """Decode a payload according to its own declared encoding.

    Args:
        ref: The payload reference.

    Returns:
        The restored array.

    Raises:
        RuntimeApiError: The ``ObjectRefId`` form is not implemented in v1.
    """
    if isinstance(ref, ObjectRefId):
        raise RuntimeApiError(
            make_error(
                ErrorCode.INVALID_ARGUMENT,
                "object-store payloads are not produced in v1",
            )
        )
    if ref.codec is PayloadCodec.PNG:
        return decode_image(ref)
    return decode_array(ref)


def check_payload_budget(
    refs: object,
    *,
    limit: int = REQUEST_PAYLOAD_LIMIT_BYTES,
    context: str = "request",
) -> int:
    """Validate that a single request's total payload does not exceed the
    hard cap.

    Args:
        refs: A single ``PayloadRef``, or any iterable/nested container.
        limit: The byte cap.
        context: The context name used in the error message.

    Returns:
        The total byte count tallied.

    Raises:
        RuntimeApiError: The cap was exceeded (``INVALID_ARGUMENT``).
    """
    total = _sum_payload_bytes(refs)
    if total > limit:
        raise RuntimeApiError(
            make_error(
                ErrorCode.INVALID_ARGUMENT,
                f"{context} payload budget exceeded: {total} > {limit} bytes",
                payload_bytes=total,
                payload_limit=limit,
            )
        )
    return total


def _sum_payload_bytes(node: object) -> int:
    if node is None:
        return 0
    if isinstance(node, (InlineBytes, ObjectRefId)):
        return payload_nbytes(node)
    if isinstance(node, bytes):
        return len(node)
    if isinstance(node, dict):
        return sum(_sum_payload_bytes(value) for value in node.values())
    if isinstance(node, (list, tuple, set, frozenset)):
        return sum(_sum_payload_bytes(item) for item in node)
    if dataclasses.is_dataclass(node) and not isinstance(node, type):
        return sum(
            _sum_payload_bytes(getattr(node, field.name))
            for field in dataclasses.fields(node)
        )
    return 0
