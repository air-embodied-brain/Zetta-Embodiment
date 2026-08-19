"""Type-driven dataclass <-> native structure encoding/decoding and canonical
digests.

This module depends only on stdlib: it converts protocol dataclasses into
"msgpack-native types" (dict / list / str / int / float / bool / bytes /
None); byte-level serialization lives in ``api.wire``. The benefit of this
split is that control-plane tests and digest computation don't need msgpack
at all.

Encoding rules:

- dataclass -> dict, with a type tag ``"@"`` written in, so the result is
  self-describing;
- ``Enum`` -> the member name as a string (stable on the wire; renaming a
  member is a breaking change);
- ``tuple`` / ``set`` / ``frozenset`` -> list, restored on decode based on
  the type annotation;
- ``bytes`` passes through unchanged (natively supported by msgpack).

Decoding is driven by the target type annotation; unknown annotations
(``Any``) take the dynamic path, relying on the ``"@"`` tag to look up the
registry. Therefore, values placed into ``dict[str, Any]`` must be
msgpack-native types or already-registered dataclasses; a bare ``Enum``
placed into an unannotated container degrades to a string.
"""

from __future__ import annotations

import base64
import dataclasses
import enum
import hashlib
import json
import types
import typing
from collections.abc import Mapping, Sequence

__all__ = [
    "TYPE_TAG",
    "canonical_json",
    "decode",
    "decode_as",
    "digest",
    "encode",
    "message_types",
    "register_messages",
]

TYPE_TAG = "@"
"""The type-tag key in a dataclass's encoded result."""

_REGISTRY: dict[str, type] = {}
_HINTS: dict[type, dict[str, typing.Any]] = {}


def register_messages(namespace: Mapping[str, typing.Any]) -> None:
    """Register the protocol dataclasses in a namespace into the decode
    registry.

    Call ``register_messages(globals())`` at the bottom of a protocol module
    to avoid decorating each class individually.

    Args:
        namespace: The module's ``globals()`` or an equivalent mapping.
    """
    for name, obj in list(namespace.items()):
        if name.startswith("_"):
            continue
        if isinstance(obj, type) and dataclasses.is_dataclass(obj):
            _REGISTRY.setdefault(obj.__name__, obj)


def message_types() -> Mapping[str, type]:
    """Return the registered protocol type table.

    Returns:
        A read-only view mapping class names to class objects.
    """
    return types.MappingProxyType(_REGISTRY)


def _resolved_hints(cls: type) -> dict[str, typing.Any]:
    cached = _HINTS.get(cls)
    if cached is None:
        cached = typing.get_type_hints(cls)
        _HINTS[cls] = cached
    return cached


def _unwrap(hint: typing.Any) -> typing.Any:
    while hasattr(hint, "__supertype__"):  # typing.NewType
        hint = hint.__supertype__
    return hint


def encode(obj: typing.Any) -> typing.Any:
    """Encode a protocol object into a msgpack-native structure.

    Args:
        obj: A protocol dataclass, enum, or native container.

    Returns:
        A structure containing only msgpack-native types.

    Raises:
        TypeError: An unsupported type was encountered.
    """
    if obj is None or isinstance(obj, (bool, int, float, str, bytes)):
        return obj
    if isinstance(obj, enum.Enum):
        return obj.name
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        encoded: dict[str, typing.Any] = {TYPE_TAG: type(obj).__name__}
        for field in dataclasses.fields(obj):
            encoded[field.name] = encode(getattr(obj, field.name))
        return encoded
    if isinstance(obj, Mapping):
        return {str(key): encode(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [encode(item) for item in obj]
    raise TypeError(f"unsupported type for runtime protocol encoding: {type(obj)!r}")


def _decode_dynamic(data: typing.Any) -> typing.Any:
    if isinstance(data, Mapping):
        tag = data.get(TYPE_TAG)
        if isinstance(tag, str):
            cls = _REGISTRY.get(tag)
            if cls is None:
                raise KeyError(f"unregistered runtime protocol type: {tag!r}")
            return _decode_dataclass(cls, data)
        return {key: _decode_dynamic(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [_decode_dynamic(item) for item in data]
    return data


def _decode_dataclass(cls: type, data: Mapping[str, typing.Any]) -> typing.Any:
    tag = data.get(TYPE_TAG)
    if isinstance(tag, str) and tag != cls.__name__:
        target = _REGISTRY.get(tag)
        if target is not None:
            cls = target
    hints = _resolved_hints(cls)
    kwargs: dict[str, typing.Any] = {}
    for field in dataclasses.fields(cls):
        if field.name not in data:
            continue
        kwargs[field.name] = decode(data[field.name], hints.get(field.name))
    return cls(**kwargs)


def decode(data: typing.Any, hint: typing.Any = None) -> typing.Any:
    """Decode a native structure back into a protocol object, driven by the
    type annotation.

    Args:
        data: The output of ``encode`` (or an equivalent msgpack-decoded
            result).
        hint: The target type annotation; ``None`` uses the ``"@"``
            tag-driven dynamic path.

    Returns:
        The decoded object.

    Raises:
        KeyError: The ``"@"`` tag points to an unregistered type.
    """
    if hint is None:
        return _decode_dynamic(data)
    hint = _unwrap(hint)
    if hint is typing.Any or hint is object or isinstance(hint, typing.TypeVar):
        # An unresolved generic parameter (e.g. the T in Result[T]) falls
        # back to tag-driven dynamic decoding.
        return _decode_dynamic(data)
    if hint is type(None):
        return None

    origin = typing.get_origin(hint)
    if origin is typing.Union or origin is types.UnionType:
        args = [arg for arg in typing.get_args(hint) if arg is not type(None)]
        if data is None:
            return None
        if len(args) == 1:
            return decode(data, args[0])
        return _decode_dynamic(data)

    if data is None:
        return None

    if origin in (list, Sequence, tuple, set, frozenset):
        args = typing.get_args(hint)
        if origin is tuple and args and args[-1] is not Ellipsis:
            items = [decode(item, arg) for item, arg in zip(data, args, strict=False)]
            return tuple(items)
        item_hint = args[0] if args else None
        items = [decode(item, item_hint) for item in data]
        if origin is tuple:
            return tuple(items)
        if origin is set:
            return set(items)
        if origin is frozenset:
            return frozenset(items)
        return items

    if origin in (dict, Mapping):
        args = typing.get_args(hint)
        value_hint = args[1] if len(args) == 2 else None
        return {key: decode(value, value_hint) for key, value in data.items()}

    if isinstance(hint, type):
        if issubclass(hint, enum.Enum):
            if isinstance(data, str):
                return hint[data]
            return hint(data)
        if dataclasses.is_dataclass(hint):
            return _decode_dataclass(hint, data)
        if hint is float and isinstance(data, int) and not isinstance(data, bool):
            return float(data)

    return data


def decode_as(cls: type, data: typing.Any) -> typing.Any:
    """Decode a native structure into a specific protocol type.

    Args:
        cls: The target protocol dataclass.
        data: The output of ``encode``.

    Returns:
        An instance of ``cls``.
    """
    return decode(data, cls)


def _json_default(value: typing.Any) -> typing.Any:
    if isinstance(value, bytes):
        return {"__b64__": base64.b64encode(value).decode("ascii")}
    raise TypeError(f"unsupported type in canonical json: {type(value)!r}")


def canonical_json(obj: typing.Any) -> str:
    """Return the canonical JSON representation of an object (sorted keys,
    no extraneous whitespace).

    Used by ``EnvSpecMsg.digest()``, ``request_digest``, and ``compat_key``;
    it must be stable across processes.

    Args:
        obj: A protocol object or native structure.

    Returns:
        The canonicalized JSON string.
    """
    return json.dumps(
        encode(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    )


def digest(obj: typing.Any, *, prefix: str = "") -> str:
    """Return the sha256 hex digest of an object's canonical representation.

    Args:
        obj: A protocol object or native structure.
        prefix: An optional domain-separation prefix, to avoid collisions
            between digest spaces used for different purposes.

    Returns:
        A 64-character hex string.
    """
    payload = f"{prefix}\x00{canonical_json(obj)}" if prefix else canonical_json(obj)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
