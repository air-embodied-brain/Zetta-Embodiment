# Copyright (c) 2026 Zetta Contributors
"""Bounded IPC carrying built-in values and RGB bytes across Python environments."""

from __future__ import annotations

import socket
import struct
import time
from typing import Any

import msgpack

PROTOCOL_VERSION = 1
MAX_PACKET_BYTES = 64 * 1024 * 1024
_HEADER = struct.Struct("!I")


def send_packet(
    channel: socket.socket, message: dict[str, Any], deadline: float
) -> None:
    data = msgpack.packb(message, use_bin_type=True, strict_types=True)
    if len(data) > MAX_PACKET_BYTES:
        raise ValueError("Genie IPC message exceeds 64 MiB")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Genie IPC send deadline expired")
    channel.settimeout(remaining)
    channel.sendall(_HEADER.pack(len(data)) + data)


def receive_packet(channel: socket.socket, deadline: float) -> dict[str, Any]:
    def read_exact(length: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < length:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Genie IPC receive deadline expired")
            channel.settimeout(remaining)
            data = channel.recv(min(length - len(chunks), 1024 * 1024))
            if not data:
                raise EOFError("Genie subprocess closed the IPC channel")
            chunks.extend(data)
        return bytes(chunks)

    (length,) = _HEADER.unpack(read_exact(_HEADER.size))
    if length == 0 or length > MAX_PACKET_BYTES:
        raise ValueError(f"invalid Genie IPC message length: {length}")
    message = msgpack.unpackb(read_exact(length), raw=False, strict_map_key=True)
    if not isinstance(message, dict):
        raise ValueError("Genie IPC message must be a mapping")
    return message
