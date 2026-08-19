# Copyright (c) 2026 Zetta Contributors
"""Canonical JSON and append-only filesystem primitives."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_write_json(path: str | Path, value: Any, *, overwrite: bool = False) -> str:
    """Write complete canonical JSON and atomically publish it.

    ``overwrite=False`` is the immutable-artifact path.  A hard-link publish
    provides no-clobber semantics even when two workers race on shared NFS.
    ``overwrite=True`` is reserved for the small recoverable state pointer.
    """

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    payload = canonical_json_bytes(value)
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, target)
        else:
            try:
                os.link(temporary, target)
            except FileExistsError:
                existing = target.read_bytes()
                if existing != payload:
                    raise FileExistsError(
                        f"immutable artifact already differs: {target}"
                    )
            finally:
                temporary.unlink(missing_ok=True)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


@contextmanager
def directory_lock(
    path: str | Path,
    *,
    timeout_s: float = 30.0,
    poll_s: float = 0.05,
) -> Iterator[None]:
    """Cross-process lock based only on atomic directory creation."""

    lock = Path(path)
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            lock.mkdir(parents=False)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out acquiring filesystem lock: {lock}")
            time.sleep(poll_s)
    try:
        yield
    finally:
        try:
            lock.rmdir()
        except FileNotFoundError:
            pass


class AppendOnlyLedger:
    """JSONL ledger with duplicate-key rejection and fsync durability."""

    def __init__(self, path: str | Path, *, key: str) -> None:
        self.path = Path(path)
        self.key = key
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = self.path.with_name(f".{self.path.name}.lock")

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        result: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"malformed ledger {self.path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(value, dict):
                    raise ValueError(
                        f"ledger record is not an object: {self.path}:{line_number}"
                    )
                result.append(value)
        return result

    def append(self, record: dict[str, Any]) -> bool:
        identity = record.get(self.key)
        if not isinstance(identity, str) or not identity:
            raise ValueError(f"ledger record requires non-empty {self.key!r}")
        payload = canonical_json_bytes(record)
        with directory_lock(self._lock):
            matches = [row for row in self.records() if row.get(self.key) == identity]
            if matches:
                if len(matches) == 1 and canonical_json_bytes(matches[0]) == payload:
                    return False
                raise ValueError(
                    f"duplicate ledger key with different payload: {self.key}={identity}"
                )
            with self.path.open("ab") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        return True
