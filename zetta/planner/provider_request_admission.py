# Copyright (c) 2026 Zetta Contributors
"""Request-scoped, cross-process admission for the provider failover pool."""

from __future__ import annotations

import asyncio
import os
import threading
import uuid
from pathlib import Path

from zetta.evolution.provider_admission import (
    ProviderAdmission,
    ProviderAdmissionConfig,
    ProviderFailure,
    ProviderLease,
)

REQUEST_ADMISSION_ROOT_ENV = "ZETTA_PROVIDER_REQUEST_ADMISSION_ROOT"
INITIAL_CONCURRENCY_ENV = "ZETTA_PROVIDER_INITIAL_CONCURRENCY"
MAX_CONCURRENCY_ENV = "ZETTA_PROVIDER_MAX_CONCURRENCY"
DEFAULT_PROVIDER_CONCURRENCY = 8
MAXIMUM_PROVIDER_CONCURRENCY = 20


def _positive_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


class ProviderRequestLease:
    """One actual provider request holding route and global capacity."""

    def __init__(
        self,
        *,
        route_context: object,
        route_lease: ProviderLease,
        global_context: object,
        global_lease: ProviderLease,
    ) -> None:
        self._route_context = route_context
        self._route_lease = route_lease
        self._global_context = global_context
        self._global_lease = global_lease
        self._lock = threading.Lock()
        self._closed = False

    async def succeed(self, *, tokens: int, latency_s: float) -> None:
        await self._finish(success=True, failure=None, tokens=tokens, latency_s=latency_s)

    async def fail(
        self,
        failure: ProviderFailure,
        *,
        tokens: int,
        latency_s: float,
    ) -> None:
        await self._finish(
            success=False,
            failure=failure,
            tokens=tokens,
            latency_s=latency_s,
        )

    async def release(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        await asyncio.to_thread(self._route_lease.release)
        await asyncio.to_thread(self._global_lease.release)
        await self._exit_contexts()

    async def _finish(
        self,
        *,
        success: bool,
        failure: ProviderFailure | None,
        tokens: int,
        latency_s: float,
    ) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("provider request lease was already finalized")
            self._closed = True
        try:
            if success:
                await asyncio.to_thread(
                    self._route_lease.succeed, tokens=tokens, latency_s=latency_s
                )
                await asyncio.to_thread(
                    self._global_lease.succeed, tokens=tokens, latency_s=latency_s
                )
            else:
                assert failure is not None
                await asyncio.to_thread(
                    self._route_lease.fail,
                    failure,
                    tokens=tokens,
                    latency_s=latency_s,
                )
                # A failed route must not cool the synthetic global limiter:
                # the same logical call may immediately continue on a backup.
                await asyncio.to_thread(self._global_lease.release)
        finally:
            await self._exit_contexts()

    async def _exit_contexts(self) -> None:
        await asyncio.to_thread(self._global_context.__exit__, None, None, None)
        await asyncio.to_thread(self._route_context.__exit__, None, None, None)


class ProviderRequestAdmission:
    """Bind admission and metrics to the route selected for each API call."""

    GLOBAL_ROUTE_ID = "global"

    def __init__(
        self,
        root: str | Path,
        *,
        initial_limit: int = DEFAULT_PROVIDER_CONCURRENCY,
        max_limit: int = MAXIMUM_PROVIDER_CONCURRENCY,
        acquire_timeout_s: float = 1800.0,
    ) -> None:
        if not 1 <= initial_limit <= max_limit:
            raise ValueError("request limits must satisfy 1 <= initial <= maximum")
        if max_limit > MAXIMUM_PROVIDER_CONCURRENCY:
            raise ValueError(
                f"provider request concurrency must not exceed {MAXIMUM_PROVIDER_CONCURRENCY}"
            )
        config = ProviderAdmissionConfig(
            initial_limit=initial_limit,
            max_limit=max_limit,
        )
        root = Path(root).expanduser().resolve()
        self.route_admission = ProviderAdmission(root / "routes", config=config)
        self.global_admission = ProviderAdmission(root / "global", config=config)
        self.global_admission.register_route(
            self.GLOBAL_ROUTE_ID,
            initial_limit=initial_limit,
            max_limit=max_limit,
        )
        self.initial_limit = initial_limit
        self.max_limit = max_limit
        self.acquire_timeout_s = acquire_timeout_s

    @classmethod
    def from_environment(cls) -> ProviderRequestAdmission | None:
        root = os.environ.get(REQUEST_ADMISSION_ROOT_ENV)
        if not root:
            return None
        initial = _positive_env(
            INITIAL_CONCURRENCY_ENV, DEFAULT_PROVIDER_CONCURRENCY
        )
        maximum = _positive_env(MAX_CONCURRENCY_ENV, MAXIMUM_PROVIDER_CONCURRENCY)
        if (
            initial > MAXIMUM_PROVIDER_CONCURRENCY
            or maximum > MAXIMUM_PROVIDER_CONCURRENCY
        ):
            raise ValueError(
                f"provider request concurrency must not exceed {MAXIMUM_PROVIDER_CONCURRENCY}"
            )
        return cls(root, initial_limit=initial, max_limit=maximum)

    async def acquire(
        self,
        route_id: str,
        *,
        estimated_tokens: int = 0,
        max_concurrency: int | None = None,
        rpm_limit: int | None = None,
        tpm_limit: int | None = None,
    ) -> ProviderRequestLease:
        owner = f"request:{os.getpid()}:{threading.get_ident()}:{uuid.uuid4().hex}"
        route_maximum = min(self.max_limit, max_concurrency or self.max_limit)
        route_initial = min(self.initial_limit, route_maximum)
        self.route_admission.register_route(
            route_id,
            initial_limit=route_initial,
            max_limit=route_maximum,
            rpm_limit=rpm_limit,
            tpm_limit=tpm_limit,
        )
        route_context = self.route_admission.acquire(
            route_id,
            owner=owner,
            timeout_s=self.acquire_timeout_s,
            estimated_tokens=estimated_tokens,
        )
        route_lease = await asyncio.to_thread(route_context.__enter__)
        try:
            global_context = self.global_admission.acquire(
                self.GLOBAL_ROUTE_ID,
                owner=owner,
                timeout_s=self.acquire_timeout_s,
            )
            global_lease = await asyncio.to_thread(global_context.__enter__)
        except BaseException:
            await asyncio.to_thread(route_context.__exit__, None, None, None)
            raise
        return ProviderRequestLease(
            route_context=route_context,
            route_lease=route_lease,
            global_context=global_context,
            global_lease=global_lease,
        )


__all__ = [
    "ProviderRequestAdmission",
    "ProviderRequestLease",
    "REQUEST_ADMISSION_ROOT_ENV",
]
