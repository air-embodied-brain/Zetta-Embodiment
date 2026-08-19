"""EnvWorker capability, capacity, and heartbeat.

Invariant: on heartbeat timeout, the rank is immediately marked unhealthy,
and any sessions on it are transitioned by the Gateway to ``LOST``. v1 does
not implement ``LOST`` recovery (a newly created environment cannot resume
physical state).

Rank selection order: can serve this env spec -> can create a slot -> lowest
load -> smallest rank (deterministic).
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Iterator, Mapping

from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.errors import RuntimeApiError, make_error
from rollout_runtime.api.messages import (
    EnvFamilyCapability,
    EnvSpecMsg,
    EnvWorkerInfo,
)

__all__ = ["DEFAULT_HEARTBEAT_TIMEOUT_SECONDS", "EnvWorkerRegistry", "WorkerEntry"]

DEFAULT_HEARTBEAT_TIMEOUT_SECONDS = 30.0
"""Heartbeat timeout threshold."""


@dataclasses.dataclass
class WorkerEntry:
    """A worker record in the registry.

    Attributes:
        info: Capability and capacity reported by the worker.
        healthy: Whether it is healthy.
        heartbeat_at: Most recent heartbeat time.
        active_sessions: Number of sessions allocated from the Gateway's viewpoint.
        can_create_slot: Whether this rank can currently attempt to
            create/reuse a slot.
        served_digests: Digests of already-built pools reported by the
            worker, used for observation only.
    """

    info: EnvWorkerInfo
    healthy: bool = True
    heartbeat_at: float = 0.0
    active_sessions: int = 0
    can_create_slot: bool = True
    served_digests: set[str] = dataclasses.field(default_factory=set)

    @property
    def worker_rank(self) -> int:
        """Return the rank.

        Returns:
            EnvWorker rank.
        """
        return self.info.worker_rank

    @property
    def free_slots(self) -> int:
        """Return the static remaining quota of the legacy observation field.

        Gateway placement no longer uses this value; it is retained only for
        compatibility with existing debugging code.

        Returns:
            ``max_sessions - active_sessions``, never below 0.
        """
        return max(0, self.info.max_sessions - self.active_sessions)


class EnvWorkerRegistry:
    """rank -> capability / capacity / heartbeat / servable EnvSpecs."""

    def __init__(
        self,
        *,
        time_source: Callable[[], float] = time.time,
        heartbeat_timeout_seconds: float = DEFAULT_HEARTBEAT_TIMEOUT_SECONDS,
    ) -> None:
        """Initialize.

        Args:
            time_source: Time source, injectable for testing.
            heartbeat_timeout_seconds: Heartbeat timeout threshold.
        """
        self._now = time_source
        self._timeout = heartbeat_timeout_seconds
        self._entries: dict[int, WorkerEntry] = {}

    def __len__(self) -> int:
        """Return the number of registered ranks.

        Returns:
            Number of records.
        """
        return len(self._entries)

    def __iter__(self) -> Iterator[WorkerEntry]:
        """Iterate over all records.

        Returns:
            Record iterator.
        """
        return iter(list(self._entries.values()))

    def register(self, info: EnvWorkerInfo) -> WorkerEntry:
        """Register or update an EnvWorker.

        Args:
            info: Capability and capacity reported by the worker.

        Returns:
            The registry record.
        """
        now = self._now()
        entry = self._entries.get(info.worker_rank)
        if entry is None:
            entry = WorkerEntry(info=info, healthy=True, heartbeat_at=now)
            self._entries[info.worker_rank] = entry
        else:
            entry.info = info
            entry.healthy = True
            entry.heartbeat_at = now
        entry.served_digests.update(info.served_env_digests)
        return entry

    def get(self, worker_rank: int) -> WorkerEntry:
        """Look up a record by rank.

        Args:
            worker_rank: Rank.

        Returns:
            The record.

        Raises:
            RuntimeApiError: The rank is not registered (``WORKER_LOST``).
        """
        entry = self._entries.get(worker_rank)
        if entry is None:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.WORKER_LOST,
                    f"env worker rank {worker_rank} is not registered",
                    worker_rank=worker_rank,
                )
            )
        return entry

    def heartbeat(self, worker_rank: int, *, at: float | None = None) -> None:
        """Record a heartbeat.

        Args:
            worker_rank: Rank.
            at: Heartbeat time; ``None`` uses the current time.
        """
        entry = self.get(worker_rank)
        entry.heartbeat_at = self._now() if at is None else at
        entry.healthy = True

    def mark_unhealthy(self, worker_rank: int) -> None:
        """Mark a rank as unhealthy.

        Args:
            worker_rank: Rank.
        """
        entry = self._entries.get(worker_rank)
        if entry is not None:
            entry.healthy = False

    def stale_ranks(self, now: float | None = None) -> list[int]:
        """List ranks whose heartbeat has timed out.

        Args:
            now: Current time; ``None`` uses the injected time source.

        Returns:
            List of timed-out ranks.
        """
        moment = self._now() if now is None else now
        return [
            entry.worker_rank
            for entry in self._entries.values()
            if entry.healthy and moment - entry.heartbeat_at > self._timeout
        ]

    def capability(
        self, worker_rank: int, env_family: str
    ) -> EnvFamilyCapability | None:
        """Query the capability declared by a rank for a family.

        Args:
            worker_rank: Rank.
            env_family: Family name.

        Returns:
            The capability, or ``None``.
        """
        return self.get(worker_rank).info.capabilities.get(env_family)

    def capabilities_for(self, env_family: str) -> EnvFamilyCapability | None:
        """Return the capability declared by any healthy rank for a family.

        Args:
            env_family: Family name.

        Returns:
            The capability, or ``None``.
        """
        for entry in self._entries.values():
            if entry.healthy and env_family in entry.info.capabilities:
                return entry.info.capabilities[env_family]
        return None

    def serves(self, entry: WorkerEntry, env_spec: EnvSpecMsg) -> bool:
        """Determine whether a rank can serve the given env spec.

        Args:
            entry: Worker record.
            env_spec: Environment specification.

        Returns:
            True if it can serve.
        """
        capability = entry.info.capabilities.get(env_spec.env_family)
        if capability is None:
            return False
        if capability.needs_accelerator and not entry.info.has_accelerator:
            return False
        node_group = env_spec.resource_hints.get("node_group")
        if node_group and entry.info.node_id and node_group != entry.info.node_id:
            return False
        return True

    def select_rank(
        self,
        env_spec: EnvSpecMsg,
        *,
        prefer_node: str | None = None,
        attempted_workers: set[int] | None = None,
    ) -> int:
        """Select an EnvWorker rank for a new session.

        Args:
            env_spec: Environment specification.
            prefer_node: Preferred node identifier (locality).
            attempted_workers: Ranks already attempted during this create call.

        Returns:
            The selected rank.

        Raises:
            RuntimeApiError: No healthy rank (``WORKER_LOST``), no capability
                match (``UNSUPPORTED_ENV_SPEC``), or currently unable to
                create a slot (``RESOURCE_EXHAUSTED``).
        """
        healthy = [entry for entry in self._entries.values() if entry.healthy]
        if not healthy:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.WORKER_LOST,
                    "no healthy env worker is registered",
                    env_family=env_spec.env_family,
                )
            )
        capable = [entry for entry in healthy if self.serves(entry, env_spec)]
        if not capable:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.UNSUPPORTED_ENV_SPEC,
                    f"no env worker can serve env_family={env_spec.env_family!r}",
                    env_family=env_spec.env_family,
                )
            )
        attempted = attempted_workers or set()
        available = [
            entry
            for entry in capable
            if entry.can_create_slot and entry.worker_rank not in attempted
        ]
        if not available:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.RESOURCE_EXHAUSTED,
                    "no capable env worker can currently create an env slot",
                    env_family=env_spec.env_family,
                    resource="gpu_memory",
                    reason="OOM",
                    attempted_workers=sorted(attempted),
                )
            )
        del prefer_node
        available.sort(key=lambda entry: (entry.active_sessions, entry.worker_rank))
        return available[0].worker_rank

    def acquire(self, worker_rank: int, env_spec_digest: str) -> None:
        """Occupy a session slot and register pool reuse info.

        Args:
            worker_rank: Rank.
            env_spec_digest: Environment specification digest.
        """
        entry = self.get(worker_rank)
        entry.active_sessions += 1
        entry.served_digests.add(env_spec_digest)

    def release(
        self, worker_rank: int, *, restore_can_create_slot: bool = True
    ) -> None:
        """Release a session slot.

        Args:
            worker_rank: Rank.
            restore_can_create_slot: Restore the can-create flag when the
                real binding was successfully released.
        """
        entry = self._entries.get(worker_rank)
        if entry is not None:
            entry.active_sessions = max(0, entry.active_sessions - 1)
            if restore_can_create_slot:
                entry.can_create_slot = True

    def mark_cannot_create_slot(self, worker_rank: int) -> None:
        """Mark a rank as currently unable to accept new slot creation requests.

        Args:
            worker_rank: Rank.
        """
        entry = self._entries.get(worker_rank)
        if entry is not None:
            entry.can_create_slot = False

    def mark_can_create_slot(self, worker_rank: int) -> None:
        """Mark a rank as able to accept new slot creation requests.

        Args:
            worker_rank: Rank.
        """
        entry = self._entries.get(worker_rank)
        if entry is not None:
            entry.can_create_slot = True

    def snapshot(self) -> Mapping[int, WorkerEntry]:
        """Return a snapshot of the registry.

        Returns:
            Copy of the rank-to-record mapping.
        """
        return dict(self._entries)
