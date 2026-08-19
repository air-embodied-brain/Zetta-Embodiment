"""Data-plane workers.

Layering constraint: this subpackage allows ``ray`` / ``rlinf``, but they
must be **lazily imported** -- ``.venv-runtime`` does not have rlinf
installed, and local tests need to be able to import these modules anyway.
Subclassing rlinf's ``Worker`` is wrapped one layer inside
``launch/ray_launch.py``.
"""

from __future__ import annotations

from rollout_runtime.workers.batch_scheduler import (
    InferenceBatchScheduler,
    PendingRequest,
    SchedulerConfig,
)
from rollout_runtime.workers.env_worker import RuntimeEnvWorker, SessionSlot
from rollout_runtime.workers.rollout_worker import RuntimeRolloutWorker

__all__ = [
    "InferenceBatchScheduler",
    "PendingRequest",
    "RuntimeEnvWorker",
    "RuntimeRolloutWorker",
    "SchedulerConfig",
    "SessionSlot",
]
