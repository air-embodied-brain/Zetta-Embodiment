"""Gateway control plane.

Layering constraint: this subpackage must not import ``ray`` / ``rlinf`` /
``torch``. The Gateway is a single-writer asyncio object inside the driver
process, depending only on the ``api`` and ``transport`` Protocols.
"""

from __future__ import annotations

from rollout_runtime.gateway.admission import AdmissionConfig, AdmissionController
from rollout_runtime.gateway.dispatcher import CommandDispatcher
from rollout_runtime.gateway.gateway import RuntimeGateway
from rollout_runtime.gateway.metrics import GatewayMetrics
from rollout_runtime.gateway.operation_registry import (
    OperationRecord,
    OperationRegistry,
    request_digest,
)
from rollout_runtime.gateway.plugin import AdapterPlugin, PluginExecutor
from rollout_runtime.gateway.session_manager import (
    ALLOWED_SESSION_TRANSITIONS,
    SessionManager,
    SessionRecord,
)
from rollout_runtime.gateway.worker_registry import EnvWorkerRegistry, WorkerEntry

__all__ = [
    "ALLOWED_SESSION_TRANSITIONS",
    "AdapterPlugin",
    "AdmissionConfig",
    "AdmissionController",
    "CommandDispatcher",
    "EnvWorkerRegistry",
    "GatewayMetrics",
    "OperationRecord",
    "OperationRegistry",
    "PluginExecutor",
    "RuntimeGateway",
    "SessionManager",
    "SessionRecord",
    "WorkerEntry",
    "request_digest",
]
