"""``runtime_api/v1``: the Runtime's external protocol.

Layering constraint: this subpackage only allows stdlib, with the sole
exception that ``api.wire`` may import ``msgpack``. numpy / torch / ray /
rlinf are forbidden.
"""

from __future__ import annotations

from rollout_runtime.api.client import Consistency, RuntimeClient
from rollout_runtime.api.enums import (
    MUTATING_OPERATIONS,
    EnvOperation,
    ErrorCode,
    OperationState,
    Priority,
    SessionState,
)
from rollout_runtime.api.errors import (
    InvalidTransition,
    RuntimeApiError,
    RuntimeErrorInfo,
    make_error,
    normalize_exception,
)
from rollout_runtime.api.ids import (
    BindingToken,
    EpisodeId,
    OperationSeq,
    RequestId,
    SessionId,
    new_binding_token,
    new_request_id,
    new_session_id,
)
from rollout_runtime.api.internal import (
    ActionResponse,
    CommandEnvelope,
    ControlEnvelope,
    InferenceRequest,
    ResultEnvelope,
    make_routing_token,
    parse_routing_token,
)
from rollout_runtime.api.messages import (
    CancelOutcome,
    CreateSessionRequest,
    EnvFamilyCapability,
    EnvSpecMsg,
    EnvWorkerInfo,
    EpisodeRequest,
    EpisodeResult,
    Observation,
    OperationStatus,
    PerStepRecord,
    PolicyInferResult,
    PolicyRequest,
    ResetSpec,
    SessionHandle,
    SessionStatus,
    StepResult,
    WorkerSummary,
)
from rollout_runtime.api.payload_ref import (
    InlineBytes,
    ObjectRefId,
    PayloadCodec,
    PayloadRef,
)
from rollout_runtime.api.result import Err, Ok, Result, err, ok, unwrap

API_VERSION = "v1"
"""The protocol version; incremented on breaking changes."""

__all__ = [
    "API_VERSION",
    "MUTATING_OPERATIONS",
    "ActionResponse",
    "BindingToken",
    "CancelOutcome",
    "CommandEnvelope",
    "Consistency",
    "ControlEnvelope",
    "CreateSessionRequest",
    "EnvFamilyCapability",
    "EnvOperation",
    "EnvSpecMsg",
    "EnvWorkerInfo",
    "EpisodeId",
    "EpisodeRequest",
    "EpisodeResult",
    "Err",
    "ErrorCode",
    "InferenceRequest",
    "InlineBytes",
    "InvalidTransition",
    "ObjectRefId",
    "Observation",
    "Ok",
    "OperationSeq",
    "OperationState",
    "OperationStatus",
    "PayloadCodec",
    "PayloadRef",
    "PerStepRecord",
    "PolicyInferResult",
    "PolicyRequest",
    "Priority",
    "RequestId",
    "ResetSpec",
    "Result",
    "ResultEnvelope",
    "RuntimeApiError",
    "RuntimeClient",
    "RuntimeErrorInfo",
    "SessionHandle",
    "SessionId",
    "SessionState",
    "SessionStatus",
    "StepResult",
    "WorkerSummary",
    "err",
    "make_error",
    "make_routing_token",
    "new_binding_token",
    "new_request_id",
    "new_session_id",
    "normalize_exception",
    "ok",
    "parse_routing_token",
    "unwrap",
]
