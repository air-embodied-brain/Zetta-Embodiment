"""Execution-core interfaces and obs / payload encoding-decoding.

Layering constraint: this subpackage must not import ``ray`` / ``rlinf``;
its dependency surface is stdlib + numpy. The real env / policy backends
are in ``rollout_runtime.backends``.
"""

from __future__ import annotations

from rollout_runtime.core.env_execution import (
    ChunkOutcome,
    EnvExecutionCore,
    EnvFamilyBehavior,
    normalize_chunk_outcome,
)
from rollout_runtime.core.env_registry import (
    ENV_FAMILY_BEHAVIORS,
    ENV_FAMILY_REGISTRY,
    EnvFamilyAdapter,
    behavior_for,
    capability_from_behavior,
    get_env_family,
    register_env_family,
    validate_env_spec,
)
from rollout_runtime.core.obs_schema import (
    ENV_OUTPUT_KEYS,
    ObsSchema,
    obs_schema_digest,
    observations_to_env_output,
    schema_of,
)
from rollout_runtime.core.payload import (
    INLINE_THRESHOLD_BYTES,
    REQUEST_PAYLOAD_LIMIT_BYTES,
    check_payload_budget,
    decode_payload,
    encode_array,
    encode_image,
    encode_payload,
)
from rollout_runtime.core.policy_inference import (
    BATCHABLE_PARAM_KEYS,
    PolicyInferenceCore,
    canonicalize_inference_parameters,
    compute_compat_key,
)

__all__ = [
    "BATCHABLE_PARAM_KEYS",
    "ENV_FAMILY_BEHAVIORS",
    "ENV_FAMILY_REGISTRY",
    "ENV_OUTPUT_KEYS",
    "INLINE_THRESHOLD_BYTES",
    "REQUEST_PAYLOAD_LIMIT_BYTES",
    "ChunkOutcome",
    "EnvExecutionCore",
    "EnvFamilyAdapter",
    "EnvFamilyBehavior",
    "ObsSchema",
    "PolicyInferenceCore",
    "behavior_for",
    "canonicalize_inference_parameters",
    "capability_from_behavior",
    "check_payload_budget",
    "compute_compat_key",
    "decode_payload",
    "encode_array",
    "encode_image",
    "encode_payload",
    "get_env_family",
    "normalize_chunk_outcome",
    "obs_schema_digest",
    "observations_to_env_output",
    "register_env_family",
    "schema_of",
    "validate_env_spec",
]
