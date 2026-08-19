"""Inference-core interface, batching-compatibility keys, and the
mixable-parameter allowlist.

``compat_key`` determines "which requests can go into the same batch". The
definition is:

``sha256(policy_id | model_version | obs_schema_digest | canonical(inference_parameters)
| device | dtype)``

``inference_parameters`` is first canonicalized: only keys that truly affect
the graph/shape enter the key (``num_steps``, ``mode``, ``max_prompt_length``,
etc.). Sampling-temperature-like parameters belong to the "mixable" allowlist
and are excluded.

This module only holds the interface and pure computation; real model
loading lives in ``backends/rlinf_policy.py``, keeping ``core``'s dependency
surface at stdlib + numpy.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any, Protocol

from rollout_runtime.api import codec
from rollout_runtime.api.internal import ActionResponse, InferenceRequest

__all__ = [
    "BATCHABLE_PARAM_KEYS",
    "DEFAULT_POLICY_FAMILY",
    "PolicyInferenceCore",
    "batchable_param_keys",
    "canonicalize_inference_parameters",
    "compute_compat_key",
]

DEFAULT_POLICY_FAMILY = "*"
"""The allowlist keys shared by all families."""

BATCHABLE_PARAM_KEYS: dict[str, frozenset[str]] = {
    DEFAULT_POLICY_FAMILY: frozenset(
        {
            "do_sample",
            "repetition_penalty",
            "seed",
            "temperature",
            "top_k",
            "top_p",
        }
    ),
    "openvla": frozenset(),
    # openvla_oft's padding="max_length" is fixed and never appears as a
    # parameter; sampling-related keys follow the default allowlist.
    # It enters compat_key as a **hard constraint** instead (see
    # backends/rlinf_policy.py::RlinfPolicyConfig.compat_key_constraints).
    "openvla_oft": frozenset(),
    # openpi (pi0 / pi0.5, the only real inference path): only keys that
    # "change sampling noise without changing the graph/shape" may be
    # mixed into the same batch. Deliberately **not** allowing `mode`
    # (train/eval take different branches), `num_steps` (the number of flow
    # sampling steps, which directly determines the number of forward
    # passes), or `actions_per_chunk` / `action_dim` (which change shape and
    # are already hard constraints). Better to over-bucket than to batch
    # incorrectly.
    "openpi": frozenset({"noise_seed"}),
    "openpi_pytorch": frozenset({"noise_seed"}),
    "pi0": frozenset({"noise_seed"}),
    "pi05": frozenset({"noise_seed"}),
    "fake": frozenset(),
}
"""Per-policy-family declarations of "mixable" parameter keys: requests
differing only in these keys may still enter the same batch.

Any key not listed is treated as affecting the graph/shape and enters
``compat_key``. Better to over-bucket than to batch incorrectly.
"""


def batchable_param_keys(policy_family: str) -> frozenset[str]:
    """Return the mixable parameter keys for a given family.

    Args:
        policy_family: The policy family name.

    Returns:
        The union of the default allowlist and the family-specific
        allowlist.
    """
    return BATCHABLE_PARAM_KEYS[DEFAULT_POLICY_FAMILY] | BATCHABLE_PARAM_KEYS.get(
        policy_family, frozenset()
    )


def canonicalize_inference_parameters(
    parameters: Mapping[str, Any], policy_family: str = DEFAULT_POLICY_FAMILY
) -> dict[str, Any]:
    """Strip mixable parameters and sort the remaining keys.

    Args:
        parameters: The raw inference parameters.
        policy_family: The policy family name.

    Returns:
        An ordered dict containing only the keys that affect graph/shape.
    """
    skip = batchable_param_keys(policy_family)
    return {key: parameters[key] for key in sorted(parameters) if key not in skip}


def compute_compat_key(
    *,
    policy_id: str,
    model_version: str,
    obs_schema_digest: str,
    inference_parameters: Mapping[str, Any],
    device: str,
    dtype: str,
    policy_family: str = DEFAULT_POLICY_FAMILY,
    constraints: Mapping[str, Any] | None = None,
) -> str:
    """Compute the batching-compatibility key.

    Args:
        policy_id: The policy identifier.
        model_version: The current model version (the version fence
            depends on it; a batch never mixes versions).
        obs_schema_digest: The observation structure digest (see
            ``core.obs_schema``).
        inference_parameters: The inference parameters; canonicalized
            internally first.
        device: The device identifier (e.g. ``"cuda:0"``).
        dtype: The compute precision (e.g. ``"bfloat16"``).
        policy_family: The policy family name, determining the mixable
            allowlist.
        constraints: Additional hard constraints (e.g. the fixed batch size
            when cuda_graph is enabled), which enter the key.

    Returns:
        A 64-bit hex sha256 digest.
    """
    parts = [
        policy_id,
        model_version,
        obs_schema_digest,
        codec.canonical_json(
            canonicalize_inference_parameters(inference_parameters, policy_family)
        ),
        device,
        dtype,
        codec.canonical_json(dict(constraints or {})),
    ]
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()


class PolicyInferenceCore(Protocol):
    """The inference-core interface.

    Completely independent of transport / session. ``backends/rlinf_policy.py``
    provides the concrete implementation (model loading, precision,
    torch_compile, cuda_graph, sampling-parameter assembly, the ``predict``
    body, and weight sync, extracted from ``huggingface_worker.py``).
    """

    @property
    def model_version(self) -> str:
        """The current model version.

        Returns:
            The version identifier; changes after ``update_weights``.
        """
        ...

    @property
    def device(self) -> str:
        """The device the model resides on.

        Returns:
            The device identifier.
        """
        ...

    @property
    def dtype(self) -> str:
        """The model's compute precision.

        Returns:
            The dtype name.
        """
        ...

    def load(self) -> None:
        """Load the model onto the accelerator assigned to the current
        rank."""
        ...

    def infer_batch(self, requests: list[InferenceRequest]) -> list[ActionResponse]:
        """Run inference on an already-bucketed batch.

        Args:
            requests: A list of requests sharing the same ``compat_key``.

        Returns:
            Per-request responses in the same order as the input.
        """
        ...

    def update_weights(self, model_version: str) -> None:
        """Switch weights at a batch boundary.

        Args:
            model_version: The target version.
        """
        ...

    def close(self) -> None:
        """Release the model and communication resources."""
        ...
