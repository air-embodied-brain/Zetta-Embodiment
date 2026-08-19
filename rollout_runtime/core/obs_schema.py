"""The unified observation schema and its digest.

Each env family's private ``_wrap_obs`` is uniformly normalized to the
5-key schema of rlinf ``EnvOutput.prepare_observations``
(``rlinf/data/embodied_io_struct.py:108``); the Runtime's ``Observation``
corresponds to it one-to-one.

``obs_schema_digest`` is half of the basis for batching compatibility:
``huggingface_worker._merge_obs_batches`` requires the obs dict structure
to be fully identical within a batch, so shape/dtype/field-presence must be
part of ``compat_key``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import numpy as np

from rollout_runtime.api import codec
from rollout_runtime.api.messages import Observation
from rollout_runtime.api.payload_ref import InlineBytes, PayloadRef
from rollout_runtime.core import payload as payload_module

__all__ = [
    "ENV_OUTPUT_KEYS",
    "OBS_FIELD_TO_ENV_OUTPUT_KEY",
    "FieldSpec",
    "ObsSchema",
    "obs_schema_digest",
    "observations_to_env_output",
    "schema_of",
]

ENV_OUTPUT_KEYS = (
    "main_images",
    "wrist_images",
    "extra_view_images",
    "states",
    "task_descriptions",
)
"""The 5 keys of rlinf ``EnvOutput.prepare_observations``."""

OBS_FIELD_TO_ENV_OUTPUT_KEY = {
    "main_image": "main_images",
    "wrist_image": "wrist_images",
    "extra_view_images": "extra_view_images",
    "state": "states",
    "instruction": "task_descriptions",
}
"""Mapping from ``Observation`` fields to the 5-key schema."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class FieldSpec:
    """The structural description of a single observation field.

    Attributes:
        present: Whether the field is present.
        shape: The logical shape.
        dtype: The dtype name.
    """

    present: bool = False
    shape: tuple[int, ...] = ()
    dtype: str = ""


@dataclasses.dataclass(frozen=True, kw_only=True)
class ObsSchema:
    """The structure of an observation (excluding values), used for
    batching-compatibility decisions.

    Attributes:
        main_image: The field description of the main view.
        wrist_image: The field description of the wrist view.
        extra_view_images: The field descriptions of the remaining views.
        state_dim: The state-vector dimension.
        has_instruction: Whether an instruction text is present.
        extra_keys: The (sorted) key set of ``extras``.
    """

    main_image: FieldSpec = FieldSpec()
    wrist_image: FieldSpec = FieldSpec()
    extra_view_images: tuple[FieldSpec, ...] = ()
    state_dim: int = 0
    has_instruction: bool = False
    extra_keys: tuple[str, ...] = ()

    def digest(self) -> str:
        """Return the structural digest.

        Returns:
            A 64-character hex sha256 digest.
        """
        return codec.digest(self, prefix="rollout_runtime/obs_schema/v1")


def _field_spec(ref: PayloadRef | None) -> FieldSpec:
    if ref is None:
        return FieldSpec()
    if isinstance(ref, InlineBytes):
        return FieldSpec(present=True, shape=tuple(ref.shape), dtype=ref.dtype)
    return FieldSpec(present=True, shape=tuple(ref.shape), dtype=ref.dtype)


def schema_of(observation: Observation) -> ObsSchema:
    """Extract the structural description of an observation.

    Args:
        observation: A single observation.

    Returns:
        The structural description.
    """
    return ObsSchema(
        main_image=_field_spec(observation.main_image),
        wrist_image=_field_spec(observation.wrist_image),
        extra_view_images=tuple(
            _field_spec(ref) for ref in observation.extra_view_images
        ),
        state_dim=len(observation.state),
        has_instruction=bool(observation.instruction),
        extra_keys=tuple(sorted(observation.extras)),
    )


def obs_schema_digest(observation: Observation) -> str:
    """Return the structural digest of an observation.

    Args:
        observation: A single observation.

    Returns:
        A 64-character hex sha256 digest.
    """
    return schema_of(observation).digest()


def observations_to_env_output(
    observations: Sequence[Observation],
) -> dict[str, Any]:
    """Convert a batch of observations into a batch dict following the
    5-key schema.

    Args:
        observations: A sequence of observations with identical structure.

    Returns:
        The five-key ``ENV_OUTPUT_KEYS`` dictionary; missing fields are
        ``None``.

    Raises:
        ValueError: The input is empty, or the structure is inconsistent
            within the batch.
    """
    if not observations:
        raise ValueError("observations must not be empty")
    reference = schema_of(observations[0])
    for index, observation in enumerate(observations[1:], start=1):
        if schema_of(observation) != reference:
            raise ValueError(
                f"observation schema mismatch at index {index}; "
                "batching requires identical obs structure"
            )

    def _stack(refs: list[PayloadRef | None]) -> np.ndarray | None:
        if any(ref is None for ref in refs):
            return None
        return np.stack([payload_module.decode_payload(ref) for ref in refs])

    main = _stack([observation.main_image for observation in observations])
    wrist = _stack([observation.wrist_image for observation in observations])
    extra: np.ndarray | None = None
    if reference.extra_view_images:
        extra = np.stack(
            [
                np.stack(
                    [
                        payload_module.decode_payload(ref)
                        for ref in observation.extra_view_images
                    ]
                )
                for observation in observations
            ]
        )
    states = (
        np.asarray(
            [observation.state for observation in observations], dtype=np.float32
        )
        if reference.state_dim
        else None
    )
    return {
        "main_images": main,
        "wrist_images": wrist,
        "extra_view_images": extra,
        "states": states,
        "task_descriptions": [observation.instruction for observation in observations],
    }
