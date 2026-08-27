# Copyright (c) 2026 Zetta Contributors
"""Remote Cosmos-Lite policy backend.

Cosmos-Lite v0.3.0 already exposes the OpenPI WebSocket policy protocol, so
this module deliberately contains no model code.  It validates the deployment
record written by Cosmos-Lite, translates the Runtime's generic observation to
the upstream RoboLab/DROID request, and converts the returned action chunk to
an :class:`ActionResponse`.

The upstream server is batch-one and serializes inference internally.  This
backend therefore performs one request at a time and never retries a request
after it has been sent: a disconnected response has an unknown server-side
outcome, even though policy inference itself has no environment side effect.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import ipaddress
import json
import re
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import msgpack
import numpy as np

from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.errors import make_error
from rollout_runtime.api.internal import ActionResponse, InferenceRequest
from rollout_runtime.core import payload as payload_module

__all__ = [
    "COSMOS_LITE_POLICY_FAMILY",
    "COSMOS_LITE_V030_REVISION",
    "CosmosLiteClient",
    "CosmosLiteModelIdentity",
    "CosmosLitePolicyConfig",
    "CosmosLitePolicyCore",
    "CosmosLiteTransport",
]

COSMOS_LITE_POLICY_FAMILY = "cosmos_lite"
COSMOS_LITE_V030_REVISION = "2b5f5f3e8c02632f04432644ac0bf31780f554b5"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
_IMAGE_LAYOUTS = frozenset({"single", "robolab_three_view"})


class CosmosLiteTransport(Protocol):
    """Small injectable boundary around the upstream WebSocket protocol."""

    def connect(self) -> Mapping[str, Any]:
        """Connect and return the server handshake metadata."""
        ...

    def infer(
        self, observation: Mapping[str, Any], *, timeout_s: float
    ) -> Mapping[str, Any]:
        """Send one observation and return one decoded response."""
        ...

    def close(self) -> None:
        """Close the connection."""
        ...


class _RequestValidationError(ValueError):
    """The Runtime request cannot be represented by the upstream contract."""


class _ResponseValidationError(RuntimeError):
    """The upstream service returned a malformed response."""


def _pack_numpy(value: Any) -> Any:
    """Encode NumPy values exactly as ``openpi_client.msgpack_numpy`` does."""
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in (
        "V",
        "O",
        "c",
    ):
        raise ValueError(f"unsupported NumPy dtype: {value.dtype}")
    if isinstance(value, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    raise TypeError(f"cannot encode value of type {type(value).__name__}")


def _unpack_numpy(value: dict[Any, Any]) -> Any:
    """Decode NumPy values encoded by ``openpi_client.msgpack_numpy``."""
    if b"__ndarray__" in value:
        return np.ndarray(
            buffer=value[b"data"],
            dtype=np.dtype(value[b"dtype"]),
            shape=tuple(value[b"shape"]),
        )
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


def _pack_message(value: Mapping[str, Any]) -> bytes:
    return msgpack.packb(value, default=_pack_numpy, use_bin_type=True)


def _unpack_message(value: bytes) -> Any:
    return msgpack.unpackb(
        value,
        object_hook=_unpack_numpy,
        raw=False,
        strict_map_key=False,
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class CosmosLiteModelIdentity:
    """Auditable identity extracted from a Cosmos-Lite deployment record."""

    repository_revision: str
    model_family: str
    artifact: str
    strategy: str | None
    manifest_sha256: str
    resolved_config_sha256: str
    profile: str
    fallback_decisions: tuple[str, ...]
    sampling: dict[str, Any]
    runtime: dict[str, Any]
    runtime_probe: dict[str, Any]
    model_version: str

    def auxiliary_output(self) -> dict[str, Any]:
        """Return a msgpack-native representation for ``ActionResponse``."""
        return {
            "repository_revision": self.repository_revision,
            "model_family": self.model_family,
            "artifact": self.artifact,
            "strategy": self.strategy,
            "manifest_sha256": self.manifest_sha256,
            "resolved_config_sha256": self.resolved_config_sha256,
            "profile": self.profile,
            "fallback_decisions": list(self.fallback_decisions),
            "runtime_probe": dict(self.runtime_probe),
        }


@dataclasses.dataclass(kw_only=True)
class CosmosLitePolicyConfig:
    """Configuration for the remote Cosmos-Lite policy service.

    ``resolved_config_path`` points to the record generated by the upstream
    ``action_policy_server_robolab_deploy`` entry point.  Initial support is
    intentionally restricted to a verified quantized bundle: the upstream
    BF16 preset follows a mutable Hugging Face revision and therefore cannot
    satisfy this backend's fail-closed identity contract.
    """

    endpoint: str = "ws://127.0.0.1:8000"
    resolved_config_path: str = ""
    expected_manifest_sha256: str = ""
    expected_repository_revision: str = COSMOS_LITE_V030_REVISION
    action_dim: int = 8
    actions_per_chunk: int = 32
    joint_position_dim: int = 7
    image_layout: str = "robolab_three_view"
    connect_timeout_s: float = 5.0
    request_timeout_s: float = 120.0
    maximum_payload_bytes: int = 8 * 1024 * 1024
    allow_insecure_remote: bool = False
    allow_dirty_upstream: bool = False
    allow_runtime_fallbacks: bool = False
    device: str = "cpu"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        parsed = urlparse(self.endpoint)
        if parsed.scheme not in ("ws", "wss") or not parsed.hostname:
            raise ValueError("endpoint must be an absolute ws:// or wss:// URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError(
                "endpoint must not contain credentials, query, or fragment"
            )
        if parsed.scheme == "ws" and not self.allow_insecure_remote:
            try:
                loopback = ipaddress.ip_address(parsed.hostname).is_loopback
            except ValueError:
                loopback = parsed.hostname.lower() == "localhost"
            if not loopback:
                raise ValueError(
                    "plain ws:// is restricted to loopback; use wss:// or set "
                    "allow_insecure_remote=true behind a trusted transport"
                )
        if not self.resolved_config_path:
            raise ValueError("resolved_config_path is required")
        manifest = self.expected_manifest_sha256.lower()
        if _SHA256_RE.fullmatch(manifest) is None:
            raise ValueError("expected_manifest_sha256 must be 64 lowercase hex chars")
        self.expected_manifest_sha256 = manifest
        revision = self.expected_repository_revision.lower()
        if _GIT_SHA_RE.fullmatch(revision) is None:
            raise ValueError(
                "expected_repository_revision must be a full 40-character Git SHA"
            )
        self.expected_repository_revision = revision
        if self.action_dim <= 0 or self.actions_per_chunk <= 0:
            raise ValueError("action_dim and actions_per_chunk must be positive")
        if self.joint_position_dim <= 0:
            raise ValueError("joint_position_dim must be positive")
        if self.image_layout not in _IMAGE_LAYOUTS:
            raise ValueError(
                f"image_layout must be one of {sorted(_IMAGE_LAYOUTS)}, "
                f"got {self.image_layout!r}"
            )
        if self.connect_timeout_s <= 0 or self.request_timeout_s <= 0:
            raise ValueError("connect_timeout_s and request_timeout_s must be positive")
        if self.maximum_payload_bytes <= 0:
            raise ValueError("maximum_payload_bytes must be positive")
        if self.device != "cpu":
            raise ValueError(
                "Cosmos-Lite is a remote backend and requires device='cpu'"
            )
        if self.dtype != "float32":
            raise ValueError("Cosmos-Lite action transport requires dtype='float32'")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> CosmosLitePolicyConfig:
        """Construct from ``rollout_worker.policy_config``."""
        payload = dict(config or {})
        known = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"unknown cosmos_lite policy_config keys: {unknown}")
        return cls(**payload)

    def compat_key_constraints(self) -> dict[str, Any]:
        """Return shape and service constraints for request bucketing."""
        return {
            "endpoint": self.endpoint,
            "expected_manifest_sha256": self.expected_manifest_sha256,
            "action_dim": self.action_dim,
            "actions_per_chunk": self.actions_per_chunk,
            "joint_position_dim": self.joint_position_dim,
            "image_layout": self.image_layout,
        }


def _as_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"resolved deployment config field {name!r} must be a map")
    return value


def _load_model_identity(config: CosmosLitePolicyConfig) -> CosmosLiteModelIdentity:
    path = Path(config.resolved_config_path).expanduser()
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise ValueError(
            f"cannot read resolved deployment config {path}: {exc}"
        ) from exc
    try:
        record = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid resolved deployment config {path}: {exc}") from exc
    root = _as_mapping(record, "root")
    if root.get("schema_version") != 1:
        raise ValueError(
            f"unsupported resolved deployment schema: {root.get('schema_version')!r}"
        )
    repository = _as_mapping(root.get("repository"), "repository")
    revision = str(repository.get("revision") or "").lower()
    if revision != config.expected_repository_revision:
        raise ValueError(
            "Cosmos-Lite repository revision mismatch: "
            f"expected {config.expected_repository_revision}, got {revision or '<missing>'}"
        )
    if repository.get("dirty") is not False and not config.allow_dirty_upstream:
        raise ValueError(
            "Cosmos-Lite repository must be clean for a verified deployment"
        )

    model = _as_mapping(root.get("model"), "model")
    artifact = str(model.get("artifact") or "")
    if artifact != "quantized_bundle":
        raise ValueError(
            "initial Cosmos-Lite backend requires artifact='quantized_bundle' "
            "so model weights have an immutable manifest"
        )
    manifest_sha256 = str(model.get("manifest_sha256") or "").lower()
    if manifest_sha256 != config.expected_manifest_sha256:
        raise ValueError(
            "Cosmos-Lite manifest mismatch: "
            f"expected {config.expected_manifest_sha256}, "
            f"got {manifest_sha256 or '<missing>'}"
        )

    bundle = _as_mapping(root.get("bundle"), "bundle")
    bundle_identity = {
        "manifest_sha256": str(bundle.get("manifest_sha256") or "").lower(),
        "model_family": str(bundle.get("model_family") or ""),
        "strategy": str(bundle.get("strategy") or ""),
    }
    model_identity = {
        "manifest_sha256": manifest_sha256,
        "model_family": str(model.get("model_family") or ""),
        "strategy": str(model.get("strategy") or ""),
    }
    if bundle_identity != model_identity:
        raise ValueError(
            "Cosmos-Lite bundle identity does not match resolved model identity"
        )

    fallbacks_value = root.get("fallback_decisions", [])
    if not isinstance(fallbacks_value, list) or not all(
        isinstance(item, str) for item in fallbacks_value
    ):
        raise ValueError("resolved deployment fallback_decisions must be a string list")
    fallbacks = tuple(fallbacks_value)
    if fallbacks and not config.allow_runtime_fallbacks:
        raise ValueError(
            "Cosmos-Lite resolved deployment contains runtime fallbacks: "
            f"{list(fallbacks)}"
        )

    effective = _as_mapping(root.get("effective"), "effective")
    profile = str(effective.get("profile") or "")
    model_family = str(model.get("model_family") or "")
    strategy_value = model.get("strategy")
    strategy = str(strategy_value) if strategy_value is not None else None
    if not profile or not model_family or not strategy:
        raise ValueError(
            "resolved deployment must include profile, model_family, and strategy"
        )
    sampling = dict(_as_mapping(effective.get("sampling"), "effective.sampling"))
    runtime = dict(_as_mapping(effective.get("runtime"), "effective.runtime"))
    if sampling.get("deterministic_seed") is not True:
        raise ValueError("Cosmos-Lite service must use deterministic_seed=true")
    runtime_probe = dict(_as_mapping(root.get("runtime_probe"), "runtime_probe"))
    if runtime_probe.get("cuda_available") is not True:
        raise ValueError("Cosmos-Lite runtime_probe must report cuda_available=true")
    capability = runtime_probe.get("compute_capability")
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in capability
        )
    ):
        raise ValueError(
            "Cosmos-Lite runtime_probe.compute_capability must be two integers"
        )

    resolved_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    stable_identity = {
        "repository_revision": revision,
        "model": {
            "artifact": artifact,
            "model_family": model_family,
            "strategy": strategy,
            "manifest_sha256": manifest_sha256,
        },
        "sampling": sampling,
        "runtime": runtime,
        "runtime_probe": runtime_probe,
        "fallback_decisions": list(fallbacks),
        "interface": {
            "action_dim": config.action_dim,
            "actions_per_chunk": config.actions_per_chunk,
            "joint_position_dim": config.joint_position_dim,
            "image_layout": config.image_layout,
        },
    }
    identity_sha256 = hashlib.sha256(
        json.dumps(
            stable_identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    model_version = f"cosmos-lite:{model_family}:{identity_sha256[:16]}"
    return CosmosLiteModelIdentity(
        repository_revision=revision,
        model_family=model_family,
        artifact=artifact,
        strategy=strategy,
        manifest_sha256=manifest_sha256,
        resolved_config_sha256=resolved_sha256,
        profile=profile,
        fallback_decisions=fallbacks,
        sampling=sampling,
        runtime=runtime,
        runtime_probe=runtime_probe,
        model_version=model_version,
    )


class CosmosLiteClient:
    """Minimal synchronous client for Cosmos-Lite's OpenPI WebSocket server."""

    def __init__(self, config: CosmosLitePolicyConfig) -> None:
        self.config = config
        self._connection: Any = None
        self._metadata: dict[str, Any] = {}

    def connect(self) -> Mapping[str, Any]:
        """Open the socket and consume the OpenPI metadata frame."""
        if self._connection is not None:
            return dict(self._metadata)
        try:
            from websockets.sync.client import connect
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Cosmos-Lite support requires the 'cosmos-lite' extra: "
                "python -m pip install -e '.[cosmos-lite]'"
            ) from exc
        connection = connect(
            self.config.endpoint,
            compression=None,
            max_size=self.config.maximum_payload_bytes,
            open_timeout=self.config.connect_timeout_s,
            close_timeout=min(self.config.connect_timeout_s, 2.0),
        )
        try:
            frame = connection.recv(timeout=self.config.connect_timeout_s)
            if isinstance(frame, str):
                raise RuntimeError(f"Cosmos-Lite handshake failed: {frame}")
            metadata = _unpack_message(bytes(frame))
            if not isinstance(metadata, Mapping):
                raise RuntimeError("Cosmos-Lite handshake metadata must be a map")
        except BaseException:
            connection.close()
            raise
        self._connection = connection
        self._metadata = dict(metadata)
        return dict(self._metadata)

    def infer(
        self, observation: Mapping[str, Any], *, timeout_s: float
    ) -> Mapping[str, Any]:
        """Send one upstream-format observation without automatic retry."""
        if self._connection is None:
            self.connect()
        assert self._connection is not None
        request = _pack_message(observation)
        if len(request) > self.config.maximum_payload_bytes:
            raise _RequestValidationError(
                "Cosmos-Lite request exceeds maximum_payload_bytes: "
                f"{len(request)} > {self.config.maximum_payload_bytes}"
            )
        try:
            self._connection.send(request)
            frame = self._connection.recv(timeout=timeout_s)
            if isinstance(frame, str):
                raise RuntimeError(f"Cosmos-Lite server error: {frame}")
            response_bytes = bytes(frame)
            if len(response_bytes) > self.config.maximum_payload_bytes:
                raise _ResponseValidationError(
                    "Cosmos-Lite response exceeds maximum_payload_bytes"
                )
            response = _unpack_message(response_bytes)
            if not isinstance(response, Mapping):
                raise _ResponseValidationError("Cosmos-Lite response must be a map")
            return response
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Close the current connection, if any."""
        connection, self._connection = self._connection, None
        self._metadata = {}
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass


class CosmosLitePolicyCore:
    """``PolicyInferenceCore`` backed by a remote Cosmos-Lite service."""

    def __init__(
        self,
        config: CosmosLitePolicyConfig,
        *,
        transport_factory: Callable[[CosmosLitePolicyConfig], CosmosLiteTransport]
        | None = None,
    ) -> None:
        self.config = config
        self._transport_factory = transport_factory or CosmosLiteClient
        self._transport: CosmosLiteTransport | None = None
        self._identity: CosmosLiteModelIdentity | None = None
        self._server_metadata: dict[str, Any] = {}
        self._lock = threading.Lock()
        self.loaded = False
        self.closed = False
        self.batch_calls = 0
        self.request_count = 0
        self.error_count = 0

    @property
    def model_version(self) -> str:
        """Return the verified deployment identity."""
        if self._identity is not None:
            return self._identity.model_version
        return "cosmos-lite-unloaded"

    @property
    def device(self) -> str:
        """Return the client device (the model lives in another process)."""
        return self.config.device

    @property
    def dtype(self) -> str:
        """Return the action transport dtype."""
        return self.config.dtype

    @property
    def policy_family(self) -> str:
        """Return the Runtime policy family name."""
        return COSMOS_LITE_POLICY_FAMILY

    def load(self) -> None:
        """Verify deployment identity and connect to the upstream service."""
        with self._lock:
            if self.loaded:
                return
            if self.closed:
                raise RuntimeError("cannot load a closed Cosmos-Lite backend")
            identity = _load_model_identity(self.config)
            transport = self._transport_factory(self.config)
            try:
                metadata = transport.connect()
            except BaseException:
                transport.close()
                raise
            self._identity = identity
            self._transport = transport
            self._server_metadata = dict(metadata)
            self.loaded = True

    def update_weights(self, model_version: str) -> None:
        """Reject hot swapping; the independently served model must restart."""
        del model_version
        raise RuntimeError(
            "Cosmos-Lite does not support runtime weight updates; restart the "
            "service and the RolloutWorker with a new verified manifest"
        )

    def close(self) -> None:
        """Close the WebSocket connection."""
        with self._lock:
            transport, self._transport = self._transport, None
            if transport is not None:
                transport.close()
            self.loaded = False
            self.closed = True

    def infer_batch(self, requests: list[InferenceRequest]) -> list[ActionResponse]:
        """Execute requests serially against the upstream batch-one server."""
        if not requests:
            return []
        self.batch_calls += 1
        self.request_count += len(requests)
        return [self._infer_one(request) for request in requests]

    def _infer_one(self, request: InferenceRequest) -> ActionResponse:
        try:
            with self._lock:
                if not self.loaded or self._transport is None or self._identity is None:
                    raise RuntimeError("Cosmos-Lite backend is not loaded")
                timeout_s = self._request_timeout(request)
                observation = self._build_observation(request)
                packed_size = len(_pack_message(observation))
                if packed_size > self.config.maximum_payload_bytes:
                    raise _RequestValidationError(
                        "Cosmos-Lite request exceeds maximum_payload_bytes: "
                        f"{packed_size} > {self.config.maximum_payload_bytes}"
                    )
                response = self._transport.infer(observation, timeout_s=timeout_s)
                actions = self._validate_response(response)
                identity = self._identity
        except _RequestValidationError as exc:
            return self._error_response(request, ErrorCode.INVALID_ARGUMENT, exc)
        except (TimeoutError, asyncio.TimeoutError) as exc:
            return self._error_response(request, ErrorCode.DEADLINE_EXCEEDED, exc)
        except BaseException as exc:  # noqa: BLE001 - worker boundary must be total
            return self._error_response(request, ErrorCode.POLICY_FAILURE, exc)

        timing = response.get("server_timing")
        timing_output = (
            {
                str(key): float(value)
                for key, value in timing.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            if isinstance(timing, Mapping)
            else {}
        )
        return ActionResponse(
            request_id=request.request_id,
            session_id=request.session_id,
            binding_token=request.binding_token,
            episode_id=request.episode_id,
            operation_seq=request.operation_seq,
            actions=payload_module.encode_array(actions),
            model_version=identity.model_version,
            auxiliary_outputs={
                "chunk": int(actions.shape[0]),
                "compat_key": request.compat_key,
                "server_timing": timing_output,
                "server_metadata_present": bool(self._server_metadata),
                "cosmos_lite_identity": identity.auxiliary_output(),
            },
        )

    def _request_timeout(self, request: InferenceRequest) -> float:
        timeout_s = self.config.request_timeout_s
        if request.deadline is not None:
            remaining = request.deadline - time.time()
            if remaining <= 0:
                raise TimeoutError("Cosmos-Lite request deadline already expired")
            timeout_s = min(timeout_s, remaining)
        return timeout_s

    def _build_observation(self, request: InferenceRequest) -> dict[str, Any]:
        observation = request.observation
        instruction = (
            request.instruction_override
            if request.instruction_override is not None
            else observation.instruction
        )
        if not isinstance(instruction, str):
            raise _RequestValidationError("instruction must be a string")

        try:
            state = np.asarray(observation.state, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise _RequestValidationError(
                f"Cosmos-Lite state is not numeric: {exc}"
            ) from exc
        expected_state_dim = self.config.joint_position_dim + 1
        if state.shape != (expected_state_dim,):
            raise _RequestValidationError(
                "Cosmos-Lite joint-position contract requires state shape "
                f"({expected_state_dim},), got {state.shape}"
            )
        if not np.isfinite(state).all():
            raise _RequestValidationError("Cosmos-Lite state contains NaN or Inf")

        upstream: dict[str, Any] = {
            "prompt": instruction,
            "observation/joint_position": np.ascontiguousarray(
                state[: self.config.joint_position_dim]
            ),
            "observation/gripper_position": np.ascontiguousarray(state[-1:]),
        }
        try:
            if self.config.image_layout == "single":
                if observation.main_image is None:
                    raise _RequestValidationError(
                        "single image_layout requires Observation.main_image"
                    )
                upstream["observation/image"] = payload_module.decode_image(
                    observation.main_image
                )
            else:
                if (
                    observation.main_image is None
                    or observation.wrist_image is None
                    or not observation.extra_view_images
                ):
                    raise _RequestValidationError(
                        "robolab_three_view requires main_image, wrist_image, "
                        "and one extra_view_image"
                    )
                upstream["observation/wrist_image_left"] = payload_module.decode_image(
                    observation.wrist_image
                )
                upstream["observation/exterior_image_1_left"] = (
                    payload_module.decode_image(observation.main_image)
                )
                upstream["observation/exterior_image_2_left"] = (
                    payload_module.decode_image(observation.extra_view_images[0])
                )
        except _RequestValidationError:
            raise
        except BaseException as exc:
            raise _RequestValidationError(
                f"cannot decode Cosmos-Lite image payload: {exc}"
            ) from exc

        self._validate_inference_parameters(request.inference_parameters)
        return upstream

    def _validate_inference_parameters(self, parameters: Mapping[str, Any]) -> None:
        assert self._identity is not None
        supported = {
            "action_dim",
            "actions_per_chunk",
            "compute_values",
            "guidance",
            "mode",
            "num_steps",
            "seed",
            "shift",
        }
        unknown = sorted(set(parameters) - supported)
        if unknown:
            raise _RequestValidationError(
                f"Cosmos-Lite does not support inference parameters {unknown}"
            )
        if parameters.get("mode", "eval") != "eval":
            raise _RequestValidationError("Cosmos-Lite only supports mode='eval'")
        if bool(parameters.get("compute_values", False)):
            raise _RequestValidationError("Cosmos-Lite does not provide value outputs")
        shape_values = {
            "action_dim": self.config.action_dim,
            "actions_per_chunk": self.config.actions_per_chunk,
        }
        for key, expected in shape_values.items():
            if key in parameters and parameters[key] != expected:
                raise _RequestValidationError(
                    f"{key} is fixed by the Cosmos-Lite service: "
                    f"expected {expected}, got {parameters[key]!r}"
                )
        sampling = self._identity.sampling
        service_values = {
            "guidance": sampling.get("guidance"),
            "num_steps": sampling.get("denoise_steps"),
            "seed": sampling.get("seed"),
            "shift": sampling.get("shift"),
        }
        for key, expected in service_values.items():
            if key in parameters and parameters[key] != expected:
                raise _RequestValidationError(
                    f"{key} is fixed by the Cosmos-Lite service: "
                    f"expected {expected!r}, got {parameters[key]!r}"
                )

    def _validate_response(self, response: Mapping[str, Any]) -> np.ndarray:
        if "action" not in response:
            raise _ResponseValidationError("Cosmos-Lite response has no 'action'")
        try:
            actions = np.asarray(response["action"], dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise _ResponseValidationError(
                f"Cosmos-Lite action is not numeric: {exc}"
            ) from exc
        expected = (self.config.actions_per_chunk, self.config.action_dim)
        if actions.shape != expected:
            raise _ResponseValidationError(
                f"Cosmos-Lite action shape mismatch: expected {expected}, "
                f"got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise _ResponseValidationError("Cosmos-Lite action contains NaN or Inf")
        return np.ascontiguousarray(actions)

    def _error_response(
        self, request: InferenceRequest, code: ErrorCode, exc: BaseException
    ) -> ActionResponse:
        self.error_count += 1
        return ActionResponse(
            request_id=request.request_id,
            session_id=request.session_id,
            binding_token=request.binding_token,
            episode_id=request.episode_id,
            operation_seq=request.operation_seq,
            model_version=self.model_version,
            error=make_error(
                code,
                f"Cosmos-Lite inference failed: {exc}",
                policy_id=request.policy_id,
                session_id=request.session_id,
                endpoint=self.config.endpoint,
            ),
        )
