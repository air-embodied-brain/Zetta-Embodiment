"""Local coverage for ``backends/rlinf_policy.py`` (mock-first).

``.venv-runtime`` has torch but no rlinf / openpi weights, so ``rlinf.models.get_model``
and ``rlinf.envs.utils.to_tensor`` are replaced with stubs here, verifying only the
**contract**:

- ``infer_batch`` produces one ``[chunk, action_dim] float32`` per request, and the
  payload is decodable;
- when the model raises, the obs schema is mixed within a batch, or actions contain
  non-finite values, each request individually returns
  ``ActionResponse(error=...)`` rather than raising (a remote exception would trigger
  ``os.kill(pid, SIGUSR1)``, killing the entire job);
- cuda_graph's fixed-batch constraint: ``load`` rejects outright if
  ``cuda_graph_batch_size`` is not provided; if provided, it pads to the fixed batch
  and trims the result back to the real request count;
- the hard constraints on ``compat_key`` are declared by the backend
  (``policy_compat_constraints``); the EnvWorker only relays them.

The real machine (real pi0.5 weights) is cross-checked by the ``@pytest.mark.remote``
cases in ``test_extension_call.py`` and by ``cli smoke`` / ``cli bench`` on a
configured GPU host.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types
from typing import Any

import numpy as np
import pytest

from rollout_runtime.api.enums import ErrorCode, Priority
from rollout_runtime.api.ids import EpisodeId, OperationSeq, RequestId, SessionId
from rollout_runtime.api.internal import InferenceRequest
from rollout_runtime.api.messages import Observation
from rollout_runtime.backends import build_policy_core, policy_compat_constraints
from rollout_runtime.core import payload as payload_module
from rollout_runtime.core.policy_inference import (
    batchable_param_keys,
    canonicalize_inference_parameters,
    compute_compat_key,
)


def _stub_module(name: str) -> types.ModuleType:
    """Build a stub module carrying ``__spec__``.

    ``rlinf_bootstrap.ensure_rlinf_importable`` uses
    ``importlib.util.find_spec("rlinf")`` to determine "is it already importable",
    and ``find_spec`` raises ``ValueError`` for modules with ``__spec__ is None``.
    So the stub must carry a spec.

    Args:
        name: The module name.

    Returns:
        The stub module.
    """
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, None)
    return module


CHUNK = 4
ACTION_DIM = 7
STATE_DIM = 8


class _StubModel:
    """``BasePolicy`` stub: implements only the four entry points used by the Runtime.

    Attributes:
        calls: The ``(batch_size, kwargs)`` of each ``predict_action_batch`` call.
        prompts: The list of prompts received each call (verifies
            ``instruction_override``).
        fail: Whether to raise an exception (verifies fault isolation).
        non_finite: Whether to produce NaN (verifies numeric checks).
        compiled: The recorded ``enable_torch_compile`` mode.
        captured: The recorded ``capture_cuda_graph`` batch sizes.
        device: The target of the most recent ``to()`` call.
    """

    def __init__(self) -> None:
        """Initialize."""
        self.calls: list[tuple[int, dict[str, Any]]] = []
        self.prompts: list[list[str]] = []
        self.fail = False
        self.non_finite = False
        self.compiled: str | None = None
        self.captured: tuple[int, int] | None = None
        self.device = "cuda"
        self.eval_called = False

    def eval(self) -> None:
        """Switch to eval mode."""
        self.eval_called = True

    def to(self, device: str) -> _StubModel:
        """Move to a device.

        Args:
            device: The target device.

        Returns:
            Self.
        """
        self.device = device
        return self

    def enable_torch_compile(self, mode: str = "") -> None:
        """Record the compile mode.

        Args:
            mode: The compile mode.
        """
        self.compiled = mode

    def capture_cuda_graph(self, train_batch_size: int, eval_batch_size: int) -> None:
        """Record the batch sizes captured by the CUDA graph.

        Args:
            train_batch_size: The training batch size.
            eval_batch_size: The inference batch size.
        """
        self.captured = (train_batch_size, eval_batch_size)

    def predict_action_batch(
        self, env_obs: dict[str, Any], **kwargs: Any
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Produce actions for a batch.

        Args:
            env_obs: The 5-key schema batch dict.
            **kwargs: Sampling parameters (``mode`` / ``compute_values``).

        Returns:
            ``([B, chunk, action_dim], result)``.

        Raises:
            RuntimeError: When ``fail`` is enabled (verifies the whole-batch failure
                path).
        """
        batch = int(np.asarray(env_obs["main_images"]).shape[0])
        self.calls.append((batch, dict(kwargs)))
        self.prompts.append(list(env_obs["task_descriptions"]))
        if self.fail:
            raise RuntimeError("stub policy exploded")
        actions = np.arange(batch * CHUNK * ACTION_DIM, dtype=np.float32).reshape(
            batch, CHUNK, ACTION_DIM
        )
        if self.non_finite:
            actions[0, 0, 0] = np.nan
        return actions, {}


@pytest.fixture
def stub_model(monkeypatch: pytest.MonkeyPatch) -> _StubModel:
    """Replace ``rlinf.models.get_model`` and ``rlinf.envs.utils.to_tensor`` with stubs.

    Args:
        monkeypatch: The pytest fixture.

    Returns:
        The stub model (tests can toggle ``fail`` / ``non_finite``).
    """
    model = _StubModel()
    import zetta.policies.openpi.factory as factory_module

    monkeypatch.setattr(factory_module, "build_openpi_model", lambda cfg: model)
    models_module = _stub_module("rlinf.models")
    models_module.get_model = lambda cfg: model  # type: ignore[attr-defined]
    utils_module = _stub_module("rlinf.envs.utils")
    utils_module.to_tensor = lambda value, device="cpu": value  # type: ignore[attr-defined]
    for name, module in {
        "rlinf": _stub_module("rlinf"),
        "rlinf.models": models_module,
        "rlinf.envs": _stub_module("rlinf.envs"),
        "rlinf.envs.utils": utils_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return model


def _core(**overrides: Any) -> Any:
    """Build a ``RlinfPolicyCore``.

    Args:
        **overrides: ``RlinfPolicyConfig`` field overrides.

    Returns:
        ``RlinfPolicyCore``.
    """
    return build_policy_core(
        backend="zetta_openpi",
        policy_config={
            "model_path": "/stub/pi05",
            "num_action_chunks": CHUNK,
            "action_dim": ACTION_DIM,
            "model_version": "pi05-stub",
            **overrides,
        },
        device="cuda",
        dtype="bfloat16",
        policy_family="openpi",
        action_dim=ACTION_DIM,
        actions_per_chunk=CHUNK,
    )


def _request(index: int, *, state_dim: int = STATE_DIM, **overrides: Any) -> Any:
    """Build an ``InferenceRequest``.

    Args:
        index: The sequence number, used in ``request_id`` / ``session_id``.
        state_dim: The state dimension (changing it creates an obs schema mismatch).
        **overrides: ``InferenceRequest`` field overrides.

    Returns:
        ``InferenceRequest``.
    """
    observation = Observation(
        session_id=SessionId(f"sess-{index}"),
        episode_id=EpisodeId(1),
        step_index=index,
        main_image=payload_module.encode_image(
            np.full((4, 4, 3), index % 256, dtype=np.uint8)
        ),
        wrist_image=payload_module.encode_image(
            np.full((4, 4, 3), (index * 3) % 256, dtype=np.uint8)
        ),
        state=[float(index)] * state_dim,
        instruction="stub: pick the cube",
    )
    payload: dict[str, Any] = {
        "request_id": RequestId(f"req-{index}"),
        "session_id": SessionId(f"sess-{index}"),
        "episode_id": EpisodeId(1),
        "operation_seq": OperationSeq(index + 1),
        "policy_id": "pi05",
        "observation": observation,
        "inference_parameters": {"mode": "eval"},
        "routing_token": "env:0",
        "compat_key": "k",
        "priority": Priority.INTERACTIVE,
    }
    payload.update(overrides)
    return InferenceRequest(**payload)


def test_infer_batch_returns_one_response_per_request(stub_model: _StubModel) -> None:
    """A single forward pass for a batch of requests, sliced per request into
    ``[chunk, action_dim] float32``.

    Args:
        stub_model: The stub model.
    """
    core = _core()
    core.load()
    assert stub_model.eval_called is True
    requests = [_request(index) for index in range(3)]
    responses = core.infer_batch(requests)

    assert [response.request_id for response in responses] == [
        request.request_id for request in requests
    ]
    assert stub_model.calls == [(3, {"mode": "eval", "compute_values": False})]
    for index, response in enumerate(responses):
        assert response.error is None
        assert response.model_version == "pi05-stub"
        block = payload_module.decode_payload(response.actions)
        assert block.shape == (CHUNK, ACTION_DIM)
        assert block.dtype == np.float32
        assert np.isfinite(block).all()
        assert response.auxiliary_outputs["chunk"] == CHUNK
        assert response.auxiliary_outputs["batch_size"] == 3
        # The per-request slice must correspond to the model output's index-th row.
        assert block[0, 0] == pytest.approx(index * CHUNK * ACTION_DIM)
    core.close()


def test_robocasa_action_dim_remaps_gripper_and_control_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``action_dim=12`` (RoboCasa contract), remaps gripper/control_mode from
    ``[-1,1]`` to ``[0,1]``; the other 10 channels pass through unchanged.

    openpi/pi0 uniformly outputs ``[-1, 1]`` for all 12 channels (following the same
    convention as position/rotation/base_motion), but
    ``robots.robocasa.action_contract.canonical_action`` requires ``[0, 1]`` only for
    ``gripper_close`` (index 6) and ``control_mode`` (index 11); out-of-range values
    there raise ``ValueError`` directly (unlike the other channels, which are clipped
    before validation).

    Args:
        monkeypatch: The pytest fixture.
    """
    robocasa_chunk = 2
    robocasa_action_dim = 12

    class _RoboCasaStubModel(_StubModel):
        def predict_action_batch(
            self, env_obs: dict[str, Any], **kwargs: Any
        ) -> tuple[np.ndarray, dict[str, Any]]:
            batch = int(np.asarray(env_obs["main_images"]).shape[0])
            self.calls.append((batch, dict(kwargs)))
            self.prompts.append(list(env_obs["task_descriptions"]))
            # All 12 channels are in [-1, 1]: position/rotation/base_motion should
            # remain unchanged, while gripper_close (6) and control_mode (11) should
            # be remapped to [0, 1].
            actions = np.full(
                (batch, robocasa_chunk, robocasa_action_dim), -0.5, dtype=np.float32
            )
            actions[:, :, 6] = -1.0
            actions[:, :, 11] = 1.0
            return actions, {}

    model = _RoboCasaStubModel()
    import zetta.policies.openpi.factory as factory_module

    monkeypatch.setattr(factory_module, "build_openpi_model", lambda cfg: model)
    models_module = _stub_module("rlinf.models")
    models_module.get_model = lambda cfg: model  # type: ignore[attr-defined]
    utils_module = _stub_module("rlinf.envs.utils")
    utils_module.to_tensor = lambda value, device="cpu": value  # type: ignore[attr-defined]
    for name, module in {
        "rlinf": _stub_module("rlinf"),
        "rlinf.models": models_module,
        "rlinf.envs": _stub_module("rlinf.envs"),
        "rlinf.envs.utils": utils_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    core = _core(num_action_chunks=robocasa_chunk, action_dim=robocasa_action_dim)
    core.load()
    responses = core.infer_batch([_request(0)])
    assert responses[0].error is None
    block = payload_module.decode_payload(responses[0].actions)
    assert block.shape == (robocasa_chunk, robocasa_action_dim)
    # Unaffected channels pass through unchanged.
    for column in range(robocasa_action_dim):
        if column in (6, 11):
            continue
        assert block[:, column] == pytest.approx(-0.5)
    # gripper_close: -1.0 -> 0.0; control_mode: 1.0 -> 1.0.
    assert block[:, 6] == pytest.approx(0.0)
    assert block[:, 11] == pytest.approx(1.0)
    core.close()


def test_non_robocasa_action_dim_is_not_remapped(stub_model: _StubModel) -> None:
    """When ``action_dim != 12`` (e.g. LIBERO's 7), no RoboCasa-specific remapping is
    applied.

    Args:
        stub_model: The stub model.
    """
    core = _core()
    core.load()
    responses = core.infer_batch([_request(0)])
    assert responses[0].error is None
    block = payload_module.decode_payload(responses[0].actions)
    # The stub model's output is entirely >= 0 (np.arange); if the RoboCasa remap were
    # mistakenly applied, the values would shift overall. This confirms the model's
    # raw output is preserved verbatim.
    assert block[0, 0] == pytest.approx(0.0)
    assert block.shape == (CHUNK, ACTION_DIM)
    core.close()


def test_instruction_override_reaches_the_model(stub_model: _StubModel) -> None:
    """``instruction_override`` only changes the prompt of that one request.

    Args:
        stub_model: The stub model.
    """
    core = _core()
    core.load()
    core.infer_batch(
        [_request(0), _request(1, instruction_override="stub: open the drawer")]
    )
    assert stub_model.prompts[-1] == [
        "stub: pick the cube",
        "stub: open the drawer",
    ]
    core.close()


def test_model_failure_is_reported_per_request(stub_model: _StubModel) -> None:
    """When the model raises for the whole batch, it still becomes a per-request
    ``POLICY_FAILURE`` and is never leaked.

    Args:
        stub_model: The stub model.
    """
    core = _core()
    core.load()
    stub_model.fail = True
    responses = core.infer_batch([_request(0), _request(1)])
    assert len(responses) == 2
    for response in responses:
        assert response.actions is None
        assert response.error is not None
        assert response.error.code is ErrorCode.POLICY_FAILURE
        assert "stub policy exploded" in response.error.message
    assert core.error_count == 2
    core.close()


def test_mixed_observation_schema_is_rejected_per_request(
    stub_model: _StubModel,
) -> None:
    """When the obs structure is mixed within a batch (violating the precondition of
    ``_merge_obs_batches``), each request individually returns an error.

    Args:
        stub_model: The stub model.
    """
    core = _core()
    core.load()
    responses = core.infer_batch([_request(0), _request(1, state_dim=STATE_DIM + 1)])
    assert [response.error is not None for response in responses] == [True, True]
    assert all(
        response.error is not None and response.error.code is ErrorCode.POLICY_FAILURE
        for response in responses
    )
    assert stub_model.calls == []
    core.close()


def test_non_finite_actions_fail_only_that_request(stub_model: _StubModel) -> None:
    """Non-finite actions only fail that one request; the rest return normally.

    Args:
        stub_model: The stub model.
    """
    core = _core()
    core.load()
    stub_model.non_finite = True
    responses = core.infer_batch([_request(0), _request(1)])
    assert responses[0].error is not None
    assert responses[0].error.code is ErrorCode.POLICY_FAILURE
    assert responses[1].error is None
    core.close()


def test_empty_batch_is_a_noop(stub_model: _StubModel) -> None:
    """An empty batch returns an empty list directly.

    Args:
        stub_model: The stub model.
    """
    core = _core()
    core.load()
    assert core.infer_batch([]) == []
    assert stub_model.calls == []
    core.close()


def test_cuda_graph_requires_a_fixed_batch_size(stub_model: _StubModel) -> None:
    """Enabling cuda_graph without a fixed batch size -> ``load`` rejects outright.

    CUDA graph capture bakes in a fixed batch size (``mlp_policy.py:392-437``); a
    silent dynamic batch would only make failures on GPU harder to diagnose.

    Args:
        stub_model: The stub model.
    """
    core = _core(enable_cuda_graph=True)
    with pytest.raises(ValueError, match="cuda_graph_batch_size"):
        core.load()


def test_cuda_graph_pads_the_batch_and_trims_the_result(
    stub_model: _StubModel,
) -> None:
    """With a fixed batch size, padding is applied and the result is trimmed back to
    the real request count.

    Args:
        stub_model: The stub model.
    """
    core = _core(enable_cuda_graph=True, cuda_graph_batch_size=4)
    core.load()
    assert stub_model.captured == (4, 4)
    responses = core.infer_batch([_request(0), _request(1)])
    assert stub_model.calls == [(4, {"mode": "eval", "compute_values": False})]
    assert len(responses) == 2
    assert core.padded_request_count == 2
    assert all(response.error is None for response in responses)
    core.close()


def test_cuda_graph_rejects_an_oversized_batch(stub_model: _StubModel) -> None:
    """When the batch exceeds the captured fixed batch size, each request returns an
    error (the scheduler was misconfigured).

    Args:
        stub_model: The stub model.
    """
    core = _core(enable_cuda_graph=True, cuda_graph_batch_size=1)
    core.load()
    responses = core.infer_batch([_request(0), _request(1)])
    assert all(response.error is not None for response in responses)
    assert "fixed_batch_size" in (
        responses[0].error.message if responses[0].error else ""
    )
    core.close()


def test_torch_compile_is_forwarded(stub_model: _StubModel) -> None:
    """``enable_torch_compile`` forwards the mode to
    ``BasePolicy.enable_torch_compile``.

    Args:
        stub_model: The stub model.
    """
    core = _core(enable_torch_compile=True, torch_compile_mode="max-autotune")
    core.load()
    assert stub_model.compiled == "max-autotune"
    core.close()


def test_offload_and_reload_move_the_model(stub_model: _StubModel) -> None:
    """``offload`` / ``reload`` move the model between CPU and accelerator
    (idempotently).

    Args:
        stub_model: The stub model.
    """
    core = _core()
    core.load()
    core.offload()
    assert stub_model.device == "cpu"
    core.offload()
    assert stub_model.device == "cpu"
    core.reload()
    assert stub_model.device == "cuda"
    core.reload()
    assert stub_model.device == "cuda"
    core.close()


def test_update_weights_relabels_without_a_checkpoint(stub_model: _StubModel) -> None:
    """Without a ``file:`` prefix, ``update_weights`` only changes the version label.

    The Runtime only serves inference and has no actor group to pull weights from;
    a real ``WeightSyncer`` will come once the Runtime is wired into the training
    loop.

    Args:
        stub_model: The stub model.
    """
    core = _core()
    core.load()
    core.update_weights("pi05-v2")
    assert core.model_version == "pi05-v2"
    responses = core.infer_batch([_request(0)])
    assert responses[0].model_version == "pi05-v2"
    core.close()


def test_infer_before_load_is_a_per_request_error(stub_model: _StubModel) -> None:
    """Running inference before ``load`` also takes the per-request error path.

    Args:
        stub_model: The stub model.
    """
    core = _core()
    responses = core.infer_batch([_request(0)])
    assert responses[0].error is not None
    assert responses[0].error.code is ErrorCode.POLICY_FAILURE


def test_unsupported_model_type_is_rejected_early() -> None:
    """Only the openpi family is currently supported; other families raise an
    explicit error instead of failing midway."""
    with pytest.raises(ValueError, match="out of M4 scope"):
        build_policy_core(backend="zetta_openpi", policy_config={"model_type": "openvla_oft"})


def test_unknown_policy_backend_is_rejected() -> None:
    """An unknown backend name raises an error."""
    with pytest.raises(ValueError, match="unknown policy backend"):
        build_policy_core(backend="tensorrt")


def test_compat_key_carries_the_policy_hard_constraints() -> None:
    """cuda_graph's fixed batch size and ``openvla_oft``'s padding must change the
    ``compat_key``."""
    dynamic = policy_compat_constraints(
        backend="zetta_openpi", policy_config={"model_path": "/x", "num_action_chunks": CHUNK}
    )
    fixed = policy_compat_constraints(
        backend="zetta_openpi",
        policy_config={
            "model_path": "/x",
            "num_action_chunks": CHUNK,
            "enable_cuda_graph": True,
            "cuda_graph_batch_size": 8,
        },
    )
    assert "fixed_batch_size" not in dynamic
    assert fixed["fixed_batch_size"] == 8
    assert policy_compat_constraints(backend="fake") == {}

    def key(constraints: dict[str, Any]) -> str:
        return compute_compat_key(
            policy_id="pi05",
            model_version="",
            obs_schema_digest="d",
            inference_parameters={"mode": "eval"},
            device="cuda",
            dtype="bfloat16",
            policy_family="openpi",
            constraints=constraints or None,
        )

    assert key(dynamic) != key(fixed)


def test_openpi_batchable_whitelist_keeps_shape_keys_in_the_key() -> None:
    """openpi's batchable whitelist only allows noise-related keys; ``mode`` /
    ``num_steps`` must remain in the key."""
    whitelist = batchable_param_keys("openpi")
    assert "noise_seed" in whitelist
    assert "temperature" in whitelist  # default whitelist
    assert "mode" not in whitelist
    assert "num_steps" not in whitelist
    canonical = canonicalize_inference_parameters(
        {"mode": "eval", "num_steps": 5, "noise_seed": 7, "temperature": 0.9}, "openpi"
    )
    assert canonical == {"mode": "eval", "num_steps": 5}
