"""Zetta LIBERO runtime facade for policy inference.

Companion to ``runtime_env_client.py`` (see that module's docstring for the
architecture context). Re-implements
``zetta/utils/vla_client.py::VLAClient``'s rollout-facing call site
(``predict_action_batch(env_obs, mode="eval")``) against
``RemoteRuntimeClient.policy_infer`` instead of an RPC connection to a
standalone Pi0.5 ``vla_server.py`` process.

Deliberately uses ``policy_infer`` (infer only), not the atomic
``policy_step`` (observe -> infer -> execute in one call):
``LiberoPrimitives`` post-processes the raw policy output itself
(``translation_scale``/``action_clip``/per-channel clipping in
``robots/libero/tools.py``) before any action reaches the environment, the
same reason ``rollout_runtime/adapters/gym_adapter.py``'s own docstring gives
for why Zetta's LIBERO adapter needs the split ``policy_infer``/``action_step``
pair rather than a single ``policy_step`` call.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rollout_runtime.adapters.zetta.runtime_env_client import SyncRuntimeLoop
from rollout_runtime.api.ids import SessionId
from rollout_runtime.api.messages import PolicyRequest
from rollout_runtime.api.result import Err
from rollout_runtime.core.payload import decode_array

__all__ = ["LiberoRuntimeVLAClient"]


class _RuntimePolicyError(RuntimeError):
    """A ``policy_infer`` call returned an ``Err`` or produced no action chunk."""


def _single(results: list[Any], *, operation: str) -> Any:
    if len(results) != 1:
        raise _RuntimePolicyError(
            f"runtime {operation} returned {len(results)} results for one session"
        )
    result = results[0]
    if isinstance(result, Err):
        info = result.error
        raise _RuntimePolicyError(
            f"runtime {operation} failed: {info.code.name}: {info.message}"
        )
    return result.value


class LiberoRuntimeVLAClient:
    """``VLAClient``-shaped, **synchronous** facade over ``policy_infer``.

    See ``runtime_env_client.py::LiberoRuntimeEnvClient`` for why this wraps
    each call in ``asyncio.run(...)`` rather than exposing an ``async``
    interface: ``LiberoPrimitives.vla_execute``/``_vlm_chunk`` and everything
    above them are synchronous.
    """

    def __init__(
        self,
        client: Any,
        session_id: SessionId,
        *,
        loop: SyncRuntimeLoop,
        policy_id: str = "pi05",
    ) -> None:
        """Wrap an already-created Runtime session.

        Args:
            client: A ``RemoteRuntimeClient``/``RuntimeClient``-shaped object.
            session_id: The rollout's session (shared with
                ``LiberoRuntimeEnvClient``: one session serves both the
                environment and the policy, matching how
                ``robots/robocasa/run_rollout.py`` uses a single
                ``RolloutSession`` for both).
            loop: The process's shared ``SyncRuntimeLoop`` (must be the same
                instance used to construct ``client``/the sibling
                ``LiberoRuntimeEnvClient``; see that class's docstring for why
                a single persistent loop is required).
            policy_id: The policy served by the Runtime's ``RolloutWorker``
                (preset default: ``"pi05"``, see
                ``rollout_runtime/config/presets/a100_libero_pi05*.yaml``).
        """
        self._client = client
        self._loop = loop
        self._session_id = session_id
        self._policy_id = policy_id

    @property
    def _ids(self) -> list[SessionId]:
        return [self._session_id]

    def healthz(self, *, timeout_s: float | None = None) -> dict[str, Any]:
        """Legacy interface parity; liveness is the Runtime's ``/healthz``, not per-session."""

        del timeout_s
        return {"status": "ok", "backend": "rollout_runtime", "policy_id": self._policy_id}

    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: str = "eval",
        **kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Infer one action chunk without touching the environment.

        Args:
            env_obs: Unused directly — the Runtime always infers against its
                own last-known session observation (the same one
                ``LiberoRuntimeEnvClient``'s most recent ``reset``/
                ``action_step`` produced), never a caller-supplied
                observation. Accepted for interface parity with
                ``VLAClient.predict_action_batch`` only.
            mode: Accepted for interface parity; folded into
                ``inference_parameters`` so a future policy backend can read
                it, but ``rollout_runtime``'s ``rlinf_policy``/``groot_policy``
                backends do not currently branch on it.
            **kwargs: ``inference_parameters`` (dict) is forwarded verbatim.

        Returns:
            ``(actions, metadata)`` matching ``VLAClient.predict_action_batch``:
            ``actions`` is ``[chunk, action_dim]`` float32, ``metadata`` carries
            the same diagnostic keys the direct-connect Pi0.5 wire response did.

        Raises:
            _RuntimePolicyError: ``policy_infer`` returned no action chunk.
        """
        del env_obs  # see docstring: the Runtime infers against its own session state.
        inference_parameters = dict(kwargs.get("inference_parameters") or {})
        inference_parameters.setdefault("mode", mode)
        result = _single(
            self._loop.run(
                self._client.policy_infer(
                    self._ids,
                    PolicyRequest(
                        policy_id=self._policy_id,
                        inference_parameters=inference_parameters,
                    ),
                )
            ),
            operation="policy_infer",
        )
        if result.actions is None:
            raise _RuntimePolicyError("policy_infer returned no action chunk")
        block = np.asarray(decode_array(result.actions), dtype=np.float32)
        if block.ndim != 2:
            raise _RuntimePolicyError(
                f"policy_infer returned shape {tuple(int(v) for v in block.shape)}, "
                "expected [chunk, action_dim]"
            )
        metadata = {
            "source": "policy_infer",
            "horizon": int(block.shape[0]),
            "model_version": result.model_version,
            "observation_step_index": int(result.observation_step_index),
            "auxiliary_outputs": dict(result.auxiliary_outputs),
            "policy_id": dict(result.info).get("policy_id", self._policy_id),
        }
        return block, metadata
