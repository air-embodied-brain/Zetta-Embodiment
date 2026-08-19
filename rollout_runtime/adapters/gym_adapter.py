"""Gym Adapter.

``RuntimeGymEnv`` is the Runtime's first client Adapter: a thin wrapper
around ``RuntimeClient``, one session per instance, **never bypassing the
Gateway** (it cannot reach the transport or the workers). Its purpose is to
demonstrate that the Runtime depends on no Agent / LIBERO semantics — only
the three actions ``reset`` / ``step`` / ``close``.

The interface is **async**: every ``RuntimeClient`` method is a coroutine,
and the Runtime's typical host (the event loop where the Gateway lives,
pytest-asyncio) is already inside a loop, so wrapping it in a synchronous
facade would deadlock. A gymnasium-style synchronous wrapper is left for
when it is actually needed (the legacy seam goes through
``adapters/zetta/runtime_env_client.py``, which is a separate duck-typed
interface).

Return values follow gymnasium's five-tuple semantics:
``reset() -> (observation, info)``,
``step(action) -> (observation, reward, terminated, truncated, info)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rollout_runtime.api.client import RuntimeClient
from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.errors import RuntimeApiError, make_error
from rollout_runtime.api.ids import SessionId
from rollout_runtime.api.messages import (
    CreateSessionRequest,
    EnvSpecMsg,
    Observation,
    PolicyRequest,
    ResetSpec,
)
from rollout_runtime.api.result import unwrap
from rollout_runtime.core import payload as payload_module

__all__ = ["RuntimeGymEnv"]


class RuntimeGymEnv:
    """A single-session, gym-style environment facade."""

    def __init__(
        self,
        client: RuntimeClient,
        env_spec: EnvSpecMsg,
        *,
        application_id: str = "gym",
        client_session_key: str = "",
        policy_id: str | None = None,
        lease_seconds: float = 300.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize (no session exists yet; it is created on ``reset``).

        Args:
            client: The Runtime API client (usually a ``RuntimeGateway``).
            env_spec: The environment spec.
            application_id: The owning application, used for quota and audit.
            client_session_key: The idempotency key; if left empty, one is
                generated from the object id, guaranteeing the same instance
                reuses the same session.
            policy_id: The policy used when ``step(None)`` goes through
                ``policy_step``.
            lease_seconds: The lease duration.
            metadata: Passthrough fields.
        """
        self._client = client
        self._env_spec = env_spec
        self._application_id = application_id
        self._client_session_key = client_session_key or f"gym-{id(self):x}"
        self._policy_id = policy_id
        self._lease_seconds = lease_seconds
        self._metadata = dict(metadata or {})
        self._session_id: SessionId | None = None
        self._closed = False
        self.episode_count = 0
        self.step_count = 0

    @property
    def session_id(self) -> SessionId | None:
        """The current session identifier.

        Returns:
            The session id; ``None`` if ``reset`` has not been called yet.
        """
        return self._session_id

    async def _ensure_session(self) -> SessionId:
        if self._closed:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.SESSION_NOT_READY, "this RuntimeGymEnv is already closed"
                )
            )
        if self._session_id is not None:
            return self._session_id
        request = CreateSessionRequest(
            application_id=self._application_id,
            client_session_key=self._client_session_key,
            env_spec=self._env_spec,
            default_policy_id=self._policy_id,
            lease_seconds=self._lease_seconds,
            metadata=self._metadata,
        )
        handle = unwrap((await self._client.create_sessions([request]))[0])
        self._session_id = handle.session_id
        return self._session_id

    async def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Observation, dict[str, Any]]:
        """Start a new episode.

        Repeated ``reset`` calls on the same ``RuntimeGymEnv`` reuse the
        same session (session and episode are two different things), so
        100 resets will never leak a session.

        Args:
            seed: The random seed.
            options: Family-private reset parameters; ``task_id`` /
                ``instruction`` / ``reset_state_id`` are promoted to the
                corresponding fields of ``ResetSpec``.

        Returns:
            ``(observation, info)``.
        """
        session_id = await self._ensure_session()
        extra = dict(options or {})
        spec = ResetSpec(
            task_id=extra.pop("task_id", None),
            seed=seed,
            instruction=extra.pop("instruction", None),
            reset_state_id=extra.pop("reset_state_id", None),
            options=extra,
        )
        result = unwrap((await self._client.reset([session_id], spec))[0])
        self.episode_count += 1
        self.step_count = 0
        if result.observation is None:
            raise RuntimeApiError(
                make_error(ErrorCode.ENV_FAILURE, "reset returned no observation")
            )
        return result.observation, {
            "episode_id": result.episode_id,
            "session_id": str(session_id),
        }

    async def step(
        self, action: np.ndarray | None = None
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        """Execute one step.

        Args:
            action: A ``[chunk, action_dim]`` (or single-step
                ``[action_dim]``) action; ``None`` means let the Runtime
                perform a ``policy_step``.

        Returns:
            ``(observation, reward, terminated, truncated, info)``.
        """
        session_id = await self._ensure_session()
        if action is None:
            result = unwrap(
                (
                    await self._client.policy_step(
                        [session_id], PolicyRequest(policy_id=self._policy_id)
                    )
                )[0]
            )
        else:
            block = np.asarray(action, dtype=np.float32)
            if block.ndim == 1:
                block = block[None, :]
            payload = payload_module.encode_array(block)
            result = unwrap(
                (await self._client.action_step([session_id], [payload]))[0]
            )
        self.step_count += 1
        if result.observation is None:
            raise RuntimeApiError(
                make_error(ErrorCode.ENV_FAILURE, "step returned no observation")
            )
        return (
            result.observation,
            result.reward,
            result.terminated,
            result.truncated,
            {
                **result.info,
                "executed_horizon": result.executed_horizon,
                "episode_id": result.episode_id,
            },
        )

    async def close(self) -> None:
        """Close the session (idempotent)."""
        if self._closed:
            return
        self._closed = True
        session_id = self._session_id
        self._session_id = None
        if session_id is not None:
            await self._client.close_sessions([session_id])

    async def __aenter__(self) -> RuntimeGymEnv:
        """Enter the context.

        Returns:
            Self.
        """
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Exit the context and close the session.

        Args:
            *exc_info: Ignored.
        """
        await self.close()
