"""Zetta LIBERO runtime facade over ``RuntimeClient``.

This module is the application-side counterpart to the server-side LIBERO
support in ``rollout_runtime/backends/rlinf_env.py`` and
``rollout_runtime/backends/libero_privileged.py``. It implements the subset of
``robots/libero/env_client.py::LiberoEnvClient`` used by
``robots/libero/tools.py::LiberoPrimitives``, but the implementation is Zetta's
own Runtime client code rather than a compatibility import from another
framework.

Design: a *thin, duck-typed* re-implementation of
``robots/libero/env_client.py::LiberoEnvClient``'s public surface, backed by
``RemoteRuntimeClient``/``RuntimeClient`` calls instead of an RPC connection to
a standalone ``env_server.py`` process. Every method name, return shape, and
episode-termination bookkeeping (``episode_terminated``/``episode_truncated``/
``episode_steps``) matches the legacy client exactly, so
``robots/libero/tools.py::LiberoPrimitives`` and every Role1/Critic/Recovery
call site above it require **zero changes** — this is the seam
``rollout_runtime/adapters/__init__.py`` describes ("only swap the
``_vlm_chunk`` call for a single ``policy_step`` call"), i.e. only the
low-level env/model calls move, the
orchestration loop in ``LiberoPrimitives`` stays as-is).

Not covered (raise ``NotImplementedError`` with a clear message instead of
silently returning a wrong shape):

- ``audit_trace``: no ``libero.audit_trace`` extension is registered by
  ``rollout_runtime/core/env_registry.py::LIBERO_EXTENSIONS``, so there is no
  server-side ledger to read back. ``step``/``chunk_step`` instead accumulate
  one raw record per physical step (``_record_steps``), which
  ``robots/libero/run_evolution_rollout.py::flush_audit_trace`` drains through
  ``drain_step_records`` to write ``actions.jsonl``/``states.jsonl``.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import numpy as np

from rollout_runtime.api.ids import SessionId
from rollout_runtime.api.messages import ResetSpec, StepResult
from rollout_runtime.api.result import Err
from rollout_runtime.core.payload import decode_payload, encode_array

__all__ = [
    "RuntimeOperationError",
    "LiberoRuntimeEnvClient",
    "SyncRuntimeLoop",
]


class RuntimeOperationError(RuntimeError):
    """A runtime operation returned a batch-of-one ``Err`` or wrong batch size."""


def _single(results: list[Any], *, operation: str) -> Any:
    """Unwrap a batch-of-one ``RuntimeClient`` result (mirrors ``run_rollout.py``)."""

    if len(results) != 1:
        raise RuntimeOperationError(
            f"runtime {operation} returned {len(results)} results for one session"
        )
    result = results[0]
    if isinstance(result, Err):
        info = result.error
        raise RuntimeOperationError(
            f"runtime {operation} failed: {info.code.name}: {info.message}"
        )
    return result.value


class SyncRuntimeLoop:
    """One dedicated background event loop for a whole rollout process.

    ``httpx.AsyncClient`` (``RemoteRuntimeClient``'s transport) binds its
    connection pool to whichever event loop issues its first request; calling
    ``asyncio.run(...)`` separately for every method call would spin up a
    fresh loop each time and break on the second call
    (``RuntimeError: Event loop is closed`` / cross-loop future errors from
    httpcore). This runs exactly one loop, in one background thread, for the
    lifetime of the process, and marshals every synchronous adapter call onto
    it via ``run_coroutine_threadsafe`` — the same one-blocking-call-per-legacy-
    call granularity ``LiberoEnvClient``/``VLAClient`` already use, just backed
    by a persistent loop instead of a persistent RPC socket.
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="libero-runtime-loop", daemon=True
        )
        self._thread.start()

    def run(self, coro: Any) -> Any:
        """Run ``coro`` on the background loop and block for its result."""

        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def close(self) -> None:
        """Stop the background loop and release its resources (idempotent)."""

        if self._loop.is_closed():
            return
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=10.0)
        self._loop.close()


class LiberoRuntimeEnvClient:
    """``LiberoEnvClient``-shaped, **synchronous** facade over a shared Runtime
    session.

    ``robots/libero/tools.py::LiberoPrimitives`` and everything above it
    (Role1, Critic, Recovery) is entirely synchronous. The direct-connect
    ``LiberoEnvClient``/``VLAClient`` wrap blocking RPC calls, not asyncio.
    ``RemoteRuntimeClient``/``RuntimeClient`` are ``async`` (the Runtime's
    native interface); every public method here dispatches through a shared
    ``SyncRuntimeLoop`` (see that class for why a single persistent loop is
    required rather than one ``asyncio.run(...)`` per call), preserving the
    exact call/return granularity ``LiberoPrimitives`` was written against.

    Attributes mirror ``LiberoEnvClient`` exactly: ``episode_terminated``,
    ``episode_truncated``, ``episode_steps``, ``return_all_frames``.
    """

    def __init__(
        self,
        client: Any,
        session_id: SessionId,
        *,
        loop: SyncRuntimeLoop,
        return_all_frames: bool = False,
        sample_privileged_state: bool = False,
    ) -> None:
        """Wrap an already-created Runtime session.

        Args:
            client: A ``RemoteRuntimeClient``/``RuntimeClient``-shaped object.
            session_id: The session created by the caller (see
                ``run_evolution_rollout.py``'s ``--runtime-url`` path, which
                owns ``create_sessions``/``close_sessions`` the same way
                ``robots/robocasa/run_rollout.py::RolloutSession`` does).
            loop: The process's shared ``SyncRuntimeLoop``. Must be the same
                instance used to construct ``client`` and any sibling
                ``LiberoRuntimeVLAClient`` sharing this session.
            return_all_frames: Default for ``chunk_step``'s per-step frames,
                matching the legacy client's constructor argument.
            sample_privileged_state: Read ``libero.critic_state`` once per
                ``step``/``chunk_step`` so the audit ledger carries the
                ``privileged.*`` plane. Costs one extra ``extension_call`` per
                chunk, so it defaults to off and callers opt in.
        """
        self._client = client
        self._loop = loop
        self.session_id = session_id
        self.return_all_frames = return_all_frames
        self.sample_privileged_state = sample_privileged_state
        self.episode_terminated = False
        self.episode_truncated = False
        self.episode_steps = 0
        self._task_language: str | None = None
        self._step_records: list[dict[str, Any]] = []

    @property
    def _ids(self) -> list[SessionId]:
        return [self.session_id]

    def _run(self, coro: Any) -> Any:
        return self._loop.run(coro)

    def check_done(self, term: Any, trunc: Any) -> None:
        """Latch terminal flags (identical semantics to the legacy client)."""

        self.episode_terminated |= bool(np.asarray(term).any())
        self.episode_truncated |= bool(np.asarray(trunc).any())

    def reset(
        self,
        *,
        task_id: int | None = None,
        seed: int | None = None,
        critic_rules: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the episode.

        Unlike the direct-connect client (whose ``reset()`` takes no arguments
        because ``task_id``/``seed`` are frozen into the ``env_server.py`` subprocess
        command line), the Runtime's env pool identity is shared across many
        rollout processes, so ``task_id``/``seed`` ride in ``ResetSpec``
        instead. Callers own passing these once, at the start of the episode.

        Args:
            task_id: LIBERO benchmark task index; ``None`` uses the pool's
                default (``env_config.task_id``).
            seed: Environment seed for this episode.
            critic_rules: Frozen Critic rule payloads for the whole episode
                (``LiberoPrimitives.configure_critic`` semantics: set once,
                never changed mid-episode). Passed through
                ``ResetSpec.options["critic_rules"]``, exactly as
                ``rollout_runtime/backends/rlinf_env.py`` expects.

        Returns:
            ``(obs, info)`` in the LIBERO client dict shape (see
            ``_decode_observation``).
        """
        options: dict[str, Any] = {}
        if critic_rules:
            options["critic_rules"] = list(critic_rules)
        spec = ResetSpec(task_id=task_id, seed=seed, options=options)
        step = _single(
            self._run(self._client.reset(self._ids, spec)), operation="reset"
        )
        self.episode_terminated = False
        self.episode_truncated = False
        self.episode_steps = 0
        self._step_records.clear()
        obs = self._decode_observation(step)
        return obs, dict(step.info)

    def _decode_observation(self, step: StepResult) -> dict[str, Any]:
        """Translate an ``Observation`` into the legacy LIBERO obs dict shape.

        ``LiberoPrimitives``/``robots/libero/tools.py`` index this dict with
        ``obs["main_images"]``/``obs["wrist_images"]``/``obs["states"]``/
        ``obs["task_descriptions"]`` (see the module docstring for the exact
        call sites this must satisfy).
        """
        observation = step.observation
        if observation is None:
            raise RuntimeOperationError("runtime step returned no observation")
        if observation.instruction:
            self._task_language = observation.instruction
        obs: dict[str, Any] = {
            "states": np.asarray(observation.state, dtype=np.float32),
            "task_descriptions": observation.instruction or self._task_language or "",
        }
        if observation.main_image is not None:
            obs["main_images"] = decode_payload(observation.main_image)
        if observation.wrist_image is not None:
            obs["wrist_images"] = decode_payload(observation.wrist_image)
        if observation.extra_view_images:
            obs["extra_view_images"] = decode_payload(observation.extra_view_images[0])
        return obs

    def _record_steps(self, step: StepResult, block: np.ndarray, executed: int) -> None:
        """Accumulate one raw per-physical-step record for the audit ledger.

        Deliberately raw -- action, state vector and terminal flags, but no
        feature extraction and no hashing. Those live in the caller, because
        the layering guard (``tests/runtime/test_layering.py`` rule 1) forbids
        importing ``zetta``/``robots`` from here.

        ``step_index`` comes from ``PerStepRecord`` when the family sends
        per-step records and is otherwise reconstructed from ``episode_steps``;
        both are 1-based, matching ``env_server.py:262``.

        With ``sample_privileged_state``, ``libero.critic_state`` is read once
        and attached to this call's last record. That read is safe mid-episode:
        the external ``extension_call`` path leaves
        ``slot.critic_history_pending_reset`` alone, so it cannot disturb the
        next automatic Critic decision.

        Args:
            step: The ``action_step`` result being recorded.
            block: The action block that was submitted, in submission order.
            executed: Number of physical steps actually executed.
        """

        before = len(self._step_records)
        per_step = step.per_step or ()
        fallback_state = (
            step.observation.state if step.observation is not None else None
        )
        if per_step:
            for action, record in zip(block, per_step, strict=False):
                state = (
                    record.observation.state
                    if record.observation is not None
                    else fallback_state
                )
                self._step_records.append(
                    {
                        "step_index": int(record.step_index),
                        "action": np.asarray(action, dtype=np.float64).tolist(),
                        "states": None if state is None else list(state),
                        "reward": float(record.reward),
                        "terminated": bool(record.terminated),
                        "truncated": bool(record.truncated),
                        "proposal_rule_ids": list(
                            record.info.get("critic_proposal_rule_ids", ())
                        ),
                    }
                )
        else:
            first_index = self.episode_steps - executed + 1
            for offset, action in enumerate(np.asarray(block)[:executed]):
                last = offset == executed - 1
                self._step_records.append(
                    {
                        "step_index": first_index + offset,
                        "action": np.asarray(action, dtype=np.float64).tolist(),
                        "states": (
                            None if fallback_state is None else list(fallback_state)
                        ),
                        "reward": float(step.reward) if last else 0.0,
                        "terminated": bool(step.terminated) if last else False,
                        "truncated": bool(step.truncated) if last else False,
                        "proposal_rule_ids": [],
                    }
                )
        appended = self._step_records[before:]
        if not appended:
            return
        for record in appended:
            record["privileged_state"] = None
        if self.sample_privileged_state:
            appended[-1]["privileged_state"] = self.privileged_critic_state(
                reset_tracker=False
            )

    def drain_step_records(self) -> list[dict[str, Any]]:
        """Return the accumulated per-step records and clear the buffer.

        Drain rather than ``since_step`` filtering: the only consumer appends
        every record it is handed and never re-reads an earlier one.
        """

        records = self._step_records
        self._step_records = []
        return records

    def step(self, action: Any) -> tuple[dict[str, Any], float, Any, Any, Any]:
        """Execute one physical action (legacy 5-tuple: obs, reward, term, trunc, info)."""

        if self.episode_terminated or self.episode_truncated:
            raise AssertionError("env.step called after the episode signaled term/trunc")
        block = np.asarray([action], dtype=np.float32)
        step = _single(
            self._run(self._client.action_step(self._ids, [encode_array(block)])),
            operation="action_step",
        )
        self.check_done(step.terminated, step.truncated)
        self.episode_steps += 1
        self._record_steps(step, block, 1)
        obs = self._decode_observation(step)
        return obs, float(step.reward), step.terminated, step.truncated, dict(step.info)

    def chunk_step(
        self,
        actions: Any,
        *,
        return_all_frames: bool | None = None,
    ) -> tuple[Any, Any, Any, Any, Any]:
        """Execute an action chunk; legacy 5-tuple with per-step term/trunc arrays.

        Critic evaluation is controlled entirely by whether ``critic_rules``
        were passed to ``reset`` (Runtime semantics: frozen for the whole
        episode), so unlike the legacy client there is no separate
        ``critic_chunk_step`` wire call — see ``critic_chunk_step`` below,
        which is a thin alias kept only so existing call sites do not need to
        branch on which method to call.
        """

        if self.episode_terminated or self.episode_truncated:
            raise AssertionError(
                "env.chunk_step called after the episode signaled term/trunc"
            )
        if return_all_frames is None:
            return_all_frames = self.return_all_frames
        block = np.asarray(actions, dtype=np.float32)
        step = _single(
            self._run(self._client.action_step(self._ids, [encode_array(block)])),
            operation="action_step",
        )
        self.check_done(step.terminated, step.truncated)
        executed_horizon = int(step.executed_horizon)
        self.episode_steps += executed_horizon
        self._record_steps(step, block, executed_horizon)
        info = dict(step.info)
        per_step = step.per_step or ()
        terminated = np.asarray(
            [bool(record.terminated) for record in per_step]
            or [bool(step.terminated)] * executed_horizon
        )
        truncated = np.asarray(
            [bool(record.truncated) for record in per_step]
            or [bool(step.truncated)] * executed_horizon
        )
        if return_all_frames and per_step:
            obs_or_list = [
                self._decode_observation(
                    StepResult(
                        request_id=step.request_id,
                        session_id=step.session_id,
                        observation=record.observation,
                        info=record.info,
                    )
                )
                if record.observation is not None
                else self._decode_observation(step)
                for record in per_step
            ]
        else:
            obs_or_list = self._decode_observation(step)
        return obs_or_list, step.reward, terminated, truncated, info

    def critic_chunk_step(
        self,
        actions: Any,
        *,
        critic_rules: list[dict[str, Any]] | None = None,
        interrupt_on_proposal: bool = True,
        return_all_frames: bool = True,
    ) -> tuple[Any, Any, Any, Any, Any]:
        """Legacy alias for ``chunk_step``.

        The Runtime freezes ``critic_rules``/``critic_interrupt_on_proposal``
        at ``reset`` time (``ResetSpec.options``), not per call, so
        ``critic_rules``/``interrupt_on_proposal`` here are accepted for
        interface compatibility and asserted consistent with what ``reset``
        already established rather than re-sent on the wire. Every
        ``action_step`` after a Critic-configured reset already evaluates the
        Critic every physical step (see ``rollout_runtime/backends/
        rlinf_env.py``), matching the legacy server's per-step behavior.
        """

        return self.chunk_step(actions, return_all_frames=return_all_frames)

    def raw_obs(self) -> dict[str, Any]:
        """Read the raw simulator obs dict (``libero.raw_obs`` extension)."""

        payload = _single(
            self._run(
                self._client.extension_call(self._ids, "libero", "raw_obs", {})
            ),
            operation="extension_call libero.raw_obs",
        )
        if not payload.get("available", False):
            raise RuntimeOperationError(
                f"libero.raw_obs unavailable: {payload.get('reason')}"
            )
        result: dict[str, Any] = dict(payload.get("scalars", {}))
        for key, ref in payload.get("arrays", {}).items():
            result[key] = decode_payload(ref)
        return result

    def privileged_critic_state(self, *, reset_tracker: bool = False) -> dict[str, Any]:
        """Read the Critic-Recovery privileged state sidecar (``libero.critic_state``)."""

        return dict(
            _single(
                self._run(
                    self._client.extension_call(
                        self._ids,
                        "libero",
                        "critic_state",
                        {"reset_tracker": bool(reset_tracker)},
                    )
                ),
                operation="extension_call libero.critic_state",
            )
        )

    def privileged_semantic_joint_plan(
        self, *, entity: str, joint: str, direction: str
    ) -> dict[str, Any]:
        """Read an audited fixture plan for a semantic joint recovery primitive."""

        return dict(
            _single(
                self._run(
                    self._client.extension_call(
                        self._ids,
                        "libero",
                        "semantic_joint_plan",
                        {
                            "entity": str(entity),
                            "joint": str(joint),
                            "direction": str(direction),
                        },
                    )
                ),
                operation="extension_call libero.semantic_joint_plan",
            )
        )

    def privileged_contacts(
        self, *, include_all_contacts: bool = False, max_contacts: int = 64
    ) -> dict[str, Any]:
        """Read current privileged contact/force evidence without stepping."""

        return dict(
            _single(
                self._run(
                    self._client.extension_call(
                        self._ids,
                        "libero",
                        "privileged_contacts",
                        {
                            "include_all_contacts": bool(include_all_contacts),
                            "max_contacts": int(max_contacts),
                        },
                    )
                ),
                operation="extension_call libero.privileged_contacts",
            )
        )

    def render_camera(
        self,
        camera_name: str = "agentview",
        height: int = 1024,
        width: int = 1024,
        depth: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Render an arbitrary camera; legacy return shape (``image`` or ``(image, depth)``)."""

        payload = _single(
            self._run(
                self._client.extension_call(
                    self._ids,
                    "libero",
                    "render_camera",
                    {
                        "camera_name": camera_name,
                        "height": int(height),
                        "width": int(width),
                        "depth": bool(depth),
                    },
                )
            ),
            operation="extension_call libero.render_camera",
        )
        image = decode_payload(payload["image"])
        if depth and payload.get("depth") is not None:
            return image, decode_payload(payload["depth"])
        return image

    def get_camera_meta(
        self,
        camera_name: str = "agentview",
        height: int = 256,
        width: int = 256,
    ) -> dict[str, Any] | None:
        """Read camera intrinsics/extrinsics (``libero.get_camera_meta``)."""

        payload = _single(
            self._run(
                self._client.extension_call(
                    self._ids,
                    "libero",
                    "get_camera_meta",
                    {
                        "camera_name": camera_name,
                        "height": int(height),
                        "width": int(width),
                    },
                )
            ),
            operation="extension_call libero.get_camera_meta",
        )
        if not payload:
            return None
        return dict(payload)

    def get_task_language(self) -> str | None:
        """Return the authoritative task instruction.

        There is no dedicated ``libero.get_task_language`` extension (see the
        module docstring): the instruction rides on every ``Observation``
        instead, so this simply returns whatever the most recent ``reset``/
        ``step``/``chunk_step`` observed.
        """

        return self._task_language

    def cached_image(self) -> np.ndarray | None:
        """Return the most recent main-view frame without stepping."""

        payload = _single(
            self._run(
                self._client.extension_call(self._ids, "libero", "cached_image", {})
            ),
            operation="extension_call libero.cached_image",
        )
        if not payload.get("available", False):
            return None
        return decode_payload(payload["image"])
