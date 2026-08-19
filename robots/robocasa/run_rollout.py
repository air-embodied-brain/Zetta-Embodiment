# Copyright (c) 2026 Zetta Contributors
"""Pure-VLA / critic-interrupted RoboCasa rollout entrypoint."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import threading
import time
import traceback
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from robots.robocasa.env_client import RoboCasaEnvClient
from robots.robocasa.groot_client import Gr00tClient
import numpy as np

from robots.robocasa.recovery_controller import RecoveryController
from robots.robocasa.role1_actor import Role1ActorError, Role1EpisodeActor
from robots.robocasa.role1_agent import (
    Role1ContractError,
    Role1DecisionStore,
    Role1ModelAdapter,
    Role1ModelError,
)
from robots.robocasa.tool_bindings import binding_for_task
from robots.robocasa.tool_catalog import DEFAULT_ROBOCASA_TOOL_CATALOG
from robots.robocasa.tool_runtime import ToolRuntime
from rollout_runtime.api.ids import SessionId
from rollout_runtime.api.messages import (
    CreateSessionRequest,
    EnvSpecMsg,
    PolicyRequest,
    ResetSpec,
    StepResult,
)
from rollout_runtime.api.result import Err, Result
from rollout_runtime.core.payload import decode_array, encode_array
from rollout_runtime.serve.client import RemoteRuntimeClient
from zetta.evolution.jsonio import atomic_write_json, canonical_sha256, read_json
from zetta.evolution.models import (
    CandidateBundle,
    EpisodeRecord,
    FailureSegment,
)
from zetta.evolution.trajectory import (
    TrajectoryArtifacts,
    index_episode_trajectory,
)
from zetta.evolution.visual_artifacts import build_episode_visual_artifacts

RUNTIME_APPLICATION_ID = "zetta-robocasa"
"""Tenant identity reported to the Gateway; the bearer token is authoritative."""

ROBOCASA_EXTENSION_NAMESPACE = "robocasa"
"""Family extension namespace declared by ``core.env_registry.ROBOCASA_EXTENSIONS``."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def _role1_inference_heartbeat(
    path: Path,
    *,
    interval_s: float,
    step_index: int,
    phase: str = "role1_inference",
) -> Iterator[None]:
    """Keep the episode watchdog informed during a bounded Role1 operation.

    The provider call has its own bounded timeout.  This heartbeat therefore
    distinguishes a live, potentially long high-reasoning request from a
    genuinely stalled rollout without weakening the environment no-progress
    watchdog.
    """

    if interval_s <= 0:
        raise ValueError("Role1 heartbeat interval must be positive")
    phase = str(phase).strip()
    if not phase:
        raise ValueError("Role1 heartbeat phase must be non-empty")
    stopped = threading.Event()
    errors: list[BaseException] = []

    def pulse() -> None:
        while not stopped.is_set():
            try:
                _append_jsonl(
                    path,
                    {
                        "phase": phase,
                        "step_index": step_index,
                        "timestamp": _now(),
                    },
                )
            except BaseException as exc:  # pragma: no cover - filesystem failure
                errors.append(exc)
                stopped.set()
                return
            stopped.wait(interval_s)

    thread = threading.Thread(
        target=pulse,
        name="role1-inference-heartbeat",
        daemon=True,
    )
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join(timeout=max(1.0, interval_s + 1.0))
        if thread.is_alive():
            raise RuntimeError("Role1 inference heartbeat did not stop")
        if errors:
            raise RuntimeError("Role1 inference heartbeat failed") from errors[0]


def _chunk_seed(policy_rng: int, chunk_index: int) -> int:
    digest = hashlib.sha256(f"{policy_rng}:{chunk_index}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _is_role1_method_failure(exc: BaseException) -> bool:
    return isinstance(exc, Role1ActorError) or (
        isinstance(exc, Role1ModelError)
        and isinstance(exc.__cause__, Role1ContractError)
    )


def _role1_artifact_index(role1_root: Path) -> dict[str, str]:
    """Expose every immutable Role1 artifact through the episode index."""

    if not role1_root.is_dir():
        return {}
    return {
        f"role1:{path.relative_to(role1_root).as_posix()}": str(path)
        for path in sorted(role1_root.rglob("*"))
        if path.is_file()
    }


class RuntimeOperationError(RuntimeError):
    """A batch entry came back as ``Err``, or the batch shape was wrong.

    Kept distinct from ``ValueError`` so that ``main()``'s
    ``infrastructure_error.json`` branch records a runtime/transport failure
    under its own name instead of hiding inside a generic ``RuntimeError``.
    """


def _single(results: Sequence[Result[Any]], *, operation: str) -> Any:
    """Unwrap a batch-of-one result.

    Args:
        results: The list returned by any ``RemoteRuntimeClient`` operation.
        operation: Operation name for the error message.

    Returns:
        The single success payload.

    Raises:
        RuntimeOperationError: The batch had a length other than one, or the
            entry is an ``Err`` (a per-item failure delivered inside a 200
            response, see ``serve/client.py``).
    """
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


def _chunk_result(step: StepResult, *, vla: dict[str, Any]) -> dict[str, Any]:
    """Normalize a ``StepResult`` into the legacy ``execute_chunk`` dict.

    Both branches of the loop (``policy_step`` and ``action_step``) produce a
    ``StepResult``; every downstream consumer in this file — ``chunks.jsonl``,
    ``actions.jsonl``, ``states.jsonl``, the strict-Gen0 attestation, the
    Critic-Recovery advance and ``EpisodeRecord`` — was written against the
    RoboCasa HTTP payload.  Translating once, here, keeps those artifact schemas
    stable so migrated and pre-migration episode records stay comparable
    (Stage 9 replays the same seeds and diffs them).

    Per-step audit fields (``applied_action`` / ``action_sha256`` /
    ``observation_sha256`` / named ``state``) ride in ``PerStepRecord.info``:
    that is the only per-step channel in the protocol, because
    ``PerStepRecord.observation`` is dropped by design (per-step frames carry no
    images and would only burn payload budget).

    Args:
        step: The runtime step result.
        vla: Inference metadata to record alongside the environment result.

    Returns:
        The legacy chunk dict.
    """
    info = dict(step.info)
    steps: list[dict[str, Any]] = []
    for record in step.per_step or ():
        record_info = dict(record.info)
        steps.append(
            {
                "step_index": int(record.step_index),
                "applied_action": record_info.get("applied_action"),
                "action_sha256": record_info.get("action_sha256"),
                "observation_sha256": record_info.get("observation_sha256"),
                "state": dict(record_info.get("raw_state") or {}),
                "reward": float(record.reward),
                "official_success": bool(record_info.get("official_success")),
                "success_latched": bool(record_info.get("success_latched")),
                "terminated": bool(record.terminated),
                "truncated": bool(record.truncated),
                "proposal_rule_ids": list(record_info.get("proposal_rule_ids") or ()),
            }
        )
    return {
        "executed_horizon": int(step.executed_horizon),
        "steps": steps,
        "critic_proposals": [dict(item) for item in info.get("critic_proposals") or ()],
        "reward": float(step.reward),
        "terminated": bool(step.terminated),
        "truncated": bool(step.truncated),
        "official_success": bool(info.get("official_success")),
        "success_latched": bool(info.get("success_latched")),
        "success_first_step": info.get("success_first_step"),
        "authoritative_success": bool(info.get("authoritative_success")),
        "task_program_enabled": bool(info.get("task_program_enabled")),
        "critic_rule_count": int(info.get("critic_rule_count", 0)),
        "video_paths": dict(info.get("video_paths") or {}),
        "environment_write_owner": info.get("environment_write_owner"),
        "vla": vla,
        "runtime": {
            "request_id": str(step.request_id),
            "episode_id": None if step.episode_id is None else int(step.episode_id),
            "operation_seq": (
                None if step.operation_seq is None else int(step.operation_seq)
            ),
            "model_version": info.get("model_version"),
            "policy_id": info.get("policy_id"),
            "side_effect_applied": bool(step.side_effect_applied),
        },
    }


def _advance_recovery_after_chunk(
    *,
    recovery_controller: RecoveryController,
    selected_tool: str | None,
    environment_step: int,
    result: dict[str, Any],
    tools_path: Path,
) -> bool:
    """Advance only a fully executed Recovery step; preserve interrupted state."""

    proposals = list(result.get("critic_proposals", ()))
    if proposals:
        _append_jsonl(
            tools_path,
            {
                "type": "recovery_step_interrupted_by_critic",
                "environment_step": environment_step,
                "selected_tool": selected_tool,
                "executed_horizon": int(result["executed_horizon"]),
                "critic_proposals": proposals,
                "recovery_advanced": False,
            },
        )
        return False
    recovery_controller.complete_current_step(
        selected_tool=selected_tool,
        environment_step=environment_step,
        executed_horizon=int(result["executed_horizon"]),
    )
    return True


class RolloutSession:
    """One rollout's view of the shared Runtime: a single session, unbatched.

    Every method here is a batch-of-one call against ``RemoteRuntimeClient``
    plus the ``Err`` unwrap; nothing else in this file touches the wire types.
    The class also owns the lease: a Goal/Long episode can run for many minutes
    of simulator time, longer than the Gateway's default lease, so the loop
    renews before the lease can expire underneath a running episode.

    Attributes:
        client: The shared runtime's HTTP client.
        session_id: This rollout's session.
        lease_seconds: Requested lease length; the server may clamp it.
        lease_expiration: Current lease deadline as a unix timestamp.
        episode_started: Whether ``reset`` has completed at least once.
    """

    def __init__(
        self,
        client: Any,
        session_id: SessionId,
        *,
        lease_seconds: float,
        lease_expiration: float,
    ) -> None:
        """Initialize the adapter.

        Args:
            client: A ``RemoteRuntimeClient``-shaped object.
            session_id: The created session.
            lease_seconds: Requested lease length.
            lease_expiration: Lease deadline reported by ``create_sessions``.
        """
        self.client = client
        self.session_id = session_id
        self.lease_seconds = float(lease_seconds)
        self.lease_expiration = float(lease_expiration)
        self.episode_started = False
        self.finalized = False
        self.closed = False

    @property
    def _ids(self) -> list[SessionId]:
        return [self.session_id]

    async def reset(self, reset_spec: ResetSpec) -> StepResult:
        """Reset the episode.

        Args:
            reset_spec: Episode parameters; family-private fields ride in
                ``options`` (see ``backends/robocasa_current.py::_EpisodeOptions``).

        Returns:
            The reset step result.
        """
        step = _single(
            await self.client.reset(self._ids, reset_spec), operation="reset"
        )
        self.episode_started = True
        return step

    async def snapshot(self, *, include_images: bool = True) -> dict[str, Any]:
        """Read the RoboCasa-native observation payload (D8 extension).

        ``Observation`` cannot carry it: its ``state`` is a flat vector (the
        named keys Role1 and the privileged-evidence contract read would be
        lost) and its images are PNG payload refs rather than the data URLs the
        audited Role1 evidence is hashed from.

        Args:
            include_images: Whether to include the three camera data URLs.

        Returns:
            ``RoboCasaSession.snapshot`` output, identical to the payload the
            debug HTTP server returns from ``POST /observation``.
        """
        return dict(
            _single(
                await self.client.extension_call(
                    self._ids,
                    ROBOCASA_EXTENSION_NAMESPACE,
                    "snapshot",
                    {"include_images": bool(include_images)},
                ),
                operation="extension_call robocasa.snapshot",
            )
        )

    async def finalize_episode(self) -> dict[str, Any]:
        """Flush episode video artifacts while keeping the environment warm.

        Returns:
            ``{"finalized": True, "video_paths": ..., "video_manifest": ...}``.
        """
        result = dict(
            _single(
                await self.client.extension_call(
                    self._ids,
                    ROBOCASA_EXTENSION_NAMESPACE,
                    "finalize_episode",
                    {},
                ),
                operation="extension_call robocasa.finalize_episode",
            )
        )
        self.finalized = True
        return result

    async def policy_step(self, policy_request: PolicyRequest) -> StepResult:
        """Atomic observe -> infer -> chunk_step (the no-Role1 fast path).

        Args:
            policy_request: Inference parameters.

        Returns:
            The step result.
        """
        return _single(
            await self.client.policy_step(self._ids, policy_request),
            operation="policy_step",
        )

    async def policy_infer(
        self, policy_request: PolicyRequest
    ) -> tuple[list[list[float]], dict[str, Any]]:
        """Infer without writing to the environment (the Role1 review path).

        Args:
            policy_request: Inference parameters.

        Returns:
            ``(actions, metadata)`` where ``actions`` is a plain nested list of
            floats — exactly the shape Role1 and ``action_step`` expect.

        Raises:
            RuntimeOperationError: The policy returned no action chunk.
        """
        result = _single(
            await self.client.policy_infer(self._ids, policy_request),
            operation="policy_infer",
        )
        if result.actions is None:
            raise RuntimeOperationError("policy_infer returned no action chunk")
        block = np.asarray(decode_array(result.actions), dtype=np.float32)
        if block.ndim != 2:
            raise RuntimeOperationError(
                f"policy_infer returned shape {tuple(int(v) for v in block.shape)}, "
                "expected [chunk, action_dim]"
            )
        metadata = {
            "source": "policy_infer",
            "horizon": int(block.shape[0]),
            "model_version": result.model_version,
            "observation_step_index": int(result.observation_step_index),
            "action_chunk_sha256": hashlib.sha256(block.tobytes()).hexdigest(),
            "auxiliary_outputs": dict(result.auxiliary_outputs),
            "policy_id": dict(result.info).get("policy_id"),
        }
        return [[float(value) for value in row] for row in block], metadata

    async def action_step(self, actions: Sequence[Sequence[float]]) -> StepResult:
        """Execute an externally decided action chunk.

        Args:
            actions: ``[chunk, action_dim]`` actions (Role1's reviewed chunk).

        Returns:
            The step result.
        """
        block = np.asarray(actions, dtype=np.float32)
        return _single(
            await self.client.action_step(self._ids, [encode_array(block)]),
            operation="action_step",
        )

    async def renew_if_needed(self, *, now: float | None = None) -> None:
        """Renew the lease before it can expire during a running episode.

        Args:
            now: Injectable clock reading; defaults to ``time.time()``.
        """
        moment = time.time() if now is None else now
        margin = max(30.0, self.lease_seconds / 4.0)
        if moment < self.lease_expiration - margin:
            return
        status = _single(
            await self.client.renew_sessions(self._ids, self.lease_seconds),
            operation="renew_sessions",
        )
        self.lease_expiration = float(status.lease_expiration)

    async def close(self) -> dict[str, Any]:
        """Close the session, returning its slot to the pool for reuse.

        This is the Runtime-native replacement for the old
        ``RoboCasaEnvClient.release()``: the binding, the env slot and the
        session state machine are all Gateway-owned now.

        Returns:
            ``{"session_closed": bool, "session_id": str}``; a failure is
            recorded rather than raised so that a cleanup error cannot mask the
            episode's own outcome.
        """
        try:
            _single(
                await self.client.close_sessions(self._ids),
                operation="close_sessions",
            )
        except Exception as exc:
            return {
                "session_closed": False,
                "session_id": str(self.session_id),
                "failure_class": type(exc).__name__,
            }
        self.closed = True
        return {"session_closed": True, "session_id": str(self.session_id)}


async def _run_with_session(
    args: argparse.Namespace,
    environment: RolloutSession,
    *,
    env_spec_digest: str = "",
) -> EpisodeRecord:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trajectory_root = output / "trajectory"
    trajectory_root.mkdir(parents=True, exist_ok=True)
    actions_path = trajectory_root / "actions.jsonl"
    states_path = trajectory_root / "states.jsonl"
    chunks_path = trajectory_root / "chunks.jsonl"
    tools_path = output / "tool_events.jsonl"
    for path in (actions_path, states_path, chunks_path, tools_path):
        path.touch(exist_ok=True)
    started_at = _now()
    started = time.time()
    episode_id = uuid.uuid4().hex
    bundle: CandidateBundle | None = None
    if args.bundle and args.bundle != "none":
        bundle = CandidateBundle.from_dict(read_json(args.bundle))
        if args.bundle_sha256 != bundle.sha256:
            raise ValueError("candidate bundle SHA does not match --bundle-sha256")
    if bundle is not None and args.role1_planner == "none":
        raise ValueError("candidate bundles require --role1-planner api or codex")
    baseline_mode = getattr(
        args,
        "baseline_mode",
        "active_bundle" if bundle is not None else "strict_pure_vla",
    )
    if baseline_mode not in {"strict_pure_vla", "active_bundle"}:
        raise ValueError(f"unsupported baseline mode: {baseline_mode}")
    if baseline_mode == "strict_pure_vla" and bundle is not None:
        raise ValueError("strict pure-VLA baseline cannot load a candidate bundle")
    if baseline_mode == "strict_pure_vla" and (
        args.generation != 0 or args.bundle_sha256 != "none"
    ):
        raise ValueError("strict pure-VLA baseline requires Gen0 and an empty bundle")
    if baseline_mode == "active_bundle" and bundle is None:
        raise ValueError("active-bundle rollout requires a frozen candidate bundle")
    enable_task_program = False
    binding = None
    tool_runtime = None
    tool_runtime_manifest: dict[str, Any] = {
        "backend": "none_pure_vla",
        "tool_count": 0,
        "tool_names": [],
        "manifest_sha256": None,
    }
    if args.role1_planner != "none" and baseline_mode != "strict_pure_vla":
        binding = binding_for_task(args.task)
        runtime_backend = getattr(args, "tool_runtime", "builtin")
        if runtime_backend == "harness":
            from robots.robocasa.harness_adapter import HarnessToolRuntimeAdapter

            tool_runtime = HarnessToolRuntimeAdapter.from_root(
                getattr(args, "harness_root", None)
            )
            tool_runtime.require_tools(binding.tool_names)
            tool_runtime_manifest = tool_runtime.describe()
        else:
            tool_runtime = ToolRuntime()
            tool_runtime_manifest = {
                "backend": "builtin",
                "tool_count": len(DEFAULT_ROBOCASA_TOOL_CATALOG.names()),
                "tool_names": list(DEFAULT_ROBOCASA_TOOL_CATALOG.names()),
                "manifest_sha256": DEFAULT_ROBOCASA_TOOL_CATALOG.digest,
            }
    # The Critic rules are frozen for the whole episode, so they are delivered
    # once through ``ResetSpec.options`` instead of per chunk: they cannot live
    # in ``env_config`` (that feeds the env-pool digest, so a per-episode value
    # would cold-start a new simulator every episode) and ``RoboCasaSession``
    # itself refuses a mid-episode rule change.
    critic_rules = [rule.as_dict() for rule in bundle.critic_rules] if bundle else []
    reset_step = await environment.reset(
        ResetSpec(
            seed=args.seed,
            instruction=args.instruction,
            options={
                "video_dir": str(output / "videos"),
                "bundle_sha256": bundle.sha256 if bundle else None,
                "critic_rules": critic_rules,
                "interrupt_on_proposal": bool(critic_rules),
                "capture_event_images": True,
                "enable_task_program": enable_task_program,
            },
        )
    )
    # ``robocasa.snapshot`` rather than ``reset_step.observation``: the audited
    # observation identity and Role1's evidence are both defined over the named
    # state dict and the JPEG data URLs that ``RoboCasaSession.snapshot``
    # produces (see ``RolloutSession.snapshot``).
    reset = await environment.snapshot(include_images=True)
    if baseline_mode == "strict_pure_vla" and (
        reset.get("task_program_enabled") is not False
        or int(reset.get("critic_rule_count", 0)) != 0
    ):
        raise RuntimeError("strict Gen0 environment attestation failed")
    instruction = args.instruction or str(
        reset.get("observation", {})
        .get("state", {})
        .get("annotation.human.task_description", args.task)
    )
    initial_state = reset.get("observation", {}).get("state", {})
    initial_images = reset.get("observation", {}).get("images", {})
    initial_observation_identity = {
        "state_sha256": canonical_sha256(initial_state),
        "camera_sha256": {
            str(key): hashlib.sha256(str(value).encode("utf-8")).hexdigest()
            for key, value in sorted(initial_images.items())
        }
        if isinstance(initial_images, dict)
        else {},
    }
    if isinstance(initial_state, dict):
        _append_jsonl(
            states_path,
            {"step_index": 0, "state": initial_state, "event": "reset"},
        )
    role1_actor: Role1EpisodeActor | None = None
    recovery_controller: RecoveryController | None = None
    if args.role1_planner != "none" and baseline_mode != "strict_pure_vla":
        decision_store = Role1DecisionStore(output / "role1" / "decisions")
        adapter = Role1ModelAdapter(
            store=decision_store,
            output_root=output / "role1" / "invocations",
            planner_type=args.role1_planner,
            model=args.role1_model or os.environ.get("ZETTA_ROLE1_MODEL"),
            reasoning_effort=args.reasoning_effort,
            base_url=os.environ.get("ZETTA_ROLE1_BASE_URL"),
            max_tokens=args.role1_max_tokens,
            timeout_s=args.role1_timeout_s,
            max_turns=args.role1_max_turns,
            require_visual_review=True,
        )
        assert binding is not None
        assert tool_runtime is not None
        role1_actor = Role1EpisodeActor(
            adapter=adapter,
            decision_store=decision_store,
            tool_runtime=tool_runtime,
            binding=binding,
            audit_root=output / "role1" / "actor",
            allow_privileged=args.allow_privileged_tools,
            maximum_decisions_without_action=args.role1_max_decisions_per_action,
        )
        if bundle is not None:
            recovery_controller = RecoveryController(
                bundle_sha256=bundle.sha256,
                audit_path=output / "role1" / "recovery-events.jsonl",
            )

    authoritative_success = bool(reset.get("authoritative_success"))
    terminated = bool(reset.get("terminated"))
    truncated = bool(reset.get("truncated"))
    last_proposals: list[dict[str, Any]] = []
    chunks = 0
    actions_executed = 0
    role1_decisions = 0
    role1_terminated = False
    role1_contract_failure = False
    while (
        not authoritative_success
        and not terminated
        and not truncated
        and actions_executed < args.max_actions
    ):
        await environment.renew_if_needed()
        inference_seed = _chunk_seed(args.policy_rng, chunks)
        policy_request = PolicyRequest(
            policy_id=args.policy_id,
            instruction_override=instruction,
            # GR00T's per-request seed contract (``groot_core.parse_inference_seed``
            # via ``backends/groot_policy.py``): the same policy_rng and chunk
            # index must reproduce the same action chunk on replay.
            inference_parameters={"seed": inference_seed},
            actions_per_chunk=args.actions_per_chunk,
        )
        recovery_suggestions: list[dict[str, Any]] = []
        if bundle and recovery_controller is not None:
            recovery_controller.activate(
                critic_proposals=last_proposals,
                recovery_rules=[rule.as_dict() for rule in bundle.recovery_rules],
                environment_step=actions_executed,
            )
            context = recovery_controller.context()
            if context is not None:
                recovery_id = str(context["recovery_id"])
                recovery_suggestions = [
                    recovery.as_dict()
                    for recovery in bundle.recovery_rules
                    if recovery.recovery_id == recovery_id
                ]
        active_recovery = (
            recovery_controller.context() if recovery_controller is not None else None
        )
        # The call path is decided *before* inference, because the two paths
        # differ in whether the environment is written by the same operation:
        # ``policy_step`` is atomic and never exposes the chunk, so a chunk that
        # Role1 may rewrite has to be inferred (``policy_infer``) and executed
        # (``action_step``) separately.
        role1_engaged = role1_actor is not None and bool(
            last_proposals or active_recovery is not None
        )
        selected_tool: str | None = None
        if not role1_engaged:
            step = await environment.policy_step(policy_request)
            vla_meta: dict[str, Any] = {
                "source": "policy_step",
                "inference_seed": inference_seed,
                "actions_per_chunk": int(args.actions_per_chunk),
                "model_version": dict(step.info).get("model_version"),
                "policy_id": dict(step.info).get("policy_id"),
            }
            result = _chunk_result(step, vla=vla_meta)
        else:
            assert role1_actor is not None
            actions, vla_meta = await environment.policy_infer(policy_request)
            actions = actions[: args.actions_per_chunk]
            vla_meta = {**vla_meta, "inference_seed": inference_seed}
            observation = await environment.snapshot(include_images=True)
            try:
                with _role1_inference_heartbeat(
                    output / "heartbeat.jsonl",
                    interval_s=float(getattr(args, "role1_heartbeat_s", 15.0)),
                    step_index=actions_executed,
                    phase="role1_actor",
                ):
                    reviewed = role1_actor.decide_action(
                        task=args.task,
                        step_index=actions_executed,
                        observation_response=observation,
                        vla_actions=actions,
                        vla_metadata=vla_meta,
                        critic_values=last_proposals,
                        recovery_suggestions=recovery_suggestions,
                        active_recovery=active_recovery,
                    )
            except (Role1ActorError, Role1ModelError) as exc:
                if not _is_role1_method_failure(exc):
                    # Provider, transport, attestation and private-image
                    # integrity failures remain infrastructure-invalid.
                    raise
                # The provider returned a complete audited decision, but the
                # method either emitted invalid contract fields or failed to
                # materialize an executable action. This is a valid controller
                # failure, not infrastructure noise: count it as zero rather
                # than retrying until a favorable sample.
                _append_jsonl(
                    tools_path,
                    {
                        "type": "role1_contract_failure",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "environment_write": False,
                    },
                )
                role1_contract_failure = True
                break
            role1_decisions += len(reviewed.decision_ids)
            selected_tool = reviewed.selected_tool
            _append_jsonl(
                tools_path,
                {
                    "type": "role1_action_boundary",
                    "decision_ids": list(reviewed.decision_ids),
                    "selected_tool": reviewed.selected_tool,
                    "terminate": reviewed.terminate,
                    "termination_reason": reviewed.termination_reason,
                    "environment_write": False,
                },
            )
            if reviewed.terminate:
                role1_terminated = True
                break
            step = await environment.action_step(list(reviewed.actions))
            result = _chunk_result(
                step,
                vla={
                    **vla_meta,
                    "role1_reviewed": True,
                    "role1_selected_tool": reviewed.selected_tool,
                },
            )
        chunks += 1
        actions_executed += int(result["executed_horizon"])
        new_proposals = list(result.get("critic_proposals", ()))
        if active_recovery is not None and recovery_controller is not None:
            _advance_recovery_after_chunk(
                recovery_controller=recovery_controller,
                selected_tool=selected_tool,
                environment_step=actions_executed,
                result=result,
                tools_path=tools_path,
            )
        if baseline_mode == "strict_pure_vla" and (
            result.get("critic_proposals")
            or result.get("task_program_enabled") is not False
            or int(result.get("critic_rule_count", 0)) != 0
        ):
            raise RuntimeError("strict Gen0 received an online Critic/task proposal")
        last_proposals = new_proposals
        _append_jsonl(
            chunks_path,
            {
                "chunk_index": chunks - 1,
                "vla": result["vla"],
                "environment": result,
            },
        )
        for step_record in result.get("steps", ()):
            if not isinstance(step_record, dict):
                raise ValueError("environment step record must be an object")
            _append_jsonl(
                actions_path,
                {
                    "step_index": step_record.get("step_index"),
                    "action": step_record.get("applied_action"),
                    "action_sha256": step_record.get("action_sha256"),
                },
            )
            _append_jsonl(
                states_path,
                {
                    "step_index": step_record.get("step_index"),
                    "state": step_record.get("state", {}),
                    "reward": step_record.get("reward"),
                    "official_success": step_record.get("official_success"),
                    "success_latched": step_record.get("success_latched"),
                    "terminated": step_record.get("terminated"),
                    "truncated": step_record.get("truncated"),
                    "proposal_rule_ids": step_record.get("proposal_rule_ids", []),
                },
            )
        _append_jsonl(
            output / "heartbeat.jsonl",
            {
                "chunk_index": chunks - 1,
                "actions_executed": actions_executed,
                "timestamp": _now(),
            },
        )
        authoritative_success = bool(result.get("authoritative_success"))
        terminated = bool(result.get("terminated"))
        truncated = bool(result.get("truncated"))

    failure = None
    if not authoritative_success:
        failure_class = (
            "role1_contract_failure"
            if role1_contract_failure
            else "role1_terminated"
            if role1_terminated
            else "critic_proposal_unrecovered"
            if last_proposals
            else "vla_task_incomplete"
        )
        failure = FailureSegment(
            segment_id=f"segment-{episode_id}",
            episode_id=episode_id,
            failure_class=failure_class,
            stage="closed_loop_execution",
            tool="vla_replan" if last_proposals else "groot",
            summary=(
                f"Task incomplete after {actions_executed} actions and {chunks} VLA chunks; "
                f"critic proposals={sorted({p['rule_id'] for p in last_proposals})}."
            ),
            earliest_divergence_step=max(0, actions_executed - 1),
            start_step=max(0, actions_executed - args.actions_per_chunk),
            end_step=max(0, actions_executed),
            severity=1.0,
            artifact_paths=(str(chunks_path),),
        )
    finalized = await environment.finalize_episode()
    released = await environment.close()
    record = EpisodeRecord(
        episode_id=episode_id,
        logical_id=args.logical_id,
        generation=args.generation,
        seed=args.seed,
        policy_rng=args.policy_rng,
        bundle_sha256=bundle.sha256 if bundle else None,
        status="valid",
        success=authoritative_success,
        started_at=started_at,
        finished_at=_now(),
        elapsed_s=time.time() - started,
        artifact_index={
            "trajectory": str(chunks_path),
            "actions": str(actions_path),
            "states": str(states_path),
            "tool_events": str(tools_path),
            "videos": finalized.get("video_paths", reset.get("video_paths", {})),
            "actions_executed": actions_executed,
            "vla_chunks": chunks,
            "role1_decisions": role1_decisions,
            "terminated": terminated,
            "truncated": truncated,
            "baseline_mode": baseline_mode,
            "active_bundle_sha256": bundle.sha256 if bundle else None,
            "safety_layer": getattr(args, "safety_layer", "interface_contract_v1"),
            "task_program_enabled": enable_task_program,
            "tool_runtime": tool_runtime_manifest,
            # The rollout runtime replaces the old two-server deployment, so the
            # audit trail records which shared runtime and which env pool served
            # this episode instead of an env/VLA endpoint pair.
            "rollout_runtime": {
                "session_id": str(environment.session_id),
                "env_spec_digest": env_spec_digest,
                "policy_id": args.policy_id,
                "reset_episode_id": (
                    None
                    if reset_step.episode_id is None
                    else int(reset_step.episode_id)
                ),
            },
            "environment_release": released,
            "initial_observation_identity": initial_observation_identity,
            **_role1_artifact_index(output / "role1"),
        },
        safety_events=("role1_contract_failure",) if role1_contract_failure else (),
        failure_segment=failure,
        failure_segments=(failure,) if failure is not None else (),
        attempt_index=args.attempt_index,
    )
    video_paths = tuple(
        str(path)
        for path in finalized.get("video_paths", reset.get("video_paths", {})).values()
        if isinstance(path, str)
    )
    trajectory_analysis = index_episode_trajectory(
        result=record,
        artifacts=TrajectoryArtifacts(
            chunks=chunks_path,
            actions=actions_path,
            states=states_path,
            tools=tools_path,
            videos=video_paths,
        ),
    )
    failure_segments = (
        trajectory_analysis.segments
        if trajectory_analysis.segments
        else (failure,)
        if failure is not None
        else ()
    )
    visual_evidence = build_episode_visual_artifacts(
        video_paths={
            str(name): str(path)
            for name, path in finalized.get(
                "video_paths", reset.get("video_paths", {})
            ).items()
            if isinstance(path, str)
        },
        states_path=states_path,
        output_root=output / "visual-evidence",
        divergence_steps=tuple(
            segment.earliest_divergence_step
            for segment in failure_segments
            if segment.earliest_divergence_step is not None
        ),
    )
    record = replace(
        record,
        artifact_index={
            **record.artifact_index,
            "trajectory_index": (
                trajectory_analysis.index.as_dict()
                if trajectory_analysis.index is not None
                else None
            ),
            "failure_segment_count": len(failure_segments),
            "visual_evidence": visual_evidence,
        },
        failure_segment=failure_segments[0] if failure_segments else None,
        failure_segments=failure_segments,
    )
    atomic_write_json(args.result_file, record.as_dict(), overwrite=False)
    return record


def _env_spec(args: argparse.Namespace) -> EnvSpecMsg:
    """Build the env spec that selects (or creates) this episode's env pool.

    Every field here enters ``EnvSpecMsg.digest()``, which is the pool key: all
    rollout processes of one campaign must pass identical values or they will
    each cold-start their own simulator pool. Per-episode values (seed, video
    directory, Critic rules) deliberately do **not** live here — they ride in
    ``ResetSpec.options``.

    Args:
        args: Parsed CLI arguments.

    Returns:
        The env spec for ``create_sessions``.
    """
    return EnvSpecMsg(
        env_family="robocasa",
        env_config={
            "task": args.task,
            "split": args.split,
            "camera_size": args.camera_size,
            "max_steps": args.env_max_steps,
            "require_isolated_renderer": bool(args.require_isolated_renderer),
            "process_isolation": bool(args.process_isolation),
        },
        pool_size=int(args.env_pool_size),
        max_dynamic_pool_size=args.env_max_pool_size,
        resource_hints={"accelerator": True},
    )


async def run(args: argparse.Namespace, *, client: Any | None = None) -> EpisodeRecord:
    """Run one episode against the shared rollout runtime.

    The session is always closed, which is what returns the env slot to the pool
    for the next rollout process (the old ``release()`` semantics).

    Args:
        args: Parsed CLI arguments.
        client: Injected ``RemoteRuntimeClient``-shaped object; tests pass an
            ASGI-backed or fake client. When ``None`` this function owns the
            client's connection pool and closes it on exit.

    Returns:
        The episode record.
    """
    owns_client = client is None
    if client is None:
        client = RemoteRuntimeClient(
            args.runtime_url,
            token=args.runtime_token,
            operation_timeout_s=args.operation_timeout_s,
            session_timeout_s=args.session_timeout_s,
        )
    environment: RolloutSession | None = None
    try:
        handle = _single(
            await client.create_sessions(
                [
                    CreateSessionRequest(
                        application_id=RUNTIME_APPLICATION_ID,
                        client_session_key=(
                            f"{args.logical_id}-attempt-{args.attempt_index}"
                        ),
                        env_spec=_env_spec(args),
                        default_policy_id=args.policy_id,
                        lease_seconds=float(args.session_lease_s),
                        metadata={
                            "task": args.task,
                            "seed": int(args.seed),
                            "generation": int(args.generation),
                            "logical_id": args.logical_id,
                        },
                    )
                ]
            ),
            operation="create_sessions",
        )
        environment = RolloutSession(
            client,
            handle.session_id,
            lease_seconds=float(args.session_lease_s),
            lease_expiration=float(handle.lease_expiration),
        )
        return await _run_with_session(
            args, environment, env_spec_digest=handle.env_spec_digest
        )
    finally:
        if environment is not None:
            cleanup_errors: list[str] = []
            if environment.episode_started and not environment.finalized:
                try:
                    await environment.finalize_episode()
                except Exception as exc:
                    cleanup_errors.append(type(exc).__name__)
            if not environment.closed:
                closed = await environment.close()
                if closed.get("session_closed") is not True:
                    cleanup_errors.append(str(closed.get("failure_class", "unknown")))
            if cleanup_errors:
                _append_jsonl(
                    Path(args.output_dir) / "cleanup_errors.jsonl",
                    {"at": _now(), "failure_classes": cleanup_errors},
                )
        if owns_client:
            await client.aclose()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one frozen RoboCasa rollout")
    parser.add_argument(
        "--runtime-url",
        required=True,
        help=(
            "Base URL of the shared rollout-runtime serve process "
            "(RemoteRuntimeClient -> Gateway -> Ray workers)."
        ),
    )
    parser.add_argument(
        "--runtime-token",
        default=os.environ.get("ZETTA_RUNTIME_TOKEN"),
        help="Bearer token for the runtime; defaults to ZETTA_RUNTIME_TOKEN.",
    )
    parser.add_argument(
        "--policy-id",
        default="groot",
        help="Policy id served by the runtime's RolloutWorker (preset: groot).",
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--split", default="target")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--policy-rng", type=int, required=True)
    parser.add_argument("--logical-id", required=True)
    parser.add_argument("--attempt-index", type=int, default=0)
    parser.add_argument("--generation", type=int, default=0)
    parser.add_argument("--bundle", default=None)
    parser.add_argument("--bundle-sha256", default="none")
    parser.add_argument(
        "--baseline-mode",
        choices=("strict_pure_vla", "active_bundle"),
        default="strict_pure_vla",
    )
    parser.add_argument(
        "--safety-layer",
        choices=("interface_contract_v1",),
        default="interface_contract_v1",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--max-actions", type=int, default=1000)
    parser.add_argument("--actions-per-chunk", type=int, default=16)
    parser.add_argument(
        "--camera-size",
        type=int,
        default=256,
        help="Camera resolution; part of the env pool identity.",
    )
    parser.add_argument(
        "--env-max-steps",
        type=int,
        default=1000,
        help="RoboCasaSession step cap; part of the env pool identity.",
    )
    parser.add_argument(
        "--require-isolated-renderer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require an isolated MuJoCo renderer; part of the env pool identity.",
    )
    parser.add_argument(
        "--process-isolation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run each env pool slot's RoboCasaSession in its own OS subprocess "
            "(robots/robocasa/session_process.py) instead of this rank's shared "
            "thread pool; part of the env pool identity. Only needed when "
            "--env-pool-size > 1 on a single rank: runtime comparison test design "
            "\u00a70.2 documents that multiple RoboCasaSession instances sharing one "
            "GPU inside one process can race inside native robosuite/MuJoCo/EGL "
            "calls during reset()/chunk_step(), nondeterministically producing "
            "EGL_BAD_ACCESS crashes or full deadlocks. Leave this off (default) "
            "for --env-pool-size=1, which has no cross-session contention and no "
            "reason to pay subprocess IPC overhead."
        ),
    )
    parser.add_argument(
        "--env-pool-size",
        type=int,
        default=1,
        help=(
            "Initial env slots for this spec. Every rollout process in one "
            "campaign must pass the same value: it enters the pool digest."
        ),
    )
    parser.add_argument(
        "--env-max-pool-size",
        type=int,
        default=None,
        help="Upper bound for dynamic slot growth; defaults to --env-pool-size.",
    )
    parser.add_argument(
        "--session-lease-s",
        type=float,
        default=1800.0,
        help="Requested session lease; renewed automatically during the episode.",
    )
    parser.add_argument(
        "--operation-timeout-s",
        type=float,
        default=900.0,
        help="Read timeout for reset / policy_step / action_step / extension_call.",
    )
    parser.add_argument(
        "--session-timeout-s",
        type=float,
        default=1800.0,
        help="Read timeout for create_sessions (the runtime may cold-start a pool).",
    )
    parser.add_argument(
        "--role1-planner",
        choices=("none", "api", "codex"),
        default="none",
    )
    parser.add_argument("--role1-model", default=None)
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high", "xhigh"),
        default=os.environ.get("ZETTA_REASONING_EFFORT"),
    )
    parser.add_argument("--role1-max-tokens", type=int, default=4096)
    parser.add_argument("--role1-timeout-s", type=int, default=900)
    parser.add_argument("--role1-heartbeat-s", type=float, default=15.0)
    parser.add_argument("--role1-max-turns", type=int, default=2)
    parser.add_argument("--role1-max-decisions-per-action", type=int, default=4)
    parser.add_argument(
        "--tool-runtime",
        choices=("builtin", "harness"),
        default="builtin",
        help="Proposal-tool runtime used by Role1; harness is loaded only explicitly.",
    )
    parser.add_argument(
        "--harness-root",
        default=os.environ.get("ZETTA_ROBOCASA_HARNESS_ROOT"),
        help="Frozen harness snapshot root used by --tool-runtime harness.",
    )
    parser.add_argument(
        "--allow-privileged-tools",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    try:
        record = asyncio.run(run(args))
        return 0 if record.status == "valid" else 2
    except Exception as exc:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            output / "infrastructure_error.json",
            {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "timestamp": _now(),
            },
            overwrite=False,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
