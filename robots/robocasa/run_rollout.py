# Copyright (c) 2026 RPent Contributors
"""Pure-VLA / critic-interrupted RoboCasa rollout entrypoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
import traceback
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from robots.robocasa.env_client import RoboCasaEnvClient
from robots.robocasa.groot_client import Gr00tClient
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
from rpent.evolution.jsonio import atomic_write_json, canonical_sha256, read_json
from rpent.evolution.models import (
    CandidateBundle,
    EpisodeRecord,
    FailureSegment,
)
from rpent.evolution.trajectory import (
    TrajectoryArtifacts,
    index_episode_trajectory,
)
from rpent.evolution.visual_artifacts import build_episode_visual_artifacts


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


def _run_with_environment(
    args: argparse.Namespace, environment: RoboCasaEnvClient
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
    vla = Gr00tClient(args.vla_endpoint, timeout_s=args.vla_timeout_s)
    reset = environment.reset(
        task=args.task,
        seed=args.seed,
        split=args.split,
        bundle_sha256=bundle.sha256 if bundle else None,
        video_dir=str(output / "videos"),
        enable_task_program=enable_task_program,
    )
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
    critic_rules = [rule.as_dict() for rule in bundle.critic_rules] if bundle else []
    role1_actor: Role1EpisodeActor | None = None
    recovery_controller: RecoveryController | None = None
    if args.role1_planner != "none" and baseline_mode != "strict_pure_vla":
        decision_store = Role1DecisionStore(output / "role1" / "decisions")
        adapter = Role1ModelAdapter(
            store=decision_store,
            output_root=output / "role1" / "invocations",
            planner_type=args.role1_planner,
            model=args.role1_model or os.environ.get("RPENT_ROLE1_MODEL"),
            reasoning_effort=args.reasoning_effort,
            base_url=os.environ.get("RPENT_ROLE1_BASE_URL"),
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
        observation = environment.observation(include_images=True)
        inference_seed = _chunk_seed(args.policy_rng, chunks)
        actions, vla_meta = vla.act(
            observation,
            instruction=instruction,
            inference_seed=inference_seed,
        )
        actions = actions[: args.actions_per_chunk]
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
        if role1_actor is not None and (last_proposals or active_recovery is not None):
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
            actions = list(reviewed.actions)
        result = environment.execute_chunk(
            actions,
            critic_rules=critic_rules,
            interrupt_on_proposal=bool(critic_rules),
            capture_event_images=True,
            enable_task_program=enable_task_program,
        )
        chunks += 1
        actions_executed += int(result["executed_horizon"])
        new_proposals = list(result.get("critic_proposals", ()))
        if active_recovery is not None and recovery_controller is not None:
            _advance_recovery_after_chunk(
                recovery_controller=recovery_controller,
                selected_tool=reviewed.selected_tool,
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
                "vla": vla_meta,
                "environment": result,
            },
        )
        for step in result.get("steps", ()):
            if not isinstance(step, dict):
                raise ValueError("environment step record must be an object")
            _append_jsonl(
                actions_path,
                {
                    "step_index": step.get("step_index"),
                    "action": step.get("applied_action"),
                    "action_sha256": step.get("action_sha256"),
                },
            )
            _append_jsonl(
                states_path,
                {
                    "step_index": step.get("step_index"),
                    "state": step.get("state", {}),
                    "reward": step.get("reward"),
                    "official_success": step.get("official_success"),
                    "success_latched": step.get("success_latched"),
                    "terminated": step.get("terminated"),
                    "truncated": step.get("truncated"),
                    "proposal_rule_ids": step.get("proposal_rule_ids", []),
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
    finalized = environment.finalize_episode()
    released = environment.release()
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
            "environment_release": {
                "binding_released": released.get("binding_released") is True,
                "released_generation": released.get("released_generation"),
            },
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


def run(args: argparse.Namespace) -> EpisodeRecord:
    """Run one episode and release a known-good persistent slot on every exit."""

    environment = RoboCasaEnvClient(args.env_endpoint, timeout_s=args.rpc_timeout_s)
    try:
        return _run_with_environment(args, environment)
    finally:
        if environment.episode_id is not None and not environment.outcome_unknown:
            cleanup_errors: list[str] = []
            try:
                environment.finalize_episode()
            except Exception as exc:
                cleanup_errors.append(type(exc).__name__)
            if not environment.outcome_unknown:
                try:
                    environment.release()
                except Exception as exc:
                    cleanup_errors.append(type(exc).__name__)
            if cleanup_errors:
                _append_jsonl(
                    Path(args.output_dir) / "cleanup_errors.jsonl",
                    {"at": _now(), "failure_classes": cleanup_errors},
                )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one frozen RoboCasa rollout")
    parser.add_argument("--env-endpoint", required=True)
    parser.add_argument("--vla-endpoint", required=True)
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
    parser.add_argument("--rpc-timeout-s", type=float, default=180.0)
    parser.add_argument("--vla-timeout-s", type=float, default=120.0)
    parser.add_argument(
        "--role1-planner",
        choices=("none", "api", "codex"),
        default="none",
    )
    parser.add_argument("--role1-model", default=None)
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high", "xhigh"),
        default=os.environ.get("RPENT_REASONING_EFFORT"),
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
        default=os.environ.get("RPENT_ROBOCASA_HARNESS_ROOT"),
        help="Frozen harness snapshot root used by --tool-runtime harness.",
    )
    parser.add_argument(
        "--allow-privileged-tools",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    try:
        record = run(args)
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
