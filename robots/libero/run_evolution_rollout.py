#!/usr/bin/env python3
# Copyright (c) 2026 Zetta Contributors
# ruff: noqa: E402
"""One strict, auditable LIBERO-Pro evolution episode."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from robots.libero.critic_runtime import extract_libero_critic_features
from robots.libero.env_client import LiberoEnvClient
from robots.libero.evolution_defaults import (
    LIBERO_WAIT_STEPS_DEFAULT,
    add_privileged_evidence_argument,
    libero_horizon_contract,
    privileged_evidence_enabled,
)
from robots.libero.role1_recovery import (
    LiberoRecoveryActorError,
    LiberoRole1RecoveryActor,
)
from robots.libero.runtime_devices import (
    attach_vla_runtime_verification,
    describe_runtime_devices,
    parse_physical_gpus,
    require_isolated_runtime_devices,
)
from robots.libero.tool_catalog import (
    DEFAULT_LIBERO_ROLE1_TOOL_CATALOG,
    LIBERO_RECOVERY_TOOL_NAMES,
)
from robots.libero.tools import LiberoPrimitives
from robots.robocasa.recovery_controller import RecoveryController
from robots.robocasa.role1_agent import (
    Role1ContractError,
    Role1DecisionStore,
    Role1ModelAdapter,
    Role1ModelError,
)
from zetta.evolution.jsonio import atomic_write_json, canonical_sha256, read_json
from zetta.evolution.models import CandidateBundle, EpisodeRecord
from zetta.evolution.trajectory import TrajectoryArtifacts, index_episode_trajectory
from zetta.evolution.visual_artifacts import (
    build_episode_visual_artifacts,
    write_video_metadata,
)
from zetta.utils.daemon import ProcessDaemon, pick_free_port
from zetta.utils.http_rpc import HttpRpcClient
from zetta.utils.rpc import RpcError, wait_for_ready
from zetta.utils.sam3_client import UnavailableSam3Client
from zetta.utils.vla_client import VLAClient

DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)
LIBERO_RECOVERY_TOOLS = LIBERO_RECOVERY_TOOL_NAMES


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _exception_traceback(exc: Exception) -> str:
    """Return the local traceback plus any traceback sent by an RPC server."""

    local_traceback = traceback.format_exc()
    remote_traceback = getattr(exc, "server_traceback", None)
    if not remote_traceback:
        return local_traceback
    return (
        f"{local_traceback}\n--- remote RPC traceback ---\n"
        f"{remote_traceback}"
    )


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _normalize_endpoint(value: str) -> str:
    value = value.strip().rstrip("/")
    return value if "://" in value else f"http://{value}"


def _normalize_task_language(value: str) -> str:
    return " ".join(value.casefold().split())


def _require_expected_task_language(
    observed: str | None, expected: str | None
) -> str:
    """Validate the live benchmark instruction against the frozen campaign."""

    task_language = str(observed or "").strip()
    if not task_language:
        raise RuntimeError("authoritative LIBERO task language is empty")
    expected_task_language = str(expected or "").strip()
    if expected_task_language and _normalize_task_language(
        task_language
    ) != _normalize_task_language(expected_task_language):
        raise RuntimeError(
            "authoritative LIBERO task language changed after reset: "
            f"expected {expected_task_language!r}, observed {task_language!r}"
        )
    return task_language


def _record_vla_runtime_verification(
    vla_rpc: HttpRpcClient, runtime_devices: dict[str, Any]
) -> None:
    try:
        vla_info = vla_rpc.call("runtime_info", timeout_s=5.0)
    except RpcError as exc:
        if "unknown RPC method" in str(exc):
            runtime_devices["vla_gpu_verification"] = "legacy_server_declared_only"
        else:
            # Runtime metadata is advisory when a legacy/shared server is busy.
            # The preregistered physical IDs still fail closed on same-GPU
            # assignments, while the first real prediction remains authoritative
            # for endpoint availability.
            runtime_devices["vla_gpu_verification"] = "server_probe_unavailable"
            runtime_devices["vla_runtime_probe_error"] = {
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
        return
    if not isinstance(vla_info, dict):
        runtime_devices["vla_gpu_verification"] = "invalid_server_report"
        runtime_devices["violations"].append(
            "VLA runtime_info response must be an object"
        )
        runtime_devices["isolation_valid"] = False
        return
    attach_vla_runtime_verification(runtime_devices, vla_info)


def _frozen_subprocess_environment(source: dict[str, str]) -> dict[str, str]:
    """Bind child servers to the same immutable repository as the runner."""

    result = dict(source)
    existing = result.get("PYTHONPATH", "")
    entries = [str(REPOSITORY_ROOT)]
    entries.extend(
        item
        for item in existing.split(os.pathsep)
        if item and Path(item).resolve() != REPOSITORY_ROOT
    )
    result["PYTHONPATH"] = os.pathsep.join(entries)
    return result


def _role1_method_failure(exc: BaseException) -> bool:
    return isinstance(exc, LiberoRecoveryActorError) or (
        isinstance(exc, Role1ModelError)
        and isinstance(exc.__cause__, Role1ContractError)
    )


@contextmanager
def _role1_inference_heartbeat(
    path: Path,
    *,
    interval_s: float,
    step_index: int,
    phase: str = "role1_inference",
) -> Iterator[None]:
    """Keep the queue watchdog live during a bounded Role1 operation."""

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
        name="libero-role1-inference-heartbeat",
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


def _run(args: argparse.Namespace) -> EpisodeRecord:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    horizon_contract = libero_horizon_contract(
        args.suite,
        max_actions=getattr(args, "max_actions", None),
        wait_steps=getattr(args, "wait_steps", LIBERO_WAIT_STEPS_DEFAULT),
    )
    args.max_actions = int(horizon_contract["policy_action_horizon"])
    args.wait_steps = int(horizon_contract["wait_steps"])
    horizon_path = output / "evaluation-horizon.json"
    atomic_write_json(horizon_path, horizon_contract, overwrite=False)
    trajectory = output / "trajectory"
    trajectory.mkdir(parents=True, exist_ok=True)
    actions_path = trajectory / "actions.jsonl"
    states_path = trajectory / "states.jsonl"
    chunks_path = trajectory / "chunks.jsonl"
    tools_path = output / "tool_events.jsonl"
    heartbeat = output / "heartbeat.jsonl"
    for path in (actions_path, states_path, chunks_path, tools_path, heartbeat):
        path.touch(exist_ok=True)
    runtime_devices_path = output / "runtime-device-assignment.json"
    if args.runtime_url:
        # GPU isolation is enforced server-side by the Runtime's placement
        # strategy (``rollout_runtime/launch/ray_launch.py``), not by this
        # process managing two separate subprocesses on chosen physical GPUs
        # — the ``--vla-endpoint``/``--vla-gpu``/``--allowed-environment-gpus``
        # contract below is specific to the direct-connect architecture.
        atomic_write_json(
            runtime_devices_path,
            {"mode": "rollout_runtime", "runtime_url": args.runtime_url},
            overwrite=False,
        )
    else:
        runtime_devices = describe_runtime_devices(
            default_environment_gpu=args.gpu,
            allowed_environment_gpus=args.allowed_environment_gpus,
            vla_gpu=args.vla_gpu,
            environment=os.environ,
            vla_endpoint=args.vla_endpoint,
        )
        vla_rpc = HttpRpcClient(_normalize_endpoint(args.vla_endpoint))
        _record_vla_runtime_verification(vla_rpc, runtime_devices)
        atomic_write_json(runtime_devices_path, runtime_devices, overwrite=False)
        require_isolated_runtime_devices(runtime_devices)
    started_at = _now()
    started = time.time()
    episode_id = f"libero-{uuid.uuid4().hex}"
    bundle: CandidateBundle | None = None
    if args.bundle != "none":
        bundle = CandidateBundle.from_dict(read_json(args.bundle))
        if bundle.sha256 != args.bundle_sha256:
            raise ValueError("candidate bundle SHA does not match --bundle-sha256")
    if args.baseline_mode == "strict_pure_vla" and (
        bundle is not None or args.generation != 0 or args.bundle_sha256 != "none"
    ):
        raise ValueError("strict LIBERO Gen0 requires an empty active bundle")
    if args.baseline_mode == "active_bundle" and bundle is None:
        raise ValueError("active-bundle LIBERO rollout requires a frozen bundle")
    if bundle is not None and args.role1_planner == "none":
        raise ValueError("candidate bundle requires a Role1 planner")

    max_episode_steps = int(horizon_contract["max_episode_steps"])
    critic_rules = [rule.as_dict() for rule in bundle.critic_rules] if bundle else []
    runtime_session_id = None
    runtime_client = None
    runtime_loop = None
    daemon = None
    if args.runtime_url:
        port = None
    else:
        gpu = int(runtime_devices["environment_gpu"])
        port = pick_free_port()
        env_vars = _frozen_subprocess_environment(os.environ.copy())
        env_vars.update(
            {
                "LIBERO_TYPE": "pro",
                "MUJOCO_GL": "egl",
                "PYOPENGL_PLATFORM": "egl",
                "ROBOT_PLATFORM": "LIBERO",
            }
        )
        env_vars.pop("CUDA_VISIBLE_DEVICES", None)
        daemon = ProcessDaemon(
            name=f"libero-{args.logical_id}",
            cmd=[
                sys.executable,
                str(Path(__file__).with_name("env_server.py")),
                "--suite",
                args.suite,
                "--task",
                str(args.task_id),
                "--seed",
                str(args.seed),
                "--max-episode-steps",
                str(max_episode_steps),
                "--transport",
                "http",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--parent-watch",
                "--cuda-device",
                str(gpu),
            ],
            env=env_vars,
            log_path=str(output / "env_server.log"),
            cwd=str(Path(__file__).resolve().parents[2]),
        )
    primitives: LiberoPrimitives | None = None
    env: LiberoEnvClient | Any = None
    method_failure = False
    role1_decisions = 0
    chunks = 0
    last_audit_step = 0

    def flush_audit_trace() -> None:
        nonlocal last_audit_step
        if env is None or not hasattr(env, "audit_trace"):
            # The Runtime path (``--runtime-url``) has no ``libero.audit_trace``
            # extension: every step already flows through ``action_step``, so
            # ``actions.jsonl``/``states.jsonl`` are appended directly at each
            # ``chunk_step``/``step`` call site below instead of being
            # reconstructed from a separate audit ledger read. See
            # ``rollout_runtime/adapters/zetta/runtime_env_client.py``'s module
            # docstring for why this extension does not exist server-side.
            return
        for row in env.audit_trace(since_step=last_audit_step):
            _append_jsonl(
                actions_path,
                {
                    "step_index": row["step_index"],
                    "action": row["action"],
                    "action_sha256": row["action_sha256"],
                },
            )
            _append_jsonl(states_path, row)
            last_audit_step = max(last_audit_step, int(row["step_index"]))

    try:
        if args.runtime_url:
            from rollout_runtime.adapters.zetta.runtime_env_client import (
                LiberoRuntimeEnvClient,
                RuntimeOperationError,
                SyncRuntimeLoop,
            )
            from rollout_runtime.adapters.zetta.runtime_policy_client import (
                LiberoRuntimeVLAClient,
            )
            from rollout_runtime.api.messages import CreateSessionRequest, EnvSpecMsg
            from rollout_runtime.api.result import Err
            from rollout_runtime.serve.client import RemoteRuntimeClient

            runtime_loop = SyncRuntimeLoop()
            runtime_client = RemoteRuntimeClient(
                args.runtime_url,
                token=args.runtime_token,
                operation_timeout_s=args.operation_timeout_s,
                session_timeout_s=args.session_timeout_s,
            )
            create_results = runtime_loop.run(
                runtime_client.create_sessions(
                    [
                        CreateSessionRequest(
                            application_id="zetta-libero",
                            client_session_key=(
                                f"{args.logical_id}-attempt-{args.attempt_index}"
                            ),
                            env_spec=EnvSpecMsg(
                                env_family="libero",
                                env_config={
                                    "task_suite_name": args.suite,
                                    "task_id": args.task_id,
                                    "max_episode_steps": max_episode_steps,
                                    "libero_variant": "pro",
                                },
                                pool_size=1,
                                resource_hints={"accelerator": True},
                            ),
                            default_policy_id=args.policy_id,
                            lease_seconds=float(args.session_lease_s),
                            metadata={
                                "task": args.suite,
                                "seed": int(args.seed),
                                "generation": int(args.generation),
                                "logical_id": args.logical_id,
                            },
                        )
                    ]
                )
            )
            create_result = create_results[0]
            if isinstance(create_result, Err):
                info = create_result.error
                raise RuntimeOperationError(
                    f"runtime create_sessions failed: {info.code.name}: {info.message}"
                )
            runtime_session_id = create_result.value.session_id
            env = LiberoRuntimeEnvClient(
                runtime_client,
                runtime_session_id,
                loop=runtime_loop,
                return_all_frames=True,
            )
            model = LiberoRuntimeVLAClient(
                runtime_client,
                runtime_session_id,
                loop=runtime_loop,
                policy_id=args.policy_id,
            )
        else:
            daemon.start()
            env_rpc = HttpRpcClient(f"http://127.0.0.1:{port}")
            wait_for_ready(env_rpc, daemon=daemon, timeout_s=300.0)
            env = LiberoEnvClient(
                env_rpc,
                expected_meta={
                    "suite": args.suite,
                    "task": args.task_id,
                    "seed": args.seed,
                    "max_episode_steps": max_episode_steps,
                },
                return_all_frames=True,
            )
            model = VLAClient(vla_rpc)
        primitives = LiberoPrimitives(
            env,
            model,
            UnavailableSam3Client("evolution rollout does not use segmentation"),
            allow_privileged_actions=privileged_evidence_enabled(args),
        )
        if args.runtime_url:
            obs, _ = env.reset(
                task_id=args.task_id, seed=args.seed, critic_rules=critic_rules
            )
        else:
            obs, _ = env.reset()
        primitives.set_obs(obs)
        primitives.configure_policy_rng(args.policy_rng)
        primitives.configure_critic(critic_rules)
        task_language = _require_expected_task_language(
            env.get_task_language(), getattr(args, "expected_task_language", None)
        )
        initial_identity = {
            "state_sha256": canonical_sha256(np.asarray(obs["states"]).tolist()),
            "camera_sha256": {
                key: hashlib.sha256(np.asarray(obs[key]).tobytes()).hexdigest()
                for key in ("main_images", "wrist_images")
                if obs.get(key) is not None
            },
        }
        primitives.start_recording()
        primitives.record_frame(obs)
        initial_privileged_state = env.privileged_critic_state(reset_tracker=True)
        _append_jsonl(
            states_path,
            {
                "step_index": 0,
                "task_language": task_language,
                "state": extract_libero_critic_features(
                    obs,
                    step_index=0,
                    reward=0.0,
                    terminated=False,
                    truncated=False,
                    privileged_state=initial_privileged_state,
                ),
                "event": "reset",
            },
        )

        recovery: RecoveryController | None = None
        actor: LiberoRole1RecoveryActor | None = None
        if bundle is not None:
            recovery = RecoveryController(
                bundle_sha256=bundle.sha256,
                audit_path=output / "role1" / "recovery-events.jsonl",
            )
            store = Role1DecisionStore(
                output / "role1" / "decisions",
                catalog=DEFAULT_LIBERO_ROLE1_TOOL_CATALOG,
            )
            adapter = Role1ModelAdapter(
                store=store,
                output_root=output / "role1" / "invocations",
                planner_type=args.role1_planner,
                model=args.role1_model,
                reasoning_effort=args.reasoning_effort,
                base_url=os.environ.get("ZETTA_ROLE1_BASE_URL"),
                max_tokens=args.role1_max_tokens,
                timeout_s=args.role1_timeout_s,
                max_turns=args.role1_max_turns,
                # LIBERO's audited privileged plane already supplies bounded
                # scalar critic evidence. Keep visual review configurable so a
                # planner transport/model that omits the optional image tool
                # call does not discard an otherwise executable recovery.
                require_visual_review=args.role1_require_visual_review,
            )
            actor = LiberoRole1RecoveryActor(
                adapter=adapter,
                audit_root=output / "role1" / "actor",
                allowed_tools=tuple(
                    name
                    for name in LIBERO_RECOVERY_TOOLS
                    if privileged_evidence_enabled(args)
                    or name != "semantic_joint_interact"
                ),
                allow_privileged_evidence=privileged_evidence_enabled(args),
            )

        def handle_recovery(proposals: Sequence[dict[str, Any]]) -> None:
            """Run one frozen Recovery to completion under Role1 control.

            Every recovery primitive evaluates the same frozen Critic on each
            physical step.  A new proposal yields back to Role1 without
            advancing the frozen recovery step; this prevents a partial tool
            call from being misreported as a completed Recovery action.
            """

            nonlocal method_failure, role1_decisions
            if not proposals:
                return
            if recovery is None or actor is None or bundle is None:
                raise RuntimeError("Critic proposal has no active Recovery runtime")
            activated = recovery.activate(
                critic_proposals=proposals,
                recovery_rules=[rule.as_dict() for rule in bundle.recovery_rules],
                environment_step=env.episode_steps,
            )
            if not activated and not recovery.active:
                _append_jsonl(
                    tools_path,
                    {
                        "type": "unmatched_critic_proposal",
                        "critic_proposals": list(proposals),
                        "environment_write": False,
                    },
                )
                method_failure = True
                return
            current_proposals = list(proposals)
            actor_calls = 0
            role1_contract_retries = 0
            while recovery.active and not env.episode_terminated and not env.episode_truncated:
                actor_calls += 1
                if actor_calls > args.max_recovery_actor_calls:
                    _append_jsonl(
                        tools_path,
                        {
                            "type": "recovery_actor_call_limit",
                            "limit": args.max_recovery_actor_calls,
                            "environment_write": False,
                        },
                    )
                    method_failure = True
                    break
                context = recovery.context()
                if context is None:
                    break
                execution = context.get("execution", {})
                trigger_rule_ids = {
                    str(value)
                    for value in execution.get("trigger_rule_ids", ())
                    if str(value)
                }
                try:
                    with primitives.suppress_recovery_rules(trigger_rule_ids):
                        with _role1_inference_heartbeat(
                            heartbeat,
                            interval_s=args.role1_heartbeat_s,
                            step_index=env.episode_steps,
                            phase="role1_actor",
                        ):
                            reviewed = actor.decide_and_execute(
                                task=args.task,
                                step_index=env.episode_steps,
                                observation=primitives._last_obs,
                                critic_values=current_proposals,
                                recovery_context=context,
                                primitives=primitives,
                            )
                except (LiberoRecoveryActorError, Role1ModelError) as exc:
                    if not _role1_method_failure(exc):
                        raise
                    _append_jsonl(
                        tools_path,
                        {
                            "type": "role1_contract_failure",
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                            "environment_write": False,
                        },
                    )
                    # A contract fault happens before the Actor acknowledges
                    # the proposal and therefore cannot have mutated the
                    # environment. Retry the same frozen event a small,
                    # explicit number of times before declaring the episode
                    # method-invalid.
                    role1_contract_retries += 1
                    if role1_contract_retries < 3:
                        continue
                    method_failure = True
                    break
                role1_contract_retries = 0
                role1_decisions += 1
                new_proposals = list(primitives._last_critic_proposals)
                _append_jsonl(
                    tools_path,
                    {
                        "type": "role1_recovery_step",
                        "decision_id": reviewed.decision_id,
                        "selected_tool": reviewed.selected_tool,
                        "executed_horizon": reviewed.executed_horizon,
                        "interrupted_by_critic": bool(new_proposals),
                        "critic_proposals": new_proposals,
                    },
                )
                if new_proposals:
                    current_proposals = new_proposals
                    continue
                recovery.complete_current_step(
                    selected_tool=reviewed.selected_tool,
                    environment_step=env.episode_steps,
                    executed_horizon=reviewed.executed_horizon,
                    no_op_verified=bool(reviewed.result.get("no_op_verified")),
                )

        for _ in range(args.wait_steps):
            if env.episode_terminated or env.episode_truncated:
                break
            primitives._step_env(DUMMY_ACTION)
            proposals = list(primitives._last_critic_proposals)
            if args.baseline_mode == "strict_pure_vla" and (
                proposals
                or primitives._critic_configured
                or "critic_rule_count" in primitives._last_chunk_info
            ):
                raise RuntimeError("strict LIBERO Gen0 Critic attestation failed")
            handle_recovery(proposals)
            if method_failure:
                break
        while (
            not env.episode_terminated
            and not env.episode_truncated
            and env.episode_steps < max_episode_steps
            and not method_failure
        ):
            _append_jsonl(
                heartbeat,
                {"phase": "vla", "step_index": env.episode_steps, "time": _now()},
            )
            primitives._vlm_chunk(
                task_language,
                mode="eval",
                actions_per_chunk=args.actions_per_chunk,
            )
            chunks += 1
            proposals = list(primitives._last_critic_proposals)
            _append_jsonl(
                chunks_path,
                {
                    "chunk_index": chunks - 1,
                    "vla": dict(primitives._last_vla_diagnostics),
                    "executed_horizon": int(
                        primitives._last_chunk_info.get("executed_horizon", 0)
                    ),
                    "critic_rule_count": int(
                        primitives._last_chunk_info.get("critic_rule_count", -1)
                    ),
                    "critic_proposals": proposals,
                },
            )
            if args.baseline_mode == "strict_pure_vla" and (
                proposals
                or primitives._critic_configured
                or "critic_rule_count" in primitives._last_chunk_info
            ):
                raise RuntimeError("strict LIBERO Gen0 Critic attestation failed")
            handle_recovery(proposals)

        video = primitives.stop_recording_and_save(
            str(output / "videos" / "episode_agentview.mp4"), fps=10
        )
        flush_audit_trace()
        success = bool(env.episode_terminated)
        video_paths = {
            "agentview": str(video["path"]),
            "wrist": str(video["wrist_path"]),
        }
        if video.get("multiview_path"):
            video_paths["multiview"] = str(video["multiview_path"])
        preliminary = EpisodeRecord(
            episode_id=episode_id,
            logical_id=args.logical_id,
            generation=args.generation,
            seed=args.seed,
            policy_rng=args.policy_rng,
            bundle_sha256=bundle.sha256 if bundle else None,
            status="valid",
            success=success,
            started_at=started_at,
            finished_at=_now(),
            elapsed_s=time.time() - started,
            artifact_index={},
            attempt_index=args.attempt_index,
        )
        trajectory_analysis = index_episode_trajectory(
            result=preliminary,
            artifacts=TrajectoryArtifacts(
                actions=actions_path,
                states=states_path,
                tools=tools_path,
                chunks=chunks_path,
                videos=tuple(Path(value) for value in video_paths.values()),
            ),
        )
        visual = build_episode_visual_artifacts(
            video_paths={
                key: value
                for key, value in video_paths.items()
                if key != "multiview"
            },
            states_path=states_path,
            output_root=output / "visual-evidence",
            divergence_steps=tuple(
                segment.earliest_divergence_step
                for segment in trajectory_analysis.segments
                if segment.earliest_divergence_step is not None
            ),
            source_fps=10,
            include_privileged_state_summary=True,
            overview_frame_count=args.visual_overview_frames,
            event_window_radius_steps=args.visual_event_window_radius,
            event_window_stride_steps=args.visual_event_window_stride,
            maximum_event_windows=args.visual_maximum_event_windows,
        )
        video_metadata = write_video_metadata(
            video_dir=output / "videos",
            video_paths=video_paths,
            visual_evidence=visual,
            suite=args.suite,
            task=args.task,
            task_id=args.task_id,
            generation=args.generation,
            logical_id=args.logical_id,
            attempt_index=args.attempt_index,
            episode_id=episode_id,
            outcome="success" if success else "failure",
            status="valid",
            seed=args.seed,
            policy_rng=args.policy_rng,
        )
        visual["artifacts"]["video_index"] = video_metadata["index"]
        visual["artifacts"]["video_readme"] = video_metadata["readme"]
        visual["artifact_sha256"].update(video_metadata["artifact_sha256"])
        visual["video_metadata"] = {
            "index": video_metadata["index"],
            "readme": video_metadata["readme"],
        }
        artifact_index: dict[str, Any] = {
            "actions": str(actions_path),
            "states": str(states_path),
            "chunks": str(chunks_path),
            "tools": str(tools_path),
            "videos": video_paths,
            # Positive rescue credit requires an actual learned intervention,
            # not ordinary stochastic divergence in the VLA trajectory.
            "candidate_intervention": role1_decisions > 0,
            "trajectory_index": (
                trajectory_analysis.index.as_dict()
                if trajectory_analysis.index is not None
                else None
            ),
            "visual_evidence": visual,
            "initial_observation_identity": initial_identity,
            "runtime_devices": str(runtime_devices_path),
            "evaluation_horizon": str(horizon_path),
        }
        if video.get("source_manifest"):
            artifact_index["video_source_manifest"] = str(video["source_manifest"])
        for path in sorted((output / "role1").rglob("*")) if (output / "role1").is_dir() else ():
            if path.is_file():
                artifact_index[f"role1:{path.relative_to(output / 'role1').as_posix()}"] = str(path)
        return EpisodeRecord(
            episode_id=episode_id,
            logical_id=args.logical_id,
            generation=args.generation,
            seed=args.seed,
            policy_rng=args.policy_rng,
            bundle_sha256=bundle.sha256 if bundle else None,
            status="valid",
            success=success,
            started_at=started_at,
            finished_at=_now(),
            elapsed_s=time.time() - started,
            artifact_index=artifact_index,
            failure_segment=(
                trajectory_analysis.segments[0]
                if trajectory_analysis.segments
                else None
            ),
            failure_segments=trajectory_analysis.segments,
            safety_events=(),
            attempt_index=args.attempt_index,
        )
    finally:
        try:
            flush_audit_trace()
        except Exception as exc:
            with contextlib.suppress(Exception):
                _append_jsonl(
                    tools_path,
                    {
                        "type": "audit_trace_flush_failure",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "environment_write": False,
                    },
                )
        if primitives is not None and primitives._recording:
            with contextlib.suppress(Exception):
                primitives.stop_recording_and_save(
                    str(output / "videos" / "partial_agentview.mp4"), fps=10
                )
        if args.runtime_url:
            if runtime_session_id is not None and runtime_client is not None:
                with contextlib.suppress(Exception):
                    runtime_loop.run(runtime_client.close_sessions([runtime_session_id]))
            if runtime_client is not None:
                with contextlib.suppress(Exception):
                    runtime_loop.run(runtime_client.aclose())
            if runtime_loop is not None:
                with contextlib.suppress(Exception):
                    runtime_loop.close()
        else:
            with contextlib.suppress(Exception):
                daemon.stop()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--policy-rng", type=int, required=True)
    parser.add_argument("--logical-id", required=True)
    parser.add_argument("--attempt-index", type=int, required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--bundle", default="none")
    parser.add_argument("--bundle-sha256", default="none")
    parser.add_argument(
        "--baseline-mode", choices=("strict_pure_vla", "active_bundle"), required=True
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--result-file", required=True)
    parser.add_argument(
        "--expected-task-language",
        help="fail closed if the live LIBERO task language differs after reset",
    )
    parser.add_argument(
        "--runtime-url",
        default=None,
        help=(
            "Base URL of a shared rollout-runtime serve process "
            "(rollout_runtime.cli serve --launch ray). When set, this episode "
            "connects to that Runtime instead of spawning a standalone "
            "env_server.py/vla_server.py subprocess pair, and "
            "--vla-endpoint/--vla-gpu/--allowed-environment-gpus are not used."
        ),
    )
    parser.add_argument(
        "--runtime-token",
        default=os.environ.get("ZETTA_RUNTIME_TOKEN"),
        help="Bearer token for the runtime; defaults to ZETTA_RUNTIME_TOKEN.",
    )
    parser.add_argument(
        "--policy-id",
        default="pi05",
        help="Policy id served by the runtime's RolloutWorker (--runtime-url only).",
    )
    parser.add_argument(
        "--session-lease-s",
        type=float,
        default=1800.0,
        help="Requested runtime session lease (--runtime-url only).",
    )
    parser.add_argument(
        "--operation-timeout-s",
        type=float,
        default=900.0,
        help="Read timeout for runtime reset/action_step/policy_infer calls.",
    )
    parser.add_argument(
        "--session-timeout-s",
        type=float,
        default=1800.0,
        help="Read timeout for runtime create_sessions (may cold-start a pool).",
    )
    parser.add_argument(
        "--vla-endpoint",
        default=None,
        help="Pi0.5 VLA server URL (direct-connect mode; ignored with --runtime-url).",
    )
    parser.add_argument(
        "--vla-gpu",
        type=int,
        default=None,
        help="Physical GPU for the VLA server (direct-connect mode only).",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--allowed-environment-gpus",
        type=parse_physical_gpus,
        default=None,
        help=(
            "comma-separated preregistered physical GPUs for LIBERO EGL workers "
            "(direct-connect mode only)"
        ),
    )
    parser.add_argument(
        "--max-actions",
        type=int,
        default=None,
        help="policy action horizon; defaults to the official value for --suite",
    )
    parser.add_argument(
        "--wait-steps", type=int, default=LIBERO_WAIT_STEPS_DEFAULT
    )
    parser.add_argument("--actions-per-chunk", type=int, default=5)
    parser.add_argument("--visual-overview-frames", type=int, default=25)
    parser.add_argument("--visual-event-window-radius", type=int, default=8)
    parser.add_argument("--visual-event-window-stride", type=int, default=2)
    parser.add_argument("--visual-maximum-event-windows", type=int, default=6)
    parser.add_argument("--role1-planner", choices=("none", "api", "codex"), default="codex")
    parser.add_argument("--role1-model", default="gpt-5.6-sol")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--role1-max-tokens", type=int, default=4096)
    parser.add_argument("--role1-timeout-s", type=int, default=1200)
    parser.add_argument("--role1-max-turns", type=int, default=2)
    parser.add_argument("--role1-heartbeat-s", type=float, default=15.0)
    parser.add_argument(
        "--role1-require-visual-review",
        dest="role1_require_visual_review",
        action="store_true",
        default=True,
        help="require Role1 to call the current-image tool before deciding",
    )
    parser.add_argument(
        "--no-role1-require-visual-review",
        dest="role1_require_visual_review",
        action="store_false",
        help="treat Role1 image review as optional for LIBERO scalar-evidence recovery",
    )
    add_privileged_evidence_argument(
        parser,
        help_text="allow audited Critic-only LIBERO scalar evidence in Role1 recovery review",
    )
    parser.add_argument("--max-recovery-actor-calls", type=int, default=16)
    args = parser.parse_args()
    if args.runtime_url:
        if args.vla_endpoint or args.vla_gpu is not None or args.allowed_environment_gpus:
            parser.error(
                "--runtime-url is exclusive with --vla-endpoint/--vla-gpu/"
                "--allowed-environment-gpus (direct-connect-only options)"
            )
    else:
        if not args.vla_endpoint or args.vla_gpu is None or not args.allowed_environment_gpus:
            parser.error(
                "--vla-endpoint, --vla-gpu, and --allowed-environment-gpus are "
                "required unless --runtime-url is set"
            )
    if args.role1_model not in ("gpt-5.6-sol", "gpt-5.6-luna") or args.reasoning_effort != "high":
        parser.error(
            "formal Role1 is frozen to gpt-5.6-sol or gpt-5.6-luna with "
            "reasoning_effort=high"
        )
    if args.role1_heartbeat_s <= 0 or args.max_recovery_actor_calls < 1:
        parser.error("Role1 heartbeat and Recovery call limit must be positive")
    result_file = Path(args.result_file)
    run_started_at = _now()
    run_started_monotonic = time.monotonic()
    try:
        record = _run(args)
    except Exception as exc:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        # Preserve both sides of the RPC boundary.  The client traceback only
        # identifies the call site; RpcError carries the server traceback that
        # explains the actual environment failure (including blank-message
        # AssertionError/StopIteration cases).
        (output / "traceback.txt").write_text(
            _exception_traceback(exc), encoding="utf-8"
        )
        partial_artifacts: dict[str, Any] = {
            "traceback": str(output / "traceback.txt")
        }
        for name, relative in (
            ("heartbeat", "heartbeat.jsonl"),
            ("environment_log", "env_server.log"),
            ("evaluation_horizon", "evaluation-horizon.json"),
            ("actions", "trajectory/actions.jsonl"),
            ("states", "trajectory/states.jsonl"),
            ("chunks", "trajectory/chunks.jsonl"),
            ("tools", "tool_events.jsonl"),
            ("runtime_devices", "runtime-device-assignment.json"),
        ):
            path = output / relative
            if path.is_file():
                partial_artifacts[name] = str(path)
        for directory in ("role1", "videos", "visual-evidence"):
            root = output / directory
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    partial_artifacts[
                        f"partial:{path.relative_to(output).as_posix()}"
                    ] = str(path)
        finished_at = _now()
        record = EpisodeRecord(
            episode_id=f"libero-infra-{uuid.uuid4().hex}",
            logical_id=args.logical_id,
            generation=args.generation,
            seed=args.seed,
            policy_rng=args.policy_rng,
            bundle_sha256=(None if args.bundle_sha256 == "none" else args.bundle_sha256),
            status="infra_invalid",
            success=None,
            started_at=run_started_at,
            finished_at=finished_at,
            elapsed_s=max(0.0, time.monotonic() - run_started_monotonic),
            artifact_index=partial_artifacts,
            invalid_reason=f"{type(exc).__name__}: {exc}",
            attempt_index=args.attempt_index,
        )
    atomic_write_json(result_file, record.as_dict(), overwrite=False)
    return 0 if record.status == "valid" else 2


if __name__ == "__main__":
    raise SystemExit(main())
