# Copyright (c) 2026 Zetta Contributors
"""One Genie Sim VLA episode, including frozen Campaign recoveries.

This is the family's sole Runtime injection point. Actions execute one control
step at a time in both paired arms, so RGB frame indexes, critic dwell and
trajectory steps have the same meaning. Physics remains free-running.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import os
import sys
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from robots.geniesim.contracts import (
    TOOL_CATALOG,
    observation_features,
    validate_bundle,
)
from rollout_runtime.api.messages import (
    CreateSessionRequest,
    EnvSpecMsg,
    PolicyRequest,
    ResetSpec,
)
from rollout_runtime.api.result import Err
from rollout_runtime.core.payload import decode_array, decode_payload, encode_array
from rollout_runtime.serve.client import RemoteRuntimeClient
from zetta.envs.geniesim.config import CAMERA_NAMES, TASK_NAME, GenieSimConfig
from zetta.envs.geniesim.session import validate_actions
from zetta.evolution.critic import TemporalCritic
from zetta.evolution.jsonio import (
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    read_json,
)
from zetta.evolution.models import CandidateBundle, EpisodeRecord
from zetta.evolution.trajectory import TrajectoryArtifacts, index_episode_trajectory
from zetta.evolution.visual_artifacts import build_episode_visual_artifacts


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _single(results: list[Any], operation: str) -> Any:
    if len(results) != 1:
        raise RuntimeError(f"{operation}: expected exactly one result")
    result = results[0]
    if isinstance(result, Err):
        raise RuntimeError(f"{operation}: {result.error}")
    value = result.value
    if getattr(value, "error", None) is not None:
        raise RuntimeError(f"{operation}: {value.error}")
    return value


def _append(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def _optional(value: str | None) -> str | None:
    return None if value in (None, "", "none") else value


def _load_bundle(args: argparse.Namespace) -> CandidateBundle | None:
    path, digest = _optional(args.bundle), _optional(args.bundle_sha256)
    if path is None and digest is None:
        return None
    if path is None or digest is None:
        raise ValueError("--bundle and --bundle-sha256 must be supplied together")
    bundle = CandidateBundle.from_dict(read_json(path))
    if bundle.sha256 != digest:
        raise ValueError("bundle digest mismatch")
    validate_bundle(bundle)
    return bundle


class FrameRecorder:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.writers: dict[str, Any] = {}
        self.paths: dict[str, str] = {}

    def capture(self, observation: Any) -> None:
        import imageio.v2 as iio

        self.directory.mkdir(exist_ok=True)
        refs = [
            observation.main_image,
            observation.wrist_image,
            *observation.extra_view_images,
        ]
        if len(refs) != 3 or any(ref is None for ref in refs):
            raise ValueError("three synchronized camera frames are required")
        for name, ref in zip(CAMERA_NAMES, refs, strict=True):
            if name not in self.writers:
                path = self.directory / f"{name}.mp4"
                self.writers[name] = iio.get_writer(
                    path,
                    fps=10,
                    codec="libx264",
                    macro_block_size=1,
                    ffmpeg_params=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"],
                )
                self.paths[name] = str(path)
            self.writers[name].append_data(decode_payload(ref))

    def close(self) -> None:
        writers, self.writers = self.writers, {}
        errors = []
        for writer in writers.values():
            try:
                writer.close()
            except Exception as error:
                errors.append(error)
        if errors:
            raise RuntimeError(f"video finalization failed: {errors[0]}")


async def _execute(
    args: argparse.Namespace,
    client: Any,
    ids: list[Any],
    output: Path,
    bundle: CandidateBundle | None,
    recorder: FrameRecorder | None,
) -> dict[str, Any]:
    paths = {
        name: output / f"{name}.jsonl"
        for name in ("chunks", "actions", "states", "tools")
    }
    for path in paths.values():
        path.touch(exist_ok=False)
    step = _single(await client.reset(ids, ResetSpec(seed=args.seed)), "reset")
    obs = step.observation
    if obs is None or obs.step_index != 0:
        raise RuntimeError("reset did not return an initial observation")
    limits = obs.extras.get("action_limits")
    validate_actions([obs.state[:16]], limits)
    identity = {
        "comparison_contract": "geniesim_g2_joint_reset_v1",
        "joint_positions": list(obs.state),
        "action_limits": limits,
        "scenario_sha256": canonical_sha256(
            {
                "env_config": read_json(args.env_config),
                "seed": args.seed,
                "instruction": obs.instruction,
            }
        ),
        "state_sha256": canonical_sha256(obs.state),
        "instruction_sha256": canonical_sha256(obs.instruction),
        "camera_sha256": {
            name: hashlib.sha256(decode_payload(ref).tobytes()).hexdigest()
            for name, ref in zip(
                CAMERA_NAMES,
                [obs.main_image, obs.wrist_image, *obs.extra_view_images],
                strict=True,
            )
        },
    }
    atomic_write_json(output / "reset.json", identity, overwrite=False)
    critic = TemporalCritic(bundle.critic_rules if bundle else ())
    pending: deque[Any] = deque()
    recovery_steps: deque[Any] = deque()
    active_step = None
    recovery_id = None
    remaining = 0
    interventions = 0
    inference_calls = 0
    renewed_at = time.monotonic()

    def record_state() -> None:
        _append(
            paths["states"],
            {
                "step_index": obs.step_index,
                "state": {
                    **observation_features(
                        obs.state, step_index=obs.step_index, extras=obs.extras
                    ),
                    "joint_positions": obs.state,
                },
                "terminated": step.terminated,
                "truncated": step.truncated,
            },
        )
        if recorder:
            recorder.capture(obs)

    record_state()
    print(json.dumps({"event": "reset", "step_index": 0}), flush=True)
    while not (step.terminated or step.truncated):
        if time.monotonic() - renewed_at >= args.session_lease_s / 4:
            _single(
                await client.renew_sessions(ids, lease_seconds=args.session_lease_s),
                "renew",
            )
            renewed_at = time.monotonic()
        if active_step is None and not recovery_steps:
            proposals = critic.evaluate(
                observation_features(
                    obs.state, step_index=obs.step_index, extras=obs.extras
                ),
                step_index=obs.step_index,
            )
            if proposals:
                triggers = {proposal["rule_id"] for proposal in proposals}
                matches = sorted(
                    (
                        rule
                        for rule in bundle.recovery_rules
                        if triggers.intersection(rule.trigger_rule_ids)
                    ),
                    key=lambda rule: rule.recovery_id,
                )
                if not matches:
                    raise ValueError("critic rejection has no executable recovery")
                recovery_id = matches[0].recovery_id
                recovery_steps.extend(matches[0].steps)
                pending.clear()
                _append(
                    paths["tools"],
                    {
                        "event": "recovery_started",
                        "step_index": obs.step_index,
                        "recovery_id": recovery_id,
                        "critic_proposals": proposals,
                    },
                )
        if active_step is None and recovery_steps:
            active_step = recovery_steps.popleft()
            remaining = active_step.parameters["max_steps"]
            pending.clear()
        source = active_step.tool if active_step else "vla"
        if active_step and active_step.tool == "geniesim.hold":
            action = np.asarray(obs.state[:16], dtype=np.float32)
        else:
            if not pending:
                parameters = active_step.parameters if active_step else {}
                horizon = min(
                    parameters.get("actions_per_chunk", args.execute_horizon),
                    remaining if active_step else 64,
                )
                policy_seed = int.from_bytes(
                    hashlib.sha256(
                        f"{args.policy_rng}:{obs.step_index}".encode()
                    ).digest()[:4],
                    "big",
                )
                infer = _single(
                    await client.policy_infer(
                        ids,
                        PolicyRequest(
                            policy_id=args.policy_id,
                            instruction_override=parameters.get("instruction"),
                            inference_parameters={"mode": "eval", "seed": policy_seed},
                            actions_per_chunk=horizon,
                        ),
                    ),
                    "policy_infer",
                )
                if (
                    infer.model_version != args.policy_model_version
                    or not infer.auxiliary_outputs.get("policy_rng_acknowledged")
                ):
                    raise ValueError(
                        "rollout requires frozen model identity and acknowledged policy RNG"
                    )
                if (
                    infer.observation_step_index != obs.step_index
                    or infer.actions is None
                ):
                    raise ValueError(
                        "policy inference is not aligned with the current observation"
                    )
                block = validate_actions(decode_array(infer.actions), limits)
                pending.extend(block[:horizon])
                inference_calls += 1
            action = pending.popleft()
        before = obs.step_index
        validate_actions([action], limits)
        step = _single(
            await client.action_step(
                ids, [encode_array(np.asarray([action], dtype=np.float32))]
            ),
            "action_step",
        )
        obs = step.observation
        if obs is None or step.executed_horizon != 1 or obs.step_index != before + 1:
            raise ValueError(
                "action execution did not advance exactly one control step"
            )
        _append(
            paths["actions"],
            {"step_index": before, "source": source, "action": action.tolist()},
        )
        _append(
            paths["chunks"],
            {
                "step_index": before,
                "executed_horizon": 1,
                "reward": step.reward,
                "terminated": step.terminated,
                "truncated": step.truncated,
                "success": step.info.get("success", False),
            },
        )
        if active_step:
            interventions += 1
            remaining -= 1
            _append(
                paths["tools"],
                {
                    "event": "recovery_action",
                    "step_index": before,
                    "tool": active_step.tool,
                    "recovery_id": recovery_id,
                    "remaining_steps": remaining,
                },
            )
            if remaining == 0 or obs.extras.get("official_success"):
                active_step = None
                pending.clear()
                if obs.extras.get("official_success"):
                    recovery_steps.clear()
        record_state()
        print(
            json.dumps({"event": "control_step", "step_index": obs.step_index}),
            flush=True,
        )
    result = _single(
        await client.extension_call(ids, "geniesim", "result", {}), "geniesim.result"
    )
    if (
        type(result.get("success")) is not bool
        or result.get("control_steps") != obs.step_index
    ):
        raise ValueError("official result does not match the executed episode")
    return {
        "success": result["success"],
        "artifact_index": {
            **{
                name if name != "tools" else "tool_events": str(path)
                for name, path in paths.items()
            },
            "initial_observation_identity": identity,
            "tool_catalog_digest": canonical_sha256(TOOL_CATALOG),
            "official_result": result,
            "steps_executed": obs.step_index,
            "inference_calls": inference_calls,
            "policy_model_version": args.policy_model_version,
            "candidate_intervention": interventions > 0,
            "recovery_steps_executed": interventions,
            "evidence_granularity": "control_step",
            "physics_between_requests": "free_running",
            "terminated": step.terminated,
            "truncated": step.truncated,
        },
    }


async def run(args: argparse.Namespace, *, client: Any = None) -> EpisodeRecord:
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(
        (output / name).exists()
        for name in (
            "episode.json",
            "chunks.jsonl",
            "actions.jsonl",
            "states.jsonl",
            "tools.jsonl",
            "video",
            "visual-evidence",
        )
    ):
        raise ValueError("rollout output directory already contains episode artifacts")
    started_at, started = _now(), time.monotonic()
    ids = []
    owned = client is None
    recorder = FrameRecorder(output / "video") if args.capture_frames else None
    failure = None
    execution: dict[str, Any] = {"artifact_index": {}}
    try:
        bundle = _load_bundle(args)
        if (
            args.env_config_sha256
            and file_sha256(args.env_config) != args.env_config_sha256
        ):
            raise ValueError("environment configuration digest mismatch")
        config = GenieSimConfig.from_mapping(read_json(args.env_config))
        if (
            args.task != config.task
            or type(args.seed) is not int
            or not 0 <= args.seed < 2**32
        ):
            raise ValueError("invalid Genie Sim task or seed")
        if not 1 <= args.execute_horizon <= 64 or not 0 <= args.policy_rng < 2**32:
            raise ValueError("invalid execute horizon or policy RNG")
        if client is None:
            client = RemoteRuntimeClient(
                args.runtime_url,
                token=os.environ.get("ROLLOUT_RUNTIME_TOKEN"),
                operation_timeout_s=args.operation_timeout_s,
                session_timeout_s=args.operation_timeout_s,
            )
        spec = EnvSpecMsg(env_family="geniesim", env_config=config.to_dict())
        handle = _single(
            await client.create_sessions(
                [
                    CreateSessionRequest(
                        application_id="zetta-geniesim",
                        client_session_key=f"{args.logical_id}-{args.attempt_index}",
                        env_spec=spec,
                        default_policy_id=args.policy_id,
                        lease_seconds=args.session_lease_s,
                    )
                ]
            ),
            "create_sessions",
        )
        ids = [handle.session_id]
        execution = await _execute(args, client, ids, output, bundle, recorder)
        execution["artifact_index"]["env_spec_digest"] = spec.digest()
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
    finally:
        for cleanup in (
            (lambda: client.close_sessions(ids)) if ids else None,
            (lambda: client.aclose()) if owned and client is not None else None,
        ):
            if cleanup:
                try:
                    outcome = await cleanup()
                    if isinstance(outcome, list):
                        _single(outcome, "close_sessions")
                except Exception as error:
                    failure = f"{failure or ''} cleanup: {error}".strip()
        if recorder:
            try:
                recorder.close()
            except Exception as error:
                failure = f"{failure or ''} video: {error}".strip()
    record = EpisodeRecord(
        episode_id=f"{args.logical_id}-{uuid.uuid4().hex[:8]}",
        logical_id=args.logical_id,
        generation=args.generation,
        seed=args.seed,
        policy_rng=args.policy_rng,
        bundle_sha256=_optional(args.bundle_sha256),
        status="infra_invalid" if failure else "valid",
        success=None if failure else execution["success"],
        started_at=started_at,
        finished_at=_now(),
        elapsed_s=time.monotonic() - started,
        artifact_index=execution["artifact_index"],
        invalid_reason=failure,
        attempt_index=args.attempt_index,
    )
    if record.status == "valid":
        try:
            videos = recorder.paths if recorder else {}
            analysis = index_episode_trajectory(
                result=record,
                artifacts=TrajectoryArtifacts(
                    chunks=output / "chunks.jsonl",
                    actions=output / "actions.jsonl",
                    states=output / "states.jsonl",
                    tools=output / "tools.jsonl",
                    videos=tuple(videos.values()),
                ),
            )
            visuals = (
                build_episode_visual_artifacts(
                    video_paths=videos,
                    states_path=output / "states.jsonl",
                    output_root=output / "visual-evidence",
                    source_fps=10,
                )
                if videos
                else {}
            )
            record = dataclasses.replace(
                record,
                artifact_index={
                    **record.artifact_index,
                    "trajectory_index": analysis.index.as_dict(),
                    "videos": videos,
                    "visual_evidence": visuals,
                },
                failure_segment=analysis.segments[0] if analysis.segments else None,
                failure_segments=analysis.segments,
            )
        except Exception as error:
            record = dataclasses.replace(
                record,
                status="infra_invalid",
                success=None,
                invalid_reason=f"evidence: {error}",
                failure_segment=None,
                failure_segments=(),
            )
    atomic_write_json(output / "episode.json", record.as_dict(), overwrite=False)
    if args.result_file and Path(args.result_file).resolve() != output / "episode.json":
        atomic_write_json(args.result_file, record.as_dict(), overwrite=False)
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-url", required=True)
    parser.add_argument(
        "--env-config", required=True, help="GenieSimConfig JSON on the worker"
    )
    parser.add_argument("--env-config-sha256")
    parser.add_argument("--policy-id", default="geniesim_vla")
    parser.add_argument("--policy-model-version", required=True)
    parser.add_argument("--task", default=TASK_NAME, choices=[TASK_NAME])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--policy-rng", type=int, required=True)
    parser.add_argument("--logical-id", default="geniesim-rollout")
    parser.add_argument("--generation", type=int, default=0)
    parser.add_argument("--attempt-index", type=int, default=0)
    parser.add_argument("--bundle", default="none")
    parser.add_argument("--bundle-sha256", default="none")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--result-file")
    parser.add_argument("--execute-horizon", type=int, default=5)
    parser.add_argument("--session-lease-s", type=float, default=3600)
    parser.add_argument("--operation-timeout-s", type=float, default=1800)
    parser.add_argument(
        "--capture-frames", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main() -> int:
    record = asyncio.run(run(build_parser().parse_args()))
    print(json.dumps(record.as_dict(), sort_keys=True), flush=True)
    return 0 if record.status == "valid" else 2


if __name__ == "__main__":
    sys.exit(main())
