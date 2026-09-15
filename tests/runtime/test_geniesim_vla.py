# Copyright (c) 2026 Zetta Contributors
"""Real Runtime and WebSocket plumbing with the supervised native-process stub."""

from __future__ import annotations

import dataclasses
import functools
import io
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import msgpack
import numpy as np
import pytest
from PIL import Image

from robots.geniesim import run_rollout
from robots.geniesim.run_rollout import build_parser, run
from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.internal import InferenceRequest
from rollout_runtime.api.messages import (
    CreateSessionRequest,
    EnvSpecMsg,
    Observation,
    PolicyRequest,
    ResetSpec,
)
from rollout_runtime.backends import build_policy_core, geniesim
from rollout_runtime.backends.geniesim_policy import (
    GENIESIM_GRIPPER_LIMIT,
    GenieSimPolicyConfig,
    GenieSimPolicyCore,
    build_payload,
    parse_actions,
)
from rollout_runtime.config.schema import load_config
from rollout_runtime.core.payload import decode_array, encode_image
from rollout_runtime.launch.local import build_local_components
from zetta.envs.geniesim.config import GenieSimConfig
from zetta.envs.geniesim.process import GenieSimProcess
from zetta.evolution.jsonio import atomic_write_json, read_json
from zetta.evolution.models import (
    CandidateBundle,
    CriticRule,
    RecoveryRule,
    RecoveryStep,
)


def reply(*, horizon=5, seed=42):
    return {
        "result": {
            "left_arm": {"kind": "JOINT_ABS", "values": [[0.25] * 7] * horizon},
            "right_arm": {"kind": "JOINT_ABS", "values": [[0.3] * 7] * horizon},
            "left_effector": [[-0.7]] * horizon,
            "right_effector": [[-0.6]] * horizon,
            "model_version": "test-g2-v1",
            "policy_rng": seed,
        }
    }


def request(**overrides):
    obs = Observation(
        session_id="s1",
        episode_id=1,
        step_index=0,
        state=[0.2] * 14 + [0.7, 0.6] + [0.0] * 5,
        main_image=encode_image(np.full((16, 16, 3), 25, np.uint8)),
        wrist_image=encode_image(np.full((16, 16, 3), 75, np.uint8)),
        extra_view_images=[encode_image(np.full((16, 16, 3), 125, np.uint8))],
        instruction="Pick the blue block",
        extras={
            "action_units": "absolute_joint_radians",
            "seed": 1001,
            "official_result": {"private": True},
        },
    )
    return InferenceRequest(
        **{
            "request_id": "r1",
            "session_id": "s1",
            "episode_id": 1,
            "policy_id": "geniesim_vla",
            "observation": obs,
            "routing_token": "env:0",
            "compat_key": "one",
            "inference_parameters": {"seed": 42},
            **overrides,
        }
    )


def test_upstream_camera_state_and_gripper_contract():
    payload = build_payload(
        request(instruction_override="Lift the block"), episode_index=0
    )
    params = payload["params"]
    assert params["states"]["gripper_states"] == [-0.7, -0.6]
    assert params["states"]["waist_joint_states"] == [0.0] * 5
    assert params["prompt"] == "Lift the block"
    assert params["policy_rng"] == 42 and "seed" not in params
    assert "official_result" not in params and params["task_progress"] == []
    assert params["robot_type"] == "G2_omnipicker"
    assert params["task_name"] == "pick_block_color"
    for name, color in (("head", 25), ("hand_left", 75), ("hand_right", 125)):
        frame = params["images"][name]
        assert frame["encoding"] == "JPEG"
        assert np.all(np.asarray(Image.open(io.BytesIO(frame["image_data"]))) == color)
    actions, _ = parse_actions(reply())
    np.testing.assert_allclose(actions[0], [0.25] * 7 + [0.3] * 7 + [0.7, 0.6])


def test_gripper_targets_are_saturated_at_native_g2_stop():
    result = reply()
    result["result"]["left_effector"] = [[-0.9]] * 5
    result["result"]["right_effector"] = [[0.1]] * 5
    actions, _ = parse_actions(result)
    assert actions[:, 14].max() == pytest.approx(GENIESIM_GRIPPER_LIMIT)
    assert actions[:, 15].min() == 0.0


@pytest.mark.parametrize(
    "edit",
    [
        lambda r: r["left_arm"].update(kind="EEF_ABS"),
        lambda r: r["left_arm"].update(values=[[0] * 6]),
        lambda r: r.update(right_effector=[[0]]),
        lambda r: r.update(left_effector=[[float("nan")]] * 5),
        lambda r: r.update(left_effector=[[True]] * 5),
        lambda r: r.update(waist={"kind": "JOINT_ABS", "values": [[0] * 5]}),
        lambda r: r.update(need_depth=True),
        lambda r: r.update(history={"interval": 1}),
    ],
)
def test_unsupported_or_malformed_actions_fail(edit):
    result = reply()
    edit(result["result"])
    with pytest.raises(ValueError):
        parse_actions(result)


def test_policy_deadline_rng_fence_and_reconnect():
    socket = Mock()
    socket.recv.return_value = msgpack.packb(reply())
    connect = Mock(return_value=socket)
    core = GenieSimPolicyCore(
        GenieSimPolicyConfig(
            endpoint="ws://localhost:8999", model_version="test-g2-v1"
        ),
        connect=connect,
    )
    core.load()
    assert core.infer_batch([request()])[0].error is None
    response = core.infer_batch([request(episode_id=2)])[0]
    assert response.error is None and connect.call_count == 2
    assert socket.close.call_count == 1
    np.testing.assert_allclose(decode_array(response.actions)[:, 14], 0.7)
    bad = reply(seed=7)
    socket.recv.return_value = msgpack.packb(bad)
    assert (
        core.infer_batch([request(episode_id=2)])[0].error.code
        == ErrorCode.POLICY_FAILURE
    )
    assert core._socket is None
    calls = socket.send.call_count
    assert core.infer_batch([request(deadline=time.time() - 1)])[0].error is not None
    assert socket.send.call_count == calls
    with pytest.raises(ValueError):
        core.update_weights("other-model")
    core.close()


def test_optional_rng_ack_cannot_forge_campaign_model_identity():
    socket = Mock()
    result = reply()
    result["result"].pop("model_version")
    socket.recv.return_value = msgpack.packb(result)
    core = GenieSimPolicyCore(
        GenieSimPolicyConfig(
            endpoint="ws://localhost:8999",
            model_version="test-g2-v1",
            require_policy_rng_ack=False,
        ),
        connect=Mock(return_value=socket),
    )
    core.load()
    response = core.infer_batch([request()])[0]
    assert response.error is None
    assert response.auxiliary_outputs["policy_rng_acknowledged"] is False
    core.close()


@pytest.fixture
def policy_server():
    from websockets.sync.server import serve

    requests = []

    def handler(socket):
        for raw in socket:
            payload = msgpack.unpackb(raw, raw=False)
            requests.append(payload)
            socket.send(msgpack.packb(reply(seed=payload["params"]["policy_rng"])))

    with serve(handler, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        yield f"ws://127.0.0.1:{server.socket.getsockname()[1]}", requests
        server.shutdown()
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.fixture
def runtime_config(tmp_path, monkeypatch, policy_server):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ZETTA_GENIESIM_TEST_MODE", raising=False)
    (tmp_path / "source/geniesim_benchmark/src/geniesim_benchmark/app/robot_cfg").mkdir(
        parents=True
    )
    config = GenieSimConfig(
        python_executable=sys.executable,
        geniesim_root=str(tmp_path),
        assets_root=str(tmp_path),
        output_root=str(tmp_path / "native"),
        max_steps=3,
        startup_timeout_seconds=10,
        request_timeout_seconds=2,
        close_timeout_seconds=2,
    )
    monkeypatch.setattr(
        geniesim,
        "GenieSimProcess",
        functools.partial(
            GenieSimProcess,
            _worker_package=Path(__file__).parent / "geniesim_worker_stub",
        ),
    )
    preset = load_config("geniesim_vla")
    return dataclasses.replace(
        preset,
        env_config=config.to_dict(),
        rollout_worker=dataclasses.replace(
            preset.rollout_worker,
            policy_config={
                "endpoint": policy_server[0],
                "model_version": "test-g2-v1",
                "actions_per_chunk": 5,
            },
        ),
    )


@pytest.mark.asyncio
async def test_real_websocket_policy_step(runtime_config):
    async with build_local_components(runtime_config) as runtime:
        client = runtime.gateway
        created = (
            await client.create_sessions(
                [
                    CreateSessionRequest(
                        application_id="vla-test",
                        client_session_key="one",
                        env_spec=EnvSpecMsg(
                            env_family="geniesim", env_config=runtime_config.env_config
                        ),
                        default_policy_id="geniesim_vla",
                    )
                ]
            )
        )[0].value
        ids = [created.session_id]
        await client.reset(ids, ResetSpec(seed=1001))
        result = (
            await client.policy_step(
                ids,
                PolicyRequest(inference_parameters={"seed": 42}, actions_per_chunk=5),
            )
        )[0]
        assert result.is_ok, result
        assert result.value.truncated and result.value.executed_horizon == 3
        np.testing.assert_allclose(result.value.observation.state[14:16], [0.7, 0.6])
        await client.close_sessions(ids)


def candidate():
    return CandidateBundle(
        candidate_id="hold-test",
        generation=1,
        parent_sha256=None,
        diagnosis_sha256="a" * 64,
        causal_hypothesis="A brief hold changes the initial motion",
        mechanism_change="Hold at reset",
        validation_plan="paired control-step comparison",
        critic_rules=(
            CriticRule(
                rule_id="initial",
                title="Initial boundary",
                feature="episode.step",
                operator="eq",
                threshold=0,
                dwell_steps=1,
                cooldown_steps=10,
                proposal="Hold briefly",
                evidence_ids=("trace",),
            ),
        ),
        recovery_rules=(
            RecoveryRule(
                recovery_id="hold",
                title="Bounded hold",
                trigger_rule_ids=("initial",),
                precondition="At initial boundary",
                steps=(
                    RecoveryStep(
                        tool="geniesim.hold",
                        parameters={"max_steps": 2},
                        stop_when="budget_exhausted_or_success",
                    ),
                ),
                safety_constraints=("Native joint limits",),
                stop_condition="official_success_or_budget",
                fallback="resume_vla",
                evidence_ids=("trace",),
            ),
        ),
    )


def rollout_args(tmp_path, runtime_config, **overrides):
    config_path = tmp_path / "env.json"
    atomic_write_json(config_path, runtime_config.env_config)
    args = build_parser().parse_args(
        [
            "--runtime-url",
            "http://127.0.0.1:1",
            "--env-config",
            str(config_path),
            "--policy-model-version",
            "test-g2-v1",
            "--seed",
            "1001",
            "--policy-rng",
            "42",
            "--output-dir",
            str(tmp_path / "episode"),
        ]
    )
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


@pytest.mark.asyncio
@pytest.mark.parametrize("with_candidate", [False, True])
async def test_campaign_episode_bundle_video_and_official_result(
    tmp_path, runtime_config, with_candidate
):
    args = rollout_args(tmp_path, runtime_config)
    Path(args.output_dir).mkdir()
    (Path(args.output_dir) / "worker.stdout.log").touch()
    if with_candidate:
        bundle = candidate()
        args.bundle = str(tmp_path / "candidate.json")
        args.bundle_sha256 = bundle.sha256
        atomic_write_json(args.bundle, bundle.as_dict())
    async with build_local_components(runtime_config) as runtime:
        record = await run(args, client=runtime.gateway)
    assert record.status == "valid", record.invalid_reason
    assert record.success is False and record.failure_segments
    assert record.artifact_index["steps_executed"] == 3
    assert record.artifact_index["candidate_intervention"] is with_candidate
    assert record.artifact_index["recovery_steps_executed"] == (
        2 if with_candidate else 0
    )
    assert len(record.artifact_index["videos"]) == 3
    assert record.artifact_index["visual_evidence"]
    assert len(Path(record.artifact_index["states"]).read_text().splitlines()) == 4
    assert read_json(Path(args.output_dir) / "episode.json")["status"] == "valid"


@pytest.mark.asyncio
async def test_invalid_bundle_is_not_scored(tmp_path, runtime_config):
    args = rollout_args(tmp_path, runtime_config, bundle="none", bundle_sha256="a" * 64)
    client = Mock()
    record = await run(args, client=client)
    assert record.status == "infra_invalid" and record.success is None
    client.create_sessions.assert_not_called()


@pytest.mark.asyncio
async def test_recovery_vla_budget_and_prompt_return_to_baseline(
    tmp_path, runtime_config, policy_server
):
    bundle = candidate()
    recovery = dataclasses.replace(
        bundle.recovery_rules[0],
        steps=(
            RecoveryStep(
                tool="geniesim.vla",
                parameters={
                    "max_steps": 2,
                    "actions_per_chunk": 1,
                    "instruction": "Lift the blue block",
                },
                stop_when="budget_exhausted_or_success",
            ),
        ),
    )
    bundle = dataclasses.replace(bundle, recovery_rules=(recovery,))
    path = tmp_path / "candidate.json"
    atomic_write_json(path, bundle.as_dict())
    args = rollout_args(
        tmp_path,
        runtime_config,
        bundle=str(path),
        bundle_sha256=bundle.sha256,
        capture_frames=False,
    )
    async with build_local_components(runtime_config) as runtime:
        record = await run(args, client=runtime.gateway)
    assert record.status == "valid", record.invalid_reason
    assert record.artifact_index["recovery_steps_executed"] == 2
    prompts = [row["params"]["prompt"] for row in policy_server[1]]
    assert prompts[:2] == ["Lift the blue block"] * 2
    assert len(prompts) == 3 and prompts[-1] != prompts[0]


@pytest.mark.asyncio
async def test_close_failure_invalidates_episode_and_closes_http_client(
    tmp_path, runtime_config, monkeypatch
):
    args = rollout_args(tmp_path, runtime_config, capture_frames=False)
    async with build_local_components(runtime_config) as runtime:
        gateway = runtime.gateway
        proxy = SimpleNamespace(
            **{
                name: getattr(gateway, name)
                for name in (
                    "create_sessions",
                    "reset",
                    "policy_infer",
                    "action_step",
                    "extension_call",
                    "renew_sessions",
                )
            },
            close_sessions=AsyncMock(
                side_effect=RuntimeError("missing close acknowledgement")
            ),
            aclose=AsyncMock(),
        )
        monkeypatch.setattr(run_rollout, "RemoteRuntimeClient", lambda *a, **kw: proxy)
        record = await run(args)
    assert record.status == "infra_invalid" and record.success is None
    assert record.failure_segments == ()
    assert "missing close acknowledgement" in record.invalid_reason
    proxy.aclose.assert_awaited_once()


def test_registration_and_single_slot_constraints():
    config = load_config("geniesim_vla")
    core = build_policy_core(
        backend="geniesim_vla",
        policy_config={"endpoint": "ws://localhost:8999", "model_version": "test"},
    )
    assert core.config.action_dim == 16
    with pytest.raises(ValueError, match="one env slot"):
        load_config(
            {**dataclasses.asdict(config), "env_worker": {"default_pool_size": 2}}
        )
