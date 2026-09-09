# Copyright (c) 2026 Zetta Contributors
"""Native process supervision and Runtime contracts without simulator dependencies."""

from __future__ import annotations

import dataclasses
import functools
import json
import os
import socket
import struct
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.errors import RuntimeApiError
from rollout_runtime.api.messages import CreateSessionRequest, EnvSpecMsg, ResetSpec
from rollout_runtime.backends import geniesim
from rollout_runtime.core.env_registry import behavior_for, requested_core_form
from rollout_runtime.core.payload import decode_payload, encode_array
from rollout_runtime.launch.local import build_local_components
from zetta.envs.geniesim import session as native_session
from zetta.envs.geniesim.config import GenieSimConfig, flatten_state
from zetta.envs.geniesim.process import GenieSimProcess, GenieSimProcessError
from zetta.envs.geniesim.session import (
    action_limits_from_articulation,
    validate_actions,
)
from zetta.envs.geniesim.wire import MAX_PACKET_BYTES, receive_packet, send_packet

STUB = Path(__file__).parent / "geniesim_worker_stub"


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ZETTA_GENIESIM_TEST_MODE", raising=False)
    (tmp_path / "source/geniesim_benchmark/src/geniesim_benchmark/app/robot_cfg").mkdir(
        parents=True
    )
    return GenieSimConfig(
        python_executable=sys.executable,
        geniesim_root=str(tmp_path),
        assets_root=str(tmp_path),
        output_root=str(tmp_path / "outputs"),
        max_steps=3,
        startup_timeout_seconds=10,
        request_timeout_seconds=2,
        close_timeout_seconds=2,
    )


@pytest.fixture
def native_stub(config, monkeypatch):
    monkeypatch.setattr(
        geniesim,
        "GenieSimProcess",
        functools.partial(GenieSimProcess, _worker_package=STUB),
    )
    geniesim.register_geniesim_env_family()
    return config


def test_declared_capability_and_unsupported_forms(native_stub):
    family = geniesim.register_geniesim_env_family()
    assert family.capability.needs_accelerator
    assert family.capability.per_step_obs_available
    assert not family.capability.supports_auto_reset
    assert behavior_for("geniesim").max_pool_size == 1
    with pytest.raises(RuntimeApiError):
        requested_core_form(
            EnvSpecMsg(
                env_family="geniesim", env_config={"core_form": "lockstep_vector"}
            ),
            behavior_for("geniesim"),
        )
    with pytest.raises(RuntimeApiError):
        family.create_core().build(
            EnvSpecMsg(env_family="geniesim", env_config=native_stub.to_dict()),
            num_envs=2,
        )


def test_native_process_lifecycle_and_absolute_paths(config):
    process = GenieSimProcess(config, _worker_package=STUB)
    assert process.pid != os.getpid()
    result = process.request("reset", {"seed": 1001})
    assert result["observation"]["seed"] == 1001
    process.close()
    process.close()
    evidence = json.loads((process.directory / "process.json").read_text())
    assert evidence == {
        "status": "closed",
        "pid": process.pid,
        "returncode": 0,
        "forced": False,
    }
    with pytest.raises(GenieSimProcessError, match="closed"):
        process.request("step", {"actions": [[0.2] * 16]})


@pytest.mark.parametrize("visible", ["", "2", "GPU-uuid"])
def test_process_cannot_escape_inherited_gpu_assignment(config, monkeypatch, visible):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    with pytest.raises(GenieSimProcessError, match="CUDA_VISIBLE_DEVICES"):
        GenieSimProcess(config, _worker_package=STUB)


def test_invalid_native_image_is_an_environment_failure(native_stub, monkeypatch):
    monkeypatch.setenv("ZETTA_GENIESIM_TEST_MODE", "invalid_image")
    core = geniesim.GenieSimEnvCore()
    core.build(
        EnvSpecMsg(env_family="geniesim", env_config=native_stub.to_dict()), num_envs=1
    )
    process = core.process
    with pytest.raises(RuntimeApiError) as error:
        core.reset([0], ResetSpec(seed=1001))
    assert error.value.info.code == ErrorCode.ENV_FAILURE
    assert not error.value.info.retryable
    assert (
        json.loads((process.directory / "process.json").read_text())["returncode"]
        is not None
    )
    core.close()


def test_inconsistent_final_step_aborts_without_replay(native_stub, monkeypatch):
    monkeypatch.setenv("ZETTA_GENIESIM_TEST_MODE", "invalid_final_step")
    core = geniesim.GenieSimEnvCore()
    try:
        core.build(
            EnvSpecMsg(env_family="geniesim", env_config=native_stub.to_dict()),
            num_envs=1,
        )
        core.reset([0], ResetSpec(seed=1001))
        with pytest.raises(RuntimeApiError) as error:
            core.chunk_step([0], [np.full((2, 16), 0.2)])
        assert error.value.info.code == ErrorCode.ENV_FAILURE
        assert not error.value.info.retryable
        evidence = json.loads((core.process.directory / "process.json").read_text())
        assert evidence["status"] == "failed" and evidence["returncode"] is not None
        assert (
            len((core.process.directory / "received.jsonl").read_text().splitlines())
            == 1
        )
    finally:
        core.close()


def test_native_termination_stops_chunk_without_truncation(native_stub, monkeypatch):
    monkeypatch.setenv("ZETTA_GENIESIM_TEST_MODE", "terminated")
    core = geniesim.GenieSimEnvCore()
    try:
        core.build(
            EnvSpecMsg(env_family="geniesim", env_config=native_stub.to_dict()),
            num_envs=1,
        )
        core.reset([0], ResetSpec(seed=1001))
        outcome = core.chunk_step([0], [np.full((5, 16), 0.2)])[0]
        assert outcome.executed_horizon == 2
        assert outcome.terminated and not outcome.truncated
        assert not outcome.info["success"]
    finally:
        core.close()


@pytest.mark.parametrize(
    "mode", ["request_exit", "request_timeout", "backend_error", "sequence"]
)
def test_request_failure_reaps_without_replay(config, monkeypatch, mode):
    monkeypatch.setenv("ZETTA_GENIESIM_TEST_MODE", mode)
    process = GenieSimProcess(config, _worker_package=STUB)
    with pytest.raises(GenieSimProcessError):
        process.request("step", {"actions": [[0.2] * 16]}, timeout=0.2)
    evidence = json.loads((process.directory / "process.json").read_text())
    assert evidence["status"] == "failed"
    assert evidence["returncode"] is not None
    assert len((process.directory / "received.jsonl").read_text().splitlines()) == 1
    process.close()


@pytest.mark.parametrize(
    "mode", ["startup_exit", "startup_timeout", "close_timeout", "close_exit"]
)
def test_startup_and_close_failures(config, monkeypatch, mode):
    monkeypatch.setenv("ZETTA_GENIESIM_TEST_MODE", mode)
    if mode.startswith("startup"):
        with pytest.raises(GenieSimProcessError):
            GenieSimProcess(
                dataclasses.replace(config, startup_timeout_seconds=0.3),
                _worker_package=STUB,
            )
    else:
        process = GenieSimProcess(
            dataclasses.replace(config, close_timeout_seconds=0.2), _worker_package=STUB
        )
        with pytest.raises(GenieSimProcessError):
            process.close()
    records = list(Path(config.output_root).glob("*/process.json"))
    assert len(records) == 1
    assert json.loads(records[0].read_text())["returncode"] is not None


def test_wire_rejects_partial_eof_and_unbounded_frames():
    for packet, error in (
        (struct.pack("!I", MAX_PACKET_BYTES + 1), ValueError),
        (struct.pack("!I", 20) + b"partial", EOFError),
    ):
        left, right = socket.socketpair()
        with left, right:
            right.sendall(packet)
            right.shutdown(socket.SHUT_WR)
            with pytest.raises(error):
                receive_packet(left, time.monotonic() + 1)
    left, right = socket.socketpair()
    with left, right:
        right.sendall(b"\x00")
        with pytest.raises(TimeoutError):
            receive_packet(left, time.monotonic() + 0.05)
        send_packet(right, {"pixels": b"\x00\xff"}, time.monotonic() + 1)


@pytest.mark.parametrize(
    "actions",
    [
        np.zeros((2, 7)),
        np.zeros((0, 16)),
        np.zeros((65, 16)),
        [[float("nan")] * 16],
        [[float("inf")] * 16],
        [[4.0] * 16],
        [[True] * 16],
        [["0"] * 16],
    ],
)
def test_bad_action_rejected_before_native_execution(native_stub, actions):
    core = geniesim.GenieSimEnvCore()
    try:
        core.build(
            EnvSpecMsg(env_family="geniesim", env_config=native_stub.to_dict()),
            num_envs=1,
        )
        core.reset([0], ResetSpec(seed=1001))
        with pytest.raises(RuntimeApiError) as error:
            core.chunk_step([0], [actions])
        assert error.value.info.code == ErrorCode.INVALID_ARGUMENT
        assert not (core.process.directory / "received.jsonl").exists()
    finally:
        core.close()


def test_mapping_cached_observe_short_chunk_and_seed_rebuild(native_stub):
    core = geniesim.GenieSimEnvCore()
    try:
        core.build(
            EnvSpecMsg(env_family="geniesim", env_config=native_stub.to_dict()),
            num_envs=1,
        )
        initial = core.reset([0], ResetSpec(seed=1001))[0]
        assert initial.state == [0.2] * 14 + [0.7, 0.7] + [0.0] * 5
        assert len(initial.state) == 21
        for image, color in zip(
            [initial.main_image, initial.wrist_image, *initial.extra_view_images],
            [25, 75, 125],
            strict=True,
        ):
            assert np.all(decode_payload(image) == color)
        assert core.observe([0])[0] is initial
        assert core.process._sequence == 1
        outcome = core.chunk_step([0], [np.full((8, 16), 0.3)])[0]
        assert outcome.executed_horizon == 3
        assert outcome.truncated and not outcome.terminated
        assert len(outcome.per_step) == 3
        assert not core.extension(0, "geniesim", "result", {})["success"]
        assert outcome.observation.state[:16] == [0.3] * 16
        with pytest.raises(RuntimeApiError):
            core.chunk_step([0], [np.full((1, 16), 0.3)])
        old = core.process
        core.reset([0], ResetSpec(seed=1001))
        assert core.process is old
        core.reset([0], ResetSpec(seed=1002))
        assert core.process.pid != old.pid
        assert (
            json.loads((old.directory / "process.json").read_text())["status"]
            == "closed"
        )
    finally:
        core.close()


def test_config_and_named_state_validation(config):
    with pytest.raises(ValueError, match="unknown"):
        GenieSimConfig.from_mapping({**config.to_dict(), "gpu": 1})
    with pytest.raises(ValueError, match="left_arm"):
        flatten_state({"left_arm": [0.1]})
    with pytest.raises(ValueError, match="limits"):
        validate_actions([[0.1] * 16], [[float("nan"), 1.0]] * 16)


def test_native_isaac51_limits_follow_named_articulation_dofs():
    properties = np.zeros(
        18,
        dtype=[("lower", float), ("upper", float), ("hasLimits", bool), ("type", int)],
    )
    properties["lower"] = -np.arange(1, 19)
    properties["upper"] = np.arange(1, 19)
    properties["hasLimits"] = True
    properties["type"] = 0
    names = [f"joint-{index}" for index in range(18)]
    articulation = SimpleNamespace(dof_names=names, dof_properties=properties)
    ordered = names[2:][::-1]
    limits = action_limits_from_articulation(articulation, ordered)
    assert limits[0] == [-18.0, 18.0]
    assert limits[-1] == [-3.0, 3.0]
    properties["type"][5] = 1
    with pytest.raises(RuntimeError, match="rotational"):
        action_limits_from_articulation(articulation, ordered)


@pytest.mark.parametrize("converges", [True, False])
def test_reset_replaces_old_drive_targets_and_requires_convergence(
    config, monkeypatch, converges
):
    clock = iter([0.0, 0.1] if converges else [0.0, 30.0])
    monkeypatch.setattr(
        native_session,
        "time",
        SimpleNamespace(monotonic=lambda: next(clock), sleep=lambda _: None),
    )
    arm = [f"arm-{index}" for index in range(14)]
    waist = [f"waist-{index}" for index in range(5)]
    grippers = ["left_gripper", "right_gripper"]
    names = arm + waist + grippers
    targets = dict(zip(names, [0.2] * 14 + [0.1] * 5 + [0.785] * 2, strict=True))
    stale = {**targets, "left_gripper": 0.66, "right_gripper": 0.66}
    courier = SimpleNamespace(
        get_joint_state_dict=Mock(
            side_effect=[stale, targets] if converges else [stale]
        )
    )
    fresh = {"states": "new observation after convergence"}
    api = SimpleNamespace(set_joint_positions_batched=Mock())
    session = native_session.GenieSimSession(
        config, None, api, Path(config.output_root)
    )
    session.env = SimpleNamespace(
        cfg={
            "arm_joints": arm,
            "waist_joints": waist,
            "head_joints": [],
            "gripper_joints": grippers,
            "init_gripper_open": [0.785, 0.785],
        },
        init_arm=[0.2] * 14,
        init_waist=[0.1] * 5,
        init_head=[],
        robot_joint_indices=dict(zip(names, range(21), strict=True)),
        data_courier=courier,
        get_observation=Mock(return_value=fresh),
        current_step=0,
    )
    if converges:
        assert session._settle_reset() is fresh
        assert courier.get_joint_state_dict.call_count == 2
        session.env.get_observation.assert_called_once_with()
    else:
        with pytest.raises(RuntimeError, match="did not settle"):
            session._settle_reset()
        session.env.get_observation.assert_not_called()
    assert session.env.current_step == 0
    api.set_joint_positions_batched.assert_called_once_with(
        [
            ([0.2] * 14, list(range(14)), True),
            ([0.1] * 5, list(range(14, 19)), True),
            ([0.785, 0.785], [19, 20], True),
        ]
    )


@pytest.mark.parametrize(
    "transport", ["inproc", pytest.param("ray_channel", marks=pytest.mark.ray)]
)
@pytest.mark.asyncio
async def test_runtime_client_lifecycle(native_stub, transport):
    runtime = build_local_components(
        {
            "env_family": "geniesim",
            "env_config": native_stub.to_dict(),
            "transport": {"kind": transport},
            "env_worker": {"placement_strategy": "packed", "max_sessions_per_rank": 1},
        }
    )
    async with runtime:
        client = runtime.gateway
        created = (
            await client.create_sessions(
                [
                    CreateSessionRequest(
                        application_id="geniesim-tests",
                        client_session_key="one",
                        env_spec=EnvSpecMsg(
                            env_family="geniesim", env_config=native_stub.to_dict()
                        ),
                    )
                ]
            )
        )[0]
        session_id = created.value.session_id
        initial = (await client.reset([session_id], ResetSpec(seed=1001)))[
            0
        ].value.observation
        action = encode_array(
            np.tile(np.asarray(initial.state[:16], dtype=np.float32), (5, 1))
        )
        step = (await client.action_step([session_id], [action]))[0].value
        assert step.truncated and not step.terminated
        assert step.observation.step_index == 3
        evidence = (
            await client.extension_call([session_id], "geniesim", "result", {})
        )[0].value
        assert evidence["control_steps"] == 3 and evidence["success"] is False
        assert (await client.reset([session_id], ResetSpec(seed=1001)))[
            0
        ].value.observation.step_index == 0
        await client.close_sessions([session_id])
    assert (
        json.loads((Path(evidence["artifact_directory"]) / "process.json").read_text())[
            "status"
        ]
        == "closed"
    )
