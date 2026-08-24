# Copyright (c) 2026 Zetta Contributors
"""The ``--runtime-url`` audit ledger.

The Runtime has no ``libero.audit_trace`` extension, so ``actions.jsonl`` and
``states.jsonl`` are rebuilt application side from the per-step records
``LiberoRuntimeEnvClient`` accumulates. Regressing that accumulation, the row
shape, or the ``audit_trace`` branch precedence silently empties the ledger,
which is how it shipped reset-only once already.

The e2e cases drive a real served Runtime (fake env family, in-process ASGI,
real msgpack, real Gateway) through the real client and the real
``runtime_ledger_rows`` -- nothing on the ledger path is mocked.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import pytest_asyncio

from robots.libero import run_evolution_rollout
from robots.libero.run_evolution_rollout import runtime_ledger_rows
from rollout_runtime.adapters.zetta.runtime_env_client import (
    LiberoRuntimeEnvClient,
    SyncRuntimeLoop,
)
from rollout_runtime.api.ids import SessionId
from rollout_runtime.api.messages import (
    CreateSessionRequest,
    EnvSpecMsg,
    Observation,
    PerStepRecord,
    StepResult,
)
from rollout_runtime.api.result import Err, ok, unwrap
from rollout_runtime.serve.app import ServeLimits
from rollout_runtime.serve.client import RemoteRuntimeClient
from rollout_runtime.serve.server import ServeOptions, build_served_runtime
from zetta.evolution.jsonio import canonical_sha256
from zetta.evolution.trajectory import TrajectoryArtifacts, index_episode_trajectory

ACTIONS_PER_CHUNK = 5
ENV_MAX_STEPS = 20
SHORT_TAIL_STEPS = 18
"""Not a multiple of ``ACTIONS_PER_CHUNK``, so the last chunk runs short."""
STATE_DIM = 8
ACTION_DIM = 7

ENV_CONFIG: dict[str, Any] = {
    "action_dim": ACTION_DIM,
    "chunk_size": ACTIONS_PER_CHUNK,
    "episode_length": ENV_MAX_STEPS,
    "image_height": 16,
    "image_width": 16,
    "state_dim": STATE_DIM,
}

ENV_SERVER_STATE_ROW_KEYS = {
    "step_index",
    "action",
    "action_sha256",
    "state",
    "observation_sha256",
    "reward",
    "libero_terminated",
    "truncated",
    "proposal_rule_ids",
}
"""The key set ``env_server.py:318-331`` writes. ``libero_terminated`` (not
``terminated``) is load-bearing: ``visual_artifacts.py:77-82`` reads it."""


# --------------------------------------------------------------------------
# unit-level: the client accumulator (W1)
# --------------------------------------------------------------------------


def _observation(*, step_index: int, state: list[float]) -> Observation:
    return Observation(
        session_id=SessionId("session-a"),
        episode_id=1,
        step_index=step_index,
        state=state,
        instruction="pick up the cup",
    )


class _FakeRuntimeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.next_step_result: StepResult | None = None
        self.next_extension_result: Any = {}

    async def reset(self, ids, spec):
        self.calls.append(("reset", spec))
        return [ok(self.next_step_result)]

    async def action_step(self, ids, actions):
        self.calls.append(("action_step", actions))
        return [ok(self.next_step_result)]

    async def extension_call(self, ids, namespace, method, args):
        self.calls.append(("extension_call", (namespace, method, args)))
        return [ok(self.next_extension_result)]


@pytest.fixture
def loop() -> Any:
    instance = SyncRuntimeLoop()
    yield instance
    instance.close()


def _chunk_result(*, first_index: int, count: int) -> StepResult:
    per_step = [
        PerStepRecord(
            step_index=first_index + offset,
            reward=float(offset),
            terminated=False,
            truncated=False,
            observation=_observation(
                step_index=first_index + offset,
                state=[float(first_index + offset)] * STATE_DIM,
            ),
            info={"critic_proposal_rule_ids": ["r1"] if offset == 1 else []},
        )
        for offset in range(count)
    ]
    return StepResult(
        request_id="req",
        session_id=SessionId("session-a"),
        observation=per_step[-1].observation,
        executed_horizon=count,
        per_step=per_step,
    )


def test_chunk_step_accumulates_one_record_per_physical_step(loop: Any) -> None:
    client = _FakeRuntimeClient()
    client.next_step_result = _chunk_result(first_index=1, count=ACTIONS_PER_CHUNK)
    env = LiberoRuntimeEnvClient(client, SessionId("session-a"), loop=loop)
    block = np.arange(ACTIONS_PER_CHUNK * ACTION_DIM, dtype=np.float32).reshape(
        ACTIONS_PER_CHUNK, ACTION_DIM
    )

    env.chunk_step(block)
    records = env.drain_step_records()

    assert [r["step_index"] for r in records] == [1, 2, 3, 4, 5]
    # actions stay aligned to their own step
    assert records[2]["action"] == block[2].astype(np.float64).tolist()
    assert records[1]["proposal_rule_ids"] == ["r1"]
    assert records[0]["proposal_rule_ids"] == []


def test_a_critic_interrupted_short_chunk_yields_executed_horizon_records(
    loop: Any,
) -> None:
    client = _FakeRuntimeClient()
    client.next_step_result = _chunk_result(first_index=1, count=3)
    env = LiberoRuntimeEnvClient(client, SessionId("session-a"), loop=loop)
    block = np.zeros((ACTIONS_PER_CHUNK, ACTION_DIM), dtype=np.float32)

    env.chunk_step(block)

    assert len(env.drain_step_records()) == 3


def test_drain_returns_then_clears(loop: Any) -> None:
    client = _FakeRuntimeClient()
    client.next_step_result = _chunk_result(first_index=1, count=2)
    env = LiberoRuntimeEnvClient(client, SessionId("session-a"), loop=loop)

    env.chunk_step(np.zeros((2, ACTION_DIM), dtype=np.float32))

    assert len(env.drain_step_records()) == 2
    assert env.drain_step_records() == []


def test_reset_clears_records_left_from_a_previous_episode(loop: Any) -> None:
    client = _FakeRuntimeClient()
    client.next_step_result = _chunk_result(first_index=1, count=2)
    env = LiberoRuntimeEnvClient(client, SessionId("session-a"), loop=loop)
    env.chunk_step(np.zeros((2, ACTION_DIM), dtype=np.float32))

    client.next_step_result = StepResult(
        request_id="req",
        session_id=SessionId("session-a"),
        observation=_observation(step_index=0, state=[0.0] * STATE_DIM),
    )
    env.reset(task_id=0, seed=1)

    assert env.drain_step_records() == []


def test_missing_per_step_observation_falls_back_to_the_chunk_final_one(
    loop: Any,
) -> None:
    client = _FakeRuntimeClient()
    base = _chunk_result(first_index=1, count=3)
    per_step = list(base.per_step or ())
    per_step[1] = PerStepRecord(
        step_index=2, reward=1.0, observation=None, info={}
    )
    client.next_step_result = StepResult(
        request_id="req",
        session_id=SessionId("session-a"),
        observation=base.observation,
        executed_horizon=3,
        per_step=per_step,
    )
    env = LiberoRuntimeEnvClient(client, SessionId("session-a"), loop=loop)

    env.chunk_step(np.zeros((3, ACTION_DIM), dtype=np.float32))
    records = env.drain_step_records()

    assert records[1]["states"] == list(base.observation.state)


def test_no_extension_call_when_privileged_sampling_is_off(loop: Any) -> None:
    """The default must add no network traffic, so callers with a fake client
    that has no ``extension_call`` support keep working."""

    client = _FakeRuntimeClient()
    client.next_step_result = _chunk_result(first_index=1, count=ACTIONS_PER_CHUNK)
    env = LiberoRuntimeEnvClient(client, SessionId("session-a"), loop=loop)
    assert env.sample_privileged_state is False

    env.chunk_step(np.zeros((ACTIONS_PER_CHUNK, ACTION_DIM), dtype=np.float32))

    assert [name for name, _ in client.calls if name == "extension_call"] == []
    assert all(r["privileged_state"] is None for r in env.drain_step_records())


def test_privileged_state_is_sampled_once_per_chunk_onto_the_last_record(
    loop: Any,
) -> None:
    client = _FakeRuntimeClient()
    client.next_extension_result = {"privileged.task.progress": 0.25}
    client.next_step_result = _chunk_result(first_index=1, count=ACTIONS_PER_CHUNK)
    env = LiberoRuntimeEnvClient(
        client, SessionId("session-a"), loop=loop, sample_privileged_state=True
    )

    env.chunk_step(np.zeros((ACTIONS_PER_CHUNK, ACTION_DIM), dtype=np.float32))
    records = env.drain_step_records()

    extension_calls = [args for name, args in client.calls if name == "extension_call"]
    assert len(extension_calls) == 1
    assert extension_calls[0] == ("libero", "critic_state", {"reset_tracker": False})
    assert records[-1]["privileged_state"] == {"privileged.task.progress": 0.25}
    assert all(r["privileged_state"] is None for r in records[:-1])


def test_privileged_sampling_reaches_the_ledger_row() -> None:
    """The sampled dict must reach the row's feature plane, not be dropped."""

    record = _raw_record(step_index=1, action=[0.0] * ACTION_DIM, z=0.5)
    record["privileged_state"] = {"privileged.task.progress": 0.75}

    rows = list(runtime_ledger_rows([record], previous_eef=None))

    features = rows[0][1]["state"]
    assert features["privileged.task.progress"] == 0.75


def test_an_empty_chunk_issues_no_privileged_sample(loop: Any) -> None:
    """No records appended means no round trip."""

    client = _FakeRuntimeClient()
    client.next_extension_result = {}
    client.next_step_result = StepResult(
        request_id="req",
        session_id=SessionId("session-a"),
        observation=_observation(step_index=0, state=[0.0] * STATE_DIM),
        executed_horizon=0,
        per_step=[],
    )
    env = LiberoRuntimeEnvClient(
        client, SessionId("session-a"), loop=loop, sample_privileged_state=True
    )

    env.chunk_step(np.zeros((ACTIONS_PER_CHUNK, ACTION_DIM), dtype=np.float32))

    assert [name for name, _ in client.calls if name == "extension_call"] == []


# --------------------------------------------------------------------------
# unit-level: the row builder (W2)
# --------------------------------------------------------------------------


def _raw_record(*, step_index: int, action: list[float], z: float) -> dict[str, Any]:
    return {
        "step_index": step_index,
        "action": action,
        "states": [0.1, 0.2, z, 0.0, 0.0, 0.0, 0.3, 0.4],
        "reward": 0.0,
        "terminated": False,
        "truncated": False,
        "proposal_rule_ids": [],
    }


def test_ledger_row_key_set_matches_env_server() -> None:
    rows = list(
        runtime_ledger_rows(
            [_raw_record(step_index=1, action=[0.0] * ACTION_DIM, z=0.5)],
            previous_eef=None,
        )
    )

    _actions_row, states_row, _eef = rows[0]
    assert set(states_row) == ENV_SERVER_STATE_ROW_KEYS
    assert "libero_terminated" in states_row
    assert "terminated" not in states_row


def test_action_sha256_matches_the_canonical_hash_of_the_action() -> None:
    action = [0.5, -0.25, 0.125, 0.0, 0.0, 0.0, -1.0]
    rows = list(
        runtime_ledger_rows(
            [_raw_record(step_index=1, action=action, z=0.5)], previous_eef=None
        )
    )

    actions_row, states_row, _eef = rows[0]
    assert actions_row["action"] == action
    assert actions_row["action_sha256"] == canonical_sha256(action)
    assert states_row["action_sha256"] == actions_row["action_sha256"]
    assert set(actions_row) == {"step_index", "action", "action_sha256"}


def test_eef_chain_advances_and_seeds_delta_available() -> None:
    records = [
        _raw_record(step_index=1, action=[0.0] * ACTION_DIM, z=0.5),
        _raw_record(step_index=2, action=[0.0] * ACTION_DIM, z=0.8),
    ]
    seeded = np.asarray([0.1, 0.2, 0.0])

    rows = list(runtime_ledger_rows(records, previous_eef=seeded))

    # seeded from the reset observation -> step 1 already has a delta
    assert rows[0][1]["state"]["robot.eef.delta_available"] is True
    assert rows[0][1]["state"]["robot.eef.delta.z"] == pytest.approx(0.5)
    assert rows[1][1]["state"]["robot.eef.delta.z"] == pytest.approx(0.3)
    assert rows[1][2].tolist() == pytest.approx([0.1, 0.2, 0.8])


def test_a_record_without_state_is_loud_not_silent() -> None:
    record = _raw_record(step_index=1, action=[0.0] * ACTION_DIM, z=0.5)
    record["states"] = None

    with pytest.raises(RuntimeError, match="no observation state"):
        list(runtime_ledger_rows([record], previous_eef=None))


def test_privileged_state_is_optional() -> None:
    """The extractor must tolerate its absence, since sampling is opt-in."""

    rows = list(
        runtime_ledger_rows(
            [_raw_record(step_index=1, action=[0.0] * ACTION_DIM, z=0.5)],
            previous_eef=None,
        )
    )

    features = rows[0][1]["state"]
    assert not [key for key in features if key.startswith("privileged.")]


# --------------------------------------------------------------------------
# e2e: real served runtime + real client + real row builder
# --------------------------------------------------------------------------


def _preset(tmp_path: Path, *, episode_length: int) -> Path:
    path = tmp_path / f"fake_issue1_{episode_length}.yaml"
    path.write_text(
        "\n".join(
            [
                "env_family: fake",
                "env_config:",
                f"  action_dim: {ACTION_DIM}",
                f"  chunk_size: {ACTIONS_PER_CHUNK}",
                f"  episode_length: {episode_length}",
                "  image_height: 16",
                "  image_width: 16",
                f"  state_dim: {STATE_DIM}",
                "transport:",
                "  kind: inproc",
                "gateway:",
                "  gateway_epoch: 11",
                "  default_lease_seconds: 600.0",
                "  heartbeat_timeout_seconds: 60.0",
                "  heartbeat_interval_seconds: 5.0",
                "env_worker:",
                "  num_ranks: 1",
                "  placement_strategy: packed",
                "  max_sessions_per_rank: 1",
                "  default_pool_size: 1",
                "rollout_worker:",
                "  num_ranks: 1",
                "  policy_id: fake",
                "  policy_family: fake",
                "  policy_backend: fake",
                "  device: cpu",
                "  dtype: float32",
                "admission:",
                "  require_auth: false",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


async def _serve(tmp_path: Path, *, episode_length: int) -> Any:
    return await build_served_runtime(
        ServeOptions(
            config=str(_preset(tmp_path, episode_length=episode_length)),
            host="127.0.0.1",
            gateway_epoch=11,
            limits=ServeLimits(
                max_lease_seconds=600.0,
                max_pool_size=2,
                max_episode_steps=ENV_MAX_STEPS,
                max_body_bytes=1 << 22,
            ),
        ),
        environ={},
    )


@pytest_asyncio.fixture(loop_scope="function")
async def served(tmp_path: Path) -> AsyncIterator[Any]:
    """A horizon that divides evenly by the chunk size (20 = 4 x 5)."""

    runtime = await _serve(tmp_path, episode_length=ENV_MAX_STEPS)
    yield runtime
    await runtime.aclose()


@pytest_asyncio.fixture(loop_scope="function")
async def served_short_tail(tmp_path: Path) -> AsyncIterator[Any]:
    """A horizon that ends mid-chunk (18 = 5 + 5 + 5 + 3)."""

    runtime = await _serve(tmp_path, episode_length=SHORT_TAIL_STEPS)
    yield runtime
    await runtime.aclose()


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    """Mirrors run_evolution_rollout.py:94-99."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


class _Episode:
    """What one scripted Runtime episode produced."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.actions_path = directory / "actions.jsonl"
        self.states_path = directory / "states.jsonl"
        self.chunks_path = directory / "chunks.jsonl"
        self.tools_path = directory / "tools.jsonl"
        for path in (
            self.actions_path,
            self.states_path,
            self.chunks_path,
            self.tools_path,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")
        self.physical_steps = 0
        self.submitted: list[list[float]] = []
        self.executed_per_chunk: list[int] = []

    @property
    def state_rows(self) -> list[dict[str, Any]]:
        return [
            json.loads(line)
            for line in self.states_path.read_text(encoding="utf-8").splitlines()
        ]

    @property
    def action_rows(self) -> list[dict[str, Any]]:
        return [
            json.loads(line)
            for line in self.actions_path.read_text(encoding="utf-8").splitlines()
        ]

    def artifacts(self) -> TrajectoryArtifacts:
        return TrajectoryArtifacts(
            chunks=self.chunks_path,
            actions=self.actions_path,
            states=self.states_path,
            tools=self.tools_path,
        )


def _run_episode(
    served_runtime: Any, directory: Path, *, episode_length: int = ENV_MAX_STEPS
) -> _Episode:
    """Drive one full episode through the real client and the real row builder.

    ``episode_length`` must match the fixture's preset: the session's own
    ``env_spec.env_config`` is what the pool is keyed on, so a mismatch would
    silently open a second pool running the wrong horizon.
    """

    env_config = dict(ENV_CONFIG, episode_length=episode_length)

    episode = _Episode(directory)
    loop = SyncRuntimeLoop()
    raw = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=served_runtime.app),
        base_url="http://serve.test",
    )
    client = RemoteRuntimeClient("http://serve.test", token=None, client=raw)
    try:
        results = loop.run(
            client.create_sessions(
                [
                    CreateSessionRequest(
                        application_id="issue1",
                        client_session_key="s-0",
                        env_spec=EnvSpecMsg(
                            env_family="fake", env_config=env_config, pool_size=1
                        ),
                        default_policy_id="fake",
                        lease_seconds=600.0,
                    )
                ]
            )
        )
        failures = [r.error for r in results if isinstance(r, Err)]
        assert not failures, f"create_sessions failed: {failures}"
        env = LiberoRuntimeEnvClient(
            client, unwrap(results[0]).session_id, loop=loop, return_all_frames=True
        )

        last_eef: np.ndarray | None = None

        def flush() -> None:
            nonlocal last_eef
            for action_row, state_row, next_eef in runtime_ledger_rows(
                env.drain_step_records(), previous_eef=last_eef
            ):
                _append_jsonl(episode.actions_path, action_row)
                _append_jsonl(episode.states_path, state_row)
                last_eef = next_eef

        obs, _info = env.reset(task_id=0, seed=4242)
        reset_states = np.asarray(obs["states"], dtype=np.float64).reshape(-1)
        last_eef = reset_states[:3].copy()
        _append_jsonl(episode.states_path, {"step_index": 0, "note": "reset row"})

        rng = np.random.default_rng(7)
        while not (env.episode_terminated or env.episode_truncated):
            block = rng.uniform(
                -1.0, 1.0, size=(ACTIONS_PER_CHUNK, ACTION_DIM)
            ).astype(np.float32)
            _frames, _reward, term, _trunc, _info = env.chunk_step(block)
            executed = int(np.asarray(term).size)
            episode.executed_per_chunk.append(executed)
            episode.submitted.extend(
                np.asarray(row, dtype=np.float64).tolist() for row in block[:executed]
            )
            episode.physical_steps += executed
            flush()
        return episode
    finally:
        loop.run(raw.aclose())
        loop.close()


def test_runtime_episode_writes_one_ledger_row_per_physical_step(
    served: Any, tmp_path: Path
) -> None:
    episode = _run_episode(served, tmp_path / "trajectory")

    state_rows = episode.state_rows
    action_rows = episode.action_rows

    # one row per physical step, plus the reset row
    assert episode.physical_steps > 0
    assert len(state_rows) == 1 + episode.physical_steps
    assert len(action_rows) == episode.physical_steps

    # the reset row stays unique and indices advance without gaps
    assert [row["step_index"] for row in state_rows[1:]] == list(
        range(1, episode.physical_steps + 1)
    )
    assert sum(row["step_index"] == 0 for row in state_rows) == 1

    # key-set parity with the direct-connect ledger
    for row in state_rows[1:]:
        assert set(row) == ENV_SERVER_STATE_ROW_KEYS

    # actions stay bound to their own step, hashes agree
    for offset, (row, action) in enumerate(zip(action_rows, episode.submitted)):
        assert row["step_index"] == offset + 1
        assert row["action"] == pytest.approx(action)
        assert row["action_sha256"] == canonical_sha256(row["action"])


def test_an_episode_ending_mid_chunk_writes_only_the_executed_steps(
    served_short_tail: Any, tmp_path: Path
) -> None:
    """A short final chunk must yield ``executed_horizon`` rows, not a full one."""

    episode = _run_episode(
        served_short_tail, tmp_path / "trajectory", episode_length=SHORT_TAIL_STEPS
    )

    assert episode.physical_steps == SHORT_TAIL_STEPS
    # the horizon does not divide evenly, so the last chunk really was short
    assert episode.executed_per_chunk[-1] < ACTIONS_PER_CHUNK
    assert sum(episode.executed_per_chunk) == SHORT_TAIL_STEPS
    # and the ledger followed the executed count, not the submitted one
    assert len(episode.action_rows) == SHORT_TAIL_STEPS
    assert len(episode.state_rows) == 1 + SHORT_TAIL_STEPS
    assert [row["step_index"] for row in episode.action_rows] == list(
        range(1, SHORT_TAIL_STEPS + 1)
    )


def _outcome() -> dict[str, Any]:
    return {
        "episode_id": "ep-issue1",
        "logical_id": "stage-issue1",
        "status": "valid",
        "success": False,
    }


def test_the_downstream_trajectory_index_sees_every_step(
    served: Any, tmp_path: Path
) -> None:
    """A populated ledger must reach the consumer, not just exist on disk.

    ``index_episode_trajectory`` is run over both a degenerate reset-only
    ledger and a full one, so the assertion is about the difference rather than
    an absolute count.
    """

    episode = _run_episode(served, tmp_path / "trajectory")

    broken_dir = tmp_path / "broken"
    broken = _Episode(broken_dir)
    _append_jsonl(broken.states_path, {"step_index": 0, "note": "reset row"})

    before = index_episode_trajectory(result=_outcome(), artifacts=broken.artifacts())
    after = index_episode_trajectory(result=_outcome(), artifacts=episode.artifacts())

    assert before.index is not None
    assert after.index is not None

    # a reset-only ledger: one state row, no actions at all
    assert before.index.action_count == 0
    assert before.index.event_count == 1

    # populated: every physical step is visible to the indexer
    assert after.index.action_count == episode.physical_steps + 1
    assert after.index.event_count == 1 + 2 * episode.physical_steps
    assert after.index.action_count > before.index.action_count
    assert after.index.event_count > before.index.event_count


def test_audit_trace_wins_when_an_env_exposes_both_accessors() -> None:
    """The direct-connect branch must never be diverted onto the newer path.

    No real env exposes both, so the precedence is pinned here rather than
    left to the reading order of two ``hasattr`` checks.
    """

    source = Path(run_evolution_rollout.__file__).read_text(encoding="utf-8")
    audit_at = source.index('if not hasattr(env, "audit_trace")')
    drain_at = source.index('if not hasattr(env, "drain_step_records")')
    assert audit_at < drain_at
