# Copyright (c) 2026 Zetta Contributors
"""End-to-end case for ``robots/robocasa/run_rollout.py`` running through the
full served Runtime.

Deliberately does **not** mock ``RemoteRuntimeClient``: what needs verifying
here is precisely "can the application side's sole call path actually run
end to end" -- swapping the client for a fake one would verify away the
very thing being tested. So this brings in the real ``serve/app.py`` via
``httpx.ASGITransport``, going through real msgpack encode/decode, a real
Gateway state machine, real ``RuntimeEnvWorker``/``RuntimeRolloutWorker``
(``launch/local.py`` + ``inproc`` transport), with the env family being the
real ``robocasa`` execution core (``RobocasaCurrentCore`` + ``RoboCasaSession``).

Two necessary stand-ins, both substituting only for "what cannot be
installed on this machine," never for the protocol:

1. ``RoboCasaSession._ensure_environment`` -> a minimal fake gym env. The
   real robocasa / robosuite / MuJoCo are not present in the unit test
   environment (same constraint as
   ``tests/test_robocasa_env_runtime.py``/``tests/runtime/test_robocasa_current_backend.py``).
2. The policy backend uses ``fake`` (12-dim action) instead of GR00T: GR00T
   needs torch + a real checkpoint. ``FakePolicyCore``'s action is uniformly
   distributed over ``[-1, 1)``, while ``action_contract`` requires
   ``gripper_close``/``control_mode`` to land in ``[0, 1]``, so these two
   columns are clamped -- this is not relaxing the contract, but making the
   fake policy satisfy it.

Path covered: ``create_sessions`` -> ``reset`` (including delivering
``video_dir`` via ``ResetSpec.options``) -> ``extension_call
robocasa.snapshot`` -> per-chunk ``policy_step`` -> ``extension_call
robocasa.finalize_episode`` -> ``close_sessions``, and the trajectory
artifacts and ``EpisodeRecord`` produced along the way.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import pytest
import pytest_asyncio

from robots.robocasa import run_rollout, session_core
from rollout_runtime.backends.fake.policy import FakePolicyCore
from rollout_runtime.serve.app import ServeLimits
from rollout_runtime.serve.client import RemoteRuntimeClient
from rollout_runtime.serve.server import ServeOptions, build_served_runtime

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason=(
        "episode video muxing (RoboCasaSession.finalize_episode_artifacts) needs "
        "ffmpeg/ffprobe, and zetta.evolution.trajectory refuses an episode whose "
        "video artifacts are missing or empty"
    ),
)

CAMERA_SIZE = 16
"""Frame size for the fake cameras; ``EpisodeVideoArtifacts`` requires an exact match."""

ENV_MAX_STEPS = 8
ACTIONS_PER_CHUNK = 4


class _FakeEnv:
    """Minimal gym-shaped env: three cameras plus two named vector state fields."""

    def __init__(self) -> None:
        self.actions: list[Any] = []
        self._step_index = 0

    def reset(self, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        self._step_index = 0
        self.seed = int(seed)
        return self._observation(), {"success": False}

    def step(self, action: dict[str, np.ndarray]) -> tuple[Any, ...]:
        self.actions.append(action)
        self._step_index += 1
        terminated = self._step_index >= 6
        return (
            self._observation(),
            1.0 if terminated else 0.0,
            terminated,
            False,
            {"success": terminated},
        )

    def close(self) -> None:
        pass

    def _observation(self) -> dict[str, Any]:
        frame = np.full((CAMERA_SIZE, CAMERA_SIZE, 3), self._step_index, dtype=np.uint8)
        return {
            "video.robot0_agentview_left": frame,
            "video.robot0_agentview_right": frame + 1,
            "video.robot0_eye_in_hand": frame + 2,
            "state.end_effector_position_relative": np.array(
                [0.1, 0.2, 0.3], dtype=np.float32
            ),
            "state.gripper_qpos": np.array([0.0, 1.0], dtype=np.float32),
            "task_descriptions": ["move the pan"],
        }


def _preset(tmp_path: Path) -> Path:
    """Write a partial preset (``load_config`` merges it onto the schema defaults)."""
    path = tmp_path / "robocasa_e2e.yaml"
    path.write_text(
        "\n".join(
            [
                "env_family: robocasa",
                "env_config:",
                "  action_dim: 12",
                f"  chunk_size: {ACTIONS_PER_CHUNK}",
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


@pytest_asyncio.fixture(loop_scope="function")
async def runtime_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[RemoteRuntimeClient]:
    """Serve a real runtime over ASGI and hand back a real ``RemoteRuntimeClient``."""

    def _fake_ensure_environment(self, task: str, split: str) -> None:
        self.env = _FakeEnv()
        self.identity = (task, split)

    original_actions_for = FakePolicyCore._actions_for

    def _contract_safe_actions(self, request: Any) -> np.ndarray:
        block = np.array(original_actions_for(self, request), dtype=np.float32)
        # gripper_close (index 6) and control_mode (index 11) are [0, 1] in
        # ``robots/robocasa/action_contract.py``.
        block[:, 6] = np.abs(block[:, 6])
        block[:, 11] = np.abs(block[:, 11])
        return block

    monkeypatch.setattr(
        session_core.RoboCasaSession, "_ensure_environment", _fake_ensure_environment
    )
    monkeypatch.setattr(FakePolicyCore, "_actions_for", _contract_safe_actions)
    # Keyframe extraction reads the muxed mp4 back through imageio, which needs the
    # optional ``imageio[ffmpeg]`` reader plugin (the muxing itself only needs the
    # ffmpeg CLI). Visual evidence has its own tests; this one is about the runtime
    # call path, so stub the reader rather than making the whole file conditional on
    # yet another optional dependency.
    monkeypatch.setattr(
        run_rollout,
        "build_episode_visual_artifacts",
        lambda **_: {"artifacts": {}, "artifact_sha256": {}},
    )

    runtime = await build_served_runtime(
        ServeOptions(
            config=str(_preset(tmp_path)),
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
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=runtime.app), base_url="http://serve.test"
    ) as raw:
        yield RemoteRuntimeClient("http://serve.test", token=None, client=raw)
    await runtime.aclose()


def _args(tmp_path: Path) -> SimpleNamespace:
    attempt = tmp_path / "attempt"
    return SimpleNamespace(
        runtime_url="http://serve.test",
        runtime_token=None,
        policy_id="fake",
        operation_timeout_s=60.0,
        session_timeout_s=60.0,
        session_lease_s=600.0,
        task="SlideDishwasherRack",
        instruction=None,
        split="target",
        seed=4242,
        policy_rng=7,
        logical_id="stage7-e2e",
        attempt_index=0,
        generation=0,
        bundle="none",
        bundle_sha256="none",
        baseline_mode="strict_pure_vla",
        safety_layer="interface_contract_v1",
        output_dir=str(attempt),
        result_file=str(attempt / "episode_record.json"),
        max_actions=ENV_MAX_STEPS,
        actions_per_chunk=ACTIONS_PER_CHUNK,
        camera_size=CAMERA_SIZE,
        env_max_steps=ENV_MAX_STEPS,
        require_isolated_renderer=False,
        process_isolation=False,
        env_pool_size=1,
        env_max_pool_size=None,
        role1_planner="none",
        role1_model=None,
        reasoning_effort=None,
        role1_max_tokens=128,
        role1_timeout_s=10,
        role1_heartbeat_s=1.0,
        role1_max_turns=1,
        role1_max_decisions_per_action=2,
        tool_runtime="builtin",
        harness_root=None,
        allow_privileged_tools=True,
    )


def _rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


async def test_strict_gen0_episode_runs_end_to_end_over_the_served_runtime(
    runtime_client: RemoteRuntimeClient, tmp_path: Path
) -> None:
    """A full Gen0 episode runs to completion solely through
    ``RemoteRuntimeClient``, producing the complete set of artifacts."""
    args = _args(tmp_path)
    record = await run_rollout.run(args, client=runtime_client)

    assert record.status == "valid"
    assert record.seed == 4242
    assert record.artifact_index["baseline_mode"] == "strict_pure_vla"
    # The session was closed == the semantics of the old release(): the env
    # slot is handed back to the EnvPool for reuse by the next rollout.
    assert record.artifact_index["environment_release"]["session_closed"] is True
    runtime_provenance = record.artifact_index["rollout_runtime"]
    assert runtime_provenance["policy_id"] == "fake"
    assert len(runtime_provenance["env_spec_digest"]) == 64
    assert runtime_provenance["reset_episode_id"] == 1

    # Per-step audit artifacts all come from PerStepRecord.info (the channel added for this).
    attempt = tmp_path / "attempt"
    actions = _rows(attempt / "trajectory" / "actions.jsonl")
    states = _rows(attempt / "trajectory" / "states.jsonl")
    chunks = _rows(attempt / "trajectory" / "chunks.jsonl")
    assert actions, "no per-step action records crossed the runtime boundary"
    assert all(len(row["action_sha256"]) == 64 for row in actions)
    assert all("action.end_effector_position" in row["action"] for row in actions)
    assert [row["step_index"] for row in actions] == list(range(1, len(actions) + 1))
    # states.jsonl's first row is the reset frame, followed by per-step rows.
    assert states[0]["event"] == "reset"
    assert "state.end_effector_position_relative" in states[0]["state"]
    assert len(states) == len(actions) + 1
    assert all(chunk["vla"]["source"] == "policy_step" for chunk in chunks)
    assert all(chunk["environment"]["critic_rule_count"] == 0 for chunk in chunks)
    assert all(
        chunk["environment"]["task_program_enabled"] is False for chunk in chunks
    )

    # Video is only written by the robocasa.finalize_episode extension; the
    # trajectory index refuses an empty video.
    videos = record.artifact_index["videos"]
    assert set(videos) == {
        "video.robot0_agentview_left",
        "video.robot0_agentview_right",
        "video.robot0_eye_in_hand",
    }
    assert all(Path(path).stat().st_size > 0 for path in videos.values())
    assert record.artifact_index["trajectory_index"] is not None
    assert not (attempt / "cleanup_errors.jsonl").exists()


async def test_second_episode_reuses_the_pool_slot_after_close_sessions(
    runtime_client: RemoteRuntimeClient, tmp_path: Path
) -> None:
    """Two consecutive episodes share the same pool (``pool_size=1``) --
    the slot must genuinely be handed back.

    This is a regression guarantee for internal runtime slot reuse: if
    ``close_sessions`` did not return the slot to the ``EnvPool``, the
    second episode would fail to create a session for lack of a free slot.
    """
    first = _args(tmp_path / "one")
    second = _args(tmp_path / "two")
    second.logical_id = "stage7-e2e-second"
    second.seed = 99

    first_record = await run_rollout.run(first, client=runtime_client)
    second_record = await run_rollout.run(second, client=runtime_client)

    assert first_record.status == "valid"
    assert second_record.status == "valid"
    assert (
        first_record.artifact_index["rollout_runtime"]["env_spec_digest"]
        == second_record.artifact_index["rollout_runtime"]["env_spec_digest"]
    )
    assert (
        first_record.artifact_index["rollout_runtime"]["session_id"]
        != second_record.artifact_index["rollout_runtime"]["session_id"]
    )
