# Copyright (c) 2026 RPent Contributors
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robots.libero.env_server import LiberoEnvFacade, build_env_cfg


class _TerminatingEnv:
    def __init__(self, terminate_at: int):
        self.terminate_at = terminate_at
        self.steps = 0

    def step(self, action):
        self.steps += 1
        terminated = self.steps == self.terminate_at
        obs = {
            "pixels": np.full((1, 1), self.steps, dtype=np.int32),
            "states": np.zeros((1, 8), dtype=np.float32),
        }
        return (
            obs,
            np.asarray([float(self.steps)]),
            np.asarray([terminated]),
            np.asarray([False]),
            {"step": self.steps},
        )


class _CriticEnv:
    def __init__(self) -> None:
        self.steps = 0

    def step(self, action):
        self.steps += 1
        state = np.zeros((1, 8), dtype=np.float32)
        state[0, 2] = float(self.steps)
        return (
            {"states": state},
            np.asarray([0.0]),
            np.asarray([False]),
            np.asarray([False]),
            {"step": self.steps},
        )


def _critic_rule(*, threshold: float = 2.0) -> dict:
    return {
        "rule_id": "critic-eef-z",
        "title": "stop at z",
        "feature": "robot.eef.z",
        "operator": "ge",
        "threshold": threshold,
        "dwell_steps": 1,
        "cooldown_steps": 0,
        "proposal": "invoke recovery",
        "evidence_ids": ["segment-1"],
        "safety_only": False,
        "activation_conditions": [],
    }


def test_chunk_step_stops_at_mid_chunk_termination():
    raw = _TerminatingEnv(terminate_at=3)
    facade = LiberoEnvFacade(raw, meta={})

    observations, rewards, terminated, truncated, info = facade.chunk_step(
        np.zeros((5, 7), dtype=np.float32), return_all_frames=True
    )

    assert raw.steps == 3
    assert len(observations) == 3
    assert rewards.tolist() == [1.0, 2.0, 3.0]
    assert terminated.tolist() == [False, False, True]
    assert truncated.tolist() == [False, False, False]
    assert info["executed_horizon"] == 3
    assert [row["step_index"] for row in facade.audit_trace()] == [1, 2, 3]
    # Ordinary chunks now retain the Critic-only availability marker so their
    # audit rows can be replayed consistently with critic_chunk_step.
    assert all(
        row["state"]["privileged.available"] is False
        for row in facade.audit_trace()
    )


class _DeadProcess:
    pid = 4321
    exitcode = -11

    def join(self, timeout: float) -> None:
        assert timeout == 0.25

    def is_alive(self) -> bool:
        return False


class _EofEnv:
    def __init__(self) -> None:
        self.env = SimpleNamespace(
            workers=[SimpleNamespace(process=_DeadProcess())]
        )

    def step(self, _action):
        raise EOFError


def test_worker_eof_reports_process_exit_status() -> None:
    facade = LiberoEnvFacade(_EofEnv(), meta={})

    with pytest.raises(
        RuntimeError,
        match=r"step 1: worker_index=0, pid=4321, exitcode=-11, alive=False",
    ):
        facade.step(np.zeros(7, dtype=np.float32))


def test_outer_wrapper_owns_episode_horizon():
    cfg = build_env_cfg(max_episode_steps=530)

    assert cfg.max_episode_steps == 530
    assert cfg.init_params.horizon == 1530
    assert cfg.init_params.ignore_done is True


def test_empty_gen0_critic_is_enabled_but_never_proposes():
    raw = _CriticEnv()
    facade = LiberoEnvFacade(raw, meta={})

    observations, _rewards, _terminated, _truncated, info = (
        facade.critic_chunk_step(
            np.zeros((4, 7), dtype=np.float32),
            critic_rules=[],
            interrupt_on_proposal=True,
            return_all_frames=True,
        )
    )

    assert raw.steps == 4
    assert len(observations) == 4
    assert info["critic_rule_count"] == 0
    assert info["critic_proposals"] == []
    assert [row["proposal_rule_ids"] for row in info["step_records"]] == [
        [],
        [],
        [],
        [],
    ]


def test_active_critic_checks_every_step_and_interrupts_at_first_proposal():
    raw = _CriticEnv()
    facade = LiberoEnvFacade(raw, meta={})

    observations, _rewards, _terminated, _truncated, info = (
        facade.critic_chunk_step(
            np.zeros((5, 7), dtype=np.float32),
            critic_rules=[_critic_rule()],
            interrupt_on_proposal=True,
            return_all_frames=True,
        )
    )

    assert raw.steps == 2
    assert len(observations) == 2
    assert info["executed_horizon"] == 2
    assert [row["step_index"] for row in info["step_records"]] == [1, 2]
    assert [row["rule_id"] for row in info["critic_proposals"]] == [
        "critic-eef-z"
    ]
    assert len(facade.audit_trace()) == 2


def test_critic_configuration_is_frozen_for_episode():
    raw = _CriticEnv()
    facade = LiberoEnvFacade(raw, meta={})
    facade.critic_chunk_step(
        np.zeros((1, 7), dtype=np.float32),
        critic_rules=[],
    )

    with pytest.raises(ValueError, match="cannot change within one LIBERO episode"):
        facade.critic_chunk_step(
            np.zeros((1, 7), dtype=np.float32),
            critic_rules=[_critic_rule()],
        )
