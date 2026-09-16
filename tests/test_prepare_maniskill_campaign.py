# Copyright (c) 2026 Zetta Contributors
"""Immutable scheduling, command binding and native-reset gate semantics."""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import pytest

from robots.maniskill.contracts import TaskContract
from robots.maniskill.run_rollout import build_parser as rollout_parser
from rollout_runtime.backends.rlinf_maniskill import ManiskillEnvConfig
from scripts.evolution.prepare_maniskill_campaign import build_parser, prepare
from zetta.evolution.gating import _same_physical_reset
from zetta.evolution.jsonio import (
    atomic_write_json,
    file_sha256,
    read_json,
)
from zetta.evolution.models import CampaignManifest


def args_for(tmp_path):
    config = ManiskillEnvConfig(
        robot_uids="panda",
        goal_visibility="visible",
        return_all_frames=True,
        extra_init_params={"enhanced_determinism": True, "reconfiguration_freq": 1},
    )
    config_path = tmp_path / "env.json"
    atomic_write_json(config_path, dataclasses.asdict(config))
    task_path = tmp_path / "task.json"
    task = TaskContract(
        policy_model_version="test-checkpoint-v1",
        checkpoint_sha256="a" * 64,
        norm_stats_sha256="b" * 64,
    )
    atomic_write_json(task_path, task.as_dict())
    return build_parser().parse_args(
        [
            "--output-root",
            str(tmp_path / "campaign"),
            "--campaign-id",
            "maniskill-test",
            "--runtime-url",
            "http://localhost:18730",
            "--env-config",
            str(config_path),
            "--code-commit",
            "a" * 40,
            "--task-contract",
            str(task_path),
            "--master-seed",
            "123",
        ]
    )


def test_preregistration_and_executable_bundle_command(tmp_path, monkeypatch):
    monkeypatch.setenv("ROLLOUT_RUNTIME_TOKEN", "must-not-be-persisted")
    args = args_for(tmp_path)
    preregistration = prepare(args)
    output = Path(args.output_root)
    manifest = CampaignManifest.from_dict(read_json(output / "manifest.json"))
    assert preregistration["manifest_sha256"] == manifest.sha256
    assert manifest.heldout_seeds == tuple(range(1, 21))
    assert len(manifest.rollout_seeds) == 50
    assert not set(manifest.rollout_seeds) & set(manifest.heldout_seeds)
    assert manifest.runtime["task_contract"]["language"]
    assert manifest.runtime["task_contract"]["suite"] == "maniskill"
    assert manifest.maximum_logical_slots == 1
    assert manifest.safety_layer.action_contract.startswith("maniskill_")
    assert "must-not-be-persisted" not in (output / "manifest.json").read_text()
    command = manifest.runtime["rollout_command"]
    assert command == manifest.runtime["same_seed_gate_rollout_command"]
    values = {
        "seed": "1001",
        "policy_rng": "42",
        "task": manifest.task,
        "logical_id": "one",
        "generation": "0",
        "attempt_index": "0",
        "bundle_file": "none",
        "bundle_sha256": "none",
        "output_dir": "/tmp/episode",
        "result_file": "/tmp/episode.json",
    }
    parsed = rollout_parser().parse_args(
        [value.format_map(values) for value in command[2:]]
    )
    assert parsed.bundle == "none" and parsed.capture_frames
    assert parsed.env_config_sha256 == file_sha256(output / "env-config.json")
    with pytest.raises(FileExistsError):
        prepare(args)


@pytest.mark.parametrize(
    "field,value",
    [
        ("generation", 1),
        ("execute_horizon", 65),
        ("population_size", 5),
        ("runtime_url", "http://user:secret@host"),
    ],
)
def test_invalid_campaign_rejected(tmp_path, field, value):
    args = args_for(tmp_path)
    setattr(args, field, value)
    with pytest.raises(ValueError):
        prepare(args)


def test_campaign_rejects_reusing_physics_contact_history(tmp_path):
    args = args_for(tmp_path)
    config = read_json(args.env_config)
    del config["extra_init_params"]["reconfiguration_freq"]
    atomic_write_json(args.env_config, config, overwrite=True)
    with pytest.raises(ValueError, match="reconfiguration_freq=1"):
        prepare(args)


def test_rollout_entrypoint_bootstraps_frozen_repository(tmp_path):
    shadow = tmp_path / "old-install" / "robots"
    shadow.mkdir(parents=True)
    (shadow / "__init__.py").write_text("raise RuntimeError('wrong source tree')\n")
    script = Path(__file__).resolve().parents[1] / "robots/maniskill/run_rollout.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(shadow.parent)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--bundle-sha256" in result.stdout


def test_pairing_requires_complete_simulator_state():
    identity = {
        "comparison_contract": "maniskill_initial_state_sha256_v1",
        "state_sha256": "robot_only",
        "initial_state_sha256": "a" * 64,
        "scenario_sha256": "b" * 64,
        "instruction_sha256": "c" * 64,
        "task_contract_sha256": "d" * 64,
    }
    assert _same_physical_reset(identity, dict(identity)) == (True, False)
    for field in (
        "initial_state_sha256",
        "scenario_sha256",
        "instruction_sha256",
        "task_contract_sha256",
    ):
        assert _same_physical_reset(identity, {**identity, field: "f" * 64})[0] is False
    with pytest.raises(ValueError, match="contracts differ"):
        _same_physical_reset(identity, {"state_sha256": "robot_only"})
    with pytest.raises(ValueError, match="invalid ManiSkill"):
        _same_physical_reset(identity, {**identity, "initial_state_sha256": "unknown"})
