# Copyright (c) 2026 Zetta Contributors
"""Immutable scheduling, command binding and native-reset gate semantics."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path

import pytest

from robots.geniesim.run_rollout import build_parser as rollout_parser
from scripts.evolution.prepare_geniesim_campaign import build_parser, prepare
from zetta.envs.geniesim.config import GenieSimConfig
from zetta.evolution.gating import _same_physical_reset
from zetta.evolution.jsonio import (
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    read_json,
)
from zetta.evolution.models import CampaignManifest


def args_for(tmp_path):
    config = GenieSimConfig(
        python_executable=sys.executable,
        geniesim_root=str(tmp_path),
        assets_root=str(tmp_path),
        output_root=str(tmp_path / "native"),
    )
    config_path = tmp_path / "env.json"
    atomic_write_json(config_path, config.to_dict())
    return build_parser().parse_args(
        [
            "--output-root",
            str(tmp_path / "campaign"),
            "--campaign-id",
            "geniesim-test",
            "--runtime-url",
            "http://localhost:18730",
            "--env-config",
            str(config_path),
            "--code-commit",
            "a" * 40,
            "--policy-model-version",
            "g2-checkpoint-v1",
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
    assert manifest.maximum_logical_slots == 1
    assert manifest.safety_layer.action_contract.startswith("geniesim_")
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


def test_rollout_entrypoint_bootstraps_frozen_repository(tmp_path):
    shadow = tmp_path / "old-install" / "robots"
    shadow.mkdir(parents=True)
    (shadow / "__init__.py").write_text("raise RuntimeError('wrong source tree')\n")
    script = Path(__file__).resolve().parents[1] / "robots/geniesim/run_rollout.py"
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


def reset_identity(offset=0.0):
    joints = [offset] * 21
    return {
        "comparison_contract": "geniesim_g2_joint_reset_v1",
        "state_sha256": canonical_sha256(joints),
        "joint_positions": joints,
        "scenario_sha256": "a" * 64,
        "camera_sha256": {"head": "same"},
    }


def test_geniesim_pairing_bounds_joint_drift_and_audits_cameras():
    parent = reset_identity()
    candidate = reset_identity(0.019)
    candidate["camera_sha256"] = {"head": "different"}
    assert _same_physical_reset(candidate, parent) == (True, True)
    assert _same_physical_reset(reset_identity(0.021), parent)[0] is False
    candidate["scenario_sha256"] = "b" * 64
    assert _same_physical_reset(candidate, parent)[0] is False
    candidate = copy.deepcopy(parent)
    candidate["joint_positions"][0] = float("nan")
    with pytest.raises(ValueError):
        _same_physical_reset(candidate, parent)
    with pytest.raises(ValueError):
        _same_physical_reset(parent, {"state_sha256": parent["state_sha256"]})
    assert (
        _same_physical_reset({"state_sha256": "a"}, {"state_sha256": "b"})[0] is False
    )
