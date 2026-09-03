# Copyright (c) 2026 Zetta Contributors
"""The RoboTwin campaign preparer.

Most of what a preparer does is shared across environments; what is worth
pinning here is the one thing RoboTwin had to change and the reasons it had to
be recorded.

RoboTwin selects its scene from the seed outright, and only its curated
``success_seeds`` are known to be solvable, so the campaign draws from that list
rather than from a dense integer range. The protocol's shape is unchanged -- a
fixed, disjoint, pre-committed held-out block -- but the values are no longer
the literal ``1..20``, so the provenance has to be in the artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "evolution"))

from prepare_robotwin_campaign import (  # noqa: E402
    build_parser,
    load_success_seeds,
    prepare,
)

COMMIT = "a" * 40
REPO_ROOT = Path(__file__).resolve().parents[1]


def _seeds_file(
    tmp_path: Path,
    count: int = 120,
    task: str = "adjust_bottle",
    name: str = "eval_seeds.json",
) -> Path:
    """Write a curated success-seed file in RLinf's shape.

    Args:
        tmp_path: Test directory.
        count: How many seeds to publish.
        task: The task name.
        name: File name, so a test can hold two different populations at once.

    Returns:
        The file path.
    """
    payload = {
        task: {
            "task_name": task,
            "success_seeds": [100100000 + index for index in range(count)],
        }
    }
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _args(tmp_path: Path, **overrides) -> argparse.Namespace:
    """Build a valid argument set.

    Args:
        tmp_path: Test directory.
        **overrides: Attribute overrides.

    Returns:
        The namespace.
    """
    args = build_parser().parse_args(
        [
            "--output-root",
            str(tmp_path / "g0000"),
            "--campaign-id",
            "robotwin-adjust-bottle-g0000",
            "--repository-root",
            str(REPO_ROOT),
            "--runtime-python",
            sys.executable,
            "--code-commit",
            COMMIT,
            "--seeds-path",
            str(_seeds_file(tmp_path)),
            "--assets-path",
            "/path/to/RoboTwin",
            "--policy-id",
            "pi05_aloha_robotwin",
            "--master-seed",
            "2026090201",
            "--runtime-url",
            "http://127.0.0.1:18730",
        ]
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


# ------------------------------------------------------------- seed source


def test_success_seeds_are_read_per_task(tmp_path: Path) -> None:
    """The population is the task's own solvable set, not a shared one."""
    seeds = load_success_seeds(_seeds_file(tmp_path), "adjust_bottle")
    assert len(seeds) == 120
    assert seeds[0] == 100100000


def test_an_unknown_task_fails_closed(tmp_path: Path) -> None:
    """Silently falling back to a dense range would break the held-out gate."""
    with pytest.raises(ValueError, match="no entry for RoboTwin task"):
        load_success_seeds(_seeds_file(tmp_path), "lift_pot")


def test_duplicate_seeds_are_collapsed(tmp_path: Path) -> None:
    """A duplicated seed would make the held-out block smaller than declared."""
    path = tmp_path / "dup.json"
    path.write_text(json.dumps({"t": {"success_seeds": [7, 7, 8]}}), encoding="utf-8")
    assert load_success_seeds(path, "t") == (7, 8)


# ------------------------------------------------------------- preparation


def test_heldout_block_comes_from_the_curated_list(tmp_path: Path) -> None:
    """The literal 1..20 is not usable here; the substitution must be exact."""
    prereg = prepare(_args(tmp_path))
    assert len(prereg["heldout_seeds"]) == 20
    assert list(prereg["heldout_seeds"]) == [100100000 + i for i in range(20)]
    # The protocol's shape survives: disjoint, and the dev block is the
    # requested size.
    assert len(prereg["rollout_seeds"]) == 50
    assert not set(prereg["heldout_seeds"]) & set(prereg["rollout_seeds"])


def test_seed_provenance_is_recorded_in_both_artifacts(tmp_path: Path) -> None:
    """A reviewer must see why the seeds are not 1..20 without reading code."""
    args = _args(tmp_path)
    prereg = prepare(args)
    provenance = prereg["seed_provenance"]
    assert provenance["population_kind"] == "robotwin_curated_success_seeds"
    assert provenance["population_size"] == 120
    assert len(provenance["source_file_sha256"]) == 64
    assert "not a gate" in provenance["rationale"]

    manifest = json.loads((Path(args.output_root) / "manifest.json").read_text())
    assert manifest["runtime"]["seed_provenance"] == provenance


def test_manifest_records_the_chunk_evidence_unit(tmp_path: Path) -> None:
    """A chunk-granular diagnosis must not be compared with a per-step one."""
    args = _args(tmp_path, execute_horizon=25)
    prereg = prepare(args)
    assert prereg["evidence_granularity"] == "chunk"
    manifest = json.loads((Path(args.output_root) / "manifest.json").read_text())
    assert manifest["runtime"]["evidence_granularity"] == "chunk"
    assert manifest["runtime"]["dwell_semantics"]["sim_steps_per_chunk"] == 25


def test_frozen_command_carries_no_bearer_token(tmp_path: Path) -> None:
    """The command lands in the manifest; a token there would be a leak."""
    args = _args(tmp_path)
    prepare(args)
    manifest = json.loads((Path(args.output_root) / "manifest.json").read_text())
    command = manifest["runtime"]["rollout_command"]
    assert "--runtime-token" not in command
    # The queue substitutes these; a preparer that resolved them early would
    # freeze one episode's identity into every episode.
    assert "{seed}" in command
    assert "{result_file}" in command
    # The Gateway owns env-slot arbitration under the Runtime, so the campaign
    # passes the shared endpoint directly instead of leasing a slot.
    assert "{env_endpoint}" not in command
    assert "http://127.0.0.1:18730" in command
    # Without frame capture the episodes cannot reach Stage 1 at all.
    assert "--capture-frames" in command


def test_catalog_artifact_surfaces_the_arm_requirement(tmp_path: Path) -> None:
    """The arm-scoped set is the expensive-to-change part of the catalog."""
    args = _args(tmp_path)
    prepare(args)
    catalog = json.loads((Path(args.output_root) / "tool-catalog.json").read_text())
    assert catalog["arm_scoped_tools"], "a bimanual binding must scope some tools"
    assert catalog["task"] == "adjust_bottle"
    assert len(catalog["digest"]) == 64


def test_role1_contract_states_the_bimanual_rules(tmp_path: Path) -> None:
    """The prompt contract is hashed into the preregistration, so pin it."""
    args = _args(tmp_path)
    prereg = prepare(args)
    contract = json.loads(
        (Path(args.output_root) / "prompt-contract.json").read_text()
    )["role1"]
    assert "names no arm" in contract
    assert "zero vector" in contract
    assert "chunk-granular" in contract
    assert len(prereg["prompt_sha256"]) == 64


def test_too_few_curated_seeds_fails_closed(tmp_path: Path) -> None:
    """A short list must not silently yield a smaller held-out block."""
    short = _seeds_file(tmp_path, count=10, name="short_seeds.json")
    args = _args(tmp_path, seeds_path=str(short))
    with pytest.raises(ValueError, match="curated success"):
        prepare(args)


def test_nonzero_generation_requires_a_parent(tmp_path: Path) -> None:
    """A generation with no parent bundle has nothing to have evolved from."""
    with pytest.raises(ValueError, match="requires --parent-bundle"):
        prepare(_args(tmp_path, generation=1))


def test_output_root_is_never_reused(tmp_path: Path) -> None:
    """A campaign root is immutable; writing into an existing one would edit it."""
    args = _args(tmp_path)
    prepare(args)
    with pytest.raises(FileExistsError):
        prepare(_args(tmp_path))


def test_frozen_interpreter_keeps_the_venv(tmp_path: Path) -> None:
    """A venv is only active when invoked through its own ``bin/python``.

    That path is a symlink to the base interpreter, so resolving it freezes the
    base interpreter into the manifest and every rollout dies at ``import
    numpy`` with the venv's site-packages nowhere on the path. Observed on a
    real campaign run before this was fixed.
    """
    real = tmp_path / "base" / "bin"
    real.mkdir(parents=True)
    interpreter = real / "python3.11"
    interpreter.write_text("#!/bin/sh\n")
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    link = venv_bin / "python"
    link.symlink_to(interpreter)

    args = _args(tmp_path, runtime_python=str(link))
    prepare(args)
    command = json.loads((Path(args.output_root) / "manifest.json").read_text())[
        "runtime"
    ]["rollout_command"]
    assert command[0] == str(link)
    assert "base/bin" not in command[0]


# --- paired-gate command contract -------------------------------------------
#
# Observed on a real campaign: Gen0 rollouts, Cluster, Diagnose and Stage2 all
# completed, and the run then died entering the same-seed gate with
# "gate rollout command omits frozen fields: ['bundle_file']".  campaign.py only
# enforces the placeholder from Gen1 on, so nothing catches it earlier.


def test_frozen_command_satisfies_the_paired_gate_contract(tmp_path: Path) -> None:
    """Every field gate_runner._command_template requires must be present."""
    args = _args(tmp_path)
    prepare(args)
    manifest = json.loads((Path(args.output_root) / "manifest.json").read_text())
    joined = "\n".join(manifest["runtime"]["rollout_command"])

    for field in (
        "seed",
        "policy_rng",
        "logical_id",
        "attempt_index",
        "bundle_sha256",
        "output_dir",
        "result_file",
    ):
        assert f"{{{field}}}" in joined, field
    # The gate accepts either spelling; the campaign substitutes both names.
    assert "{bundle_file}" in joined or "{bundle}" in joined
