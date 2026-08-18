# Copyright (c) 2026 RPent Contributors
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "deployment" / "prepare_experiment.sh"


def test_prepare_experiment_shell_is_valid() -> None:
    bash = shutil.which("bash")
    if bash is None or os.name == "nt":
        return
    subprocess.run([bash, "-n", str(SCRIPT)], check=True)


def test_worker_stability_deadline_is_nounset_safe() -> None:
    bash = shutil.which("bash")
    if bash is None or os.name == "nt":
        return
    text = SCRIPT.read_text(encoding="utf-8")
    function = text.split("wait_process_stable() {", 1)[1].split("\n}", 1)[0]
    program = f"set -u\nwait_process_stable() {{{function}\n}}\nwait_process_stable worker 0\n"
    subprocess.run([bash, "-c", program], check=True)


def test_prepare_experiment_covers_complete_service_topologies() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    for token in (
        "start_provider",
        "start_libero_vla",
        "start_groot",
        "start_robocasa_farm",
        "run_provider_probes",
        "run_libero_smoke",
        "run_robocasa_smoke",
        "prepare_libero_campaign.py",
        "prepare_robocasa_campaign.py",
        "rpent.evolution.cli worker",
        "run_experiment.sh",
        "stop_experiment.sh",
        "--environment-ready-manifest",
        "--slot-broker-root",
    ):
        assert token in text
    assert "write_broker_client_env" in text
    assert "broker_client_file=${state_root}/broker-client.env" in text
    assert "sanitize_provider_config_for_broker_client" in text
    assert 'source "${broker_client_file}"' in text
    assert 'source "${provider_file}"' not in text.split("write_launchers() {", 1)[1]
    assert '--expected-checkpoint-sha256 "$GROOT_CHECKPOINT_SHA256"' in text
    assert '"${ROBOCASA_ACTIONS_PER_CHUNK:-16}"' in text
    assert '"${PROVIDER_PROBE_WIRE_API:-responses}"' in text


def test_service_shutdown_refuses_unowned_pids() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "RPENT_EXPERIMENT_SERVICE=" in text
    assert "refusing to stop unowned pid" in text
    assert "kill -TERM -- \"-$pid\"" in text
    assert '[[ "$state" != Z && "$state" != X ]]' in text
    assert "for _ in $(seq 1 40)" in text
    assert "acquire_prepare_lock" in text
    assert "another preparation is active" in text
    assert "ensure_port_available" in text
    assert "lost ownership" in text
    assert "verify_prepared_services" in text
    assert "wait_process_stable worker" in text
    assert 'sha256sum "$config_file" "$provider_file"' in text
    assert text.index("trap cleanup_failure EXIT") < text.index("config_sha=$({")


def test_examples_are_secret_free_and_match_script_contract() -> None:
    for family in ("libero", "robocasa"):
        path = ROOT / "deployment" / "experiments" / f"{family}.env.example"
        text = path.read_text(encoding="utf-8")
        assert f"EXPERIMENT_FAMILY={family}" in text
        assert "PROVIDER_ENV_FILE=/abs/path/to/providers.env" in text
        assert "sk-" not in text
        assert "/mnt/" not in text
        assert "C:\\" not in text
    provider = (
        ROOT / "deployment" / "experiments" / "providers.env.example"
    ).read_text(encoding="utf-8")
    assert "<provider-api-key>" in provider
    assert "RPENT_API_PROVIDERS" in provider
    assert "sk-" not in provider
    assert "/mnt/" not in provider


def test_robocasa_uses_separate_simulator_and_groot_pythons() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    example = (
        ROOT / "deployment" / "experiments" / "robocasa.env.example"
    ).read_text(encoding="utf-8")
    assert "ROBOCASA_PYTHON=" in example
    assert "GROOT_PYTHON=" in example
    assert "GROOT_CHECKPOINT_SHA256=" in example
    assert "ROBOCASA_ACTIONS_PER_CHUNK=16" in example
    assert 'groot_python=$(resolve_python "${GROOT_PYTHON:-$common_python}")' in script
    assert 'robocasa_python=$(resolve_python "${ROBOCASA_PYTHON:-$common_python}")' in script
    family_block = script.split('if [[ "$family" == libero ]]; then', 1)[1]
    assert "export RPENT_LIBERO_GPU=" in family_block
    assert "local campaign_python=$libero_python" not in script


def test_python_resolution_preserves_virtualenv_symlink() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    function = text.split("resolve_python() {", 1)[1].split("\n}", 1)[0]
    assert 'resolved=$value' in function
    assert 'realpath "$value"' not in function


def test_robocasa_formal_chunk_default_is_sixteen() -> None:
    source = (
        ROOT / "scripts" / "evolution" / "prepare_robocasa_campaign.py"
    ).read_text(encoding="utf-8")
    assert 'parser.add_argument("--actions-per-chunk", type=int, default=16)' in source
