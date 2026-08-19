# Copyright (c) 2026 Zetta Contributors
from __future__ import annotations

import pytest

from robots.libero.run_evolution_rollout import _record_vla_runtime_verification
from robots.libero.runtime_devices import (
    attach_vla_runtime_verification,
    describe_runtime_devices,
    parse_physical_gpus,
    preregister_device_contract,
    require_isolated_runtime_devices,
    vla_runtime_info,
)
from zetta.utils.rpc import RpcError


def test_preregistered_libero_devices_require_physical_gpu_isolation() -> None:
    assert parse_physical_gpus("6,5") == (6, 5)
    contract = preregister_device_contract(environment_gpus=(5, 6), vla_gpu=7)
    assert contract["environment_gpus"] == [5, 6]
    assert contract["vla_gpu"] == 7
    with pytest.raises(ValueError, match="must be isolated"):
        preregister_device_contract(environment_gpus=(6, 7), vla_gpu=7)


def test_runtime_worker_gpu_must_be_preregistered_and_isolated() -> None:
    valid = describe_runtime_devices(
        default_environment_gpu=5,
        allowed_environment_gpus=(5, 6),
        vla_gpu=7,
        environment={"ZETTA_LIBERO_GPU": "6"},
        vla_endpoint="http://127.0.0.1:18811",
    )
    require_isolated_runtime_devices(valid)
    assert valid["environment_gpu"] == 6
    assert valid["environment_gpu_source"] == "ZETTA_LIBERO_GPU"

    invalid = describe_runtime_devices(
        default_environment_gpu=5,
        allowed_environment_gpus=(5, 6),
        vla_gpu=7,
        environment={"ZETTA_LIBERO_GPU": "4"},
        vla_endpoint="http://127.0.0.1:18811",
    )
    with pytest.raises(RuntimeError, match="not preregistered"):
        require_isolated_runtime_devices(invalid)


def test_reported_vla_gpu_must_match_preregistration() -> None:
    assignment = describe_runtime_devices(
        default_environment_gpu=6,
        allowed_environment_gpus=(6,),
        vla_gpu=7,
        environment={},
        vla_endpoint="http://127.0.0.1:18811",
    )
    attach_vla_runtime_verification(assignment, {"physical_gpu": 6})
    assert assignment["vla_gpu_verification"] == "server_reported"
    with pytest.raises(RuntimeError, match="server reports physical GPU 6"):
        require_isolated_runtime_devices(assignment)


def test_vla_runtime_info_reports_single_visible_physical_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    assert vla_runtime_info(backend="pi0.5") == {
        "schema_version": 1,
        "backend": "pi0.5",
        "cuda_visible_devices": "7",
        "physical_gpu": 7,
    }


def test_busy_legacy_vla_probe_does_not_invalidate_isolated_assignment() -> None:
    class BusyClient:
        def call(self, method: str, *, timeout_s: float) -> object:
            raise RpcError(method, "HTTP request failed: timed out")

    assignment = describe_runtime_devices(
        default_environment_gpu=6,
        allowed_environment_gpus=(6,),
        vla_gpu=7,
        environment={},
        vla_endpoint="http://127.0.0.1:18811",
    )
    _record_vla_runtime_verification(BusyClient(), assignment)  # type: ignore[arg-type]
    require_isolated_runtime_devices(assignment)
    assert assignment["vla_gpu_verification"] == "server_probe_unavailable"
    assert assignment["vla_runtime_probe_error"] == {
        "error_type": "RpcError",
        "message": "runtime_info: HTTP request failed: timed out",
    }
