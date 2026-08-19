# Copyright (c) 2026 Zetta Contributors
"""Auditable physical-GPU isolation for LIBERO environment rollouts."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

LIBERO_DEVICE_ISOLATION_CONTRACT = "libero_env_vla_physical_gpu_isolation_v1"


def parse_physical_gpus(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        tokens = [token.strip() for token in value.split(",") if token.strip()]
        if not tokens:
            raise ValueError("environment GPU list must not be empty")
        if any(not token.isdecimal() for token in tokens):
            raise ValueError("physical GPU identifiers must be non-negative integers")
        devices = tuple(int(token) for token in tokens)
    else:
        devices = tuple(int(device) for device in value)
    if not devices or any(device < 0 for device in devices):
        raise ValueError("physical GPU identifiers must be non-negative integers")
    if len(set(devices)) != len(devices):
        raise ValueError("environment GPU list contains duplicates")
    return devices


def preregister_device_contract(
    *, environment_gpus: str | Sequence[int], vla_gpu: int
) -> dict[str, object]:
    allowed = parse_physical_gpus(environment_gpus)
    vla = int(vla_gpu)
    if vla < 0:
        raise ValueError("VLA physical GPU must be a non-negative integer")
    if vla in allowed:
        raise ValueError(
            "LIBERO environment and VLA physical GPUs must be isolated; "
            f"VLA GPU {vla} appears in environment GPUs {list(allowed)}"
        )
    return {
        "schema_version": 1,
        "contract": LIBERO_DEVICE_ISOLATION_CONTRACT,
        "environment_gpus": list(allowed),
        "vla_gpu": vla,
        "same_gpu_forbidden": True,
    }


def describe_runtime_devices(
    *,
    default_environment_gpu: int,
    allowed_environment_gpus: str | Sequence[int],
    vla_gpu: int,
    environment: Mapping[str, str],
    vla_endpoint: str,
) -> dict[str, object]:
    contract = preregister_device_contract(
        environment_gpus=allowed_environment_gpus,
        vla_gpu=vla_gpu,
    )
    allowed = tuple(int(device) for device in contract["environment_gpus"])
    fallback = int(default_environment_gpu)
    raw_worker_gpu = environment.get("ZETTA_LIBERO_GPU")
    actual = int(raw_worker_gpu) if raw_worker_gpu is not None else fallback
    violations: list[str] = []
    if fallback not in allowed:
        violations.append(
            f"fallback environment GPU {fallback} is not preregistered in {list(allowed)}"
        )
    if actual not in allowed:
        violations.append(
            f"runtime environment GPU {actual} is not preregistered in {list(allowed)}"
        )
    if actual == int(vla_gpu):
        violations.append(
            f"runtime environment GPU {actual} equals VLA GPU {int(vla_gpu)}"
        )
    return {
        **contract,
        "environment_gpu": actual,
        "environment_gpu_source": (
            "ZETTA_LIBERO_GPU" if raw_worker_gpu is not None else "--gpu"
        ),
        "default_environment_gpu": fallback,
        "vla_endpoint": str(vla_endpoint),
        "isolation_valid": not violations,
        "violations": violations,
    }


def require_isolated_runtime_devices(assignment: Mapping[str, object]) -> None:
    violations = assignment.get("violations")
    if violations:
        detail = "; ".join(str(value) for value in violations)
        raise RuntimeError(f"LIBERO runtime device isolation failed: {detail}")


def vla_runtime_info(*, backend: str) -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    physical_gpu = int(visible) if visible is not None and visible.isdecimal() else None
    return {
        "schema_version": 1,
        "backend": backend,
        "cuda_visible_devices": visible,
        "physical_gpu": physical_gpu,
    }


def attach_vla_runtime_verification(
    assignment: dict[str, object], runtime_info: Mapping[str, object]
) -> None:
    reported = runtime_info.get("physical_gpu")
    assignment["vla_runtime_info"] = dict(runtime_info)
    if reported is None:
        assignment["vla_gpu_verification"] = "server_device_not_detectable"
        return
    reported_gpu = int(reported)
    assignment["vla_gpu_verification"] = "server_reported"
    if reported_gpu != int(assignment["vla_gpu"]):
        violations = assignment.setdefault("violations", [])
        if not isinstance(violations, list):
            raise TypeError("runtime device violations must be a list")
        violations.append(
            f"VLA server reports physical GPU {reported_gpu}, "
            f"but campaign preregistered GPU {assignment['vla_gpu']}"
        )
        assignment["isolation_valid"] = False
