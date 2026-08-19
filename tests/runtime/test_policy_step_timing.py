"""Regression tests for ``policy_step``'s internal timing instrumentation.

The client-side methodology has a known limitation: ``policy_step`` is an
atomic Gateway call (observe -> infer -> chunk_step in a single RPC), so the
client cannot see the boundary between the two internal phases. This test
verifies that server-side instrumentation writes both phase durations into
``StepResult.info`` (a family-private info dict that can be freely extended
without changing the external RPC contract):

- ``info["inference_latency_s"]``: duration of the ``request_inference`` phase.
- ``info["env_step_latency_s"]``: duration of the ``_step_slot`` (chunk_step) phase.

Both fields must be non-negative floats, and their sum should be <= the
overall ``policy_step`` duration measured by the client (it is fine for the
sum to be slightly smaller than the overall duration due to fixed overhead,
but it must not be much larger -- that would indicate a timing boundary bug).
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from rollout_runtime.api.messages import PolicyRequest, ResetSpec
from rollout_runtime.api.result import unwrap
from rollout_runtime.launch.local import LocalRuntime
from tests.runtime.conftest import open_sessions

POLICY = PolicyRequest(policy_id="fake")


async def test_policy_step_info_reports_inference_and_env_step_latency(
    local_runtime: LocalRuntime, fake_env_spec: Any
) -> None:
    """The ``info`` returned by ``policy_step`` must include separate durations for the inference and env.step phases."""

    gateway = local_runtime.gateway
    spec = fake_env_spec(episode_length=12)
    (session_id,) = await open_sessions(local_runtime, spec)
    unwrap((await gateway.reset([session_id], ResetSpec(seed=7)))[0])

    started = time.perf_counter()
    step = unwrap((await gateway.policy_step([session_id], POLICY))[0])
    wall_elapsed = time.perf_counter() - started

    assert "inference_latency_s" in step.info
    assert "env_step_latency_s" in step.info
    inference_latency = step.info["inference_latency_s"]
    env_step_latency = step.info["env_step_latency_s"]

    assert isinstance(inference_latency, float)
    assert isinstance(env_step_latency, float)
    assert inference_latency >= 0.0
    assert env_step_latency >= 0.0
    # Fixed overhead (scheduling/serialization, etc.) is allowed, but the sum of
    # both phases must not far exceed the overall duration measured by the
    # client (exceeding it by more than 2x would indicate a timing boundary
    # bug, e.g. counting lock-wait time as well).
    assert inference_latency + env_step_latency <= wall_elapsed * 2 + 0.01

    # Pre-existing fields (model_version/policy_id) must remain unchanged: the new instrumentation must not crowd out existing info.
    assert step.info["model_version"] == "fake-v1"
    assert step.info["policy_id"] == "fake"


async def test_policy_infer_info_has_no_env_step_latency(
    local_runtime: LocalRuntime, fake_env_spec: Any
) -> None:
    """``policy_infer`` (inference only, no execution) must not report ``env_step_latency_s``.

    This case ensures the new instrumentation is only added to the merged
    ``policy_step`` path, and does not affect ``policy_infer``/``action_step``,
    which are already naturally separate entry points (they don't need this
    instrumentation since the client can already time them independently).
    """

    gateway = local_runtime.gateway
    spec = fake_env_spec(episode_length=12)
    (session_id,) = await open_sessions(local_runtime, spec)
    unwrap((await gateway.reset([session_id], ResetSpec(seed=7)))[0])

    result = unwrap((await gateway.policy_infer([session_id], POLICY))[0])
    assert "env_step_latency_s" not in result.info
