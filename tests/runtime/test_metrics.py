"""Metrics completeness for ``GatewayMetrics`` and ``/metrics`` exposure.

Historical context: ``GatewayMetrics`` has six metrics, but only ``record_operation`` /
``record_error`` were actually being written; the two gauges ``sessions`` /
``inflight_operations`` were never written, ``record_payload`` had no call site, and
``/metrics`` was never exposed.

This covers three things:

1. **State-type metrics are actually sampled**: session state distribution, in-flight,
   queue depth, transport bytes, payload bytes, batch utilization, rank health
   distribution, epoch;
2. **Delta conversion for cumulative quantities**: the source is a monotonic counter
   while the prometheus ``Counter`` only supports ``inc``, so ``observe_*`` must only
   add the delta, and must **not go backwards** when the source is reset;
3. ``/metrics`` actually exposes the prometheus text format, and **requires
   authentication**.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import pytest_asyncio

from rollout_runtime.api.messages import PolicyRequest, ResetSpec
from rollout_runtime.api.result import Err
from rollout_runtime.gateway.metrics import GatewayMetrics, prometheus_available
from rollout_runtime.serve.app import ServeLimits
from rollout_runtime.serve.server import ServeOptions, build_served_runtime

from .conftest import open_sessions

TOKEN = "metrics-token"


@pytest_asyncio.fixture(loop_scope="function")
async def served_metrics() -> AsyncIterator[Any]:
    """Start a served runtime with authentication (fake backend).

    Yields:
        ``(ServedRuntime, httpx.AsyncClient)``.
    """
    runtime = await build_served_runtime(
        ServeOptions(
            config="local_fake",
            host="127.0.0.1",
            gateway_epoch=4242,
            limits=ServeLimits(max_pool_size=2),
        ),
        environ={"RR_AUTH_TOKEN": f"metrics:{TOKEN}"},
    )
    transport = httpx.ASGITransport(app=runtime.app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://metrics.test"
    ) as client:
        yield runtime, client
    await runtime.aclose()


def sample(metrics: GatewayMetrics, name: str, **labels: str) -> float | None:
    """Read one metric sample.

    Args:
        metrics: The metrics collection.
        name: Metric name (including the ``rr_`` prefix).
        **labels: Labels.

    Returns:
        The sample value; ``None`` if the sample does not exist.
    """
    assert metrics.registry is not None
    return metrics.registry.get_sample_value(name, labels or None)


def test_prometheus_is_available_locally() -> None:
    """The local environment must have ``prometheus_client`` installed (otherwise the assertions below are meaningless)."""
    assert prometheus_available()


def test_counters_only_advance_by_the_delta() -> None:
    """Cumulative quantities ``inc`` by delta; repeated sampling doesn't double-count, and a source reset doesn't go backwards."""
    metrics = GatewayMetrics(namespace="rr", registry=None)
    snapshot = {
        "encoded_count": 3,
        "encoded_bytes": 300,
        "decoded_count": 1,
        "decoded_bytes": 100,
        "oversize_count": 1,
        "oversize_bytes": 900,
    }
    metrics.observe_payload_stats(snapshot)
    metrics.observe_payload_stats(snapshot)  # Sample the same snapshot again
    assert sample(
        metrics, "rr_payload_bytes_total", direction="out", kind="encoded"
    ) == (300.0)
    assert sample(metrics, "rr_payload_events_total", kind="encoded") == 3.0
    assert sample(metrics, "rr_payload_events_total", kind="oversize") == 1.0

    metrics.observe_payload_stats({**snapshot, "encoded_bytes": 500})
    assert sample(
        metrics, "rr_payload_bytes_total", direction="out", kind="encoded"
    ) == (500.0)

    # The source is reset (e.g. the test's ``PayloadStats.reset()``): the counter holds
    # steady and does not go backwards.
    metrics.observe_payload_stats(dict.fromkeys(snapshot, 0))
    assert sample(
        metrics, "rr_payload_bytes_total", direction="out", kind="encoded"
    ) == (500.0)


def test_session_gauge_zeroes_states_that_disappeared() -> None:
    """When a state that once appeared disappears, the gauge must be explicitly set to 0 instead of staying at the old value."""
    metrics = GatewayMetrics(namespace="rr", registry=None)
    metrics.observe_sessions({"ready": 3, "lost": 1})
    assert sample(metrics, "rr_sessions", state="ready") == 3.0
    assert sample(metrics, "rr_sessions", state="lost") == 1.0
    metrics.observe_sessions({"ready": 2})
    assert sample(metrics, "rr_sessions", state="ready") == 2.0
    assert sample(metrics, "rr_sessions", state="lost") == 0.0


def test_scheduler_snapshot_reports_average_batch_size() -> None:
    """Batch utilization = number of batched requests / number of batches (observability for the related optimization work)."""
    metrics = GatewayMetrics(namespace="rr", registry=None)
    metrics.observe_scheduler({"batch_count": 4, "batched_count": 16, "queue_depth": 2})
    assert sample(metrics, "rr_inference_batches_total") == 4.0
    assert sample(metrics, "rr_inference_batched_requests_total") == 16.0
    assert sample(metrics, "rr_inference_batch_size") == 4.0
    assert sample(metrics, "rr_inference_queue_depth") == 2.0


def test_render_returns_prometheus_text() -> None:
    """``render()`` returns ``(body, Content-Type)``."""
    metrics = GatewayMetrics(namespace="rr", registry=None)
    metrics.observe_gateway_epoch(11)
    rendered = metrics.render()
    assert rendered is not None
    body, content_type = rendered
    assert b"rr_gateway_epoch 11.0" in body
    assert "text/plain" in content_type


def test_metrics_degrade_to_noop_without_prometheus(monkeypatch) -> None:
    """When ``prometheus_client`` is missing, degrade to a no-op implementation with the same interface; ``render()`` returns ``None``.

    Args:
        monkeypatch: pytest fixture.
    """
    monkeypatch.setitem(sys.modules, "prometheus_client", None)
    metrics = GatewayMetrics(namespace="rr")
    assert not metrics.enabled
    assert metrics.registry is None
    # All entry points must remain callable (the Gateway must still start in a minimal-dependency environment).
    metrics.record_operation("reset", "succeeded", 0.1)
    metrics.record_error("INTERNAL")
    metrics.record_payload("out", "image", 10)
    metrics.observe_sessions({"ready": 1})
    metrics.observe_inflight(1)
    metrics.observe_rejections({"QUEUE_FULL": 1})
    metrics.observe_queue_depth({0: 1})
    metrics.observe_transport_bytes(sent=1, received=1)
    metrics.observe_payload_stats({"encoded_bytes": 1})
    metrics.observe_scheduler({"batch_count": 1, "batched_count": 1})
    metrics.observe_env_workers(healthy=1, unhealthy=0)
    metrics.observe_late_results(absorbed=0, orphaned=0)
    metrics.observe_gateway_epoch(1)
    assert metrics.render() is None


async def test_refresh_metrics_samples_the_live_gateway(
    local_runtime, fake_env_spec
) -> None:
    """After running a real operation, both state-type and event-type metrics have values.

    Args:
        local_runtime: In-process runtime fixture.
        fake_env_spec: env spec factory.
    """
    gateway = local_runtime.gateway
    metrics = gateway.metrics
    session_ids = await open_sessions(
        local_runtime, fake_env_spec(pool_size=2), count=2
    )
    await gateway.reset(session_ids, ResetSpec(seed=1))
    results = await gateway.policy_step(session_ids, PolicyRequest(policy_id="fake"))
    assert not [item for item in results if isinstance(item, Err)]
    gateway.refresh_metrics()

    assert sample(metrics, "rr_sessions", state="ready") == 2.0
    assert sample(metrics, "rr_env_workers", state="healthy") == 1.0
    assert sample(metrics, "rr_env_workers", state="unhealthy") == 0.0
    assert sample(metrics, "rr_inflight_operations") == 0.0
    assert sample(metrics, "rr_gateway_epoch") == float(gateway.gateway_epoch)
    assert (
        sample(
            metrics,
            "rr_operations_total",
            operation="policy_step",
            outcome="succeeded",
        )
        == 2.0
    )
    latency_count = sample(
        metrics, "rr_operation_latency_seconds_count", operation="policy_step"
    )
    assert latency_count == 2.0
    # Payload bytes come from ``core.payload.stats()`` (a getter function injected by the launch layer).
    encoded = sample(metrics, "rr_payload_bytes_total", direction="out", kind="encoded")
    assert encoded is not None and encoded > 0.0
    # Batch utilization comes from this process's rollout scheduler.
    batches = sample(metrics, "rr_inference_batches_total")
    assert batches is not None and batches >= 1.0

    # After closing the sessions, the gauge should follow suit (proving it's a **sample**, not a one-time write).
    await gateway.close_sessions(session_ids)
    gateway.refresh_metrics()
    assert sample(metrics, "rr_sessions", state="ready") == 0.0
    assert sample(metrics, "rr_sessions", state="closed") == 2.0


async def test_admission_rejections_show_up_in_metrics(
    local_runtime, fake_env_spec
) -> None:
    """Quota rejections show up in ``rr_admission_rejected_total``.

    Args:
        local_runtime: In-process runtime fixture.
        fake_env_spec: env spec factory.
    """
    gateway = local_runtime.gateway
    gateway.admission.config.max_sessions_per_application = 1
    spec = fake_env_spec(pool_size=2)
    await open_sessions(local_runtime, spec, count=1, key_prefix="quota")
    from rollout_runtime.api.messages import CreateSessionRequest

    refused = await gateway.create_sessions(
        [
            CreateSessionRequest(
                application_id="test",
                client_session_key="quota-over",
                env_spec=spec,
                lease_seconds=60.0,
            )
        ]
    )
    assert isinstance(refused[0], Err)
    gateway.refresh_metrics()
    assert (
        sample(gateway.metrics, "rr_admission_rejected_total", code="QUOTA_EXCEEDED")
        == 1.0
    )


async def test_metrics_endpoint_requires_auth_and_returns_text(
    served_metrics,
) -> None:
    """``/metrics`` requires authentication, and returns prometheus text once authenticated.

    Args:
        served_metrics: fixture.
    """
    runtime, client = served_metrics
    assert (await client.get("/metrics")).status_code == 401

    response = await client.get(
        "/metrics", headers={"Authorization": f"Bearer {TOKEN}"}
    )
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    body = response.text
    for family in (
        "rr_operations_total",
        "rr_errors_total",
        "rr_sessions",
        "rr_inflight_operations",
        "rr_admission_rejected_total",
        "rr_command_queue_depth",
        "rr_transport_bytes_total",
        "rr_payload_bytes_total",
        "rr_payload_events_total",
        "rr_operation_latency_seconds",
        "rr_inference_batch_size",
        "rr_env_workers",
        "rr_gateway_epoch",
    ):
        assert f"# HELP {family}" in body, f"missing metric family: {family}"
    assert f"rr_gateway_epoch {float(runtime.epoch)}" in body
    assert 'rr_env_workers{state="healthy"} 1.0' in body


async def test_metrics_endpoint_reflects_http_traffic(served_metrics) -> None:
    """Operations performed over HTTP can be seen in ``/metrics``.

    Args:
        served_metrics: fixture.
    """
    _runtime, client = served_metrics
    headers = {"Authorization": f"Bearer {TOKEN}"}
    from rollout_runtime.api import wire
    from rollout_runtime.api.messages import CreateSessionRequest, EnvSpecMsg

    created = wire.decode_bytes(
        (
            await client.post(
                "/v1/sessions",
                content=wire.encode_bytes(
                    {
                        "requests": [
                            CreateSessionRequest(
                                application_id="",
                                client_session_key="m1",
                                env_spec=EnvSpecMsg(
                                    env_family="fake",
                                    env_config={"chunk_size": 2},
                                    pool_size=1,
                                ),
                                lease_seconds=60.0,
                            )
                        ]
                    }
                ),
                headers=headers,
            )
        ).content
    )
    session_id = created[0].value.session_id
    await client.post(
        "/v1/reset",
        content=wire.encode_bytes(
            {"session_ids": [session_id], "reset_spec": ResetSpec(seed=1)}
        ),
        headers=headers,
    )
    body = (await client.get("/metrics", headers=headers)).text
    assert (
        'rr_operations_total{operation="create_session",outcome="succeeded"} 1.0'
        in (body)
    )
    assert 'rr_operations_total{operation="reset",outcome="succeeded"} 1.0' in body
    assert 'rr_sessions{state="ready"} 1.0' in body


def test_batch_size_gauge_reflects_the_sampling_window_not_the_lifetime() -> None:
    """``rr_inference_batch_size`` must be the mean of the **current window**, not the lifetime mean.

    Independent audit finding: the original implementation used a ratio of cumulative
    values; after 1000 size-1 batches followed by 10 size-8 batches, the gauge only
    moved from 1.00 to 1.07 — meaning production could never reveal a change to the
    batching parameter, which is exactly the number that needs to be watched.
    """
    metrics = GatewayMetrics(namespace="rr", registry=None)
    metrics.observe_scheduler({"batch_count": 1000, "batched_count": 1000})
    assert sample(metrics, "rr_inference_batch_size") == 1.0
    metrics.observe_scheduler({"batch_count": 1010, "batched_count": 1080})
    assert sample(metrics, "rr_inference_batch_size") == 8.0
    # An empty window (no new batches) retains the previous value instead of resetting to 0.
    metrics.observe_scheduler({"batch_count": 1010, "batched_count": 1080})
    assert sample(metrics, "rr_inference_batch_size") == 8.0
    # The counter still only adds the delta.
    assert sample(metrics, "rr_inference_batches_total") == 1010.0
    assert sample(metrics, "rr_inference_batched_requests_total") == 1080.0


def test_oversize_bytes_are_not_double_counted_into_the_out_direction() -> None:
    """``oversize_*`` is a subset of ``encoded_*`` and must never share ``direction="out"`` with it.

    ``core/payload.py`` accumulates both encoded and oversize on the **same** encode
    call, so the earlier `sum(direction="out")` would double-count bytes past the
    threshold (independent audit measured a 2x factor).
    """
    metrics = GatewayMetrics(namespace="rr", registry=None)
    metrics.observe_payload_stats(
        {
            "encoded_count": 1,
            "encoded_bytes": 1_000_000,
            "oversize_count": 1,
            "oversize_bytes": 1_000_000,
        }
    )
    assert (
        sample(metrics, "rr_payload_bytes_total", direction="out", kind="encoded")
        == 1_000_000.0
    )
    assert (
        sample(metrics, "rr_payload_bytes_total", direction="out", kind="oversize")
        is None
    ), "oversize must not share the direction=out sum"
    assert sample(metrics, "rr_payload_oversize_bytes_total") == 1_000_000.0
    assert sample(metrics, "rr_payload_events_total", kind="oversize") == 1.0


async def test_one_broken_sampler_does_not_hide_the_others(local_runtime) -> None:
    """When one sampler breaks, only that one error is recorded once, and the other metrics still get written (the earlier implementation had one blanket except that wiped everything).

    Args:
        local_runtime: In-process runtime fixture.
    """
    gateway = local_runtime.gateway

    def explode() -> None:
        raise RuntimeError("sampler is broken")

    gateway._sample_sessions = explode  # type: ignore[assignment]
    gateway.refresh_metrics()
    assert (
        sample(gateway.metrics, "rr_metrics_sampling_errors_total", sampler="sessions")
        == 1.0
    )
    # The subsequent samplers still ran (epoch is one of the last things written).
    assert sample(gateway.metrics, "rr_gateway_epoch") == float(gateway.gateway_epoch)
    assert sample(gateway.metrics, "rr_env_workers", state="healthy") == 1.0


async def test_control_plane_late_results_do_not_pollute_late_counters(
    local_runtime,
) -> None:
    """Late control-plane replies (e.g. a heartbeat that returns after a health-check timeout) must not be counted into ``late_results``.

    Heartbeats go through ``send_control`` and never enter ``OperationRegistry``, so
    when one arrives late, ``operations.find()`` is necessarily empty — it used to be
    recorded as ``orphaned``, and this metric would be completely dominated by
    heartbeats during rank churn (per independent audit).

    Args:
        local_runtime: In-process runtime fixture.
    """
    from rollout_runtime.api.enums import EnvOperation, OperationState
    from rollout_runtime.api.internal import ResultEnvelope

    gateway = local_runtime.gateway
    before = (gateway.late_result_count, gateway.orphan_result_count)
    await gateway._absorb_late_result(
        ResultEnvelope(
            request_id="req-late-heartbeat",
            operation=EnvOperation.HEARTBEAT,
            state=OperationState.SUCCEEDED,
        )
    )
    assert (gateway.late_result_count, gateway.orphan_result_count) == before
    # A late result on the data plane is still recorded as before (control group).
    await gateway._absorb_late_result(
        ResultEnvelope(
            request_id="req-late-policy-step",
            operation=EnvOperation.POLICY_STEP,
            state=OperationState.SUCCEEDED,
        )
    )
    assert gateway.late_result_count == before[0] + 1
    assert gateway.orphan_result_count == before[1] + 1


def test_metrics_use_a_private_registry() -> None:
    """Two Gateway metrics collections don't interfere with each other (not using the global default registry)."""
    first = GatewayMetrics(namespace="rr")
    second = GatewayMetrics(namespace="rr")
    assert first.registry is not second.registry
    first.record_error("INTERNAL")
    assert sample(first, "rr_errors_total", code="INTERNAL") == 1.0
    assert sample(second, "rr_errors_total", code="INTERNAL") is None


def test_metrics_namespace_is_configurable() -> None:
    """The metrics prefix follows ``gateway.metrics_namespace``."""
    metrics = GatewayMetrics(namespace="custom")
    metrics.observe_gateway_epoch(3)
    rendered = metrics.render()
    assert rendered is not None
    assert b"custom_gateway_epoch 3.0" in rendered[0]


@pytest.mark.parametrize("bad", ["", "not-a-metric"])
def test_sample_helper_returns_none_for_unknown_metrics(bad: str) -> None:
    """The sample helper returns ``None`` for unknown metrics (ensures the assertions above are not false positives).

    Args:
        bad: A metric name that does not exist.
    """
    metrics = GatewayMetrics(namespace="rr")
    assert sample(metrics, bad) is None
