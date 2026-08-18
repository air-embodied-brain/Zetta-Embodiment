# Copyright (c) 2026 RPent Contributors
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.models import Model

from rpent.planner.base import build_planner
from rpent.planner.provider_pool import (
    PROVIDERS_ENV,
    FailoverModel,
    ProviderModelRoute,
    ProviderPoolConfigError,
    ProviderRouteSpec,
    is_retryable_provider_error,
    load_provider_pool_config,
    sanitize_provider_config_for_broker_client,
)
from rpent.planner.provider_request_admission import ProviderRequestAdmission


class _FakeModel(Model):
    def __init__(self, name: str, outcomes: list[Any], calls: list[str]) -> None:
        self._name = name
        self._outcomes = outcomes
        self._calls = calls
        super().__init__()

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def system(self) -> str:
        return "openai"

    async def request(self, messages, model_settings, model_request_parameters):
        self._calls.append(self._name)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _FakeStreamingModel(Model):
    def __init__(self, name: str, stream: Any, calls: list[str]) -> None:
        self._name = name
        self._stream = stream
        self._calls = calls
        super().__init__()

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def system(self) -> str:
        return "openai"

    async def request(self, messages, model_settings, model_request_parameters):
        del messages, model_settings, model_request_parameters
        raise AssertionError("streaming test must use request_stream")

    @asynccontextmanager
    async def request_stream(
        self, messages, model_settings, model_request_parameters, run_context=None
    ):
        del messages, model_settings, model_request_parameters, run_context
        self._calls.append(self._name)
        yield self._stream


class _Clock:
    def __init__(self, value: float = 1_000.0) -> None:
        self.value = value
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.value

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.value += seconds


@pytest.mark.parametrize(
    "document",
    [
        [
            {
                "name": "one",
                "provider": "openai-chat",
                "base_url": "https://one.example/v1",
                "api_key_env": "UPSTREAM_ONE",
            }
        ],
        {
            "providers": [
                {
                    "name": "one",
                    "provider": "openai-chat",
                    "base_url": "https://one.example/v1",
                    "api_key": "literal-secret",
                }
            ]
        },
        {
            "routes": [
                {
                    "name": "one",
                    "provider": "openai-chat",
                    "base_url": "https://one.example/v1",
                    "key": "legacy-secret",
                }
            ]
        },
        {
            "one": {
                "provider": "openai-chat",
                "base_url": "https://one.example/v1",
                "api_key_env": "UPSTREAM_ONE",
            }
        },
        {
            "name": "one",
            "provider": "openai-chat",
            "base_url": "https://one.example/v1",
            "api_key": "literal-secret",
        },
    ],
)
def test_broker_client_sanitizer_covers_every_provider_document_shape(
    document: object,
) -> None:
    sanitized = sanitize_provider_config_for_broker_client(json.dumps(document))

    assert "UPSTREAM_ONE" not in sanitized
    assert "literal-secret" not in sanitized
    assert "legacy-secret" not in sanitized
    assert "broker-managed-placeholder" in sanitized
    config = load_provider_pool_config(
        default_model="openai-chat:gpt-5.6-sol",
        env={PROVIDERS_ENV: sanitized},
    )
    assert config is not None
    assert len(config.routes) == 1


def _spec(
    name: str,
    *,
    key: str,
    price: float,
    cooldown: float = 30.0,
) -> ProviderRouteSpec:
    return ProviderRouteSpec(
        name=name,
        model="openai-responses:gpt-test",
        provider="openai-responses",
        base_url=f"https://{name}.invalid/v1",
        api_key=key,
        price=price,
        cooldown_seconds=cooldown,
    )


def _route(spec: ProviderRouteSpec, outcomes: list[Any], calls: list[str]):
    return ProviderModelRoute(
        spec=spec,
        model=_FakeModel(spec.name, outcomes, calls),
    )


def test_list_configuration_is_sorted_by_price_and_hides_keys(tmp_path: Path):
    raw = [
        {
            "name": "expensive",
            "provider": "openai-responses",
            "base_url": "https://expensive.invalid/v1",
            "api_key": "secret-expensive",
            "price": 9.4,
        },
        {
            "name": "cheap",
            "provider": "openai-responses",
            "base_url": "https://cheap.invalid/v1",
            "api_key": "secret-cheap",
            "price": 0.125,
        },
    ]
    config = load_provider_pool_config(
        default_model="openai-responses:gpt-test",
        env={
            PROVIDERS_ENV: json.dumps(raw),
            "RPENT_API_PROVIDER_STATE_FILE": str(tmp_path / "state.json"),
        },
    )
    assert config is not None
    assert [route.name for route in config.routes] == ["cheap", "expensive"]
    assert "secret-cheap" not in repr(config)
    assert config.state_file == (tmp_path / "state.json").resolve()


def test_mapping_configuration_and_api_key_env_are_supported(tmp_path: Path):
    document = {
        "cooldown_seconds": 31,
        "state_file": str(tmp_path / "shared.json"),
        "primary": {
            "provider": "openai-responses",
            "url": "https://primary.invalid/v1",
            "api_key_env": "PRIMARY_KEY",
            "price": 0.125,
            "max_concurrency": 20,
            "rpm": 5000,
            "tpm": 500000,
        },
        "secondary": {
            "provider": "openai-responses",
            "url": "https://secondary.invalid/v1",
            "key": "secondary-key",
            "price": 0.15,
        },
    }
    config = load_provider_pool_config(
        default_model="openai-responses:gpt-test",
        env={PROVIDERS_ENV: json.dumps(document), "PRIMARY_KEY": "primary-key"},
    )
    assert config is not None
    assert [route.name for route in config.routes] == ["primary", "secondary"]
    assert config.routes[0].api_key == "primary-key"
    assert config.routes[0].cooldown_seconds == 31
    assert config.routes[0].max_concurrency == 20
    assert config.routes[0].rpm_limit == 5000
    assert config.routes[0].tpm_limit == 500000


def test_same_url_with_two_keys_produces_distinct_routes():
    shared = "https://shared.invalid/v1"
    config = load_provider_pool_config(
        default_model="openai-responses:gpt-test",
        env={
            PROVIDERS_ENV: json.dumps(
                [
                    {
                        "name": "key-a",
                        "provider": "openai-responses",
                        "base_url": shared,
                        "api_key": "a-secret",
                        "price": 1,
                    },
                    {
                        "name": "key-b",
                        "provider": "openai-responses",
                        "base_url": shared,
                        "api_key": "b-secret",
                        "price": 2,
                    },
                ]
            )
        },
    )
    assert config is not None
    assert config.routes[0].route_id != config.routes[1].route_id


def test_full_openai_endpoint_urls_are_normalized_to_base_urls():
    config = load_provider_pool_config(
        default_model="openai-chat:gpt-test",
        env={
            PROVIDERS_ENV: json.dumps(
                [
                    {
                        "name": "chat",
                        "provider": "openai-chat",
                        "url": "https://chat.invalid/v1/chat/completions",
                        "api_key": "chat-key",
                        "price": 1,
                    },
                    {
                        "name": "responses",
                        "provider": "openai-responses",
                        "url": "https://responses.invalid/v1/responses",
                        "api_key": "responses-key",
                        "price": 2,
                    },
                ]
            )
        },
    )
    assert config is not None
    assert config.routes[0].base_url == "https://chat.invalid/v1"
    assert config.routes[1].base_url == "https://responses.invalid/v1"


def test_full_azure_responses_endpoint_derives_endpoint_and_api_version():
    config = load_provider_pool_config(
        default_model="azure-responses:gpt-test",
        env={
            PROVIDERS_ENV: json.dumps(
                [
                    {
                        "name": "azure",
                        "provider": "azure-responses",
                        "url": "https://azure.invalid/openai/responses?api-version=2025-04-01-preview",
                        "api_key": "azure-key",
                        "price": 1,
                    }
                ]
            )
        },
    )
    assert config is not None
    assert config.routes[0].base_url == "https://azure.invalid"
    assert config.routes[0].api_version == "2025-04-01-preview"


@pytest.mark.asyncio
async def test_429_cools_first_route_and_falls_back(tmp_path: Path):
    calls: list[str] = []
    first = _spec("cheap", key="secret-a", price=0.1)
    second = _spec("backup", key="secret-b", price=0.2)
    model = FailoverModel(
        [
            _route(
                first,
                [
                    ModelHTTPError(
                        429,
                        "gpt-test",
                        body={"message": "secret-a should never be logged"},
                        headers={"retry-after": "45"},
                    )
                ],
                calls,
            ),
            _route(second, ["ok"], calls),
        ],
        state_file=tmp_path / "health.json",
    )
    assert await model.request([], None, None) == "ok"
    assert calls == ["cheap", "backup"]
    state_text = (tmp_path / "health.json").read_text(encoding="utf-8")
    assert "secret-a" not in state_text
    state = json.loads(state_text)
    assert state["routes"][first.route_id]["cooldown_until"] > 0


@pytest.mark.asyncio
async def test_request_admission_is_bound_to_each_actual_fallback_route(
    tmp_path: Path,
):
    calls: list[str] = []
    first = _spec("cheap", key="secret-a", price=0.1)
    second = _spec("backup", key="secret-b", price=0.2)
    admission = ProviderRequestAdmission(
        tmp_path / "admission", initial_limit=2, max_limit=4
    )
    model = FailoverModel(
        [
            _route(first, [ModelHTTPError(429, "gpt-test")], calls),
            _route(second, ["ok"], calls),
        ],
        state_file=tmp_path / "health.json",
        request_admission=admission,
    )
    assert await model.request([], None, None) == "ok"
    first_state = admission.route_admission.snapshot(first.route_id)
    second_state = admission.route_admission.snapshot(second.route_id)
    global_state = admission.global_admission.snapshot(
        ProviderRequestAdmission.GLOBAL_ROUTE_ID
    )
    assert first_state.routes[first.route_id].failed_requests == 1
    assert second_state.routes[second.route_id].successful_requests == 1
    assert first_state.active_leases == 0
    assert second_state.active_leases == 0
    assert global_state.active_leases == 0
    assert global_state.routes[ProviderRequestAdmission.GLOBAL_ROUTE_ID].successful_requests == 1


@pytest.mark.asyncio
async def test_stream_is_marked_success_only_after_full_context_exit(tmp_path: Path):
    calls: list[str] = []
    spec = _spec("stream", key="secret", price=0.1)
    stream = SimpleNamespace(
        usage=lambda: SimpleNamespace(request_tokens=10, response_tokens=5)
    )
    admission = ProviderRequestAdmission(
        tmp_path / "admission", initial_limit=1, max_limit=1
    )
    model = FailoverModel(
        [
            ProviderModelRoute(
                spec=spec,
                model=_FakeStreamingModel(spec.name, stream, calls),
            )
        ],
        state_file=tmp_path / "health.json",
        request_admission=admission,
    )
    async with model.request_stream([], None, None) as received:
        assert received is stream
        snapshot = admission.route_admission.snapshot(spec.route_id)
        assert snapshot.routes[spec.route_id].successful_requests == 0
        assert snapshot.active_leases == 1
    route = admission.route_admission.snapshot(spec.route_id).routes[spec.route_id]
    assert route.successful_requests == 1
    assert route.rolling_tpm == 15
    assert route.active_leases == 0


@pytest.mark.asyncio
async def test_partial_stream_failure_is_not_replayed_and_cools_actual_route(
    tmp_path: Path,
):
    calls: list[str] = []
    first = _spec("stream", key="secret", price=0.1)
    second = _spec("backup", key="backup", price=0.2)
    admission = ProviderRequestAdmission(
        tmp_path / "admission", initial_limit=1, max_limit=1
    )
    model = FailoverModel(
        [
            ProviderModelRoute(
                spec=first,
                model=_FakeStreamingModel(first.name, object(), calls),
            ),
            ProviderModelRoute(
                spec=second,
                model=_FakeStreamingModel(second.name, object(), calls),
            ),
        ],
        state_file=tmp_path / "health.json",
        request_admission=admission,
    )
    with pytest.raises(RuntimeError, match="consumer lost stream"):
        async with model.request_stream([], None, None):
            raise RuntimeError("consumer lost stream")
    assert calls == ["stream"]
    route = admission.route_admission.snapshot(first.route_id).routes[first.route_id]
    assert route.failed_requests == 1
    assert route.rolling_stream_errors == 1
    assert route.active_leases == 0


@pytest.mark.asyncio
async def test_shared_cooldown_is_seen_by_another_model_instance(tmp_path: Path):
    state_file = tmp_path / "shared-health.json"
    first = _spec("cheap", key="a", price=0.1)
    second = _spec("backup", key="b", price=0.2)
    clock = _Clock()

    calls_one: list[str] = []
    model_one = FailoverModel(
        [
            _route(first, [ModelAPIError("gpt-test", "network")], calls_one),
            _route(second, ["first-request"], calls_one),
        ],
        state_file=state_file,
        clock=clock,
        sleeper=clock.sleep,
    )
    assert await model_one.request([], None, None) == "first-request"

    calls_two: list[str] = []
    model_two = FailoverModel(
        [
            _route(first, ["must-not-run"], calls_two),
            _route(second, ["second-request"], calls_two),
        ],
        state_file=state_file,
        clock=clock,
        sleeper=clock.sleep,
    )
    assert await model_two.request([], None, None) == "second-request"
    assert calls_two == ["backup"]


@pytest.mark.asyncio
async def test_preferred_route_is_retried_after_cooldown(tmp_path: Path):
    state_file = tmp_path / "health.json"
    clock = _Clock()
    first = _spec("cheap", key="a", price=0.1, cooldown=30)
    second = _spec("backup", key="b", price=0.2)
    calls: list[str] = []
    model = FailoverModel(
        [
            _route(
                first,
                [ModelHTTPError(429, "gpt-test"), "cheap-again"],
                calls,
            ),
            _route(second, ["backup-now"], calls),
        ],
        state_file=state_file,
        clock=clock,
        sleeper=clock.sleep,
    )
    assert await model.request([], None, None) == "backup-now"
    clock.value += 31
    assert await model.request([], None, None) == "cheap-again"
    assert calls == ["cheap", "backup", "cheap"]


@pytest.mark.asyncio
async def test_non_retryable_request_error_does_not_replay(tmp_path: Path):
    calls: list[str] = []
    first = _spec("first", key="a", price=1)
    second = _spec("second", key="b", price=2)
    model = FailoverModel(
        [
            _route(first, [ModelHTTPError(400, "gpt-test")], calls),
            _route(second, ["must-not-run"], calls),
        ],
        state_file=tmp_path / "health.json",
    )
    with pytest.raises(ModelHTTPError) as captured:
        await model.request([], None, None)
    assert captured.value.status_code == 400
    assert calls == ["first"]


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ModelHTTPError(429, "gpt-test"), True),
        (ModelHTTPError(503, "gpt-test"), True),
        (ModelHTTPError(401, "gpt-test"), True),
        (ModelHTTPError(400, "gpt-test"), False),
        (ValueError("application bug"), False),
    ],
)
def test_retryable_error_classification(error: Exception, expected: bool):
    assert is_retryable_provider_error(error) is expected


def test_configuration_errors_do_not_echo_api_keys():
    secret = "this-is-a-very-secret-key"
    with pytest.raises(ProviderPoolConfigError) as captured:
        load_provider_pool_config(
            default_model="openai-responses:gpt-test",
            env={
                PROVIDERS_ENV: json.dumps(
                    [{"name": "bad name", "api_key": secret, "base_url": "x"}]
                )
            },
        )
    assert secret not in str(captured.value)


def test_build_planner_uses_provider_pool_when_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv(
        PROVIDERS_ENV,
        json.dumps(
            [
                {
                    "name": "test-route",
                    "provider": "openai-responses",
                    "base_url": "https://provider.invalid/v1",
                    "api_key": "test-secret",
                    "price": 1,
                }
            ]
        ),
    )
    monkeypatch.setenv("RPENT_API_PROVIDER_STATE_FILE", str(tmp_path / "health.json"))
    planner = build_planner(
        "api",
        output_dir=tmp_path,
        recipe_tag="test",
        env_name="libero",
        model="openai-responses:gpt-test",
    )
    assert isinstance(planner._model, FailoverModel)


def test_build_planner_uses_external_central_broker_when_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv(
        PROVIDERS_ENV,
        json.dumps(
            [
                {
                    "name": "test-route",
                    "provider": "openai-responses",
                    "base_url": "https://provider.invalid/v1",
                    "api_key": "upstream-secret",
                    "price": 1,
                }
            ]
        ),
    )
    monkeypatch.setenv("RPENT_API_PROVIDER_STATE_FILE", str(tmp_path / "health.json"))
    monkeypatch.setenv(
        "RPENT_API_PROVIDER_BROKER_URL", "http://127.0.0.1:4110"
    )
    monkeypatch.setenv("RPENT_API_PROVIDER_BROKER_API_KEY", "local-broker-key")
    planner = build_planner(
        "api",
        output_dir=tmp_path,
        recipe_tag="test",
        env_name="libero",
        model="openai-responses:gpt-test",
    )
    assert not isinstance(planner._model, FailoverModel)
    assert str(planner._model.base_url).rstrip("/") == "http://127.0.0.1:4110/v1"
