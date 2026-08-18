# Copyright (c) 2026 RPent Contributors
from __future__ import annotations

import json
import sys
from typing import Any

import pytest

from scripts.evolution.probe_provider_runtime import _probe, main


class _Response:
    status = 200

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    @staticmethod
    def read() -> bytes:
        return b'{"choices":[{"message":{"content":"API_TEST_OK"}}]}'


def _route(index: int) -> dict[str, Any]:
    return {
        "name": f"route-{index}",
        "provider": "openai-chat",
        "base_url": "https://provider.invalid/v1",
        "api_key_env": f"ROUTE_KEY_{index}",
        "model": "openai-chat:ignored-model",
        "price": float(index),
        "max_concurrency": 8,
    }


def test_probe_uses_passed_model_and_reasoning_without_leaking_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []
    monkeypatch.setenv("ROUTE_KEY_1", "fixture-secret-value")

    def fake_urlopen(request: Any, *, timeout: float) -> _Response:
        assert timeout == 12.0
        requests.append(json.loads(request.data))
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result = _probe(
        _route(1),
        12.0,
        model="gpt-5.6-sol",
        reasoning_effort="high",
    )

    assert result["ok"] is True
    assert requests == [
        {
            "model": "gpt-5.6-sol",
            "input": "Reply only with OK.",
            "max_output_tokens": 16,
            "reasoning": {"effort": "high"},
            "stream": False,
        }
    ]
    assert "fixture-secret-value" not in json.dumps(result)


def test_main_probes_exactly_eight_routes_at_eight_concurrency(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    routes = [_route(index) for index in range(8)]
    for index in range(8):
        monkeypatch.setenv(f"ROUTE_KEY_{index}", f"fixture-secret-{index}")
    monkeypatch.setenv("RPENT_API_PROVIDERS", json.dumps({"providers": routes}))
    monkeypatch.setattr("urllib.request.urlopen", lambda *_args, **_kwargs: _Response())
    monkeypatch.setattr(sys, "argv", ["probe_provider_runtime.py"])

    assert main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["concurrency"] == 8
    assert report["model"] == "gpt-5.6-sol"
    assert report["reasoning_effort"] == "high"
    assert report["wire_api"] == "responses"
    assert len(report["routes"]) == 8
    assert all(route["ok"] for route in report["routes"])


def test_main_rejects_concurrency_above_eight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["probe_provider_runtime.py", "--concurrency", "9"])
    with pytest.raises(ValueError, match="between 1 and 8"):
        main()
