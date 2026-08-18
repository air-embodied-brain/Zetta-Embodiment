# Copyright (c) 2026 RPent Contributors
from __future__ import annotations

import json
from pathlib import Path

from rpent.planner.codex import PROVIDER_ENV_KEY, CodexPlanner
from rpent.planner.provider_pool import PROVIDERS_ENV


def test_codex_uses_external_broker_instead_of_per_solve_proxy(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(
        PROVIDERS_ENV,
        json.dumps(
            [
                {
                    "name": "upstream",
                    "provider": "openai-responses",
                    "base_url": "https://upstream.invalid/v1",
                    "api_key": "upstream-secret",
                    "price": 1,
                }
            ]
        ),
    )
    monkeypatch.setenv(
        "RPENT_API_PROVIDER_BROKER_URL", "http://127.0.0.1:4110"
    )
    monkeypatch.setenv("RPENT_API_PROVIDER_BROKER_API_KEY", "local-broker-key")
    planner = CodexPlanner(
        output_dir=str(tmp_path),
        model="gpt-5.6-sol",
        reasoning_effort="high",
    )

    assert planner._external_provider_broker == (
        "http://127.0.0.1:4110",
        "local-broker-key",
    )
    config = planner._build_config("http://127.0.0.1:9999")
    overrides = "\n".join(config.config_overrides)
    assert 'model_providers.rpent_proxy.base_url="http://127.0.0.1:4110/v1"' in overrides
    assert 'model_reasoning_effort="high"' in overrides
    assert config.env[PROVIDER_ENV_KEY] == "local-broker-key"
    assert "upstream-secret" not in overrides
