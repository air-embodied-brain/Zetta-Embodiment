# Copyright (c) 2026 Zetta Contributors
from __future__ import annotations

import sys
import urllib.error
from pathlib import Path

import pytest

from scripts.experiments.probe_api_channels import _post, main


def test_transport_error_does_not_persist_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args, **_kwargs):
        raise urllib.error.URLError("https://sensitive-endpoint.invalid/v1")

    monkeypatch.setattr("urllib.request.urlopen", fail)
    result = _post(
        "https://sensitive-endpoint.invalid/v1/responses",
        "secret-key",
        {"model": "gpt-test"},
    )
    assert result == {"transport_error": "URLError"}


def test_probe_requires_injected_base_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api_file = tmp_path / "keys.txt"
    api_file.write_text("no keys", encoding="utf-8")
    monkeypatch.delenv("ZETTA_CHANNEL_PROBE_BASE_URL", raising=False)
    monkeypatch.setattr(sys, "argv", ["probe_api_channels.py", "--api-file", str(api_file)])
    with pytest.raises(ValueError, match="base-url"):
        main()
