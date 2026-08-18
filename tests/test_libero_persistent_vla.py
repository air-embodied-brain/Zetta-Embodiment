from __future__ import annotations

import argparse

from robots.libero import _add_cli_args


def test_vla_endpoint_defaults_to_runtime_environment(monkeypatch):
    monkeypatch.setenv("RPENT_VLA_ENDPOINT", "http://127.0.0.1:18091")
    parser = argparse.ArgumentParser()
    _add_cli_args(parser, use_dashboard=True)

    args = parser.parse_args([])

    assert args.vla_endpoint == "http://127.0.0.1:18091"


def test_explicit_vla_endpoint_overrides_runtime_environment(monkeypatch):
    monkeypatch.setenv("RPENT_VLA_ENDPOINT", "http://127.0.0.1:18091")
    parser = argparse.ArgumentParser()
    _add_cli_args(parser, use_dashboard=True)

    args = parser.parse_args(["--vla-endpoint", "http://127.0.0.1:19091"])

    assert args.vla_endpoint == "http://127.0.0.1:19091"
