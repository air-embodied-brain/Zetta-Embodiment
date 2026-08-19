# Copyright (c) 2026 Zetta Contributors
from __future__ import annotations

import gzip
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest

from zetta.planner.provider_pool import ProviderPoolConfig, ProviderRouteSpec
from zetta.planner.provider_proxy import (
    ProviderPoolProxy,
    load_provider_broker_connection,
)


class _Upstream:
    def __init__(
        self,
        statuses: list[int],
        *,
        delay_s: float = 0.0,
        gzip_response: bool = False,
    ) -> None:
        self.statuses = statuses
        self.delay_s = delay_s
        self.gzip_response = gzip_response
        self.requests: list[dict[str, Any]] = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()
        upstream = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                size = int(self.headers.get("content-length", "0"))
                body = self.rfile.read(size)
                with upstream.lock:
                    upstream.active += 1
                    upstream.max_active = max(upstream.max_active, upstream.active)
                    upstream.requests.append(
                        {
                            "path": self.path,
                            "authorization": self.headers.get("authorization"),
                            "api_key": self.headers.get("api-key"),
                            "body": json.loads(body),
                        }
                    )
                    status = upstream.statuses.pop(0)
                time.sleep(upstream.delay_s)
                payload = json.dumps({"ok": status == 200}).encode()
                if upstream.gzip_response:
                    payload = gzip.compress(payload)
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                if upstream.gzip_response:
                    self.send_header("content-encoding", "gzip")
                if status == 429:
                    self.send_header("retry-after", "45")
                self.end_headers()
                self.wfile.write(payload)
                with upstream.lock:
                    upstream.active -= 1

            def log_message(self, _format: str, *args: Any) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def _route(
    name: str,
    upstream: _Upstream,
    *,
    key: str,
    price: float,
    provider: str = "openai-responses",
    api_version: str | None = None,
) -> ProviderRouteSpec:
    base_url = upstream.base_url
    if provider.startswith("azure"):
        base_url = base_url.removesuffix("/v1")
    return ProviderRouteSpec(
        name=name,
        model=f"{provider}:route-model",
        provider=provider,
        base_url=base_url,
        api_key=key,
        price=price,
        cooldown_seconds=30,
        api_version=api_version,
    )


def test_proxy_falls_back_and_shares_cooldown(tmp_path: Path):
    first = _Upstream([429])
    second = _Upstream([200, 200])
    config = ProviderPoolConfig(
        routes=(
            _route("cheap", first, key="secret-a", price=1),
            _route("backup", second, key="secret-b", price=2),
        ),
        state_file=tmp_path / "health.json",
    )
    proxy = ProviderPoolProxy(config)
    proxy_url = proxy.start()
    try:
        with httpx.Client(trust_env=False) as client:
            response_one = client.post(
                proxy_url + "/v1/responses",
                json={"model": "client-model", "input": "test"},
                headers={"authorization": f"Bearer {proxy.api_key}"},
            )
            response_two = client.post(
                proxy_url + "/v1/responses",
                json={"model": "client-model", "input": "test"},
                headers={"authorization": f"Bearer {proxy.api_key}"},
            )
    finally:
        proxy.stop()
        first.close()
        second.close()

    assert response_one.status_code == 200
    assert response_two.status_code == 200
    assert len(first.requests) == 1
    assert len(second.requests) == 2
    assert second.requests[0]["authorization"] == "Bearer secret-b"
    assert second.requests[0]["body"]["model"] == "route-model"
    state = (tmp_path / "health.json").read_text(encoding="utf-8")
    assert "secret-a" not in state
    assert "secret-b" not in state
    assert first.base_url not in state


def test_proxy_uses_azure_responses_path_and_api_key_header(tmp_path: Path):
    upstream = _Upstream([200])
    config = ProviderPoolConfig(
        routes=(
            _route(
                "azure",
                upstream,
                key="azure-secret",
                price=1,
                provider="azure-responses",
                api_version="2025-04-01-preview",
            ),
        ),
        state_file=tmp_path / "health.json",
    )
    proxy = ProviderPoolProxy(config)
    proxy_url = proxy.start()
    try:
        response = httpx.post(
            proxy_url + "/v1/responses",
            json={"model": "client-model", "input": "test"},
            headers={"authorization": f"Bearer {proxy.api_key}"},
            trust_env=False,
        )
    finally:
        proxy.stop()
        upstream.close()

    assert response.status_code == 200
    assert upstream.requests[0]["path"] == (
        "/openai/responses?api-version=2025-04-01-preview"
    )
    assert upstream.requests[0]["api_key"] == "azure-secret"
    assert upstream.requests[0]["authorization"] is None


def test_proxy_decodes_gzip_when_content_encoding_is_not_forwarded(tmp_path: Path):
    upstream = _Upstream([200], gzip_response=True)
    config = ProviderPoolConfig(
        routes=(_route("gzip", upstream, key="secret", price=1),),
        state_file=tmp_path / "health.json",
    )
    proxy = ProviderPoolProxy(config)
    proxy_url = proxy.start()
    try:
        response = httpx.post(
            proxy_url + "/v1/responses",
            json={"model": "client-model", "input": "test"},
            headers={"authorization": f"Bearer {proxy.api_key}"},
            trust_env=False,
        )
    finally:
        proxy.stop()
        upstream.close()

    assert response.status_code == 200
    assert response.headers.get("content-encoding") is None
    assert response.json() == {"ok": True}


def test_proxy_rejects_missing_local_credential(tmp_path: Path):
    upstream = _Upstream([200])
    config = ProviderPoolConfig(
        routes=(_route("only", upstream, key="upstream-secret", price=1),),
        state_file=tmp_path / "health.json",
    )
    proxy = ProviderPoolProxy(config)
    proxy_url = proxy.start()
    try:
        response = httpx.post(
            proxy_url + "/v1/responses",
            json={"model": "client-model", "input": "test"},
            trust_env=False,
        )
    finally:
        proxy.stop()
        upstream.close()

    assert response.status_code == 401
    assert not upstream.requests


def test_proxy_uses_one_bounded_fifo_and_fixed_dispatch_workers(tmp_path: Path):
    upstream = _Upstream([200] * 6, delay_s=0.1)
    config = ProviderPoolConfig(
        routes=(_route("only", upstream, key="upstream-secret", price=1),),
        state_file=tmp_path / "health.json",
    )
    proxy = ProviderPoolProxy(config, max_concurrency=2, queue_capacity=16)
    proxy_url = proxy.start()

    def send(index: int) -> int:
        response = httpx.post(
            proxy_url + "/v1/responses",
            json={"model": "client-model", "input": f"request-{index}"},
            headers={"authorization": f"Bearer {proxy.api_key}"},
            timeout=10,
            trust_env=False,
        )
        return response.status_code

    try:
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(send, index) for index in range(6)]
            observed_queue_depth = 0
            for _ in range(100):
                observed_queue_depth = max(
                    observed_queue_depth, proxy.stats()["queue_depth"]
                )
                if proxy.stats()["accepted"] == 6:
                    break
                time.sleep(0.005)
            statuses = [future.result() for future in futures]
        stats = proxy.stats()
    finally:
        proxy.stop()
        upstream.close()

    assert statuses == [200] * 6
    assert upstream.max_active == 2
    assert observed_queue_depth >= 1
    assert stats["max_concurrency"] == 2
    assert stats["max_inflight_observed"] == 2
    assert stats["accepted"] == stats["completed"] == 6
    assert stats["queue_depth"] == stats["inflight"] == stats["rejected"] == 0


def test_external_broker_connection_is_atomic():
    assert load_provider_broker_connection({}) is None
    assert load_provider_broker_connection(
        {
            "ZETTA_API_PROVIDER_BROKER_URL": "http://127.0.0.1:4110/",
            "ZETTA_API_PROVIDER_BROKER_API_KEY": "local-key",
        }
    ) == ("http://127.0.0.1:4110", "local-key")
    with pytest.raises(ValueError, match="configured together"):
        load_provider_broker_connection(
            {"ZETTA_API_PROVIDER_BROKER_URL": "http://127.0.0.1:4110"}
        )
