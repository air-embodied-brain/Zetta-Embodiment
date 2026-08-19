# Copyright (c) 2026 Zetta Contributors
"""Small client for the persistent RoboCasa chunk server."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
import uuid
from typing import Any

from robots.robocasa.operation_protocol import payload_sha256


class RoboCasaEnvClient:
    def __init__(self, base_url: str, *, timeout_s: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.session_id = "session-" + uuid.uuid4().hex
        self.binding_token: str | None = None
        self.episode_id: str | None = None
        self.next_operation_seq = 0
        self.outcome_unknown = False

    def _request(
        self,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        retry_transport: bool = False,
    ) -> dict[str, Any]:
        body = None
        method = "GET"
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            method = "POST"
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            headers=headers,
            method=method,
        )
        attempts = 2 if retry_transport else 1
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout_s
                ) as response:
                    value = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                try:
                    terminal = json.loads(detail).get("_operation", {})
                except (AttributeError, json.JSONDecodeError):
                    terminal = {}
                if (
                    payload is not None
                    and "_operation" in payload
                    and terminal.get("outcome") == "OUTCOME_UNKNOWN"
                ):
                    self.outcome_unknown = True
                raise RuntimeError(
                    f"RoboCasa HTTP {exc.code} {path}: {detail}"
                ) from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt + 1 == attempts:
                    if payload is not None and "_operation" in payload:
                        self.outcome_unknown = True
                    raise RuntimeError(
                        f"RoboCasa request failed for {path}: {exc}"
                    ) from exc
        if not isinstance(value, dict):
            raise ValueError(f"RoboCasa endpoint {path} returned a non-object")
        return value

    def _ensure_binding(self) -> None:
        if self.binding_token is not None:
            return
        health = self.health()
        protocol = health.get("write_protocol")
        if not isinstance(protocol, dict):
            raise RuntimeError("RoboCasa server does not expose write protocol state")
        token = protocol.get("binding_token")
        if not isinstance(token, str) or not token:
            raise RuntimeError("RoboCasa server returned an invalid binding token")
        self.binding_token = token

    def _write(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        starts_episode: bool = False,
        releases_binding: bool = False,
    ) -> dict[str, Any]:
        if releases_binding and self.outcome_unknown:
            raise RuntimeError(
                "cannot release a binding after an unknown write outcome"
            )
        self._ensure_binding()
        if starts_episode:
            episode_id = "episode-" + uuid.uuid4().hex
            sequence = 0
        else:
            if self.episode_id is None:
                raise RuntimeError(
                    "reset must complete before another environment write"
                )
            episode_id = self.episode_id
            sequence = self.next_operation_seq
        request_id = "request-" + uuid.uuid4().hex
        body = dict(payload)
        body["_operation"] = {
            "request_id": request_id,
            "session_id": self.session_id,
            "binding_token": self.binding_token,
            "episode_id": episode_id,
            "operation_seq": sequence,
            "payload_sha256": payload_sha256(body),
        }
        result = self._request(path, payload=body, retry_transport=True)
        terminal = result.get("_operation")
        if not isinstance(terminal, dict) or terminal.get("request_id") != request_id:
            raise RuntimeError("RoboCasa server returned an unrelated write result")
        if terminal.get("outcome") != "COMMITTED":
            raise RuntimeError(
                f"RoboCasa write did not commit: {terminal.get('outcome', 'unknown')}"
            )
        if starts_episode:
            self.episode_id = episode_id
            self.next_operation_seq = 1
            self.outcome_unknown = False
        elif releases_binding:
            if result.get("binding_released") is not True:
                raise RuntimeError("RoboCasa server did not release the binding")
            self.session_id = "session-" + uuid.uuid4().hex
            self.binding_token = None
            self.episode_id = None
            self.next_operation_seq = 0
        else:
            self.next_operation_seq += 1
        return result

    def health(self) -> dict[str, Any]:
        return self._request("/health")

    def schema(self) -> dict[str, Any]:
        return self._request("/schema")

    def reset(
        self,
        *,
        task: str,
        seed: int,
        split: str = "target",
        bundle_sha256: str | None = None,
        video_dir: str | None = None,
        action_scale: dict[str, float] | None = None,
        enable_task_program: bool = False,
    ) -> dict[str, Any]:
        return self._write(
            "/reset",
            {
                "task": task,
                "seed": seed,
                "split": split,
                "bundle_sha256": bundle_sha256,
                "video_dir": video_dir,
                "action_scale": action_scale or {},
                "enable_task_program": enable_task_program,
            },
            starts_episode=True,
        )

    def observation(self, *, include_images: bool = True) -> dict[str, Any]:
        return self._request("/observation", payload={"include_images": include_images})

    def execute_chunk(
        self,
        actions: list[Any],
        *,
        critic_rules: list[dict[str, Any]] | None = None,
        interrupt_on_proposal: bool = True,
        capture_event_images: bool = True,
        enable_task_program: bool = False,
    ) -> dict[str, Any]:
        return self._write(
            "/execute_chunk",
            {
                "actions": actions,
                "critic_rules": critic_rules or [],
                "interrupt_on_proposal": interrupt_on_proposal,
                "capture_event_images": capture_event_images,
                "enable_task_program": enable_task_program,
            },
        )

    def finalize_episode(self) -> dict[str, Any]:
        """Flush episode artifacts without destroying the persistent env."""

        return self._write("/finalize_episode", {})

    def release(self) -> dict[str, Any]:
        """Release this slot for another rollout process without closing it."""

        return self._write("/release", {}, releases_binding=True)

    def close(self) -> dict[str, Any]:
        return self._write("/close", {})
