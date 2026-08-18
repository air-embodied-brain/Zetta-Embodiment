# Copyright (c) 2026 RPent Contributors
"""LIBERO env client that forwards calls over an RPC transport.

Lives in :mod:`robots.libero` because the methods exposed
here (``raw_obs`` / ``render_camera`` / ``cached_image`` / …)
reference LIBERO-specific obs dict keys and camera names. The generic
transport layer lives in :mod:`rpent.utils.socket_rpc`.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from rpent.utils.rpc import RpcClient

_TIMEOUT_S = {
    "default": 30.0,
    # MuJoCo/EGL reset can wait behind the shared VLA process during formal
    # paired gates; an infrastructure timeout must not become a policy result.
    "env.reset": 900.0,
    "env.step": 60.0,
    "env.chunk_step": 120.0,
    "env.render_camera": 120.0,
}


class LiberoEnvClient:
    """Remote implementation of the LIBERO env protocol."""

    def __init__(
        self,
        client: RpcClient,
        *,
        expected_meta: dict,
        return_all_frames: bool = False,
    ):
        self._client = client
        self.return_all_frames = return_all_frames
        self.episode_terminated = False
        self.episode_truncated = False
        self.episode_steps = 0
        server_meta = self._client.call(
            "env.get_env_meta", timeout_s=_TIMEOUT_S["default"]
        )
        assert server_meta == expected_meta, (
            f"env_meta mismatch: expected={expected_meta!r} "
            f"actual={server_meta!r}. The env_server was launched with "
            "different args than this client expects — kill the stale "
            "env_server and relaunch."
        )
        self.reset()

    def check_done(self, term, trunc) -> None:
        self.episode_terminated |= bool(np.asarray(term).any())
        self.episode_truncated |= bool(np.asarray(trunc).any())

    def reset(self) -> tuple[dict, Any]:
        ret = self._client.call("env.reset", timeout_s=_TIMEOUT_S["env.reset"])
        self.episode_terminated = False
        self.episode_truncated = False
        self.episode_steps = 0
        return ret

    def step(self, action) -> tuple[dict, Any, np.ndarray, Any, Any]:
        assert not (self.episode_terminated or self.episode_truncated), (
            "env.step called after the episode signaled term/trunc"
        )
        ret = self._client.call(
            "env.step", args=(action,), timeout_s=_TIMEOUT_S["env.step"]
        )
        _, _, term, trunc, _ = ret
        self.check_done(term, trunc)
        self.episode_steps += 1
        return ret

    def chunk_step(self, actions, *, return_all_frames: bool | None = None) -> tuple[Any, Any, Any, Any, Any]:
        """Run an action chunk in one RPC. Returns the 5-positional tuple
        ``(obs_or_list, reward, terminated, truncated, info)``.

        ``obs`` is ``list[Obs]`` when ``return_all_frames`` is True
        (one entry per chunk step), otherwise the final ``Obs`` dict.
        Terminated / truncated have shape ``[chunk_size]`` after the
        server strips the env dim.
        """
        assert not (self.episode_terminated or self.episode_truncated), (
            "env.chunk_step called after the episode signaled term/trunc"
        )
        if return_all_frames is None:
            return_all_frames = self.return_all_frames
        ret = self._client.call(
            "env.chunk_step",
            args=(actions,),
            kwargs={"return_all_frames": return_all_frames},
            timeout_s=_TIMEOUT_S["env.chunk_step"],
        )
        _, _, term, trunc, _ = ret
        self.check_done(term, trunc)
        info = ret[4]
        self.episode_steps += int(
            info.get("executed_horizon", np.asarray(term).size)
            if isinstance(info, dict)
            else np.asarray(term).size
        )
        return ret

    def critic_chunk_step(
        self,
        actions,
        *,
        critic_rules: list[dict[str, Any]],
        interrupt_on_proposal: bool = True,
        return_all_frames: bool = True,
    ) -> tuple[Any, Any, Any, Any, Any]:
        """Execute a chunk while the environment evaluates Critic every step."""

        assert not (self.episode_terminated or self.episode_truncated), (
            "env.critic_chunk_step called after the episode signaled term/trunc"
        )
        ret = self._client.call(
            "env.critic_chunk_step",
            args=(actions,),
            kwargs={
                "critic_rules": critic_rules,
                "interrupt_on_proposal": bool(interrupt_on_proposal),
                "return_all_frames": bool(return_all_frames),
            },
            timeout_s=_TIMEOUT_S["env.chunk_step"],
        )
        _, _, term, trunc, info = ret
        self.check_done(term, trunc)
        self.episode_steps += int(info.get("executed_horizon", np.asarray(term).size))
        return ret

    def audit_trace(self, *, since_step: int = 0) -> list[dict[str, Any]]:
        return self._client.call(
            "env.audit_trace",
            kwargs={"since_step": int(since_step)},
            timeout_s=_TIMEOUT_S["default"],
        )

    def raw_obs(self) -> dict:
        return self._client.call("env.raw_obs", timeout_s=_TIMEOUT_S["default"])

    def privileged_contacts(
        self,
        *,
        include_all_contacts: bool = False,
        max_contacts: int = 64,
    ) -> dict:
        """Read current privileged contact/force evidence without stepping."""
        return self._client.call(
            "env.privileged_contacts",
            kwargs={
                "include_all_contacts": include_all_contacts,
                "max_contacts": max_contacts,
            },
            timeout_s=_TIMEOUT_S["default"],
        )

    def privileged_semantic_joint_plan(
        self,
        *,
        entity: str,
        joint: str,
        direction: str,
    ) -> dict[str, Any]:
        """Read an audited fixture plan for an internal recovery primitive."""

        return self._client.call(
            "env.privileged_semantic_joint_plan",
            kwargs={
                "entity": str(entity),
                "joint": str(joint),
                "direction": str(direction),
            },
            timeout_s=_TIMEOUT_S["default"],
        )

    def privileged_critic_state(
        self, *, reset_tracker: bool = False
    ) -> dict[str, Any]:
        """Read the simulator semantic sidecar reserved for Critic rules."""

        return self._client.call(
            "env.privileged_critic_state",
            kwargs={"reset_tracker": bool(reset_tracker)},
            timeout_s=_TIMEOUT_S["default"],
        )

    def render_camera(
        self,
        camera_name: str = "agentview",
        height: int = 1024,
        width: int = 1024,
        depth: bool = False,
    ):
        return self._client.call(
            "env.render_camera",
            kwargs={
                "camera_name": camera_name,
                "height": height,
                "width": width,
                "depth": depth,
            },
            timeout_s=_TIMEOUT_S["env.render_camera"],
        )

    def get_camera_meta(
        self,
        camera_name: str = "agentview",
        height: int = 256,
        width: int = 256,
    ) -> dict | None:
        return self._client.call(
            "env.get_camera_meta",
            kwargs={"camera_name": camera_name, "height": height, "width": width},
            timeout_s=_TIMEOUT_S["default"],
        )

    def get_task_language(self) -> str | None:
        return self._client.call(
            "env.get_task_language", timeout_s=_TIMEOUT_S["default"]
        )

    def cached_image(self) -> np.ndarray | None:
        return self._client.call(
            "env.cached_image", timeout_s=_TIMEOUT_S["default"]
        )
