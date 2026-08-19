"""Adapter Plugin interface.

v1 **only defines the interface plus a built-in registry; it does not open
third-party plugin loading**. Plugins can only call the same ``RuntimeClient``
as a regular Adapter; they cannot access the transport, nor bypass the
``SessionManager`` to operate the EnvWorker directly.

Sinking primitives such as ``pi0_pick`` / ``vla_execute`` into plugins is an
optimization slated for after v1; in v1 they remain in ``LiberoPrimitives``
(Adapter side).
"""

from __future__ import annotations

import types
from collections.abc import Mapping
from typing import Any, Protocol

from rollout_runtime.api.client import RuntimeClient
from rollout_runtime.api.enums import ErrorCode
from rollout_runtime.api.errors import RuntimeApiError, make_error

__all__ = ["AdapterPlugin", "PluginExecutor"]


class AdapterPlugin(Protocol):
    """High-level Adapter plugin."""

    @property
    def name(self) -> str:
        """Plugin name.

        Returns:
            Unique name used for registration.
        """
        ...

    async def invoke(
        self, client: RuntimeClient, args: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute the plugin logic.

        Args:
            client: Only permitted to access the Runtime through the Runtime API.
            args: Plugin arguments.

        Returns:
            Structured result.
        """
        ...


class PluginExecutor:
    """Built-in plugin registry and invocation entry point."""

    def __init__(self, plugins: Mapping[str, AdapterPlugin] | None = None) -> None:
        """Initialize.

        Args:
            plugins: Built-in plugin table; ``None`` means empty.
        """
        self._plugins: dict[str, AdapterPlugin] = dict(plugins or {})

    def register(self, plugin: AdapterPlugin) -> None:
        """Register a built-in plugin.

        Args:
            plugin: Plugin instance.

        Raises:
            ValueError: A plugin with the same name already exists.
        """
        if plugin.name in self._plugins:
            raise ValueError(f"plugin already registered: {plugin.name!r}")
        self._plugins[plugin.name] = plugin

    def plugins(self) -> Mapping[str, AdapterPlugin]:
        """Return a read-only view of the registry.

        Returns:
            Mapping of plugin name to plugin.
        """
        return types.MappingProxyType(self._plugins)

    async def invoke(
        self, name: str, client: RuntimeClient, args: dict[str, Any]
    ) -> dict[str, Any]:
        """Invoke the named plugin.

        Args:
            name: Plugin name.
            client: Runtime API client.
            args: Plugin arguments.

        Returns:
            Plugin result.

        Raises:
            RuntimeApiError: The plugin is not registered (``UNSUPPORTED_EXTENSION``).
        """
        plugin = self._plugins.get(name)
        if plugin is None:
            raise RuntimeApiError(
                make_error(
                    ErrorCode.UNSUPPORTED_EXTENSION,
                    f"unknown plugin: {name!r}",
                    known_plugins=sorted(self._plugins),
                )
            )
        return await plugin.invoke(client, args)
