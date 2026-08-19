"""Rollout Runtime: a session-based execution runtime for embodied environments.

A top-level standalone package with no import dependency on ``zetta`` /
``robots``. Subpackage layering:

- ``api``       Pure protocol (stdlib; ``api.wire`` is an exception, allowing msgpack)
- ``core``      Execution core interfaces and obs/payload encode/decode (stdlib + numpy)
- ``gateway``   Control plane (stdlib + api; ``ray`` / ``rlinf`` / ``torch`` forbidden)
- ``workers``   Data-plane workers (``ray`` / ``rlinf`` allowed, lazily imported)
- ``transport`` ``base``/``inproc`` are pure stdlib; ``ray_channel`` allows rlinf
- ``backends``  Env and policy backends
- ``adapters``  Application-side adapters
- ``config``    Configuration schema and presets
- ``launch``    Local / Ray launchers
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0.dev0"
