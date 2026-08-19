"""Launchers.

- ``local.py``: ``build_local_runtime(config) -> RuntimeGateway``. The gateway
  and both worker groups run in the same process; ``transport.kind`` decides
  whether the data plane is ``inproc`` or a real rlinf ``Channel``.
- ``ray_launch.py``: launches both worker groups via ``Cluster`` + placement,
  with workers actually running in separate Ray processes, following the
  pattern in ``examples/embodiment/train_async.py:80-95``; group names come
  from config. With 0 local accelerators, ``NodePlacementStrategy`` must be
  used.

``ray_launch`` is **not re-exported here**: importing it requires rlinf, which
is not installed in ``.venv-runtime`` (it is located via ``rlinf_bootstrap``
under ``third_party/rlinf``). Callers that need it should import explicitly:
``from rollout_runtime.launch.ray_launch import build_ray_runtime``.
"""

from __future__ import annotations

from rollout_runtime.launch.local import (
    LocalRuntime,
    build_local_components,
    build_local_runtime,
)

__all__ = ["LocalRuntime", "build_local_components", "build_local_runtime"]
