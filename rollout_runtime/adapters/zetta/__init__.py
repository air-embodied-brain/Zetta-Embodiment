"""The "zetta seam": legacy client-shaped adapters over the shared Runtime.

See ``rollout_runtime/adapters/__init__.py``'s milestone table. This
subpackage exists so ``robots/libero/run_evolution_rollout.py`` can drive a
LIBERO episode through ``rollout_runtime.cli serve --launch ray`` with the
existing ``robots/libero/tools.py::LiberoPrimitives`` orchestration loop
unchanged, instead of spawning a standalone ``env_server.py``/``vla_server.py``
subprocess pair.
"""

from __future__ import annotations

__all__: list[str] = []
