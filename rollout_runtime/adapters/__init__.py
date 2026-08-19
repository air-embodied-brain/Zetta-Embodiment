"""Application-side adapters.

v1 only ships three client adapters, all thin wrappers around
``RuntimeClient``, and **none of them may bypass the Gateway**:

| Adapter | File |
|---|---|
| Gym | ``gym_adapter.py`` |
| Evaluation | ``eval_adapter.py`` |
| Zetta LIBERO runtime facade | ``zetta/runtime_env_client.py``, ``zetta/runtime_policy_client.py`` |

The Agent Toolkit adapter is **not in v1**: the loop and stopping conditions
for ``vla_execute`` / ``pi0_pick`` / ``pi0_doubled`` remain in
``LiberoPrimitives``; only ``_vlm_chunk`` inside them is replaced with a
single ``policy_step`` call.
"""

from __future__ import annotations

__all__: list[str] = []
