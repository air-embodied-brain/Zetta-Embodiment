# Third-Party Notices

## RLinf-derived embodied runtime code

Zetta includes selectively adapted files from the Apache-2.0 licensed
[RLinf project](https://github.com/RLinf/RLinf), source commit `9ad44393`.

- `rlinf/envs/action_utils.py` and `rlinf/envs/utils.py` → `zetta/compat/`
- `rlinf/envs/venv/venv.py`, `rlinf/envs/libero/venv.py`,
  `rlinf/envs/libero/utils.py`, and `rlinf/envs/libero/libero_env.py` →
  `zetta/envs/libero/`
- `rlinf/envs/maniskill/maniskill_env.py` and `rlinf/envs/maniskill/utils.py` →
  `zetta/envs/maniskill/`
- `rlinf/models/embodiment/openpi/` → `zetta/policies/openpi/`
- `rlinf/models/embodiment/base_policy.py`, the OpenPI-referenced files in
  `rlinf/models/embodiment/modules/`, `rlinf/utils/{nested_dict_process,pytree,rot6d}.py`,
  and `rlinf/envs/robocasa365/utils.py` → `zetta/policies/openpi/_compat/`

The adaptations replace RLinf package imports and registries with Zetta-local
interfaces, narrow the supported model surface to OpenPI, and make optional
simulator imports lazy. Source files retain their original copyright and
Apache-2.0 notices where applicable. See `LICENSE` for Zetta's license; the
RLinf-derived portions remain subject to Apache License 2.0.
