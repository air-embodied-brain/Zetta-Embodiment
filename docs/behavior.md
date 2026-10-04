# BEHAVIOR / OmniGibson

The `behavior` Runtime family provides the environment side of the physical
intelligence BEHAVIOR policy integration already present in
`zetta/policies/openpi/policies/behavior_policy.py`. It creates one R1Pro agent
in an OmniGibson `BehaviorTask` per Runtime slot and exposes:

- a 224×224 main RGB image;
- two named wrist views, kept in left/right order;
- the full R1Pro proprioception vector (the Pi0.5 transform selects its 23
  policy state values); and
- a finite 23-dimensional action chunk.

OmniGibson and BDDL are optional external checkouts. They are intentionally
lazy: importing `rollout_runtime` and running contract tests does not import a
GPU simulator. Install the lightweight package extra, then install matching
upstream simulator checkouts and assets in a separate environment:

```bash
python -m pip install -e ".[behavior]"
python scripts/deployment/prepare_behavior.py \
  --omnigibson-root /path/to/OmniGibson \
  --bddl-root /path/to/bddl \
  --assets-root /path/to/behavior-assets
```

Copy `rollout_runtime/config/presets/behavior_pi05.yaml`, replace its external
paths, and run the lifecycle check before a rollout:

```bash
python scripts/deployment/smoke_behavior.py \
  --config rollout_runtime/config/presets/behavior_pi05.yaml
```

The preset is a wiring example. Activity names, camera names, OmniGibson
revision, task assets, and model checkpoints must be frozen together for a
reproducible campaign. This repository does not download or vendor those
artifacts.
