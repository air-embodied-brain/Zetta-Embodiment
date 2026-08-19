# LIBERO Recovery Tool Additions

This branch keeps the original LIBERO recovery tools and adds a proposal and
bounded-motion layer. The public Role1 names are namespaced as
`libero.<tool>`; persisted recovery bundles continue to use the short names.

| Tool | Role | Current backend | Advances LIBERO |
| --- | --- | --- | --- |
| `graspgen` | grasp proposal | optional `GRASPGEN_URL` service; otherwise structured `unconfigured` result | no |
| `candidate_freshness` | candidate lifecycle check | local EEF age/displacement check | no |
| `curobo_reachability` | reachability gate | local workspace/distance fallback (`certificate_level=workspace_only`) | no |
| `curobo_motiongen_pregrasp` | bounded pregrasp | audited LIBERO OSC fallback | yes |
| `mink_reach` | constrained local reach | audited LIBERO OSC fallback after reachability gate | yes |
| `mink_precontact` | open-gripper precontact | audited LIBERO OSC fallback | yes |
| `mink_engage_close` | close and micro-advance | audited LIBERO OSC fallback | yes |
| `mink_pull` | incremental closed-gripper pull | audited LIBERO OSC fallback | yes |
| `progress_liveness` | progress check | local motion-history check | no |

No fallback is reported as a full CuRobo collision/path certificate or a full
Mink IK/contact solve. A real backend can be added behind the same tool names
once its endpoint contract and runtime resource are available.

## Role1 and privilege boundary

All additions are in the LIBERO Role1 catalog and complete Toolkit. Proposal-only
tools return `no_op_verified=true` and `environment_advanced=false`; the Actor
accepts that result without weakening the existing hard check for mutating
recovery tools. Simulator privileged state remains local to Critic and the
audited primitive. Role1 may receive bounded scalar progress/contact evidence
when `--allow-privileged-evidence` is enabled; absolute coordinates remain
withheld from Actor-visible payloads.

`base_se2_astar` is intentionally not added: LIBERO Panda uses a fixed base,
so RoboCasa mobile-base staging has no equivalent action surface here.

## Video layout

Each rollout writes to its unique attempt directory:

```text
<attempt>/
  videos/
    episode_agentview.mp4
    episode_wrist.mp4
    episode_agentview_multiview.mp4   # when available
    VIDEO_INDEX.json
    README.md
  visual-evidence/
    visual-evidence-manifest.json
    privileged-state-summary.json
```

`videos/VIDEO_INDEX.json` maps camera, task, generation, logical rollout,
attempt, seed, and outcome to the actual files. Existing historical output
directories are never reused or overwritten.

## Validation status

The additions are covered by unit and Role1 wiring tests. Real score changes
require configured VLA/Role1 services and (for non-fallback planning) actual
CuRobo/Mink runtimes; this checkout currently has no such endpoints configured.
