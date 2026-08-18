# LIBERO-PRO Environment Calibration

## Current LIBERO MCP Runtime Contract

Use this file as a calibration reference only. For current MCP-based runs, use
structured MCP tools, do not issue file-based driver commands, and do not manually manage
`env_server.py`. Do not read BDDL files or hidden task definition files to infer
coordinates. Do not expect object world coordinates in `states.json`; localize
objects through images_cam + depth/back_project, segment, and wrist/high-res
artifacts when available.

Measured 2026-05-20 on `libero_10_with_mug` t0 (LIVING_ROOM frame) and t8
(KITCHEN frame). All probes use `move_to` with `gripper=-1` and tight
`step_clip≤0.015`.

## Key finding: PRO scene frames, picked per-task by table fixture

Each task scene uses one of the table fixtures below, which sets the entire
world-frame z origin. The OSC workspace and all pick/place altitudes shift
accordingly. **Check `states.json[0].state.robot0_eef_pos[2]` in the initial
state and branch on it.** Do not read BDDL files for this; use runtime state and
visual evidence.

| Fixture | eef home z | Table top z | Used by tasks |
|---|---|---|---|
| `living_room_table` | ≈ 0.68 | ≈ 0.43 | libero_10 t0, t1, t2, t4, t6, t7 (basket / plates / pudding) |
| `kitchen_table` | ≈ 1.17 | ≈ 0.90 | libero_10 t3, t5, t8, t9 (stove / cabinet / drawer / microwave) |
| `object` (low table) | ≈ 0.26 | ≈ 0.0 | `libero_object` grocery-into-basket (see altitudes below) |

(The two libero_10 frames match base LIBERO-10's LIVING_ROOM_SCENE vs
KITCHEN_SCENE split — the PRO BDDLs rename the prefix to `MAIN_TABLE_SCENE1` but
reuse the same underlying table assets. The `object` frame is a third, lower
table specific to `libero_object`; its per-item pick/place altitudes are in
`resources/libero/memory/project_libero_object_pro_done.md`.)

## OSC reachable workspace — LIVING_ROOM frame (eef home z=0.68)

Probed with `xy=(-0.20, 0.10)`, gripper open, `step_clip=0.008–0.015`.

### z bounds

| Target z | Final z | dist (m) | Note |
|---|---|---|---|
| 0.65 | 0.654 | 0.008 | safe |
| 0.62 | 0.628 | 0.008 | safe |
| 0.60 | 0.606 | 0.006 | safe |
| 0.58 | 0.588 | 0.008 | safe |
| 0.55 | 0.556 | 0.006 | safe |
| 0.53 | 0.538 | 0.008 | safe |
| 0.52 | 0.527 | 0.007 | safe |
| 0.51 | 0.516 | 0.006 | safe |
| 0.50 | 0.506 | 0.006 | safe |
| 0.48 | 0.486 | 0.006 | safe |
| 0.46 | 0.468 | 0.008 | safe |
| 0.44 | 0.448 | 0.008 | **floor edge** |
| **0.42** | **0.446** | **0.026** | **STALL** (eef refuses below 0.446) |
| | | | |
| 0.85 | 0.841 | 0.009 | safe |
| 1.00 | 0.991 | 0.010 | safe |
| 1.15 | 1.138 | 0.012 | soft upper edge |
| **1.25** | **1.236** | **0.016** | **STALL** at 1.236 |

**Effective z range: `[0.44, 1.15]` clean, hard upper at 1.25.**

### xy bounds at z=0.65

| Target xy | Final xy | dist (m) | Note |
|---|---|---|---|
| (−0.40, 0.10) | (−0.015, 0.107) | 0.55 | **catastrophic** — OSC flips IK, eef went +x |
| (+0.40, 0.10) | (+0.303, 0.097) | 0.10 | stalled at x≈+0.30 |
| (0.00, −0.35) | (+0.136, −0.362) | 0.14 | stalled, slight x drift |
| (0.00, +0.40) | (+0.223, −0.053) | 0.51 | **catastrophic** — OSC flipped, eef went −y |
| (+0.30, −0.30) | (+0.257, −0.291) | 0.10 | borderline corner OK |
| (−0.30, +0.30) | (−0.215, +0.198) | 0.13 | borderline corner OK |

**Effective xy region (single-step):** roughly `x ∈ [−0.30, +0.30]`,
`y ∈ [−0.30, +0.30]`. Aggressive commands beyond ±0.30 can flip the IK
branch and throw the eef to the wrong half-space. Stage long moves in
small steps and prefer paths that stay within ±0.20.

## OSC reachable workspace — KITCHEN frame (eef home z=1.17)

### z bounds (KITCHEN)

| Target z | Final z | dist (m) | Note |
|---|---|---|---|
| 0.93 | 0.942 | 0.012 | safe |
| 0.90 | 0.909 | 0.009 | floor edge |
| **0.88** | **0.909** | **0.029** | **STALL** at 0.909 |
| 0.85 | 0.909 | 0.059 | stalled (same floor) |
| 0.83 | 0.909 | 0.079 | stalled (same floor) |

**Effective z range: `[0.91, 1.30+]`** — kitchen floor is 0.91, ~0.47m higher
than living-room floor.

## Object-level reference heights

### LIVING_ROOM frame (z origin ≈ 0.43 table top)

| Object class | obs z (center) | est top z | safe pre-pos eef z | safe carry z | safe release eef z |
|---|---|---|---|---|---|
| flat box (cream_cheese, butter) | 0.445 | ≈ 0.48 | 0.56 | 0.65 | 0.56 |
| can (alphabet_soup, tomato_sauce) | 0.475 | ≈ 0.50 | 0.56 | 0.65 | 0.57 |
| tall bottle/carton (milk, ketchup, OJ) | 0.506 | ≈ 0.54 | 0.60 | 0.70 | 0.60 |
| basket (rim) | 0.432 (base) | ≈ 0.50 | n/a | n/a | n/a (target) |
| libero_mug (distractor) | 0.437 | ≈ 0.48 | n/a | n/a | n/a |

Carry/lift height for traversal: **z=0.65–0.95** (well below upper soft
limit 1.15). My libero_10 t0 used z=0.95 for travel — safe and consistent.

### KITCHEN frame (z origin ≈ 0.90 table top)

| Object class | obs z (center) | est top z | safe pre-pos eef z | safe carry z | safe release eef z |
|---|---|---|---|---|---|
| moka pot | 0.966 | ≈ 1.01 | 1.05 | 1.30 | 1.05–1.08 |
| cabinet drawer interior | varies | ≈ 1.09 | 1.20 | 1.30 | 1.13 |
| cabinet top side | n/a | ≈ 1.12 | 1.25 | 1.30 | 1.25 |
| stove cook_region | n/a | ≈ 0.93 | 1.05 | 1.20 | 1.00 |

## Practical rules going forward

1. **Always read `states.json[0].state.robot0_eef_pos[2]` before computing any z target.**
   ≈ 0.68 → LIVING_ROOM; ≈ 1.17 → KITCHEN; ≈ 0.26 → OBJECT. Use the matching
   frame table above.
2. **Never command an eef z below the per-frame floor.** Going to z=0.42
   in LIVING_ROOM stalls (eef refuses). Going to z=0.49 with prior xy
   drift may crash the env worker (`EOFError`).
3. **Never command xy outside ±0.30 in a single `move_to`.** OSC can flip
   IK branches and throw the eef to the wrong half-space.
4. **`set_gripper` after a stalled `move_to` is unsafe.** The previous
   t0 attempt closed the gripper above the bottle (eef stalled high)
   then opened it; this returned ok but the env was effectively desynced.
   Treat any `move_to` with `final_dist_m > 0.02` as a failure and
   recover (back to safe altitude, re-plan) before proceeding.
5. **Drop height matters for basket tasks.** Release at eef z=0.58 in
   LIVING_ROOM frame caused basket displacement Δ≈4–5 cm; z=0.53 keeps
   basket Δ<1 mm. Use z=0.53 for basket releases.
6. **Pi0 may refuse to grasp in cluttered scenes after a partial completion.**
   In libero_10_with_mug t0, after the first can entered the basket Pi0
   failed to pick the second (chunks=25, peak_lift=0). Workarounds: re-pre-pos
   precisely above the remaining target at z=`floor + 0.07`; re-localize and
   recover in the current episode if safe; or use an LLM-scripted grasp via
   `move_to` + `set_gripper` (last resort; unreliable for objects <6 cm — see
   `resources/libero/memory/feedback_scripted_pick_limits.md`).

## Calibration log files

Raw probe logs are preserved in:
- `{output_dir}/states.json` (one step entry per command — the per-command
  audit; each entry has `command`, `result`, `state`, `elapsed_s`).
- Only kept for the most recent driver session; reproduce by re-running
  the calibration with the snippet in the next section.

## Reproducer

Legacy calibration notes below describe the old file-driver flow. In the
current MCP runtime, use the runner-managed environment and call structured MCP
tools instead.

```bash
# For each z in 0.65 .. 0.42, call the MCP tool:
move_to({"xyz": [-0.20, 0.10, 0.65],
         "gripper": -1, "tol": 0.008, "step_clip": 0.010,
         "max_steps": 80})
# Then read states.json entry NN for final_eef_pos & final_dist_m.
```
