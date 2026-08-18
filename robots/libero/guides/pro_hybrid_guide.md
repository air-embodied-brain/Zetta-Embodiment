# LIBERO-Pro Hybrid (Pi0.5 + LLM-in-the-loop) — Perception-Isolated Guide

You are picking up the **LIBERO-Pro** evaluation track in **perception-isolated**
mode — the only mode in this repository (the legacy oracle-state mode is not
included here).

> **Pi0.5 only does the grasp (`pi0_pick`). The LLM (you) handles every motion
> (`move_to`), every release, sequencing, retries — and you do not get GT object
> coordinates. You localize objects yourself from the depth + camera calibration
> the runtime dumps each step.**

This document layers on the base playbook. Read it first:

- [`strict_hybrid_guide.md`](./strict_hybrid_guide.md) — the perception protocol:
  back-projection localization, the perception artifacts
  (`images_cam/image_cam_NN.png` / `depths/depth_NN.npy` / `camera_meta.json`),
  the primitive vocabulary, the Rules (0/1/2/4/5), and the `strict_perception`
  audit format. **This is the source of truth for *how you localize*.**
- **This file** — LIBERO-Pro–specific setup, the four perturbation axes, the Pi0
  fullshot baseline, the frame split, and how perception isolation changes the
  P2-swap story.

> Your task comes from `states.json[0].task_language`; the BDDL is FORBIDDEN.
> Read the authoritative instruction (the BDDL `:language` tag, coord-free) that
> the runtime injects and obey it verbatim. Do **not** scrape the BDDL or import
> the benchmark: that is error-prone (wrong task-map index → wrong task) and a
> perception-isolation breach (the `:init` block holds the GT coordinates this
> mode withholds). For swap fixtures, localize them **visually** (§3.5), never
> from `:init`. You never hand-roll projection math — `back_project` applies
> `K⁻¹` + the cam→world extrinsic and returns the surface `world_xyz` under a
> pixel; just use it.

Whenever this guide says "see the perception protocol" or "the localization
snippet", it means `strict_hybrid_guide.md`. **Every rule there applies here**;
this guide only *adds* PRO constraints and tooling. Runner contract: call the
structured MCP tools (the runner owns the env server — do not start/stop it, and
issue no file-based commands); it is a single-episode run — no `reset`/`exit`,
recover in place or write an honest failure audit and `finish`.

## 0. What's different from oracle PRO mode (read this first)

| | oracle mode (not in this repo) | **perception (this guide)** |
|---|---|---|
| how launched | oracle-state run | `rpent/cli/main.py --libero-type pro` (or `LIBERO_TYPE=pro`); perception artifacts always dumped, coords withheld |
| `states.json` objects | full `objects:{name:[x,y,z]}` | **`object_names:[…]` only — NO coords** |
| how you learn the task | env prompt / scrape BDDL | **`states.json[0].task_language`** (authoritative `:language`, coord-free) — never read the BDDL |
| extra obs artifacts | agentview RGB only | **+ `images_cam/`, `depths/`, `world/` (agentview); `images_wrist/`, `depths_wrist/`, `world_wrist/` (wrist); hi-res pairs; `camera_meta.json`** — all via `back_project` |
| cameras | agentview only | **agentview (fixed, ~1m → ±8-13cm) + eye-in-hand wrist (moves with gripper, ±1-2cm when <20cm to target)** — see §3.6 |
| how P2 swap is solved | read swapped coords from `states.json[0]` | **localize the swapped objects by `back_project`** |
| how a swapped *fixture* site is found | read the swap BDDL `:init` block | **localize the fixture visually** (see §3.5) |
| run budget | short | **larger** — perception localization + manipulation is slower (raise `--max-turns` / `--planner-timeout-s`) |
| audit `regime` | `strict` | `strict_perception` |
| which image you pick pixels in | agentview RGB | `images_cam/image_cam_NN.png` (or hi-res) for **pixel-picking**; `images/image_NN.png` only for a sanity glance |

**The single most important conceptual shift.** In oracle PRO mode the headline
is "the hybrid beats Pi0 on P2 because it reads the *swapped* coordinates straight
out of `states.json[0]`, while Pi0 is prompt-/memory-blind." In **perception**
mode there are no coordinates to read — so the hybrid's P2 win now comes from
**seeing where the object is and back-projecting it**. This is a *stronger* claim
(no oracle state at all), and it is the whole point of running PRO in perception
mode: P1 (Task) is still won by reading the *language*; P2 (Position) is now won by
*perception*, not by an oracle.

## 1. Why LIBERO-Pro

LIBERO-Pro
([paper](https://arxiv.org/pdf/2510.03827), [repo](https://github.com/Zxy-MLlab/LIBERO-PRO))
perturbs each base task along five axes; all end-to-end VLAs (OpenVLA / Pi0 /
Pi0.5 / UniVLA) collapse on the two strongest:

| Axis | Suffix | Paper column | What changes | Headline result |
|---|---|---|---|---|
| **Task** | `_task` | **P1** | Instruction + goal predicate inverted | All VLAs ≈ 0.0 |
| **Position** | `_swap` | **P2** | Object **and fixture** initial positions swapped | All VLAs 0.0–0.4 |
| Semantic | `_lan` | — | Instruction paraphrased; goal unchanged | VLAs handle (memorize visual) |
| Object | `_object` | — | Object appearance / colour / scale | VLA visual policy stressed |
| Environment | `_environment` | — | Table / scene swapped | Visual policy stressed |

The agentic hybrid wins on **P1 and P2** by routing the language channel and the
*perceived* spatial-state channel through the LLM. Object and Environment
perturbations enter through the Pi0 vision channel and the hybrid inherits the
VLA's weakness there — declare that upfront, don't oversell. **In perception
mode, the Object/Environment axes also stress your own localization** (a
recolored or rescaled object is harder to pixel-pick), so lean on the
multi-pixel-median tip from the perception protocol.

## 2. Setup (do these once on a fresh checkout)

Run the idempotent installer from the repo root:

```bash
bash scripts/install_libero_pro_plus.sh
```

It does all four steps below: the liberopro editable install, applies the
benchmark-registration patch, syncs the authoritative HF dataset snapshot, and
verifies with the `get_benchmark(...).get_task(0).language` check. The perception
observables (depth + both cameras + hi-res) are unconditional once the runner
launches with `--libero-type pro`; there is nothing perception-specific to
install beyond the PRO setup itself.

### 2.1. LIBERO-PRO repo

Cloned at `${LIBERO_PRO_PATH:-/path/to/LIBERO-PRO}/` from
`https://github.com/RLinf/LIBERO-PRO.git` and installed editable into the openpi
venv:

```bash
python -m pip show liberopro
# Name: liberopro  Version: 0.1.0  Location: ${LIBERO_PRO_PATH:-/path/to/LIBERO-PRO}
```

### 2.2. Apply the benchmark-registration patch

The upstream `__init__.py` does **not** expose the 16 perturbation suites through
`get_benchmark()`. Our patch
[`scripts/liberopro_register_perturbations.patch`](../../../../scripts/liberopro_register_perturbations.patch)
adds them and overrides `Task.language` to read each BDDL's actual `:language`
tag (so the perturbed instruction reaches Pi0 / hybrid).

```bash
cd ${LIBERO_PRO_PATH:-/path/to/LIBERO-PRO}
git apply <repo-root>/scripts/liberopro_register_perturbations.patch
```

If already applied (likely), `git status -s` shows clean. If you reinstall
liberopro, re-apply.

### 2.3. Huggingface dataset (authoritative)

The LIBERO-PRO git repo ships **incomplete / broken** init files for several
perturbation suites (e.g. `libero_spatial_swap` has 0 BDDLs; some
`libero_spatial_task` `.pruned_init` files are 0 bytes). Treat the git repo as
unreliable for perturbation data. The full, correct set lives on Huggingface
([`zhouxueyang/LIBERO-Pro`](https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro)),
persisted locally at:

```
${LIBEROPRO_DATASET_PATH:-/path/to/liberopro_hf}/
├── bddl_files/                    16 perturbation suites, 10 BDDLs each
└── init_files/                    16 perturbation suites, 10 init files each
```

Covers `{libero_spatial, libero_object, libero_goal, libero_10} × {swap, task,
lan, object}`. The installer syncs this into the liberopro install (overwriting
the broken upstream files); if the persistent copy is gone, re-download with:

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='zhouxueyang/LIBERO-Pro', repo_type='dataset',
                  local_dir='${LIBEROPRO_DATASET_PATH:-/path/to/liberopro_hf}',
                  allow_patterns=['bddl_files/**','init_files/**'])"
```

### 2.4. Verify

```bash
LIBERO_TYPE=pro python -c "
import liberopro.liberopro.benchmark as bench
for n in ['libero_spatial_task','libero_spatial_swap','libero_spatial_lan']:
    b = bench.get_benchmark(n)(); t = b.get_task(0)
    print(f'{n} t0: {t.language!r}  trials={len(b.get_task_init_states(0))}')"
```

Expected:
```
libero_spatial_task t0: 'Pick the akita black bowl not between the plate and the ramekin and place it on the plate'  trials=50
libero_spatial_swap t0: 'Pick the akita black bowl between the plate and the ramekin and place it on the plate'  trials=50
libero_spatial_lan  t0: 'lift the black bowl between the plate and ramekin and set it on the plate'  trials=50
```

## 3. PRO-specific environment gotchas

Everything in `strict_hybrid_guide.md` applies. The following are **additional**
PRO constraints (most live in detail at [`env_calibration.md`](./env_calibration.md)).

### 3.1. Three scene frames, picked per-task

PRO scenes use one of three table fixtures; the eef home z differs by up to
~0.9 m.

| Fixture | eef home z | Table top z | xy reachable | Where |
|---|---|---|---|---|
| `living_room_table` | ≈ 0.68 | ≈ 0.43 | `(x∈±0.30, y∈±0.30)` | basket / plate / pudding |
| `kitchen_table` | ≈ 1.17 | ≈ 0.90 | `(x∈±0.30, y∈±0.30)` | stove / cabinet / drawer / microwave |
| `object` (low table) | ≈ 0.26 | ≈ 0.0 | `(x∈±0.30, y∈±0.30)` | `libero_object` grocery-into-basket |

**Mandatory check at session start: read `states.json[0].state.robot0_eef_pos[2]`**
(via `view_driver_state({"step": 0})`). ≈ 0.68 → LIVING_ROOM; ≈ 1.17 → KITCHEN;
≈ 0.26 → OBJECT. Pick `pre_pos_z` / `carry_z` / `release_z` accordingly (per-item
OBJECT-frame altitudes are in
`resources/libero/memory/project_libero_object_pro_done.md`). Sending a
wrong-frame z (e.g. KITCHEN coordinates while the env is in LIVING_ROOM frame)
crashes the env worker (EOFError, silent state loss).

> **Perception note.** This proprioceptive z is *not* an object coordinate — it's
> the robot's own pose, which `states.json` still gives you in perception mode.
> Reading it to pick the frame is fine. It also doubles as a free depth sanity
> check: your back-projected table z should sit ≈0.25 m below eef home z (≈0.43 in
> LIVING_ROOM, ≈0.90 in KITCHEN). If a back-projection lands far from that, you
> picked a wrong pixel.

For `libero_spatial_*` you are always in **KITCHEN frame**. Standard heights:

```
pre_pos_z = 1.05   # ~7 cm above objects at z≈0.97
carry_z   = 1.10   # safe traversal, well under upper limit 1.15
release_z = 1.01   # ~7 cm above plate top at z≈0.90
```

These are *altitude* knobs (robot-frame), not object positions — you still derive
the object's **xy** by `back_project` every time.

### 3.2. xy single-step ±0.30 cap

OSC flips IK branches if you command `|x|>0.30` or `|y|>0.30` in a single
`move_to` (the eef lands in the wrong half-space and corrupts the run). **Never
command beyond ±0.30 in a single move** — split into carry-z waypoints. (Detail
in `env_calibration.md`.)

### 3.3. Slow long-distance carry for swap variants

P2 swap can move a large object 15 cm across the table. `step_clip=0.025` lets it
slip in the gripper and the object ends up centimetres off target. Mitigation
(proven on `_swap` t0, carries over directly):

- `carry_z = 1.15` (higher than usual 1.10)
- `step_clip = 0.020` (slower)
- **Re-localize mid-travel** instead of trusting a cached `object_xyz - eef_xyz`
  offset: `Read images_cam/image_cam_NN.png` (or the hi-res `images_cam_hi/`)
  mid-carry and call `back_project` on the carried object's pixel; if the
  perceived offset drifted >5 mm from post-pick, re-pre-position and re-`pi0_pick`.

### 3.4. Task language is from BDDL, not filename

After the patch, `get_task(i).language` returns the perturbed `:language` tag, and
the env passes it to Pi0 as the prompt (surfaced to you as
`states.json[0].task_language`). For `_task` and `_lan` this is the perturbed
instruction. **Don't override it** — falsifying the VLA's prompt-blindness is the
point. You read the same language to decide *which* object to localize and place.

### 3.5. ⚠ Swap moves FIXTURES too — and you must localize them visually

This is the biggest perception-mode trap, and it is **specific to `_swap`**. The
P2 perturbation does not only swap loose objects; for `libero_goal_swap` it swaps
entire **fixtures** (stove ↔ cabinet ↔ wine_rack), so a goal predicate like
`On(bowl, flat_stove_1_cook_region)` now points at wherever the *stove* was
relocated to, and there are no coordinates in `states.json` to read. See
`resources/libero/memory/feedback_swap_perturbs_fixtures.md`.

In **oracle** mode the documented fix is to read the swap BDDL `:init` block and
recompute the fixture site's world coordinates. **That is forbidden here** — the
`:init` block is ground-truth geometry. In **perception** mode you instead:

1. Identify the target fixture by name from the (perturbed) task language and
   `state.object_names`.
2. **Localize the fixture's predicate site visually** in
   `images_cam_hi/image_cam_hi_NN.png`: pick pixels on the stove's cook region /
   cabinet top surface / rack top shelf, then `back_project` (sample 3–5 pixels,
   median the xy — fixtures are large and the surface you want is the *placement*
   surface, not the nearest edge).
3. Carry target xy = perceived site xy; descend to the perceived site z + a small
   clearance, `release`, then **retreat with gripper open** — the predicate often
   fires *during* the settle, not at the release step itself.

So the swap-fixture problem becomes "find the fixture in the image" rather than
"read where the BDDL put it." Loose-object swaps (e.g. `libero_spatial_swap`,
where only bowls/plates move) are simpler: just localize each object by
`back_project` as usual; their new positions fall straight out of the depth.

### 3.6. Two cameras — agentview = IDENTITY, wrist = GEOMETRY

Do **not** restate the two-camera protocol here — the strict guide's **First-step
perception protocol** is the source of truth: agentview / agentview-hi is the
semantic identity authority (decides *which* object/surface satisfies the task
language + relation); wrist / wrist-hi refines geometry for the *same* candidate
(accept only within ~3–5 cm of the agentview anchor, never average, basket/cavity
excepted). Both maps share one world frame, read via
`back_project({"camera":"wrist"})`; the optional `segment` honours both cameras
(`{"camera":"agentview"|"wrist"}`, `min_score` default 0.2, or `point:[row,col]`).

**PRO implication:** `_swap` (and `_task`) can *invert* target or destination
semantics, so **never reuse base-task or recipe coordinates** — the object
satisfying the relation may now sit where the base task never put it. Let
agentview decide *what*; the near-vertical wrist is BAD at identity (locks onto
look-alikes) and only refines *where*.

### 3.6c. Mandatory pre-task perception pass

Also owned by the strict guide's First-step perception protocol: before ANY
pick/place, build the localization table (one row per task-relevant entity) and
pass the FINAL READY CHECK. **PRO implication:** in single-attempt mode a
wrong-target first grab is unrecoverable, and under `_swap` the "right" target is
exactly the one you'd get wrong by habit — so localizing all entities up front is
cheap insurance, not optional.

### 3.6b. Hi-res perception channel (1024×1024)

Also owned by the strict guide's Hi-res perception channel section (1024×1024
pairs each step; prefer the hi image for identification; `back_project` defaults
`resolution="high"`; never mix pixel grids). **PRO implication:** hi-res fixes
*which* object you point at — decisive for the recolored/rescaled Object axis and
for telling swapped same-shape groceries apart — but does NOT change metric
accuracy or replace the wrist coarse→fine refinement. **Identity caveat:** SAM3
gives two same-shaped, different-brand objects the SAME category (it labelled the
tomato-sauce and alphabet-soup cans identically) — use its mask only as a category
candidate, then READ the label yourself in the hi-res crop to assign identity;
never let SAM3's category settle a brand choice.

## 4. The four-cell experiment per (base task, seed)

For each base task you claim coverage on, generate four runs — every run is a
perception cell:

| Suite | Variant | Perception-mode expectation |
|---|---|---|
| `libero_spatial`        | base sanity | Pi0 and hybrid both pass |
| `libero_spatial_task`   | **P1 Task** | Pi0 ✗ (picks base target); hybrid ✓ (LLM flips target from instruction, then localizes it) |
| `libero_spatial_swap`   | **P2 Position** | Pi0 mixed; hybrid ✓ — **by localizing the swapped object/fixture from depth, not from state** |
| `libero_spatial_lan`    | Semantic | Both pass (paraphrase invariant) |

Replace `spatial` with `object`, `goal`, or `10` for the other base suites.

### 4.1. Hybrid run — the MCP runner

Launch a cell with the CLI; the runner owns `env_server.py`, exposes the
structured tools, and runs single-attempt:

```bash
python rpent/cli/main.py --env libero --suite <suite> --task <n> --seed <k> \
    --libero-type pro --planner claude_code --model claude-opus-4-8

# e.g. --suite libero_spatial_task 0 (P1) · libero_spatial_swap 0 (P2) ·
#      libero_goal_swap 2 (P2 fixture swap) · libero_10_task 5 (long horizon)
```

`--libero-type pro` may be given as `LIBERO_TYPE=pro` instead.

Audit + recipe land in the run's `output_dir` (`output_dir` and `recipe_tag`
arrive in your first message):

```
{output_dir}/{recipe_tag}.json           <- you write this (write_text_file)
{output_dir}/recipe_{recipe_tag}.jsonl   <- exported automatically by the runner
```

Do NOT write into `resources/libero/results_*_pert/` — that tree is a **read-only
seed-0 reference corpus**, not a write target.

### 4.2. Environment server is runner-owned

There is no manual driver to launch and no REPL to drive: the MCP runner starts,
manages, and tears down `env_server.py` for you and blocks each tool call until
the next `states.json` entry is dumped. Do not start, stop, or background it, and
do not poll for readiness.

### 4.3. Pi0 fullshot baseline

The baseline is Pi0.5 driving the task end-to-end with the runtime's own
(perturbed) `task_language`. There is **no standalone baseline CLI in this repo**;
the numbers to compare against are the recorded full-shot results (the team's
`SUCCESS_RATES` table). **Do not invent a `pi0_baseline.py` path.**

Pi0 never sees object coords in either mode, so there is no perception variant of
the baseline. Expected behavior:

- **P1 (task):** Pi0 "succeeds at the wrong task" — it picks the *base*-task
  target object, places it on the plate, and `libero_terminated=False` because the
  goal predicate names a different object. This is exactly the gap the hybrid
  closes.
- **P2 (position):** Pi0 picks / places at the *base* (un-swapped) location.

### 4.4. Audit JSON — PRO + perception fields

Start from the `strict_perception` audit schema in the perception protocol, and
add the PRO fields:

```jsonc
{
  "suite": "libero_spatial_swap",
  "task_id": 0,
  "seed": 0,
  "regime": "strict_perception",
  "perturbation_type": "swap (P2 Position perturbation)",
  "perturbed_task_language": "<actual :language from states.json[0].task_language>",
  "perturbation_semantics": "<what swapped — objects and/or fixtures>",
  "expected_baseline_behavior": "<predicted Pi0 failure: e.g. picks base location>",
  "strategy_notes": "HOW you localized — which pixel(s) in images_cam, back-projected world xyz; for swap, how you found the relocated object/fixture",
  "pick_result": { /* the pi0_pick step's result */ },
  "final_state": { /* latest states.json entry's `state` field */ },
  "libero_terminated": true
}
```

`strategy_notes` **must** describe the localization (pixel → `back_project` →
world xyz). For swap cells, explicitly note that the relocated object/fixture was
found by perception, not by reading coords. If unrecoverable after honest
exploration, write `libero_terminated: false` with what you tried and which step
failed — never warp (teleport primitives are deleted; see Rule 4 in the
perception protocol). Write the audit with `write_text_file` to
`{output_dir}/{recipe_tag}.json`, then call `finish`.

## 5. Perception-protocol rules — PRO clarifications

- **Rule 0 (use images for reasoning).** Even more critical under PRO: P2 swap can
  move a large object/fixture clear across the table, and you have *no*
  coordinates to fall back on. `images_cam/image_cam_NN.png` + depth are your only
  spatial truth — open `images_cam/` and describe the scene before deciding
  targets.
- **Rule 1 (no `pi0_end_to_end`).** Pi0 does the grasp via `pi0_pick`; the LLM
  scripts every motion + release. Under PRO this is doubly important — handing
  back to Pi0 means handing back to the prompt-blind / memorized-place habit you
  are trying to falsify.
- **Rule 2 (single-episode current run).** This is a one-shot eval: do not call
  `reset` or `exit`. Recover *within* the episode when safe (re-localize,
  re-pre-position, re-`pi0_pick`, walk the prompt ladder,
  `rotate_pitch`/`move_pose`); otherwise write an honest stuck/failure audit and
  `finish`. `_swap` typically needs an in-episode retry — document it.
- **Rule 4 (no teleport).** `set_object_pose`, `articulate_to`, `js_move_to`,
  `carry_object` are deleted. A goal past OSC reach with no physical approach →
  honest `libero_terminated:false`.
- **Rule 5 (assume solvable).** A localization that moves the gripper into thin
  air means you picked a wrong pixel (a reflection, a rim, a decoy object under
  the Object perturbation), not that the cell is unreachable. Re-look, re-pick,
  re-`back_project` before concluding failure.

## 6. Existing corpus

```
robots/libero/guides/
├── pro_hybrid_guide.md                 <- this file
├── strict_hybrid_guide.md              <- perception protocol + Rules (source of truth)
└── env_calibration.md                  <- OSC frame bounds + safe altitudes
scripts/
└── liberopro_register_perturbations.patch
resources/libero/memory/                <- MEMORY.md index + feedback_*/project_* notes
resources/libero/results_spatial_pert/  <- read-only seed-0 reference corpus
resources/libero/results_{object,goal,10}_pert/   <- same, other suites (seed-0)
```

Before you start, **read the auto-memory**: `resources/libero/memory/MEMORY.md`
(one-line hooks, auto-injected via CLAUDE.md). For perception PRO cells always
open `feedback_no_teleport_rule.md` and — for any `_swap` cell —
`feedback_swap_perturbs_fixtures.md` (what swaps, and why you re-find the
relocated fixture visually). For bowl→plate spatial tasks also read
`feedback_bowl_eef_y_offset.md`; for cluttered picks, `feedback_pi0_pick_full_prompt.md`;
after two failed retries, `feedback_failure_forensics.md`. The
`resources/libero/results_*_pert/` recipes are **inputs** (technique priors) —
consult them for prompt ladders, staging, and target zones, but never reuse their
coordinates (re-derive every xyz from THIS scene) and never write there.

## 7. What to do next (priority order)

1. **Extend spatial to all 10 tasks at seed 0**, four perception cells each
   (base / `_task` / `_swap` / `_lan`). For hybrid runs, use the seed-0 reference
   recipes in `resources/libero/results_spatial_pert/` as *technique* starting
   points — the pick step is usually identical; the place target changes for
   `_swap`, the target object changes for `_task`. Never reuse their coordinates.
2. **Scale to seeds beyond 0** (50 trials per task). Recipes must re-localize per
   scene; a perception recipe is *data-flow* (perceive → plan → act), never
   hard-coded xyz.
3. **Replicate on `libero_object`, `libero_goal`, `libero_10`.** Frame split
   applies (read `states.json[0].state.robot0_eef_pos[2]`). For
   `libero_goal_swap`, apply §3.5 — localize the **swapped fixture** visually.
4. **Aggregate into a main table** `(suite × perturbation × {Pi0, hybrid})`. The
   headline number is the conditional: of the seeds Pi0 fails on, what fraction
   does the *perception* hybrid solve? Because there is no oracle state anywhere in
   the hybrid's reasoning here, this is the strongest single statistic the agentic
   decomposition can claim.

## 8. Quick reference for a brand-new session

```bash
# 1. Sanity-check the liberopro patch (perturbed language must show)
LIBERO_TYPE=pro python -c \
  "import liberopro.liberopro.benchmark as b; print(b.get_benchmark('libero_spatial_task')().get_task(0).language)"
# -> must read 'Pick the akita black bowl not between ...' (the perturbed text)

# 2. Read the auto-memory: resources/libero/memory/MEMORY.md

# 3. Launch a perception cell (runner owns env_server; single-attempt)
python rpent/cli/main.py --env libero --suite libero_spatial_swap --task <N> --seed 0 \
    --libero-type pro --planner claude_code --model claude-opus-4-8
```

Then, inside the run:

4. `view_driver_state({"step": 0})` → read `state.robot0_eef_pos[2]` to pick the
   frame (§3.1). `Read images_cam/image_cam_00.png` (or hi-res) and
   `view_camera_meta`. Localize the target (and, for `_swap`, the relocated
   object/fixture) with `back_project` — run the mandatory pre-task perception pass
   (§3.6c). Plan, then execute one structured tool at a time.
5. `write_text_file` the audit to `{output_dir}/{recipe_tag}.json`
   (`regime: strict_perception`) before `finish`; the recipe
   `{output_dir}/recipe_{recipe_tag}.jsonl` is exported automatically by the runner.

When in doubt about *how to localize* or a primitive, the source of truth is
[`strict_hybrid_guide.md`](./strict_hybrid_guide.md); about *PRO setup /
perturbation semantics*, see
[`scripts/install_libero_pro_plus.sh`](../../../../scripts/install_libero_pro_plus.sh)
and §2.
