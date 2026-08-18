# Strict Hybrid LLM + Pi0.5 — Perception-Isolated Guide

You are taking over a hybrid LIBERO experiment in **perception-isolated** mode
— the only mode in this repository (the legacy oracle-state mode, where the
state JSON carried GT object coordinates, is not included here).

> **Pi0.5 only does the grasp (`pi0_pick`). The LLM (you) handles every motion
> (`move_to`), every release, sequencing, retries — and you do not get GT
> object coordinates. You localize objects yourself from the depth + camera
> calibration the driver dumps each step.**

## What's different from legacy oracle mode (read this first)

| | legacy oracle mode (not in this repo) | **perception (this guide)** |
|---|---|---|
| how object coords are withheld | (none — full GT coords in `state`) | **the runner withholds them: `states.json` carries `object_names` only, no coordinates** |
| `states.json` objects | full `objects:{name:[x,y,z]}` | **`object_names:[…]` only — NO coords** |
| how you learn the task | env prompt / BDDL | **`states.json[NN].task_language`** (authoritative `:language`, coord-free) — never scrape the BDDL |
| extra obs artifacts | `images/image_NN.png` only | **+ `images_cam/`, `depths/`, `world/` (agentview); `images_wrist/`, `depths_wrist/`, `world_wrist/`, `wrist_meta/` (wrist); top-level `camera_meta.json`; hi-res `images_cam_hi/`, `world_hi/`, `images_wrist_hi/`, `world_wrist_hi/`** |
| cameras | agentview only | **agentview (fixed, ~1m → ±8–13 cm) + eye-in-hand wrist (moves with gripper, ±1–2 cm when <20 cm to target)** — coarse→fine, see below |
| how you get an object's xyz | read `state["objects"][name]` | **pick the object pixel, then `back_project({"row":ROW,"col":COL,"step":NN})` (K⁻¹+extrinsic already done); refine with the wrist map up close** |
| how you confirm the grasp | GT object-lift oracle | **grasp-only, no oracle — you judge the grasp from gripper width + wrist cam (Rule 1 / 1b)** |
| how you pick which of two identical objects | read their distinct coords | **by SPATIAL RELATION from `task_language` (elevation / left-right), never by `_1`/`_2` name** |
| cell budget | 600 (short suites) | **1200** — perceptual localization + manipulation is slower |
| audit `regime` | `strict` | `strict_perception` |

How localization works (the core of this mode):

The driver already back-projects EVERY pixel for you into world coordinates and
saves them as `world/world_NN.npy` (agentview) and `world_wrist/world_wrist_NN.npy`
(wrist) — both in the SAME world frame. **You do NOT write back-projection math;
you pick a pixel and call `back_project`, which indexes the map for you.**

> **Path convention:** below, a bare file name refers to the file inside its own
> `output_dir` subdirectory — e.g. `image_cam_hi_NN.png` lives at
> `images_cam_hi/image_cam_hi_NN.png`, `world_NN.npy` at `world/world_NN.npy`,
> `image_wrist_hi_NN.png` at `images_wrist_hi/image_wrist_hi_NN.png` (see the obs
> artifacts row above for the full directory list). `NN` is the zero-padded step
> index. In practice you rarely hand-build these paths — each primitive tool
> already returns the resolved `image_cam_hi_path` etc. for the new step.

1. Read `image_cam_hi_NN.png` — agentview RGB **in the calibration frame**
   (vertical-flip of the raw buffer). This is the image you pick pixels in.
   `image_NN.png` is the Pi0-rotation frame; **do not pick pixels there**.
2. Find the target object visually → pixel `(row, col)`.
3. Read its world xyz directly: `back_project({"row":ROW,"col":COL,"step":NN})`.
   Sample 3–5 pixels on the object's top surface and median the returned xy
   (robust to one mis-picked rim/edge pixel). That's the object's surface point
   in world frame.

### Hi-res perception channel (ON BY DEFAULT, 1024×1024)

The driver dumps, every step, a 1024×1024 pair per camera IN ADDITION to the
256 files: `images_cam_hi/image_cam_hi_NN.png` + `world_hi/world_hi_NN.npy`
(agentview) and `images_wrist_hi/image_wrist_hi_NN.png` +
`world_wrist_hi/world_wrist_hi_NN.npy` (wrist).
**PREFER the hi pair for looking and identification** — a far object spans 4×
more pixels, and package text/label art is actually legible (e.g. "Cream
Cheese" vs "BUTTER" boxes are readable at 1024 and indistinguishable smears
at 256). If the hi files are absent, everything below works unchanged at 256.

- A hi-res pixel `(row,col)` indexes ONLY the hi-res world map (`back_project`
  default `resolution:"high"`; float16, same world frame). Never index a 1024
  pixel into the 256 map or vice versa — pass `resolution:"low"` only for a
  pixel taken from a 256 image (convert by dividing/multiplying by 4 if needed).
- The `segment` command automatically uses the hi-res frame when present
  (its `centroid_pixel`/`box` are then in 1024 coords).
- Metric accuracy of mask-median localization is the SAME at both resolutions
  (the residual ~2 cm error is the surface-vs-center offset, not pixel size) —
  the hi channel is for **identification**, not for replacing the wrist-cam
  fine-localization protocol.
- Disk note: hi files keep only the LAST ~5 steps (rolling window); the 256
  history is complete. If the hi files are absent, everything below works
  unchanged at 256.

### Two cameras — agentview = IDENTITY, wrist = GEOMETRY

**Roles are NOT symmetric (this is the core discipline, from the 80-task
localization sweep):**
- **agentview** (`image_cam_hi_NN.png` + `world_hi/world_hi_NN.npy`, ~1 m away):
  the **semantic IDENTITY authority** — decides WHICH object/surface satisfies
  the task language + spatial relation. At 1024 a far object spans enough pixels
  to read labels/shape. Its metric precision is only ±8–13 cm, but identity is
  its job, not millimetres.
- **wrist** (`image_wrist_hi_NN.png` + `world_wrist_hi/world_wrist_hi_NN.npy`,
  <20 cm → ±1–2 cm): a **GEOMETRY refinement** camera for the SAME candidate
  agentview already chose. It is near-vertical and **bad at identity** — it
  cannot read side labels or tell duplicate/similar items apart, and in failed
  probes it locked onto a look-alike hundreds of pixels away. **NEVER let the
  wrist freely re-identify a non-basket target.**

Protocol (non-basket objects/surfaces):
1. **Identity (agentview)** — in `image_cam_hi_NN.png` choose the target by RGB /
   label / shape / global spatial relation; sample 3–8 pixels on it and median
   `back_project` → the **identity anchor** (rough xy ok).
2. **Approach** — `move_to` ~15–20 cm directly above that anchor xy (driver
   re-renders both cams after every primitive).
3. **Geometry refine (wrist)** — pick the SAME candidate's pixel in
   `image_wrist_hi_NN.png`, `back_project` it with `camera:"wrist"`. **Accept
   this corrected xy ONLY if it is within ~3–5 cm of the agentview anchor**; if
   it jumps >5 cm, REJECT it (it hit a look-alike/background) and keep the
   agentview xyz. The wrist may sharpen coordinates but may NOT override the
   agentview semantic choice. Never average the two.
- **Basket / cavity special case:** for `basket`/cavity the wrist MAY also
  confirm/refine the true interior centre (basket failures are rim/edge bias,
  not semantic confusion).

### Mandatory pre-task perception pass — localize EVERYTHING, THEN act

Before ANY pick/place, build a localization table in your reasoning — one row
per task-relevant entity (every movable target, every destination/support/
fixture, every relation landmark named by `task_language`), each with:
`name_or_role · agentview_evidence (why this candidate) · agentview_pixels ·
agentview_xyz (median back_project on hi-res) · wrist_refine
accepted|rejected|basket_confirmed · final_xyz · uncertainty`. Do the
**FINAL READY CHECK** (every entity has a final xyz; non-basket wrist
refinements are spatially consistent with agentview; basket points are
interior-centred) — only then start manipulating. Rationale: in single-attempt
mode a **wrong-target first grab is unrecoverable** (it tips/displaces both the
grabbed object and the target zone), so the cheap insurance is to identify all
entities up front instead of recovering later.

> The underlying math (already done for you): `cam = [(col−cx)·z/fx,
> (row−cy)·z/fy, z]` with `z=depth[row,col]`, then `P_world =
> extrinsic_cam2world @ [cam,1]`. You must invert `K` BEFORE the extrinsic — the
> old `E @ [col·z, row·z, z, 1]` recipe (no `K⁻¹`) is **wrong** (metres off).
> The `world/` and `world_wrist/` maps bake in the correct `K⁻¹`+extrinsic, and
> `back_project` reads them for you — that is precisely why `back_project`
> exists; you never write this math yourself. Forward world→pixel was verified
> 5/5 vs GT at design time (plate Δ=6 mm). The wrist map may be all-table/null
> early (the wrist only sees gripper + table until you move it over a target).

## Before you start: READ THE AUTO-MEMORY

Operating wisdom lives in the in-repo memory:

```
resources/libero/memory/MEMORY.md
```

Scan the ~30 one-line hooks. For perception cells **always** open:
- `feedback_no_teleport_rule.md` — the deleted primitives.
- `feedback_redo_cell_timeout_1200.md` — why the cell budget is long here.
(The grasp is oracle-free too: there is no GT-lift oracle — you judge the grasp
from gripper width + the wrist cam, see Rule 1 / 1b.)

For bowl→plate spatial tasks always also read `feedback_bowl_eef_y_offset.md`
(bowl-eef y-offset 4.5 cm: place at `eef_y = plate_y + 0.045`, not `plate_y`).

## Rule 1 — `pi0_pick` is grasp-only (no oracle)

`pi0_end_to_end` is FORBIDDEN. `pi0_pick` is for the grasp only; you script every
`move_to` and every `release`. Use:

```
pi0_pick({
  "prompt": "<carefully chosen prompt>",
  "max_chunks": 8,
  "lift_thresh": 0.05,
  "gripper_closed_thresh": 0.06
})
```

`pi0_pick` takes **no object-tracking / oracle argument** — it is grasp-only
and reads NO GT object pose. Passing a name would do nothing even if you tried.
You judge "did I grab the target?" yourself (Rule 1b). NEVER let Pi0 finish the
place — YOU do every `move_to` and the `release`.

## Rule 1b — JUDGE THE GRASP from perception, NOT from a name

After a pick, decide "did I grab the target?" from two coord-free signals:

- **Gripper** (`state.robot0_gripper_qpos` from the latest `states.json`
  entry): fingers closed but NOT fully shut (~0.01–0.05 gap) ⇒ holding an
  object; fully closed (~0.0) ⇒ grasped air.
- **Wrist cam**: after the lift, read `image_wrist_hi_NN.png` — the target
  should be raised into the gripper and the spot it came from now empty (its
  surface z jumps up by your lift distance). If needed, `back_project` wrist
  pixels to confirm the surface z jumped.

`pi0_pick`'s returned `success` (an eef-lift + gripper-closure heuristic, pure
proprioception) is a HINT, not proof — confirm with the wrist cam before
carrying. The only authoritative TASK-success signal is
`state.libero_terminated` (the benchmark predicate), which is name-independent.

## Rule 4 — NO TELEPORT primitives (physics-only)

`set_object_pose`, `articulate_to`, `js_move_to`, `carry_object` are **deleted
from the codebase**. They are not callable and are not in the tool list. If a
goal is past OSC reach and no physical approach works, write an honest
`libero_terminated:false` audit — never warp.

## Rule 0 — Use images for reasoning, not just JSON state

After every primitive tool call, inspect the returned state and image paths
(the tool returns them — no separate read needed). Read the new
`image_cam_hi_NN.png` path (calibration frame — the one you pick pixels in)
and, when close to a target, the `image_wrist_hi_NN.png` path. Even more
important here than in oracle mode: this is *the* signal you use to find
objects.

`image_cam_hi_NN.png` (and its 256 counterpart `image_cam_NN.png`) is the
calibration-frame RGB — same scene as `image_NN.png` but vertically flipped so
that pixel coordinates align with the camera matrices in `camera_meta.json`.
Pick object pixels from these; `states.json` alone gives only proprioception +
object names.

## Rule 2 — SINGLE EPISODE, NO RESET

This is a **one-shot** evaluation: you get **exactly ONE episode**. Do NOT reset
and do NOT restart the episode. You MAY recover *within* this one episode
(re-localize — objects may have moved; re-pre-position; re-`pi0_pick` a missed
grasp; walk the next rung of the Pi0 prompt ladder in Rule 3; re-firm the grip;
`rotate_pitch`/`move_pose`) — that is all one continuous attempt. But the
instant you would want to start over, **STOP instead and write the audit**
(success or an honest `libero_terminated:false`), then call `finish`. Never
warp; never reset.

## Rule 5 — Assume every task is physically solvable

Same as oracle mode. A localization that "looks right" but moves the gripper
into thin air usually means your pixel was on a wrong surface (e.g. picked the
bowl's reflection on the table). Re-look at `image_cam_hi_NN.png`, pick a
different pixel firmly on the target's top, re-`back_project`. Don't conclude
"unreachable" until you've validated localization.

## Rule 6 — Your task is `task_language`; the BDDL is FORBIDDEN

Each `states.json[NN]` entry carries a **`task_language`** field — the
authoritative task instruction (the BDDL's `:language` tag, which contains
**no** coordinates). **Read it and obey it verbatim.** Do not infer the task
from object names, from sibling recipes, or by guessing a `task_map` index —
that produced *wrong-task* runs (an agent solving a task it was never assigned).

> ⚠️ **Never read the BDDL files, import the benchmark, or query env object
> poses.** The BDDL is the one place the task language *and* the `:init`
> ground-truth coordinates live together — reading it to get the language also
> leaks the coordinates this mode exists to withhold (a perception-isolation
> breach; observed on several swap cells in earlier runs). You already have the
> task from `task_language`; you get object **positions** ONLY by depth
> back-projection. The runtime strips coords from `state` but does **not**
> sandbox the BDDL — that discipline is on you.

## Rule 7 — Ground the target by its spatial RELATION, not its name

When the task names a relation ("the bowl **ON THE COOKIES BOX**", "the mug
**LEFT OF** the plate"), the target is whichever object SATISFIES that relation
in the scene — find it by perception. Identical objects (`akita_black_bowl_1` vs
`_2`) carry NO perceptual difference in their names, so the name can't choose
for you; the RELATION disambiguates:

- "on the cookies box" ⇒ the bowl that is **elevated** (~0.03–0.06 m above the
  table, on the box) — distinguish it by its higher world-z from `back_project`
  vs the table-level bowl.
- "left/right/front/back of X" ⇒ compare back-projected world xy to X's xy.

Pick the target purely from where things ARE. (You never need to know which
`_N` name the target is — no primitive in this mode asks for one.)

## Mental model

1. **The runner (`rpent/cli/main.py`) owns a long-lived env server** — Pi0.5 + a
   single-env LIBERO sim. It launches and manages the server; you do NOT start,
   stop, or restart it.
2. **You call one structured MCP tool per step.** The tool BLOCKS until the
   driver runs that one primitive and dumps the new step, then RETURNS the new
   state + log + image paths. There is no file bus and no polling — the tool's
   return value IS your signal. Each dumped step appends an entry to
   `states.json` (entry `[NN]`) and writes `images/image_NN.png` +
   `images_cam/image_cam_NN.png` + `depths/depth_NN.npy` + `world/world_NN.npy`
   (+ the wrist and hi-res dirs) and, once, top-level `camera_meta.json`.
3. **You read the returned state, localize via `back_project`, decide the next
   move, and call the next tool.**

## Launch a session

The runner (`rpent/cli/main.py`) launches and owns the env server (Pi0.5 + single-env
sim) — do not start/stop it. You call MCP tools; begin by reading step 0 via
`view_driver_state({"step":0})`.

## The perception artifacts you read each step

| artifact | what's in it |
|---|---|
| `states.json` (entry `[NN]`) | `step_idx`, `libero_terminated`, **`task_language` (your authoritative task instruction — the BDDL `:language` tag, coord-free; obey it verbatim)**, `state.{robot0_eef_pos, robot0_eef_quat, robot0_gripper_qpos, object_names}`, and the merged `command`/`result`/`elapsed_s` for that step. **No object coordinates.** Read via `view_driver_state({"step": NN})` (omit `step` = latest). |
| `images/image_NN.png` | RGB in Pi0 frame (180° rotated). *Do not pick pixels here for back-projection.* |
| `images_cam/image_cam_NN.png` | agentview RGB in **calibration frame** (vertical flip). Pick object pixels HERE (256 grid → `back_project` with `resolution:"low"`). |
| `images_cam_hi/image_cam_hi_NN.png` | **HI-RES 1024×1024** agentview RGB, calibration frame. **PREFER this for looking / identification**; `back_project` its pixels at the default `resolution:"high"`. |
| `depths/depth_NN.npy` | `(256, 256) float32` agentview metric depth (m), calibration frame. Same row/col as `image_cam_NN.png`. |
| `world/world_NN.npy` | `(256, 256, 3) float32` — **precomputed agentview world xyz per pixel** (K⁻¹+extrinsic done). Prefer `back_project`; read this manually only for debugging. Fixed cam (~1 m). |
| `world_hi/world_hi_NN.npy` | `(1024, 1024, 3) float16` — precomputed agentview world xyz per hi-res pixel. |
| `images_wrist/image_wrist_NN.png` (+ `images_wrist_hi/`) | **wrist (eye-in-hand) RGB**, calibration frame. Moves with the gripper. |
| `depths_wrist/depth_wrist_NN.npy` | wrist metric depth (m). |
| `world_wrist/world_wrist_NN.npy` (+ `world_wrist_hi/`) | `(256, 256, 3)` / `(1024, 1024, 3)` **precomputed wrist world xyz per pixel**, SAME world frame as agentview. ±1–2 cm when <20 cm to target. May be all-table/null until you move over the target. |
| `wrist_meta/wrist_meta_NN.json` | wrist intrinsics + extrinsic **for THAT step only** (the wrist cam moves). Read via `view_camera_meta({"camera":"wrist","step":NN})`. |
| `camera_meta.json` (top-level) | agentview `intrinsic_K` (3×3), `extrinsic_cam2world` (4×4), `depth_near/far`, projection recipe. Read via `view_camera_meta({"camera":"agentview"})`. |
| `segments/segment_NN_XX.json` | (after a completed `segment` response) `{found, mode, prompt|point, camera, source_step, score, box, mask_shape, centroid_pixel, world_xyz, n_pixels}` — SAM3's top mask back-projected via the matching world map. `world_xyz` is a robust MEDIAN over the whole mask. A valid no-detection response carries `{found:false, error}`. `XX` is a per-step index. |
| `segments/segment_overlay_NN_XX.png` | (only after a successful `segment` call) the segmented mask tinted red on the source image — read it to confirm SAM3 grabbed the right object. |

The command + its result + `elapsed_s` are merged INTO the `states.json[NN]`
entry (and echoed in each primitive tool's return value) — there is no separate
per-step log file to read.

## Localization with `back_project` (coarse agentview → fine wrist)

You index the precomputed map through `back_project` — no K⁻¹ math. Coarse
(agentview) then, once the gripper is parked over the target, fine (wrist):

```
# COARSE: agentview hi-res (default resolution:"high") — choose which object / rough xy
back_project({"row": ROW, "col": COL, "step": NN})

# FINE: wrist — after move_to ~15-20cm above the target, pick its pixel in
# image_wrist_hi_NN.png and back_project it (±1-2cm; refines the SAME candidate).
back_project({"row": ROW, "col": COL, "step": NN, "camera": "wrist"})
```

(Pass `"resolution":"low"` only when `ROW,COL` came from a 256 image.)

Tips:

- Sample 3–5 pixels on the object (centre + a couple of edge pixels) and
  median the back-projected xy — robust to a single mis-picked pixel. Avoid
  pixels on the thin rim/edge or the gap to the table (those index a
  background/edge depth and give a world point metres away).
- The returned point is the **visible surface** under your chosen pixel.
  For a flat object (plate, basket) that surface ≈ the place target. For a
  bowl/bottle, the surface is the top of the object; the rim's xy is what
  you want for `release`, not the grasp's eef_y (apply
  `feedback_bowl_eef_y_offset` — bowl `eef_y_target = perceived_plate_y + 0.045`).
- For the **table z** (when you need a known floor to compare against),
  `back_project` a pixel on bare table near the object.

## The command vocabulary

Call one structured tool per step. The full primitive set (only the *control
signals* are listed here) — every one blocks until the step is dumped and
returns the new state + log + image paths:

```jsonc
// === physics-only primitives (the entire allowed set) =====================

// Scripted EEF servo. action_scale=0.05 is the env's units; step_clip caps
// per-step Δxyz (m) BEFORE division by action_scale — smaller = slower.
move_to({"xyz": [x, y, z], "gripper": -1,
         "tol": 0.012, "step_clip": 0.025, "max_steps": 80,
         "action_scale": 0.05, "target_yaw": null})   // gripper: -1 open, +1 close

// Pi0.5 closed-loop pick. Grasp-only — no oracle/tracking arg (Rule 1).
// You judge the grasp from gripper width + wrist cam (Rule 1b).
pi0_pick({"prompt": "pick up the X", "max_chunks": 8,
          "lift_thresh": 0.05, "gripper_closed_thresh": 0.06})

// Pi0.5 for a contact skill (knob turn, drawer/door open-close — rare here).
// success mirrors libero_terminated only; inspect image/state for intermediates.
pi0_doubled({"prompt": "turn off the stove", "max_chunks": 20})

// Open gripper to place. Triggers libero termination if the On/In predicate met.
release({"max_steps": 20})

// Hold pose + drive gripper. Use to firm a grip mid-carry.
set_gripper({"gripper": 1, "steps": 5})   // -1 open, +1 close

// Wrist yaw (world-z). Provide target_yaw (absolute) OR delta_yaw (relative).
rotate_wrist({"target_yaw": 0.0, "gripper": 1,
              "max_steps": 40, "tol": 0.02, "step_clip": 0.10})

// Tilt eef pitch (axis-angle X). Cavity entry / micro-aiming. target_pitch OR
// delta_pitch. Use before threading a narrow opening whose face normal is ±y.
rotate_pitch({"target_pitch": 0.9, "gripper": 1,
              "max_steps": 40, "tol": 0.02, "step_clip": 0.10})

// Co-vary xyz + pitch + yaw — threads cabinet-front IK singularity that move_to
// walls at. gripper defaults to -1 (OPEN) — pass gripper:1 while holding.
move_pose({"xyz": [x, y, z], "target_pitch": 0.0, "target_yaw": 0.0,
           "gripper": 1, "step_clip": 0.02, "pitch_step": 0.08, "yaw_step": 0.08,
           "tol": 0.012, "ori_tol": 0.05, "max_steps": 150})

// SAM3-grounded localization — does NOT move the robot and does NOT
// replace manual back-projection. Segments the most-recent dumped image for the
// prompt, back-projects the mask through the matching world map, and writes
// segments/segment_NN_XX.json {score, box, centroid_pixel, world_xyz (robust MEDIAN over
// the whole mask), n_pixels} + segments/segment_overlay_NN_XX.png. Read world_xyz
// directly instead of eyeballing a pixel. camera "wrist" uses the wrist world
// map for fine refinement (move the eef over the target first). Pass
// "point":[row,col] for a point prompt instead of text; prompt and point are
// mutually exclusive. min_score default 0.2.
// RPent manages the SAM3 service. If one call fails or finds nothing, the result is
// an {"error":...,"fallback":...} dict — fall back to picking a pixel in
// image_cam_hi_NN.png and calling back_project.
segment({"prompt": "the black bowl on the stove", "camera": "agentview",
         "point": null, "min_score": 0.2})

// === FORBIDDEN — DELETED FROM THE CODE — DO NOT EMIT ============
//   set_object_pose, articulate_to, js_move_to, carry_object
```

### How to use `segment` (SAM3) — practical tips

`segment` is the fastest way to localize: one call gives you a `world_xyz`
that is a robust MEDIAN over the whole object mask (hundreds of pixels), which
beats eyeballing 3–5 pixels. Workflow:

1. Read `image_cam_hi_NN.png`, decide the target by the task's spatial RELATION.
2. Call `segment` with a **plain visual phrase + the relation**, then read the
   returned `world_xyz` and the `segments/segment_overlay_NN_XX.png` to confirm the mask
   is on the right object before you move.
3. **Two camera views — YOUR choice via the `"camera"` field** (`segment` works
   on either; default `"agentview"`):
   - `"camera":"agentview"` — the fixed ~1 m cam (±8–13 cm). Use it to choose
     WHICH object (global layout / spatial relation) and get a rough xy.
   - `"camera":"wrist"` — the eye-in-hand cam (±1–2 cm), reads the wrist world
     map in the SAME world frame. Use it to PRECISELY localize once the gripper
     is parked over the target. ⚠ It is null / all-table until you `move_to`
     ~15–20 cm over the target (the wrist only sees the gripper+table before
     that), so segment agentview first, approach, THEN segment wrist.
   Typical coarse→fine: agentview select → `move_to` above → wrist refine → grasp.
   You decide when the wrist refinement is worth it (small / closely-spaced /
   identical objects benefit most; a big isolated plate may not need it).

**Prompt phrasing (important — SAM3 is sensitive):**
- ✅ Use plain colour + shape + spatial relation: `"the black bowl on the stove"`,
  `"the white plate"`, `"the bowl on the cookies box"`.
- ❌ Do NOT use the object's internal/brand NAME from `object_names`/BDDL:
  `"the akita black bowl"` scores ~0.03 (no detection) because SAM3 can't ground
  "akita"; the same object as `"the black bowl on the stove"` scores ~0.76. Strip
  proper nouns (`akita`, `glazed_rim_porcelain_…`) — say what it *looks like*.
- For two identical objects, the relation in the prompt (`…on the stove`,
  `…left of the plate`) usually steers SAM3 to the right instance; verify via the
  overlay and the world-z (an object *on* a fixture has a higher z than a
  table-level twin). The two-camera relation protocol still applies when SAM3
  can't disambiguate from text alone.
- If `segment` returns `{"error":...}` (low score / service down), walk the
  prompt (drop the brand word, simplify, add the relation) or fall back to
  manual pixel → `back_project`.

## The strict-hybrid recipe (perception variant)

A typical bowl→plate cell looks like:

1. `view_driver_state({"step":0})`; inspect `task_language`,
   `image_cam_hi_00.png`, and `camera_meta.json`.
2. Identify the target by the task's spatial RELATION, not its name (Rule 7).
3. Localize it — pick its pixel in `image_cam_hi_00.png`,
   `back_project({"row":ROW,"col":COL,"step":0})` (coarse). Refine with the
   wrist cam after pre-positioning (fine).
4. **Pre-position** ~15–20 cm above it: `move_to([obj_x, obj_y, carry_z])`,
   gripper open. Re-`back_project` a wrist pixel here and correct the xy before
   grasping.
5. `pi0_pick` with the right prompt (start "pick up the {object}", escalate per
   the Pi0 ladder — see below). Confirm the grasp via gripper + wrist (Rule 1b).
6. `set_gripper({"gripper":1,"steps":5})` to firm the grip.
7. Localize the placement region (basket / plate / drawer slot) the same way.
8. `move_to([place_x, place_y, carry_z])` to traverse at constant height.
9. Optionally descend `move_to([place_x, place_y, place_z])`.
10. `release` — predicate (`On`/`In`) checks → `libero_terminated=True` if hit.
11. Light retreat (`move_to` upward) so the next step's image is clean.

> **Predicate fire timing.** Most LIBERO `On(X, Y)` predicates fire on
> `release` if `X` is above `Y`'s region. `In(X, container)` needs `X` to
> have actually entered the container's volume before release. If a release
> fires `term=False` but the object is on top of the right region, descend
> 1–2 cm more and re-`release`.

## Pi0 prompt ladder (Rule 3 — Pi0 IS the delivery service)

Try in order; each rung uses a slightly more specific prompt or a re-pre-pos.

1. `"pick up the {object}"` — generic sub-instruction.
2. The full `task_language` verbatim (e.g. `"Pick the akita black bowl on the
   cookies box and place it on the plate"`).
3. Add a spatial qualifier (`"…on the wooden cabinet"`, `"…next to the
   basket"`).
4. Re-pre-position 5 cm lower or shifted, then retry rung 2 or 3.

Empirically Pi0 sometimes needs the **full prompt with the spatial
qualifier** for elevated picks (stove, cabinet-top, drawer). See
`feedback_pi0_pick_full_prompt.md` and the prompt-ladder note in MEMORY.

## Key hyperparameters

- Single-step `xyz` within ±0.30 m of current eef or OSC flips IK. Split
  long traversals into 2–3 carry-z waypoints.
- `lift_thresh`: 0.05 (flat/stable) / 0.08 (slippery tall bottles).
- `step_clip`: 0.025 (empty / box) / 0.015 (cans) / 0.012 (tall bottles).
- Frame z (from `state.robot0_eef_pos[2]` at step 00):
  ≈ 0.68 → LIVING_ROOM, ≈ 1.17 → KITCHEN, ≈ 0.26 → OBJECT.
- BOWL: `eef_y_target = perceived_plate_y + 0.045` (bowl-eef y-offset).
- TALL BOTTLES: carry at `z=0.30`, release without descending.
- Approach high-then-vertical; recover by re-`pi0_pick`, not by hovering.

## Reading state

After every primitive tool call, the return value already carries the new
`state`, the merged `command`/`result`/`elapsed_s` log, and the image paths —
you do **not** need a separate read. When you need an older step, call
`view_driver_state({"step":NN})` (omit `step` for the latest):

1. Check `result.success`, `final_dist_m`, `peak_lift_m`, etc. in the returned
   log.
2. Read the returned `image_cam_hi_NN.png` path → visual confirmation; pick
   pixels for any new localization.
3. Check `state.robot0_eef_pos` / `robot0_gripper_qpos` / `libero_terminated`
   in the returned `state`.

You **do not** open the depth maps yourself — feed a pixel to `back_project`.
Don't call `view_driver_state` immediately after a primitive that already
returned the new state.

## Common failure modes

- **Pick missed (gripper closed empty).** `result.peak_lift_m` < `lift_thresh`,
  `min_gripper_opening` ≈ 0. Re-pre-position 1–2 cm lower or shifted; retry
  Pi0 with next prompt-ladder rung.
- **Object slipped mid-carry.** `release` returns `term=False` and the object
  is no longer where you `release`d it. `release`, re-pre-pos above it,
  `pi0_pick` again, traverse again.
- **OSC stuck.** `move_to` returns `final_dist_m > 0.05` at `max_steps` and
  same xy twice → try `rotate_pitch`, split into more waypoints, or
  `move_pose` (co-varying) for the cabinet-front singularity.
- **Placement off because localization was wrong.** The release puts the
  object on bare table instead of on the plate. Re-read `image_cam_hi_NN.png`,
  pick a different pixel on the target region (sample 3 pixels on the plate's
  flat top, median the back-projected xy), redo. Tip: if depth at your chosen
  pixel is much closer than the table z, you picked the camera's near edge /
  an object rim — pick again.

## Verifying strict compliance

Before saving the audit, confirm your command history is physics-only. The
teleport primitives are not even in the tool list, but audit it anyway: read
the `command` field of every `states.json` entry (via
`view_driver_state({"step":NN})` per step, or `read_text_file` on
`{output_dir}/states.json`) and confirm every `command.action` is one of the
allowed physics primitives (`move_to`, `pi0_pick`, `pi0_doubled`, `release`,
`set_gripper`, `rotate_wrist`, `rotate_pitch`, `move_pose`) — no
`set_object_pose` / `articulate_to` / `js_move_to` / `carry_object` appears.

## Persisting successful runs as audit JSONs

When `state.libero_terminated == true`:

a. The working command recipe (`{output_dir}/recipe_{recipe_tag}.jsonl`) is
   **auto-exported by the runner** from the non-error primitive commands in
   `states.json` plus successful segment calls recorded in
   `segments/segment_*.json`, merged in execution order — you do NOT
   hand-write it.
b. Write a minimal audit JSON with `write_text_file` to
   `{output_dir}/{recipe_tag}.json` with at least: `suite`, `task_id`, `seed`,
   `regime: "strict_perception"`, `strategy_notes` (mention HOW you localized —
   which pixel, depth, back-projected world xyz), `pick_result` (the `result`
   from your `pi0_pick`), `final_state` (the latest `states.json` entry's
   `state` field), `libero_terminated: true`.
c. Call `finish({"status":"success","summary":"…"})`.

If unrecoverable after honest exploration in this one episode, write
`{output_dir}/{recipe_tag}.json` with `libero_terminated: false` +
`strategy_notes` describing what you tried, the back-projected xyz you used, and
which step failed. Then call `finish` (NO reset, NO second attempt).

> The `regime: "strict_perception"` tag distinguishes these audits from the
> oracle-state `strict` regime in mixed-mode datasets.

## Iteration heuristics

- After 2 failed retries on the same step, **stop tuning numerics** and
  inspect images: read `image_cam_hi_NN.png` at pick / pre-release /
  post-release. The visual disagreement is usually the bug (you picked the
  wrong pixel, or Pi0 grabbed the decoy). This is the lesson of
  `feedback_failure_forensics.md` — applies even more strongly here.
- If a release reports `term=False` but the object is visibly on the target
  region, descend 1–2 cm more and `release` again; predicate often needs
  contact, not just hover.
- Once you use `back_project`, trust it: if a localization "feels off", it's
  because you picked the wrong pixel — not because the depth / calibration is
  wrong. (`back_project` inverts `K` before the extrinsic; the raw
  `E @ [col·z, row·z, z, 1]` form skips it and is wrong — which is why you use
  `back_project` and never hand-roll the math.)

## What "strict_perception" means concretely

- **No GT object coordinates anywhere in your reasoning.** The `state` you
  read has none; the only legitimate sources of object xyz are the camera
  images + depth + the precomputed `world/` and `world_wrist/` maps (via
  `back_project`).
- **No teleport primitives.** The four are deleted.
- **Pi0 only does the grasp.** You script every motion + release.
- **Fully oracle-free, including the grasp.** There is no GT-lift oracle —
  `pi0_pick` reads NO GT object pose and takes no tracking argument. You judge
  the grasp from gripper width + the wrist cam, and TASK success from
  `state.libero_terminated` (the benchmark predicate).
- **Single attempt.** One episode, no reset (Rule 2).
- The expected audit `regime` is `strict_perception`.

## Reference cases

The seed-0 sweep results live under `resources/libero/results_*_pert/`
(`results_10_pert`, `results_object_pert`, `results_spatial_pert`,
`results_goal_pert` — PRO swap+task, t0–t9). Each solved cell has an audit JSON
+ a `recipe_{tag}.jsonl` command sequence. The consistent winning pattern:
localize → pre-pos → `pi0_pick` → `set_gripper` → move → `release` in 6–12
commands.

When you write a new audit, browse a sibling cell's `recipe_{tag}.jsonl` as a
*technique* template — but never paste its xyz; re-derive every position via
`back_project` from THIS scene's depth.

Begin by reading `resources/libero/memory/MEMORY.md`, then call
`view_driver_state({"step":0})` and inspect the returned `image_cam_hi_00.png`
(+ `camera_meta.json` via `view_camera_meta`); localize the target object via
`back_project`, then plan and execute.
