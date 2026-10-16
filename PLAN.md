# WorldArm plan

Team of 4, two arms. Demos are **Saturday at 3:30 PM**. Doors close at 9 PM on Friday, with no overnight hacking.

## How we work

1. **Reuse first.** Low-level pieces already exist (drivers, saved poses, block detection, pick and place, trajectory record and replay). We configure them; we don't rewrite them.
2. **Skeleton first.** A thin layer of our code connects those pieces into one full, logged episode: `python run_episode.py`. It already runs in simulation (`--fake`); it goes onto the arms this afternoon.
3. **Four owners, one robot.** Everyone improves one part of the **same running system** and proves it with a number on the scoreboard (`python scoreboard.py`).

## Who does what

The task splits into the robot's three skills, plus the memory and the loop that tie them together:

| | Owner | Goal | Files | Score | Demo moment |
|---|---|---|---|---|---|
| **1. Find and pick up any block** | **Waj** | Find every block on the table, whatever its shape, and pick it up | `worldarm/vision.py` `pick.py` `perception.py` `frames.py`; new `eval_perception.py` | Detection recall, color accuracy, position error (mm); pick success per shape | Cylinders and triangles get sorted, not just cubes |
| **2. Open and close the drawers** | **Riyaj** | Pull each drawer to **75% of its travel, never further**, and push it closed | `worldarm/drawers.py` `safety.py`; knob poses and recordings | Open success (within ±10 mm of the 75% mark); close success; drawers pulled out: 0 | Drawers opening smoothly to the mark |
| **3. Sort the blocks into the drawers by color** | **Sandeep** | Put each block into its color's drawer and confirm it landed there | `worldarm/placing.py`; drop poses; the color → drawer assignment | Place success (seen in the right drawer); share of misses caught | Each color in its own drawer, confirmed on screen |
| **4. World model, integration & demo** | **Jithendra** | Remember what's in closed drawers, notice tampering, find things; run the whole loop on both arms; measure and present | `worldarm/world_model.py` `planner.py` `episode.py` `robot.py` `manipulation.py` `sim.py`; `run_episode.py` `try_skill.py` `scoreboard.py`; docs | End-to-end success judged on where the blocks really are; belief accuracy; surprise precision and recall | "Where's blue?" and the surprise |

**Jithendra's arm work:**
- Bring up every skill on the arm with its owner.
- Run the first full episodes on both arms.
- Run the hidden-hand and "where's blue?" demos.
- Stretch: a retrieve skill that takes a block back out of its drawer.

**Shared files:**
- `worldarm/schema.py` (actions and fact keys) changes only at a sync. Adding a new fact key is fine; just announce it.
- `worldarm/manipulation.py` holds the shared arm helpers: saved poses, safe moves, the gripper, recording playback. Ask Jithendra before changing it.

### Arms

- **Arm A:** armfarm03 with the drawer organizer. This is Riyaj's and Sandeep's arm.
- **Arm B:** the second machine (name TBD) with three open bins. Set `WORLDARM_CONTAINERS=bins` in its env file. This is Waj's arm; Sandeep also develops the drawer check on its bins.

Only one person drives an arm at a time, and the driver stays at the arm with the E-stop in reach for every motion. **Two people** are needed for manual-mode recording and for the first run of any new motion.

**Friday**

| Time | Arm A (drawers) | Arm B (bins) | Off the arm |
|---|---|---|---|
| 1:30–2:30 | **Riyaj + Sandeep:** tape the organizer down; measure each drawer's maximum travel and mark 75% on its side; test by hand (**G1**); save `home-pose`, `table-view`, `organizer-view`, `knob-1..3-pre`, then open each drawer to its mark by hand and save `drop-1..3` | **Waj + Jithendra:** add the services to *both* machines in the Viam app; Arm B env file and `verify_camera.py`; save Arm B poses; `--dry-run` (**G2**) | — |
| 2:30–3:30 | **Riyaj + Sandeep:** record `grasp-k`, `pull-k` (to the mark) and `push-k` for each drawer; replay each once | **Waj:** ground-truth sheet and perception test set (every shape) | **Jithendra:** world model in simulation (hidden-hand protocol) |
| 3:30–4:30 | **Riyaj:** `try_skill.py open k` and `close k`, 3× per drawer | **Jithendra + Waj:** `try_skill.py observe`, then `pick blue`, then `open 1 pick blue place 1` | **Sandeep:** the drawer check, on saved captures. **Waj:** shape-finder baseline |
| 4:30 | **Sync** | | |
| 4:30–5:15 | **Sandeep + Riyaj:** `try_skill.py open k pick <color> place k close k` for each drawer | **Jithendra:** first full episode on the bins (**G3**) | **Waj:** depth-first detector |
| 5:15–6:00 | **Jithendra + Riyaj:** first full drawer episode (**G4**) | **Waj:** pick trials, 10 per shape | **Sandeep:** wire the check into `place` |
| 6:00–7:00 | **Dinner + sync:** the v0 numbers on the scoreboard become everyone's starting numbers | | |
| 7:00–8:00 | **Riyaj:** travel measurement, straight-line pull | **Waj:** any-shape pick, live | **Sandeep:** release height and position. **Jithendra:** demo UI |
| 8:00–8:45 | **Sandeep:** place and check on the drawers | **Jithendra:** logged episodes, hidden-hand runs, **backup video** | **Riyaj, Waj:** fixes |
| 8:45 | **Sync:** push everything; agree tomorrow's first task (**G5**) | | |

**Saturday**

| Time | Arm A (drawers) | Arm B (bins) | Off the arm |
|---|---|---|---|
| 9:30 | **Standup at the scoreboard** | | |
| 9:45–11:00 | **Riyaj** (to 10:20), then **Sandeep** | **Waj** | **Jithendra:** probabilistic belief, demo UI |
| 11:00–12:30 | **Jithendra:** hidden-hand runs (H3), "where's blue?", full drawer episodes. Riyaj and Sandeep nearby for fixes | **Sandeep** (to 11:45): H1 runs, checks vs. `--blind`. Then **Waj:** any-shape episodes | |
| 12:30 | **Feature freeze (G6).** Lunch. Jithendra runs the final evaluation on frozen code. | | |
| 1:30–2:30 | Results slide: each owner gives one row. Demo UI polish. | | |
| 2:30–3:30 | Rehearse the demo twice, with one person trying to break it | | |
| 3:30 | **Demo** | | |

### Checkpoints

| | When | Owner | Passes when | If it fails |
|---|---|---|---|---|
| **G1** | Fri 2:30 | Riyaj | By hand: the fingers hold the knob stem; the drawer pulls to its 75% mark without the organizer sliding; with a block inside, the open drawer doesn't tip | Pull less (as far as is safe), or Arm A also uses bins with lids |
| **G2** | Fri 2:30 | Waj | `run_episode.py --dry-run` sees all three colors (cubes and bars) from `table-view` on both arms | Waj writes a color-threshold detector first |
| **G3** | Fri 5:15 | Jithendra | First logged real episode on Arm B (bins) | Run single skills with `try_skill.py` and fix the first failure |
| **G4** | Fri 6:00 | Jithendra, Riyaj, Sandeep | First logged real drawer episode on Arm A; v0 numbers on the scoreboard | Use Arm B's numbers as the baseline; drawers get Saturday morning |
| **G5** | Fri 8:45 | Jithendra | ≥10 real episodes logged, backup video recorded, everything pushed | — |
| **G6** | Sat 12:30 | Everyone | Feature freeze: after this, only fixes | — |

### 1. Find and pick up any block (Waj)

**Research question:** does a depth-first detector beat shape-finder on recall and position error, and does gripping each shape along the right axis raise pick success? Shape-finder only sees squares and rectangles. Depth-first means: find anything that sticks up above the fitted table plane, then classify its color.

**Until 6 PM**
- [ ] With Jithendra: add the services in [docs/viam-config.md](docs/viam-config.md) §1 to both machines; `--dry-run` sees all three colors (**G2**)
- [ ] **Ground-truth sheet:**
  1. Tape a paper with 9 crosses, 5 cm apart, to Arm B's table.
  2. In manual mode, touch 3 crosses with a fingertip and save each with `teach.py` (`s cross-1` …). With the gripper pointing straight down, fingertip z = gripper z − 60 mm.
  3. Work out every cross's world position from those three.
- [ ] **Test set:**
  1. Capture ≥15 layouts with blocks centered on crosses, every shape, saved with `teach.py`, `c layout-01`.
  2. Record which block sits on which cross in `data/perception/labels.json`.
  3. Images stay out of git because they're too big; share the folder.
- [ ] **Baselines:** shape-finder's recall per shape and color, and its position error (median and 90th percentile). Pick success for cubes and bars: `try_skill.py pick <color>`, 10 tries each.

**Then**
- [ ] Depth-first detector in `vision.py`, **same return format**, covering cylinders and triangles
- [ ] Grip axis per block in `pick.py`: across two parallel faces. For a cylinder that's across its diameter; for a triangular prism, across its two triangular ends.
- [ ] Retry a failed grab once, from a fresh detection

**Score:** recall, color accuracy and position error on the test set; pick success per shape on the arm. Target: ≤ 5 mm median, 100% color, ≥ 95% picks for every shape.

### 2. Open and close the drawers (Riyaj)

**The rule:** pull each drawer to **75% of its maximum travel, never further.** The drawers detach, and the arm can't put a detached drawer back.

**Research question:** how reliable are recorded demonstrations? And can a straight-line pull from the knob, measured by the arm itself, hit the 75% mark within ±10 mm just as reliably?

**Until 6 PM (Arm A, with Sandeep)**
- [ ] Tape the organizer down. Measure each drawer's maximum travel, interior height, knob diameter and how far the knob sticks out, and write them under "Organizer and blocks" below.
- [ ] Mark 75% on each drawer's side with tape, so everyone can see the target
- [ ] Test by hand (**G1**)
- [ ] Save poses `home-pose`, `table-view`, `organizer-view`, `knob-1..3-pre` ([docs/viam-config.md](docs/viam-config.md) §2)
- [ ] Record `grasp-k`, `pull-k` (to the mark) and `push-k` for k = 1..3 (§3); replay each once with the E-stop ready
- [ ] `try_skill.py open k` and `close k`, 3× per drawer

**Then**
- [ ] **Measure the travel.** The fingers hold the knob during the pull, so travel = how far the gripper moved.
  - Report `drawer_open:<k>` (with the mm in the detail), so opens stop showing as "unverified" on the scoreboard.
  - A slipped knob shows up in the holding check.
- [ ] **Straight-line pull** to exactly the mark from the knob pose (`motion.move` with a linear constraint) instead of replaying the recording; compare the two
- [ ] **Close check:** is the drawer flush? (Travel back to zero, or depth at `organizer-view`.)
- [ ] Stretch: find the red knob from `organizer-view`, so the organizer can be placed anywhere

**Score:** open success (within ±10 mm of the mark), close success, drawers pulled out (must stay 0). Target ≥ 95%.

### 3. Sort the blocks into the drawers by color (Sandeep)

**Research question:** does looking into the drawer after placing catch the misses the gripper sensor can't, and how much does that raise end-to-end success? In simulation it takes success from 51% to 99% (see "The skeleton" below). This is H1 on real hardware.

**Until 6 PM**
- [ ] On Arm A with Riyaj: open each drawer to its mark by hand and save `drop-1..3` above the exposed part, low enough that the block falls ≤ 3 cm
- [ ] **Look into a container** (`placing.py`):
  - From `drop-k`, back off about 150 mm and run shape-finder's 2D detections (colors only; no depth needed). Report `contents:<k>`.
  - Develop it on saved captures and Arm B's bins first.
  - Check that the drawer's interior isn't detected as a colored object.
- [ ] With Riyaj: `try_skill.py open k pick <color> place k close k` for each drawer

**Then**
- [ ] Wire the check into `place` (and into `open`, for the world model) with Jithendra
- [ ] **Reliable placing:** release height and position inside the exposed part. A tall block must end up lying below the drawer's rim, or closing will jam.
- [ ] **H1 on real hardware:** the same system with checks vs. `--blind`, ≥10 episodes each

**Score:** place success (seen in the right drawer), share of misses caught, H1 difference. Target: ≥ 95% placed, every miss caught.

### 4. World model, integration & demo (Jithendra)

**Research question:** with sensor and action reliability estimated from our own logs, does a probabilistic belief do three things?
- (a) Stay correct about what's in closed drawers (H2).
- (b) Flag tampering with high precision and recall (H3).
- (c) Find a block in fewer drawer openings than v0's fixed search.

**Until 6 PM**
- [ ] With Waj: services and Arm B setup (1:30–2:30)
- [ ] **Hidden-hand protocol, in simulation:**
  - Scripted perturbations (move a block to another drawer, remove one, leave a drawer open) on a seeded schedule.
  - A `--perturbed` note in each run's meta, so the scoreboard can compute precision and recall.
  - The hidden hand puts blocks in the front part of a drawer, which is the part visible when it's open.
- [ ] Bring-up on Arm B with Waj: `try_skill.py observe`, then `pick blue`, then `open 1 pick blue place 1`. On bins, `open` does nothing physically, but the belief needs it: placing into a container believed closed is refused.
- [ ] First full episode on Arm B (**G3**) and on Arm A (**G4**)
- [ ] Keep `main` working: before a merge, `python -m pytest` passes and `run_episode.py --fake` ends `done`

**Then**
- [ ] ≥10 real episodes on Friday; backup video before 9 PM (**G5**)
- [ ] Hidden-hand runs (H3) and "where's blue?" on the drawers
- [ ] Probabilistic belief and active search: open the most likely drawer first
- [ ] Stretch, on the arm: **retrieve.** Pick a block out of its open drawer using Waj's detection and pick, and put it on the table.
- [ ] Demo UI: live camera next to the belief, with a banner when there's a surprise. Results slide.

**Score:** end-to-end success judged on where the blocks really are (≥ 8/10), belief accuracy, surprise precision and recall.

### Rules

1. **Own your files.** Ask before changing someone else's. Wiring a new measurement into `robot.py` is a one-line change; tell Jithendra.
2. **Branches.** Work on `<name>/<thing>`. Before merging to `main`, `python -m pytest` must pass and `python run_episode.py --fake` must end `done`. `git pull --rebase` often.
3. **No unlogged experiments.** Every real run goes through `run_episode.py` or `try_skill.py`; the results slide comes from `runs/`.
4. **A change counts when the number moves:** on real runs (or the perception test set), with n stated. Note the number before and after.
5. **Safety:** the driver stays at the arm with the E-stop in reach. New motions are tested high first. Drawers never go past the 75% mark.
6. **Blocked for 20 minutes? Say so** in the team chat. Don't wait for the next sync.

**Syncs:** 5 minutes at the scoreboard: Fri 4:30, 6:00, 8:45; Sat 9:30, 12:30. Each owner says their number, their next change, and what's blocking them.

## Reuse first: what we take, what we build

| Layer | We reuse | Notes |
|---|---|---|
| Drivers | `viam:ufactory` (arm, gripper), `viam:realsense` | Already configured |
| Saved poses | `erh:vmodutils:arm-position-saver` (switch: `set_position(2)` = go to the saved pose) | Teach in the Viam app; replay from code |
| Obstacles and motion | `erh:vmodutils:obstacle`, built-in motion service | Our code still enforces its own table floor; the configured table is 25 mm too low |
| Block detection, v0 | `devrel:shape-finder:detector` → `viam:vision:detections-to-segments` (3D objects) | Labels like `square-blue`. **Only four-sided shapes**, fixed color ranges. No cylinders, triangles or round knobs. |
| Pick, v0 | The pick pattern from Viam's [pick-and-place workshop](https://docs.viam.com/tutorials/pick-and-place/) ([code](https://github.com/viam-devrel/pick-and-place/blob/main/scripts/reference-solution.py)) | Same hardware. Plans the approach in the camera frame, then descends in the gripper frame; fingertips are about 60 mm past the gripper frame. |
| Drawer moves, v0 | `devrel:arm-recorder:recorder`: record a pull or push by hand in manual mode, replay it | **Arm only**: the xArm gripper doesn't support `get_position`, so our code does grab and open between recordings |
| Data and training | Viam data manager; [`viam-devrel/yolo-training`](https://github.com/viam-devrel/yolo-training) + `viam-labs:vision:yolov8` | If Waj wants a trained detector |
| Precision and markers | `viam:camera-calibration:handeye`, `viam:apriltag:pose_tracker` | Only if a track needs them |
| AI tooling | [`viam-devrel/agent-skills`](https://github.com/viam-devrel/agent-skills) (Claude Code plugins); Viam MCP server `https://app.viam.com/mcp` | Viam SDK knowledge verified against the source |

**What we build, which is the research:**
- **World model:** a belief state over hidden drawer contents, a predicted effect for every action, surprise detection when a prediction doesn't match what we see, recovery, and "where is X?" retrieval.
- **Observation model:** the checks the world model needs. Is the block gone from the table? How far is the drawer open? What's in this open drawer?
- **Perception beyond shape-finder:** depth-based segmentation that works for any shape (cylinders, triangles), red-knob detection, and robustness to lighting.
- **Drawers anywhere:** move from the detected knob instead of replaying recordings, so the organizer can be placed anywhere.
- **Evaluation:** seeded episodes, logs, metrics, and comparisons that switch one component on or off.

Configuration for the reused pieces: **[docs/viam-config.md](docs/viam-config.md)**.

## What we claim, and how we'll show it

| | Hypothesis | Measured by |
|---|---|---|
| **H1** | Checking every action against its prediction raises end-to-end success | The **same system** with checks on vs. off, ≥10 episodes each, 95% Wilson intervals |
| **H2** | The belief state tracks hidden drawer contents correctly | Fraction of blocks whose believed location matches reality at episode end |
| **H3** | The robot detects when the world was changed behind its back | Precision and recall of "surprise" flags over scripted perturbations (block moved, drawer left open, block removed) |

**Episode protocol and logging**
- Block start positions and colors come from a seeded generator, which prints a layout for the person resetting the table.
- Every episode is logged; no cherry-picking.
- Each step writes one JSON line to `runs/<timestamp>/episode.jsonl`: belief before, action, predicted effect, observation, outcome, and paths to the before and after images.
- Failures are sorted into categories (perception, grasp, drawer, planner) for the results slide.

## The skeleton

**Status (Fri 1:30 PM):**
- Runs end to end in simulation (`python run_episode.py --fake`); 26 tests pass.
- The real-arm code is split by owner: `vision.py` and `pick.py` (Waj), `drawers.py` (Riyaj), `placing.py` (Sandeep), and `robot.py` + `manipulation.py` (Jithendra). It is **not yet validated on hardware**.
- `try_skill.py` is the bring-up tool.

```
worldarm/  schema · world_model · planner · episode · sim · robot · manipulation
           vision · pick · drawers · placing · safety · config · logger · metrics
run_episode.py   one full episode: --fake | --dry-run | real;  --goal sort | find:<color>;  --blind
try_skill.py     a short script of skills on a real arm, checked and logged (bring-up)
scoreboard.py    success rates with 95% intervals, per-action results, surprise counts
teach.py         manual-mode helper: gripper open/grab, pose readout, captures
```

**First simulated result (H1).** 100 seeds, every skill failing 15% of the time, scored on whether the blocks truly ended up sorted:

| Checks after each action | Truth success | 95% interval |
|---|---|---|
| Full: drawer contents and drawer state | 99% | 95–100% |
| Real v0: gripper sensor and table only | 51% | 41–61% |
| None (`--blind`) | 38% | 29–48% |

These failure rates are assumptions, and the simulated checks never make mistakes. What it does show is where checking helps most: **looking into the drawer**, which is Sandeep's first task.

**Version 0 of each stage:**

| Stage | v0 |
|---|---|
| Find and pick | The `vision-segment` service from the `table-view` pose; the workshop pick pattern |
| Drawers | Switch `knob-k-pre` → recording `grasp-k` → grab → recording `pull-k` (to the 75% mark) → open. Close is recording `push-k`. Organizer taped in place. |
| Placing | Switch `drop-k`, then open the gripper |
| World model | Deterministic belief with predicted effects; checks use only `IsHoldingSomething` and "block gone from the table" |
| Planner | Fixed order: for each block on the table, open drawer → pick → place → close |

## Interfaces

Defined in `worldarm/schema.py`. A robot (real or simulated) implements one method:

```python
async def execute(action: Action) -> Observation
# Action: observe_table | open(k) | close(k) | pick(color) | place(k)     (drawer 1 = top)

@dataclass
class Observation:
    facts: dict       # measured AFTER the action; checked against the prediction
    pre_facts: dict   # measured BEFORE it (e.g. table detections at pick time); checked against the prior
    detail: dict      # anything else worth logging

# fact keys
holding          -> color | None
drawer_open:<k>  -> bool
contents:<k>     -> [colors seen in open drawer k]
on_table         -> [colors seen on the table]
```

**A fact you don't measure, you leave out.** It can't cause a false surprise, and the scoreboard counts that step as "unverified". Adding a new measurement means adding a fact key: for example, `contents:<k>` from a look into the open drawer. The world model needs no other change.

For a bin (Arm B, `WORLDARM_CONTAINERS=bins`), `open` and `close` do nothing, so the same planner runs on both arms.

## Organizer and blocks

**What we know:**
- The organizer has 3 **opaque** drawers, each with a **rounded red knob**.
- The slides are smooth, and **there's no stop**: a drawer slides all the way out and detaches.
- The blocks are **blue, yellow and green**, in mixed shapes.

**Decisions:**
- **Red only ever means a knob.**
- **Knob grasp:** approach horizontally and close the fingers on the knob's **stem, behind the head**. When we pull, the head can't slip through the closed fingers.
- **Pull each drawer to 75% of its maximum travel, never further.** A detached drawer is very hard for the arm to put back. Mark the 75% point on each drawer.
- **Tape the organizer down.**
- **v0 blocks are cubes and bars,** because shape-finder only sees four-sided shapes. Every other shape is Waj's goal.

**Still open:** each drawer's maximum travel, interior height, knob diameter and how far the knob sticks out (Riyaj measures these); the second arm's machine name and address.
