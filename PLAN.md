# WorldArm plan

Team of 4, two arms. Demos are **Saturday at 3:30 PM**. Doors close at 9 PM on Friday, with no overnight hacking.

## How we work

1. **Reuse first.** Low-level pieces already exist (drivers, saved poses, block detection, pick and place, trajectory record and replay). We configure them; we don't rewrite them.
2. **Skeleton first.** A thin layer of our code connects those pieces into one full, logged episode: `python run_episode.py`. It already runs in simulation (`--fake`); it goes onto the arms this afternoon.
3. **Four owners, one robot.** Everyone improves one part of the **same running system** and proves it with a number on the scoreboard (`python scoreboard.py`). Who does what is below.

## Who does what

| Role | Owner | Mission | Your files | Your score | Your demo moment |
|---|---|---|---|---|---|
| **Lead: integration, evaluation & demo** | Jithendra | Get the full loop running on both arms, keep `main` working, turn the logs into results | `run_episode.py` `try_skill.py` `scoreboard.py` `worldarm/robot.py` `episode.py` `logger.py` `metrics.py` `config.py`, docs | End-to-end success, judged on where the blocks really are; episode time | The opening sort run; the results slide |
| **Perception** | _name_ | Know exactly what's on the table, for any block shape | `worldarm/vision.py` `perception.py` `frames.py`, new `eval_perception.py` | Detection recall, color accuracy, position error (mm) | "Any shape": a cylinder and a triangle sorted too |
| **Manipulation** | _name_ | Make every arm skill reliable: pick, place, open, close | `worldarm/manipulation.py` `safety.py`, poses and recordings in `docs/viam-config.md` | Success rate per skill | Drawers that open and close cleanly; stretch: the organizer moved and it still works |
| **World model** | _name_ | Make the robot's memory trustworthy: know what's in closed drawers, notice tampering, find things fast | `worldarm/world_model.py` `planner.py` `sim.py`, new `inspection.py` | Belief accuracy (H2), surprise precision and recall (H3), drawer openings needed to find a block | "Where's blue?" and the surprise |

**Who fits where:**
- **Perception:** comfortable with OpenCV and numpy.
- **Manipulation:** hands-on and patient with hardware; spends the most time at the arm.
- **World model:** likes probability and algorithms; can do most of the work in simulation.

**Shared:** `worldarm/schema.py` (actions and fact keys) changes only at a sync. Adding a new fact key is fine; just announce it.

### Arms

- **Arm A:** armfarm03 with the drawer organizer.
- **Arm B:** the second machine (name TBD) with three open bins. In its env file, set `WORLDARM_CONTAINERS=bins`; open and close then do nothing, and everything else runs unchanged.

Only one person drives an arm at a time, and the driver stays at the arm with the E-stop in reach for every motion. **Two people** are needed for manual-mode recording and for the first run of any new motion.

**Friday**

| Time | Arm A (drawers) | Arm B (bins) | Off the arm |
|---|---|---|---|
| 1:30–2:30 | **Manipulation + Lead:** tape and measure the organizer; test the drawers by hand (**G1**); save Arm A poses | **Perception:** add the services to *both* machines in the Viam app; Arm B env file, `verify_camera.py`, save Arm B poses | **World model:** inspection skill, in simulation |
| 2:30–3:30 | **Manipulation + Lead:** record the 9 drawer sessions; replay each once | **Perception + World model:** perception test set (one moves blocks, one captures) | — |
| 3:30–4:30 | **Manipulation:** `try_skill.py open k` and `close k`, 3× per drawer | **Lead + World model:** `try_skill.py` observe → pick → place; first look into a bin | **Perception:** shape-finder baseline numbers; **G2** |
| 4:30 | **Sync** | | |
| 4:30–6:00 | **Lead + Manipulation:** first full drawer episode (**G4**) | **World model:** full bin episodes with the inspection check (**G3** by 5:00) | **Perception:** depth-first detector |
| 6:00–7:00 | **Dinner + sync:** the v0 numbers on the scoreboard become everyone's starting numbers | | |
| 7:00–8:00 | **Manipulation:** drawer-state measurement, pick fixes | **Perception:** live test of the new detector | **Lead:** demo UI. **World model:** H3 protocol |
| 8:00–8:45 | **Lead:** logged drawer episodes, **backup video** | **World model:** hidden-hand runs on the bins | Fixes |
| 8:45 | **Sync:** push everything; agree tomorrow's first task (**G5**) | | |

**Saturday**

| Time | Arm A (drawers) | Arm B (bins) | Off the arm |
|---|---|---|---|
| 9:30 | **Standup at the scoreboard** | | |
| 9:45–11:00 | **Manipulation** | **Perception** | **Lead:** demo UI. **World model:** probabilistic belief, in simulation |
| 11:00–12:30 | **World model:** H3 hidden-hand runs and the "find" demo, on drawers | **Lead:** H1 runs, checks vs. `--blind`, ≥10 each | Owners confirm their improvement on real runs |
| 12:30 | **Feature freeze (G6).** Lunch. Lead runs the final evaluation on frozen code. | | |
| 1:30–2:30 | Results slide: each owner gives one row. Demo UI polish. | | |
| 2:30–3:30 | Rehearse the demo twice, with one person trying to break it | | |
| 3:30 | **Demo** | | |

### Checkpoints

| | When | Owner | Passes when | If it fails |
|---|---|---|---|---|
| **G1** | Fri 2:30 | Manipulation | By hand, the fingers hold the knob stem and pull the drawer 50–60% open without the organizer sliding | Arm A also uses bins with lids |
| **G2** | Fri 3:30 | Perception | `run_episode.py --dry-run` sees all three colors from `table-view` on both arms | Perception writes a color-threshold detector first |
| **G3** | Fri 5:00 | Lead | First logged real episode on Arm B (bins) | Run single skills with `try_skill.py` and fix the first failure |
| **G4** | Fri 6:00 | Lead | First logged real drawer episode on Arm A; v0 numbers on the scoreboard | Use Arm B's numbers as the baseline; drawers get Saturday morning |
| **G5** | Fri 8:45 | Lead | ≥10 real episodes logged, backup video recorded, everything pushed | — |
| **G6** | Sat 12:30 | Everyone | Feature freeze: after this, only fixes | — |

### Lead: integration, evaluation & demo (Jithendra)

**Until 6 PM**
- [ ] Help Manipulation with recordings on Arm A (1:30–3:30)
- [ ] Bring-up on Arm B with `try_skill.py`: `observe`, then `pick blue`, then `open 1 pick blue place 1`. On bins, `open` does nothing physically, but the belief needs it: placing into a container believed closed is refused.
- [ ] First full episode on Arm B (**G3**) and on Arm A (**G4**)
- [ ] Keep `main` working: before a merge, `python -m pytest` passes and `run_episode.py --fake` ends `done`

**Then**
- [ ] ≥10 real episodes on Friday; backup video before 9 PM (**G5**)
- [ ] H1 on real hardware: the same system with checks vs. `--blind`, ≥10 episodes each
- [ ] Demo UI: live camera next to the belief (what's in each drawer), plus a banner when there's a surprise
- [ ] Results slide: each owner's before and after numbers, with n and intervals

**Score:** end-to-end success judged on where the blocks really are (target ≥ 8/10), plus median episode time.

### Perception (owner: _name_)

**Research question:** does a depth-first detector beat shape-finder's color-contour detector on recall and 3D position error, for every block shape? Depth-first means: find anything that sticks up above the fitted table plane, then classify its color.

**Until 6 PM**
- [ ] Add the services in [docs/viam-config.md](docs/viam-config.md) §1 to **both** machines; `--dry-run` sees all three colors (**G2**)
- [ ] **Ground-truth sheet:**
  1. Tape a paper with 9 crosses, 5 cm apart, to Arm B's table.
  2. In manual mode, touch 3 crosses with a fingertip and save each with `teach.py` (`s cross-1` …). With the gripper pointing straight down, fingertip z = gripper z − 60 mm.
  3. Work out every cross's world position from those three.
- [ ] **Test set:**
  1. Capture ≥15 layouts with blocks centered on crosses, mixed shapes, saved with `teach.py`, `c layout-01`.
  2. Record which block sits on which cross in `data/perception/labels.json`.
  3. Images stay out of git because they're too big; share the folder.
- [ ] **Baseline:** shape-finder's recall per shape and color, and its position error (median and 90th percentile)
- [ ] Capture the organizer from `organizer-view` while you're on Arm A (3:30–4:30), for knob detection

**Then**
- [ ] Depth-first detector in `vision.py`, **same return format**; compare on the same test set
- [ ] Cylinders and triangles, which shape-finder can't see
- [ ] Grasp angle (yaw) for each block, for Manipulation
- [ ] Red-knob detection from `organizer-view`, for Manipulation's "organizer anywhere"

**Score:** recall, color accuracy and position error against ground truth. Target: ≤ 5 mm median, 100% color, every shape.

### Manipulation (owner: _name_)

**Research question:** how reliable can recorded demonstrations make drawer handling, and can straight-line moves driven by perception match them while letting the organizer move?

**Until 6 PM (Arm A)**
- [ ] Tape the organizer down. Measure depth, interior height, knob diameter and how far the knob sticks out, and write them under "Organizer and blocks" below.
- [ ] Test the drawers by hand in manual mode (**G1**)
- [ ] Save poses: `home-pose`, `table-view`, `organizer-view`, `knob-1..3-pre`, `drop-1..3` ([docs/viam-config.md](docs/viam-config.md) §2)
- [ ] Record `grasp-k`, `pull-k` and `push-k` for k = 1..3 (§3), and replay each once with the E-stop ready
- [ ] `try_skill.py open k` and `close k`, 3× per drawer, for the first success numbers

**Then**
- [ ] **Drawer-state check:** report `drawer_open:<k>` from depth at `organizer-view`, so open and close stop showing as "unverified" on the scoreboard
- [ ] **Pick:** grip across the block's short side (yaw from Perception); retry a failed grab once
- [ ] **Organizer anywhere:** approach and pull in straight lines from the detected knob (`motion.move`) instead of replaying recordings. Compare success rates against the recordings.

**Score:** success rate of each skill (pick, place, open, close) among verified steps. Target ≥ 95%. Stretch: still works with the organizer moved 5 cm.

### World model (owner: _name_)

**Research question:** with sensor and action reliability estimated from our own logs, does a probabilistic belief do three things?
- (a) Stay correct about what's in closed drawers.
- (b) Flag tampering with high precision and recall.
- (c) Find a block in fewer drawer openings than v0's fixed search.

**Until 6 PM**
- [ ] **Look into the container** (new `worldarm/inspection.py`):
  - After `place k` and `open k`, view the open drawer or bin and report `contents:<k>`.
  - shape-finder's 2D detections are enough (colors only, no depth). For example, from `drop-k`, back off 150 mm and detect.
  - Wire it into `robot.py` with Lead, and test on Arm B's bins first.
  - In simulation, this check takes truth success from 51% to 99% (see "The skeleton" below).
- [ ] Check whether shape-finder sees the drawer's interior as a colored object. If it does, mask it out.
- [ ] **H3 protocol:**
  - A list of scripted perturbations: move a block to another drawer, remove a block, leave a drawer open.
  - A seeded schedule for running them.
  - A `--perturbed` note in each run's meta, so the scoreboard can compute precision and recall.
  - Rule: the hidden hand puts blocks in the front half of a drawer, which is the part visible when it's open.

**Then**
- [ ] **Probabilistic belief:** a distribution over locations for each block, updated with Bayes' rule using hit and miss rates measured from our logs, instead of "the latest measurement always wins"
- [ ] **Active search** for `find:<color>`: open the most likely drawer first; measure how many openings it takes to find the block
- [ ] Memory that persists across episodes (`state/`), for the "where's blue?" demo

**Score:** belief accuracy (H2), surprise precision and recall (H3), drawer openings needed to find a block. Target: 100% belief accuracy, ≥ 90% surprise recall.

### Rules

1. **Own your files.** Ask before changing someone else's. Wiring a new measurement into `robot.py` is a one-line change; tell Lead.
2. **Branches.** Work on `<name>/<thing>`. Before merging to `main`, `python -m pytest` must pass and `python run_episode.py --fake` must end `done`. `git pull --rebase` often.
3. **No unlogged experiments.** Every real run goes through `run_episode.py` or `try_skill.py`; the results slide comes from `runs/`.
4. **A change counts when the number moves:** on real runs (or the perception test set), with n stated. Note the number before and after.
5. **Safety:** the driver stays at the arm with the E-stop in reach. New motions are tested high first.
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
| Data and training | Viam data manager; [`viam-devrel/yolo-training`](https://github.com/viam-devrel/yolo-training) + `viam-labs:vision:yolov8` | For the perception track |
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

**Status (Fri 1:15 PM):**
- Runs end to end in simulation (`python run_episode.py --fake`); 26 tests pass.
- The real-arm code is split by owner: `vision.py` (Perception), `manipulation.py` (Manipulation), and `robot.py`, which wires them together (Lead). It is **not yet validated on hardware**.
- `try_skill.py` is the bring-up tool.

```
worldarm/  schema · world_model · planner · episode · sim · vision · manipulation · robot · safety · config · logger · metrics
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

These failure rates are assumptions, and the simulated checks never make mistakes. What it does show is where checking helps most: **looking into the drawer** is the observation the real robot doesn't have yet. That's the world-model owner's first task.

**Version 0 of each stage:**

| Stage | v0 |
|---|---|
| Perception | The `vision-segment` service, from the `table-view` pose |
| Pick and place | The workshop pick pattern; place by going to switch `drop-k` and opening |
| Drawers | Switch `knob-k-pre` → recording `grasp-k` → grab → recording `pull-k` → open. Close is recording `push-k`. Organizer taped in place. |
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
- The slides are smooth, and **there's no stop**: a drawer can come out completely.
- The blocks are **blue, yellow and green**, in mixed shapes.

**Decisions:**
- **Red only ever means a knob.**
- **Knob grasp:** approach horizontally and close the fingers on the knob's **stem, behind the head**. When we pull, the head can't slip through the closed fingers.
- **Pull about 50–60% of the drawer's depth**, never "until it stops", because nothing stops it.
- **Tape the organizer down.**
- **v0 blocks are cubes and bars,** because shape-finder only sees four-sided shapes. Other shapes are a perception goal.

**Still open:** organizer depth, drawer interior height, knob diameter and how far the knob sticks out (Manipulation measures these); the second arm's machine name and address.
