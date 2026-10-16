# WorldArm plan

Team of 4, two arms. Demos are **Saturday at 3:30 PM**. Doors close at 9 PM on Friday, with no overnight hacking.

## How we work

1. **Reuse first.** Low-level pieces already exist (drivers, saved poses, block detection, pick and place, trajectory record and replay). We configure them; we don't rewrite them.
2. **Skeleton (Jithendra, until about 4 PM Friday).** A thin layer of our code connects those pieces into one full, logged episode on a real arm: `python run_episode.py`. `--dry-run` plans without moving; `--fake` uses a simulated robot.
3. **Improvement tracks (everyone, from about 4 PM Friday).** Each person owns one stage of the running system. Each stage has a score computed from the episode logs, a starting number from the skeleton, and a target. Every improvement shows up in the team's end-to-end success rate. `python metrics.py` prints the scoreboard.

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
- **Drawers anywhere:** adapt the recorded drawer moves to the *detected* knob position, so the organizer can be placed anywhere.
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

## Phase 1: the skeleton

Our code is a thin layer over the services above:

```
worldarm/
  config.py       machine env, service and switch names, safety limits
  safety.py       our own table floor + workspace box, checked before every motion.move
  robot.py        connect; go_to(switch); move_to(pose) via motion + safety; gripper; play(recording)
  observe.py      observation model: blocks via vision-segment; drawer open/contents via depth
  world_model.py  belief state; predict(action) → expected; update(observation) → surprises
  planner.py      next action for the goal (sort everything / retrieve one color)
  skills.py       look, pick, place, open_drawer, close_drawer (each returns what it observed)
  logger.py       runs/<timestamp>/episode.jsonl + before/after images
run_episode.py    one full episode   (--dry-run, --fake, --goal sort | retrieve:blue)
metrics.py        scoreboard from runs/
teach.py          manual-mode helper: gripper open/grab, pose readout, captures
```

**Version 0 of each stage:**

| Stage | v0 |
|---|---|
| Perception | The `vision-segment` service, from the `table-view` pose |
| Pick and place | The workshop pick pattern; place by going to switch `drop-k` and opening |
| Drawers | Switch `knob-k-pre` → recording `grasp-k` → grab → recording `pull-k` → open. Close is recording `push-k`. Organizer taped in place. |
| World model | Deterministic belief with predicted effects; checks use only `IsHoldingSomething` and "block gone from the table" |
| Planner | Fixed order: for each block on the table, open drawer → pick → place → close |

### What the rest of the team does during Phase 1

| Who | Task | Feeds into |
|---|---|---|
| One person | Add the modules and services in [docs/viam-config.md](docs/viam-config.md) to **both** machines; check that `vision-segment` finds the blocks in the app's Test panel | Perception v0 |
| Two people on Arm A | Tape the organizer down and measure it. In manual mode, save the switch poses, then record `grasp-k`, `pull-k` and `push-k` for each drawer (see the config doc) | Drawers v0 |
| One person on Arm B | Env file, `verify_camera.py`, save switch poses with 3 bins; captures of blocks and knobs from both views (`teach.py`, command `c NAME`), with ruler-measured ground truth | Perception test set |

## Phase 2: improvement tracks

| Track | Owner | Score (from episode logs) | Starts from | Ways to improve | Target |
|---|---|---|---|---|---|
| **Perception** | _ | Block position error vs. ruler (mm); color accuracy; knob localization error | shape-finder (four-sided shapes only) | Depth segmentation that works for any shape; knob detection; tolerance to lighting; compare against the shape-finder baseline | ≤ 5 mm, 100% color, every shape |
| **Pick & place** | _ | Pick success (`IsHoldingSomething` after the lift); place success (block seen in drawer) | Workshop pick pattern | Grip across the short side; grasps per shape; retry after a failed grab | ≥ 95% |
| **Drawers** | _ | Open success (travel within ±10 mm of target); close success (flush) | Recorded demonstrations, organizer taped | Adapt the recordings to the detected knob, so the organizer can be anywhere; straight-line pulls | ≥ 95%, organizer placed anywhere |
| **World model** | _ | Belief accuracy; surprise precision and recall; recovery success | Holding check only | Look into each opened drawer and compare with the belief; on a surprise, replan; "where is X?" | 100% belief, ≥ 90% surprise recall |
| **Integration, evaluation & demo** | Jithendra | End-to-end episode success; episode time | — | Demo UI (camera feed next to the belief state), results slide, backup video | ≥ 8/10 episodes |

**Rules for tracks**
- Change your stage's internals freely. Interfaces only change by agreement.
- A change counts when the score improves on real episodes, not when it looks better. Keep before and after numbers for the results slide.
- **Two arms:** Arm A has the organizer. Arm B runs the same episodes with three open bins in place of drawers.

## Interfaces

```python
@dataclass
class BlockObs:          # observe → world model
    color: str           # "blue" | "yellow" | "green"  (no red: the knobs are red)
    xyz_mm: np.ndarray   # world frame, center of the block
    size_mm: tuple[float, float, float]
    confidence: float

@dataclass
class DrawerObs:
    index: int           # 1 = top
    open_mm: float       # measured travel
    visible_colors: list[str]  # only meaningful when open

# skills: each returns what was observed, never just True/False
pick(block: BlockObs) -> PickResult            # holding: bool
place(container: int) -> PlaceResult
open_drawer(index: int) -> DrawerObs
close_drawer(index: int) -> DrawerObs
```

For a bin (Arm B), `open_drawer` and `close_drawer` do nothing, so the same planner runs on both arms.

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
- **v0 blocks are cubes and bars,** because shape-finder only sees four-sided shapes. Other shapes are a perception-track goal.

**Still open:** organizer depth, drawer interior height, knob diameter and how far the knob sticks out; the second arm's machine name and address.

## Timeline

| When | Skeleton (Jithendra) | Rest of the team |
|---|---|---|
| **Fri → 2:00** | Package, safety, robot I/O, belief, planner, logger, simulated robot | Configure services on both machines, tape and measure the organizer, save poses, record drawer moves, Arm B captures |
| Fri 2:00 → 4:00 | Skeleton on the real arm → **first full episode** | Finish recordings and captures; pick tracks; read the skeleton code |
| Fri 4:00 → 6:00 | Run Integration, evaluation & demo | **Tracks start.** Everyone runs episodes on both arms |
| Fri 7:00 → 9:00 | ≥10 logged episodes; **backup video before 9 PM** | Tracks |
| Sat 9:30 → 12:30 | Demo UI, evaluation runs for H1–H3 | Tracks; retrieval and surprise demo |
| Sat 12:30 → 1:30 | **Feature freeze.** Lunch. | |
| Sat 1:30 → 3:30 | Results slide, rehearse the demo twice with someone actively trying to break it | |
