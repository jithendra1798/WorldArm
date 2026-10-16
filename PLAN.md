# WorldArm plan

Team of 4, two arms. Demos are **Saturday at 3:30 PM**. Doors close at 9 PM on Friday, with no overnight hacking.

## How we work: build the whole system thin first, then everyone improves one stage

**Phase 1: skeleton (Jithendra, until about 4 PM Friday).**
- The whole system runs end to end on a real arm. Every stage exists, and each one is as simple as possible.
- One command runs a full, logged episode: `python run_episode.py`.
- `--dry-run` plans and prints the actions without moving anything. `--fake` uses a simulated robot, so you can work without an arm.

**Phase 2: improvement tracks (everyone, from about 4 PM Friday).**
- Each person owns one stage of the same system.
- Each stage has a score computed from the shared episode logs, a starting number from the skeleton, and a target.
- Everyone runs full episodes on the real arms, and every improvement shows up in the team's end-to-end success rate.
- `python metrics.py` prints the scoreboard.

**Why:** nobody builds a part in isolation, we see a real robot running from Friday afternoon, and there's no big merge on Saturday morning.

## What we claim, and how we'll show it

We don't just say it works: each claim gets a baseline, a number, and the number of trials behind it.

| | Hypothesis | Measured by |
|---|---|---|
| **H1** | Checking every action against its prediction raises end-to-end success compared with running the same sequence blind | Success rate over ≥10 episodes each, verified vs. blind, with 95% Wilson intervals |
| **H2** | The belief state tracks hidden drawer contents correctly | Fraction of blocks whose believed location matches reality at episode end |
| **H3** | The robot detects when the world was changed behind its back | Precision and recall of "surprise" flags over scripted perturbations (block moved, drawer left open, block removed) |

**Episode protocol**
- Block start positions and colors come from a seeded generator, which prints a layout for the person resetting the table.
- Every episode is logged; no cherry-picking.
- Failures are sorted into categories (perception, grasp, drawer, planner) for the results slide.

**Logging**
- Each step writes one JSON line to `runs/<timestamp>/episode.jsonl` containing: belief before, action, predicted effect, observation, outcome, and paths to the before and after images.

## Phase 1: the skeleton

```
worldarm/
  config.py       machine env, taught poses (poses/<machine>.json), safety limits
  safety.py       rejects any target below the fingertip floor or outside the workspace box
  robot.py        connect; move_to(pose) via the motion service + safety check; gripper; capture
  frames.py       camera → world
  perception.py   blocks (height above table → color, xyz, yaw); knobs (red blob + depth)
  world_model.py  belief state; predict(action) → expected state; update(observation) → surprises
  planner.py      next action for the goal (sort everything / retrieve one color)
  skills.py       look, pick, place, open_drawer, close_drawer
  logger.py       runs/<timestamp>/episode.jsonl + before/after images
run_episode.py    one full episode   (--dry-run, --fake, --goal sort | retrieve:blue)
metrics.py        scoreboard from runs/
teach.py          manual-mode helper: gripper, record poses, captures
```

**Version 0 of each stage (deliberately simple):**

| Stage | v0 |
|---|---|
| Perception | Color thresholds plus a depth mask of what sticks up above the table, from the table view. Knobs aren't detected; their poses are taught. |
| Skills | Taught poses. Pull and drop positions are fixed per drawer. |
| World model | Deterministic belief with predicted effects. Checks only use `IsHoldingSomething` and "the block is gone from the table". |
| Planner | Fixed order: for each block on the table, open drawer → pick → place → close. |

### What the rest of the team does during Phase 1

| Who | Task | Feeds into |
|---|---|---|
| Two people on Arm A | Tape the organizer down, measure it, test the drawers by hand, then **teach the poses below** with `teach.py` | Skills v0 |
| One person on Arm B | Set up the env file and run `verify_camera.py`; save captures of blocks and knobs from both views (`teach.py` command `c NAME`); write down the ruler-measured ground truth for each capture | Perception test set |

**Poses to teach** (drawers are numbered 1 = top to 3 = bottom; use these exact names):

| Name | What it is |
|---|---|
| `home` | A safe pose clear of everything |
| `table_view` | Camera pointing straight down, about 38 cm above the block area |
| `organizer_view` | Camera facing the drawer fronts from about 30 cm away |
| `knob{k}_grasp` | Gripper horizontal, fingers around the knob's **stem** behind the head, drawer closed |
| `knob{k}_open` | Same grasp, after pulling the drawer out to the chosen distance (50–60% of its depth) |
| `drop{k}` | Gripper above the exposed part of the open drawer, where the block gets released |

The approach pose for each knob is computed automatically: 60 mm back along the gripper's pointing direction.

## Phase 2: improvement tracks

| Track | Owner | Score (from episode logs) | Skeleton v0 | Ways to improve | Target |
|---|---|---|---|---|---|
| **Perception** | _ | Block position error vs. ruler (mm); color accuracy; knob localization error | Color thresholds + depth mask | Shape-aware angle estimate; detect knobs so taught knob poses aren't needed; tolerance to lighting changes | ≤ 5 mm, 100% color |
| **Pick & place** | _ | Pick success (`IsHoldingSomething` after the lift); place success (block seen in drawer) | Grab from above at the block's center | Grip across the short side; grasps per shape; retry after a failed grab | ≥ 95% |
| **Drawers** | _ | Open success (travel within ±10 mm of target); close success (flush) | Taught knob poses, fixed pull | Move to the *detected* knob so the organizer can be anywhere; straight-line pulls | ≥ 95%, organizer placed anywhere |
| **World model** | _ | Belief accuracy; surprise precision and recall; recovery success | Holding check only | Look into each opened drawer and compare with the belief; on a surprise, replan; "where is X?" retrieval | 100% belief, ≥ 90% surprise recall |
| **Integration, evaluation & demo** | Jithendra | End-to-end episode success; episode time | — | Demo UI (camera feed next to the belief state), results slide, backup video | ≥ 8/10 episodes |

**Rules for tracks**
- Change your stage's internals freely. Interfaces only change by agreement.
- A change counts when the score improves on real episodes, not when it looks better. Keep before and after numbers for the results slide.
- **Two arms:** Arm A has the organizer, so the drawers track needs it. Arm B runs the same episodes with three open bins in place of drawers, for the perception, pick & place and world-model tracks.

## Interfaces

```python
@dataclass
class BlockObs:          # perception → world model
    color: str           # "blue" | "yellow" | "green"  (no red: the knobs are red)
    xyz_mm: np.ndarray   # world frame, top-center of the block
    yaw_deg: float       # grasp axis about world z
    size_mm: tuple[float, float, float]
    confidence: float

@dataclass
class DrawerObs:
    index: int           # 1 = top
    open_mm: float       # measured travel
    knob_xyz_mm: np.ndarray    # world frame, center of the red knob
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
- The organizer has 3 **opaque** drawers, so closed contents are truly hidden. Each drawer has a **rounded red knob**.
- The slides are smooth, and **there's no stop**: a drawer slides the full depth of the organizer and can come out completely.
- The blocks are **blue, yellow and green** (the red one was swapped out), in mixed shapes.

**Decisions:**
- **Red only ever means a knob.** That gives us a free marker for locating handles.
- **Knob grasp:** approach horizontally and close the fingers on the knob's **stem, behind the rounded head**. When we pull, the head can't slip through the closed fingers, so we aren't relying on friction against a round surface.
- **Pull a fixed distance**, about 50–60% of the drawer's depth, never "until it stops", because nothing stops it.
- **Tape the organizer down.**
- **Start with these block shapes:** ones with parallel faces when seen from above (cube, bar, upright cylinder) that fit the drawer's interior height. Other shapes come later, as a test of whether the system generalizes.

**Still open:** organizer depth, drawer interior height, knob diameter and how far the knob sticks out; the second arm's machine name and address.

## Timeline

| When | Skeleton (Jithendra) | Rest of the team |
|---|---|---|
| **Fri → 1:00** | Package, safety, robot I/O, belief, planner, logger, simulated robot | Drawer test by hand, measurements, **teach poses**; Arm B setup and perception captures |
| Fri 2:00 → 4:00 | Skeleton on the real arm: v0 perception + taught-pose skills → **first full episode** | Finish poses and captures; pick tracks; read the skeleton code |
| Fri 4:00 → 6:00 | Run Integration, evaluation & demo | **Tracks start.** Everyone runs episodes on both arms |
| Fri 7:00 → 9:00 | ≥10 logged episodes; **backup video before 9 PM** | Tracks |
| Sat 9:30 → 12:30 | Demo UI, evaluation runs for H1–H3 | Tracks; retrieval and surprise demo |
| Sat 12:30 → 1:30 | **Feature freeze.** Lunch. | |
| Sat 1:30 → 3:30 | Results slide, rehearse the demo twice with someone actively trying to break it | |
