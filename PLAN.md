# WorldArm plan

Team of 4, two arms. Demos are **Saturday at 3:30 PM**. Doors close at 9 PM on Friday, with no overnight hacking.

## What we claim, and how we'll show it

We don't just say it works: each claim gets a baseline, a number, and the number of trials behind it.

| | Hypothesis | Measured by |
|---|---|---|
| **H1** | Checking every action against its prediction raises end-to-end success compared with running the same sequence blind | Success rate over ≥10 episodes each, verified vs. blind, with 95% Wilson intervals |
| **H2** | The belief state tracks hidden drawer contents correctly | Fraction of blocks whose believed location matches reality at episode end (target ≥95%) |
| **H3** | The robot detects when the world was changed behind its back | Precision and recall of "surprise" flags over scripted perturbations (block moved, drawer left open, block removed) |

**Per-step reliability**, reported as successes/attempts:

| Step | Counts as success when |
|---|---|
| Pick | `IsHoldingSomething` is true after the lift |
| Open | Drawer travel measured from depth is ≥ the target |
| Place | The block is visible in the drawer before closing |
| Close | The drawer front is flush (depth check) |

**Episode protocol**
- Block start positions and colors come from a seeded generator, which prints a layout for the person resetting the table.
- The drawer for each color is fixed per episode.
- Every episode is logged; no cherry-picking.
- Failures are sorted into categories (perception, grasp, drawer, planner) for the results slide.

**Logging**
- Each step writes one JSON line to `runs/<timestamp>/episode.jsonl` containing: belief before, action, predicted effect, observation, outcome, and paths to the before and after images.
- `runs/` stays out of git. Summaries go on the results slide.

## Roles

| # | Owner | Owns | First deliverable |
|---|---|---|---|
| 1 | Perception | Block detection (height above table plane → segments → color, position, angle); organizer and handle localization; "is the drawer open, what's inside" observations | Block positions within ±5 mm of ruler ground truth |
| 2 | Manipulation | Safety layer (minimum height, workspace box, all moves via the motion service); pick, place, open drawer, close drawer; taught poses relative to the organizer | One reliable drawer open/close |
| 3 | World model and planner | Belief state; predicted effect per action; compare predicted vs. observed; surprise → replan; "where is X?" retrieval | Belief and planner running on logged or fake observations, no arm needed |
| 4 | Evaluation, data and demo | Episode runner and seeded layouts; JSONL logging; metrics script; demo UI (camera feed next to the belief state); results slide; backup video | Logger and metrics script, running on the first real trials |

Owners 3 and 4 can start without an arm. Write against the interfaces below and use fake observations until the real ones land.

## Two arms: parallel testing

There's only one drawer organizer, so the arms split by risk:

| | Arm A (armfarm03): drawer arm | Arm B (second machine): block arm |
|---|---|---|
| Fri → 4 PM | Drawer feasibility, then the open, close and place skills (owner 2) | Block perception, then pick and place into **open bins** standing in for drawers (owners 1 and 4) |
| Fri 4 → 9 PM | Full episodes with drawers (owners 2 and 3) | Full episodes with bins: tests planner, belief, logging and metrics without the drawer risk (owners 3 and 4) |
| Sat | Evaluation runs and the demo | Extra evaluation runs (doubles our trial count) and backup demo |

- The skill interface treats a container as **drawer or bin** (for a bin, open and close do nothing), so the same planner code runs on both arms.
- **Calibration is per arm.** Run `verify_camera.py` on Arm B first; its table height and camera frame will differ from Arm A's.

## Interfaces (agree on these before splitting up)

```python
@dataclass
class BlockObs:          # perception → world model
    color: str           # "red" | "green" | "blue" | ...
    xyz_mm: np.ndarray   # world frame, top-center of the block
    yaw_deg: float       # grasp axis about world z
    size_mm: tuple[float, float, float]
    confidence: float

@dataclass
class DrawerObs:
    index: int           # 0 = top
    open_mm: float       # measured travel
    visible_colors: list[str]  # only meaningful when open

# skills: each returns what was observed, never just True/False
pick(block: BlockObs) -> PickResult            # holding: bool
place(container: int) -> PlaceResult
open_drawer(index: int, travel_mm: float) -> DrawerObs
close_drawer(index: int) -> DrawerObs
```

## Timeline

| When | Gate (we don't move on until it passes) |
|---|---|
| **Fri → 12:30** | **Drawer feasibility, tested by hand in manual mode:** organizer taped down; handle can be pinched or hooked; drawer opens far enough to lower a block in from above; maximum travel measured. Interfaces agreed. **If it fails, switch to bins with lids.** |
| Fri 12:30 → 4:00 | Block perception within ±5 mm; reliable pick (Arm B); drawer open and close (Arm A) |
| Fri 4:00 → 6:00 | Full episode end to end on both arms; logging running |
| Fri 7:00 → 9:00 | ≥10 logged episodes; fix the top failure categories; **record backup video before 9 PM** |
| Sat 9:30 → 12:30 | Retrieval ("where is X?"), surprise detection, demo UI; H1–H3 evaluation runs |
| Sat 12:30 → 1:30 | **Feature freeze.** Lunch. |
| Sat 1:30 → 3:30 | Results slide, rehearse the demo twice with someone actively trying to break it |

## Open questions

- What does the drawer handle look like (knob, lip, recessed slot), and how far does each drawer travel?
- What are the second arm's machine name and address?
- Which three block colors and shapes? Start with blocks that stand flat and can be grabbed from above, like cubes and rectangular bars.
