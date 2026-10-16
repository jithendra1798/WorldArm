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
| 1 | Perception | Block detection (height above table plane → segments → color, position, angle); knob localization (red blob + depth); "is the drawer open, what's inside" observations | Block positions within ±5 mm of ruler ground truth |
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
    color: str           # "yellow" | "green" | "blue" (no red: the knobs are red)
    xyz_mm: np.ndarray   # world frame, top-center of the block
    yaw_deg: float       # grasp axis about world z
    size_mm: tuple[float, float, float]
    confidence: float

@dataclass
class DrawerObs:
    index: int           # 0 = top
    open_mm: float       # measured travel
    knob_xyz_mm: np.ndarray    # world frame, center of the red knob
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
| **Fri → 12:30** | **Drawer feasibility, tested by hand in manual mode:** organizer taped down; fingers close on the knob stem from a horizontal approach and pull the drawer out without slipping; pull distance chosen so a block can be lowered in and the drawer doesn't tip. Interfaces agreed. **If it fails, switch to bins with lids.** |
| Fri 12:30 → 4:00 | Block perception within ±5 mm; reliable pick (Arm B); drawer open and close (Arm A) |
| Fri 4:00 → 6:00 | Full episode end to end on both arms; logging running |
| Fri 7:00 → 9:00 | ≥10 logged episodes; fix the top failure categories; **record backup video before 9 PM** |
| Sat 9:30 → 12:30 | Retrieval ("where is X?"), surprise detection, demo UI; H1–H3 evaluation runs |
| Sat 12:30 → 1:30 | **Feature freeze.** Lunch. |
| Sat 1:30 → 3:30 | Results slide, rehearse the demo twice with someone actively trying to break it |

## Organizer and blocks

**What we know:**
- The organizer has 3 drawers, each with a **rounded red knob**.
- The slides are smooth, and **there's no stop**: a drawer slides the full depth of the organizer and can come out completely.
- The organizer can be taped down.
- The blocks are red, yellow and green, in mixed shapes. We can swap them.

**Decisions:**
- **Use no red blocks; swap the red one for blue.** The knobs are red, so a red block would make every color check also have to rule out knobs. With no red blocks, anything red on the organizer face is a knob, which gives us a free marker for locating handles. If we keep red, mask out the organizer's area before running block detection.
- **Knob grasp:** approach horizontally with the gripper pointing at the organizer face, and close the fingers on the knob's **stem, behind the rounded head**. When we pull, the head can't slip through the closed fingers, so we aren't relying on friction against a round surface. Fallback: hook the closed fingers behind the head and pull.
- **Pull a fixed distance, never "until it stops"**, because nothing stops it. Open to about 50–60% of the drawer's depth: far enough to lower a block in from above, short enough that the drawer doesn't tip or fall out. Pick the exact distance during the drawer test.
- **Locate knobs from about 30 cm away** with the camera facing the organizer: the red blob plus depth gives the knob's 3D center. Then move in on a straight line without vision; depth fails closer than about 26 cm.
- **Start with these block shapes:** ones with parallel faces when seen from above (cube, rectangular bar, upright cylinder) that fit the drawer's interior height. Triangles and arches come later, as a test of whether the system generalizes.
- **Hidden must really mean hidden.** If the drawers are see-through, cover their fronts and sides with paper. Otherwise "remembers what it can't see" isn't true.

**Still open:**
- Measure with a ruler: organizer depth, drawer interior height, knob diameter, and how far the knob sticks out (needed for the pull distance and the grasp).
- Are the drawers transparent?
- What are the second arm's machine name and address?
