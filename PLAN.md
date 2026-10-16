# WorldArm plan

Team of 4, two arms. Demos are **Saturday at 3:30 PM**; feature freeze at 12:30. Doors close at 9 PM on Friday.

## The story

**Robots that model their world:** they remember what they can't see, and predict where things will be.

- **Chapter 1, memory.** One arm sorts colored blocks into a 3-drawer organizer, one color per drawer.
  - Once a drawer closes, its contents are hidden, so the robot keeps a belief about where everything is.
  - It predicts what each action will do, and flags a *surprise* when what it sees disagrees.
- **Chapter 2, physics.** Two arms face each other across one table, under an overhead depth camera, and pass a black rubber ball.
  - The same world model now predicts *motion*: where a rolling or tossed ball will be, and when.
  - The ball's physics is learned from the robot's own trials.
  - The ball is just another object in the belief, a toss is an action with a predicted outcome, and a miss is a surprise.

**The demo, in order:**
1. **Sort:** blocks go into their color's drawer.
2. **Where's green?** The robot opens the drawer it remembers.
3. **Surprise:** a judge secretly moves a block; the robot notices when it looks.
4. **Ball into a held drawer:** the other arm holds a drawer open, the ball goes in, the drawer closes. Then "where's the ball?"
5. **Intercept:** a judge rolls the ball across the table, and the robot puts a cup where the ball *will be*, before it gets there.

## Where we are (Fri 6:45 PM)

- **Chapter 1 skills work in isolation:** picking, opening and closing the drawers, and dropping blocks. The code is in the repo as Viam modules and scripts.
- **The world-model loop** (`run_episode.py`) runs end to end in simulation, but isn't connected to those skills yet.
- **The overhead camera works** (`overview-cam` on armfarm03; details under Stage 2). Only armfarm03's half of the table is in its view so far.
- **The second arm, armfarm05**, is at the other end of the table.

## Stage 1: integrate and benchmark chapter 1 (tonight, on armfarm03)

This is the guaranteed demo, so it comes before chapter 2.

The drawer service is deployed on armfarm03 as `PickBottle`. Waj's pick-and-drop (`move_one_object`) accepts any world-frame drop point.

1. **Deploy the drawer split.** ✅ Deployed Friday evening and checked with `status`. `modules/drawer_service.py` now has `open` (pull 12 cm, let go, return a drop point inside the drawer), `close`, and `status` and `forget`, which never move. Paste it into the inline module behind `PickBottle` and deploy. Its logic is tested with fake hardware (`tests/test_drawer_service.py`).
2. **Set up armfarm03 for Waj's scene parser:**
   - Put **one** blue-tape square in the wrist camera's overview, with the blocks in it. The drawers are the destination, so no second square is needed (`VIAM_TAPE_ZONES=1`; armfarm05 keeps its two).
   - Colors default to **green → 1, yellow → 2, black → 3** (`WORLDARM_ASSIGNMENT`). His detector names green, yellow, orange, red and black. Blue is the tape, and red is the knobs.
   - Put the arm at the overview pose and save it: `python sort_bench.py --teach-view`.
3. **Run:** `python sort_bench.py --preview`, then `python sort_bench.py --episodes 10`.
   - Each block goes drawer `open` → pick and drop → `close`.
   - Every step's own success report is logged, and you record where each block really ended up.
   - `python scoreboard.py` shows per-step and end-to-end rates.
4. **Freeze:** tag `v1-sorting` in git and record a backup video. Chapter 2 work then happens on a branch, so v1 can't break.

Waj's code defaulted to armfarm05 in two places: its side wall (y = −500; armfarm03's is at +500) and one shared overview-pose file. Both can now be set per arm, and `sort_bench.py` applies armfarm03's values.

**Gate (8:45 PM):** per-skill numbers and ≥10 full episodes on the scoreboard, plus the backup video.

## Stage 2: one shared world (tonight and Saturday early)

**Camera roles:**
- **The overhead camera tracks the ball and judges every outcome.**
- **The wrist cameras are each arm's close-up eyes:** picking up the ball or cup, and checking what's inside a cup or drawer.
- **The cameras are combined in the world model, not at the pixel level.** Each camera reports what it measured, with its timestamp, and the belief merges them. Cameras on two different machines can't be synchronized in hardware anyway.

**The overhead camera sees the whole ball path,** from the release point to the cup or drawer, with margin:
- At 30 fps, a toss gives only about 9 frames end to end; a partial view leaves 3–4.
- The start lets the model predict early. The landing lets it check the prediction and learn.
- The predicted-vs-actual arc is the demo's best visual.

**Placement:**
- Keep it overhead, but widen the view until both arms' hand-off areas fit.
- The white table is the best possible background for a black ball.
- Ball height in the air comes from three sources: the thrower's own release position and speed; the ball's apparent size (closer looks bigger); and the point cloud, where black rubber returns depth.

**Background:** the camera is fixed, so we ignore everything outside a fixed outline of the table. On the table, we pick out the ball by its dark color, its size (about 30 px at this height), its motion, and **agreement with the predicted path**.

**Measured on `overview-cam`** (Orbbec Astra 2):

| Measurement | Value |
|---|---|
| Image | 1280×720 color, fx = fy = 834 px |
| Table distance | ≈ 1.31 m below the camera (from its point cloud) |
| 3D | **Use the point cloud.** The depth image comes back rounded to whole meters (1 or 2), so it's unusable. |
| Speed through the SDK from a laptop | 3.5 fps with depth; 5.8 fps color only. 1 in 10 requests takes > 400 ms. |

That's far too slow and jittery for a toss. So:
- **To learn:** record at 30 fps on the robot with Viam's data capture.
- **For live tracking:** run the tracker on armfarm03's computer.
- `webcam-1` reads the same camera's color stream through a second driver. Earlier, one of the two dropped out, so keep only `overview-cam` once its color is confirmed stable.

**Calibration:** touch 4–6 marked table points with each arm's fingertip. This maps overhead pixels to table coordinates for both arms. Report the leftover error in mm; target ≤ 10 mm.

**Two-arm safety:**
- Each arm stays in its own half; the middle is for the ball only. Each arm's half goes in the other machine's config as an obstacle.
- Two E-stops staffed whenever both arms can move.
- **Speed caps:** 15°/s for the arm and 0.9 m/s for the ball, from the catcher module. Raise them only with Viam staff's OK.

**Measure before designing:** frame rate on the robot; command delay to each arm; how long each arm takes to move a given distance at the capped speed.

**What the speed caps mean physically:**
- At 0.9 m/s, a ball thrown at 45° travels about 8 cm; released from 30 cm up, it travels about 22 cm. A throw across the table needs 2.5–3 m/s.
- So "throws" are short tosses between grippers that meet in the middle, or the ball rolls.
- The catcher can't chase at 15°/s. It has to be in place before the ball arrives, which means **predicting is required**.

## Stage 3: the two-arm ladder (Saturday)

Each rung is benchmarked before moving to the next.

| Rung | What happens | World model | Score |
|---|---|---|---|
| **A. Ball into a held drawer** | Arm 1 pulls a drawer to 75% and keeps holding the knob. Arm 2 drops or tosses the ball in from 20–30 cm away. The drawer closes. | The ball becomes an object in the belief: "ball in drawer 2". Then "where's the ball?" | Success rate; belief accuracy |
| **B. Intercept a rolling ball** | A judge rolls the ball across the table; the catcher places a cup where the ball will be, before it arrives | A rolling model, with its slowdown learned from our rolls, predicts where and when. The catcher only commits to points it can reach in time. | Intercept rate, predicting vs. chasing the last seen position (N ≥ 20 each) |
| **C. Toss into a placed bowl** | Arm 2 tosses with adjustable release settings; the overhead camera records the landing; a fitted model aims the next toss. Arm 1 puts the bowl at the predicted landing spot. | A learned toss model with uncertainty | Landing error vs. number of practice tosses; catch rate |
| D. Stretch | Correcting during flight; tosses across the table | Only if staff allow higher speeds and the measured delays allow it | — |

## What we claim, and how we'll show it

| | Hypothesis | Measured by |
|---|---|---|
| **H1** | Checking every action against its prediction raises end-to-end success | The same system with checks on vs. off, ≥10 episodes each |
| **H2** | The belief tracks hidden drawer contents correctly | Share of blocks whose believed location matches reality at episode end |
| **H3** | The robot detects when the world changed behind its back | Precision and recall of surprises over scripted perturbations |
| **H4** | Learned ball physics predicts where the ball will stop or land better than "it keeps its current velocity" | Prediction error vs. time before arrival, both models |
| **H5** | Going to the predicted point beats chasing the last seen position | Intercept/catch rate, N ≥ 20 each |
| **H6** | The robot still knows where the ball is after it's hidden in a drawer | Belief accuracy for the ball, including scripted perturbations |

Every number has n and a 95% interval. Every run is logged to `runs/` (seeded layouts, no cherry-picking), and failures are sorted into categories for the results slide.

## Who does what

**Chapter 1** (skills done; integration tonight):

| Package | Owner |
|---|---|
| Find and pick up any block | **Waj** |
| Open and close the drawers (75%, never further) | **Sandeep** |
| Sort the blocks into the drawers by color (dropping) | **Riyaj** |
| World model, integration, benchmark & demo | **Jithendra** |

**Chapter 2** (proposed; confirm at the next sync):

| Package | Owner |
|---|---|
| Ball tracking and prediction on the overhead camera (builds on his catcher's tracking) | **Waj** |
| Holding a drawer open as the target; the two-arm zones | **Sandeep** |
| The release: drops, tosses and rolls from arm 2; practice-toss data | **Riyaj** |
| Stage 1 integration and benchmark; shared calibration; the program that coordinates both machines; the ball in the world model; evaluation; demo screen | **Jithendra** |

## Timeline

| When | What |
|---|---|
| **Fri 7:00–8:30** | Stage 1: per-skill benchmarks, integration, full episodes on armfarm03. In parallel: widen the overhead view to both arms; touch-point calibration. |
| **Fri 8:30–8:45** | Backup video, tag `v1-sorting`, push (**gate**) |
| **Fri 8:45–9:00** | Sync: confirm the chapter 2 split; first task for Saturday |
| **Sat 9:30–11:00** | Rungs A and B, with benchmarks |
| **Sat 11:00–12:30** | Rung C; H4–H6 runs |
| **Sat 12:30** | **Feature freeze.** Lunch. Final evaluation on frozen code. |
| **Sat 1:30–2:30** | Results slide; demo screen |
| **Sat 2:30–3:30** | Rehearse twice, with one person trying to break it |
| **Sat 3:30** | **Demo** |

## Rules

1. **Own your files and functions.** Ask before changing someone else's.
2. **Branches.** Work on `<name>/<thing>`. Before merging to `main`, `python -m pytest` must pass and `python run_episode.py --fake` must end `done`.
3. **No unlogged experiments.** Results come from `runs/`.
4. **A change counts when the number moves:** on real runs, with n stated.
5. **Safety:** someone at every arm that can move, E-stop in reach. New motions are tested slow and high first. Drawers never go past the 75% mark.
6. **Blocked for 20 minutes? Say so.**

## Reference

### Reuse first

| Layer | We reuse |
|---|---|
| Drivers | `viam:ufactory` (arm, gripper), `viam:realsense` (wrist), `viam:orbbec` (overhead) |
| Saved poses, obstacles, motion | `erh:vmodutils` (arm-position-saver, obstacle), built-in motion service |
| Detection | YOLOv8 (`viam-labs:YOLOv8`), color detectors → `detections-to-segments` (3D objects) |
| Skills | The team's modules, controlled through their service commands (DoCommand): pick, drawers, catch, transfer between zones |

### Interfaces

Defined in `worldarm/schema.py`. A robot implements `execute(action) -> Observation`.
- **Actions:** `observe_table | open(k) | close(k) | pick(color) | place(k)`.
- **Observation facts:** `holding`, `drawer_open:<k>`, `contents:<k>`, `on_table`.
- **A fact you don't measure, you leave out:** it can't cause a false surprise.
- **Chapter 2 adds:**
  - the ball as an object with locations `in_air | table | cup | drawer:<k>`;
  - actions `hold_open(k)`, `drop`, `toss`, `roll`;
  - facts `ball_at` and `landing_xy`.

### Organizer and blocks

- **The organizer:** 3 opaque drawers with rounded red knobs. The smooth slides have no stop, so a pulled-out drawer detaches.
- **Pulls:** each drawer goes to 75% of its maximum travel, never further.
- **Red only ever means a knob.** The sorting blocks are green, yellow and black by default; blue is the tape color Waj's scene parser uses for its work squares.
- **The ball:** black rubber. Find it in color (dark on white); depth on black rubber is unreliable.
