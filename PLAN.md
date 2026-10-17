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

## Stage 1: chapter 1 on armfarm03, piece by piece

This is the guaranteed demo, so it comes before chapter 2. Each piece is its own module with its own real
test (`drill.py`), so a failing piece is tested and fixed alone; `sort.py` runs them together.

**What the robot does** (`sort.py`, rules in `worldarm/task/policy.py`). Every step starts with one look from
the overview pose: where the blocks are, where the organizer is, each drawer shut, partly open or open,
and what's inside the open ones (shut drawers' contents are remembered). Then:
- one drawer open at a time; a partly open drawer is pulled out to 12 cm first;
- an open drawer with mixed colors keeps the color it has most of; the others are taken out onto the table;
- an open drawer with one color takes that color's table blocks; none left, it's closed;
- an empty open drawer takes the color with the most table blocks that has no drawer yet;
- nothing open: open the mixed drawer, else the drawer holding a color still on the table, else a drawer not
  yet looked into, else an empty one for a new color;
- nothing left to organize (every drawer seen and single-colored, table clear): stop.
Grips are planned from each block's measured size and height (across its short side, pads on its middle,
fingertips never on the table), every descent stops on contact, and every grip is checked by finger width.

**Set up (physical):**
- **Tape the drawer box to the table.** Closing pushes on it; on Friday it slid 11 cm.
- The overview must show all three knobs and the blocks; keep blocks 16 cm from the knobs.
- Measure with a ruler and put in `config/armfarm03.json`: finger thickness (`finger_depth_mm`, along the
  closing direction) and width (`finger_width_mm`), the knob's centre in front of the drawer face
  (`knob_protrusion_mm`). Defaults are guesses (12, 18, 20 mm).

**Morning run, in this order** (each logs to `runs/`, `python scoreboard.py` shows pass rates):

| # | Command | Tests | Pass when |
|---|---|---|---|
| 1 | `python drill.py check` | camera, depth, arm, gripper, joint loads (no motion) | PASS, load noise low |
| 2 | `python drill.py view --teach` (only if `scan` shows a problem) | overview pose: tilt, organizer and blocks in view | saved |
| 3 | `python drill.py scan --times 3` | perception: blocks, organizer, drawer states, contents | every block and all 3 knobs found; repeatability < 3 mm |
| 4 | `python drill.py calibrate --save` | camera offset (wrist turned 180°, two heights) | rms after < 4 mm |
| 5 | `python drill.py touch --save` | fingertip length on bare table | 40-80 mm, saved |
| 6 | `python drill.py pick --times 5` | grasp planning + pick (puts each block back) | 5/5 held, firm |
| 7 | `python drill.py drawer 2 --cycle --times 3` (each drawer) | open to 12 cm, close without shoving the box | ok, box moved < 15 mm |
| 8 | `python drill.py drop 1` | a table block into an open drawer | seen inside |
| 9 | `python drill.py unload 1` | a block out of a drawer onto the table | on the table |
| 10 | `python drill.py policy` | the decision from one look (no motion) | sensible |
| 11 | `python sort.py --dry-run`, then `python sort.py` | the whole task | SORTED |

`python drill.py scan --frame runs/<dir>/<file>.npz` re-runs perception on any saved look with no robot.
`sort.py --drawers service` uses the deployed drawer service (PickBottle) instead of our drawer skill.
`sort_bench.py` (the previous pipeline) still works and is left as it was.

**Gate:** steps 1-10 pass, then ≥5 full `sort.py` runs on the scoreboard, plus the backup video. Then tag
`v1-sorting`; chapter 2 work happens on a branch.

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
| **C. Toss into a placed bowl** | armfarm05 throws: one synchronized swing, the ball held barely and released at 45° while the swing is at full speed (the acceleration is tuned so the release falls in the constant-speed part). First ~0.8 m into a large cloth-lined bowl, slow swings, until the landing is repeatable. Then arm 1 holds the bowl at the predicted landing spot. | Physics, no learning: the thrower's settings (joint speed, base angle) give the landing point, through three calibrated constants (speed factor, release delay, the transform between the arms). Change the thrower and the catcher moves itself. | Landing spread at one setting (repeatability); landing error vs. prediction; catch rate |
| D. Stretch | Correcting during flight; tosses across the table | Only if staff allow higher speeds and the measured delays allow it | — |

**Rung B, piece by piece** (`ball.py` on armfarm03; settings in `config/ball_armfarm03.json`, logs in `runs/`).
A paddle in armfarm03's gripper defends a straight lane low over the table. The overhead camera tracks the
ball, the rolling model predicts where and when it will cross the lane, and the paddle goes there. The
camera also judges each roll: blocked, goal, or short. Commands that move ask "type go"; Ctrl-C stops the arm.

| # | Command | Tests | Pass when |
|---|---|---|---|
| 1 | `python ball.py check` | overhead camera speed, ball found on the table, table outline (no motion) | ~20+ fps, ball found in nearly every frame |
| 2 | `python ball.py lane --teach a`, then `--teach b` | the lane's two ends: the arm's pose at that moment, paddle in the gripper, low over the table (no motion) | both saved |
| 3 | `python ball.py calibrate` | camera-to-table map: the arm sets the ball down on a grid | leave-one-out error < 5 mm |
| 4 | `python ball.py lane-test` (ball off the table) | paddle speed, start and retarget delays, how straight it stays on the lane, lookalikes at both ends | saved, no lookalikes |
| 5 | `python ball.py record --rolls 15` | people roll the ball across the field, not at the paddle; tracks saved (no motion) | 15 rolls |
| 6 | `python ball.py fit` | the learned slowdown, and H4 on the recorded rolls (each predicted with the slowdown learned from the others) | learned arrival time clearly closer than constant velocity |
| 7 | `python ball.py sim`, then `--realistic` | offline: rates expected with the measured lane, arm and slowdown | predict well above chase |
| 8 | `python ball.py goalie --rolls 10 --mode predict --confirm`, then `--mode chase`, alternating to 20+ each | the demo, and the H5 baseline; `--confirm` lets the operator correct the camera's call | — |
| 9 | `python ball.py report` | H5 with 95% intervals, the camera's agreement with the operator, H4 on real rolls | predict > chase |

`--speed` is deg/s along the lane. The default, 30, is above the 15°/s cap in Rules: get the staff OK or pass
`--speed 15`. Alternating predict and chase in blocks of 10 keeps the comparison fair if the rolls drift.
The simulation behind the H4 and H5 slides runs with no robot: `caffeinate -i .venv/bin/python ball_study.py`
(~50 min, one process), then the charts: `/opt/homebrew/bin/python3.14 ball_study.py --report-only runs/<stamp>-ball-study`.

**Rung C, piece by piece** (`toss.py`; settings in `config/toss.json`, logs in `runs/`). armfarm05 throws, the
overhead camera on armfarm03 finds where the ball came to rest, then armfarm03 holds the bowl at the predicted
landing. Physics, no learning: the arm's own kinematics and the ball's flight predict the landing; three
calibrated constants (speed factor, release delay, where the arms stand) make it match.

**Set up (physical):**
- Tape-measure where armfarm05's base is in armfarm03's frame (x ahead of armfarm03's base, y to its left) and
  which way armfarm05 faces: `thrower_in_catcher` in `config/toss.json`. `calibrate` refines it.
- Put any wall, rail or screen near armfarm05 in `walls` (boxes in armfarm05's frame, mm) and set
  `ceiling_z_mm`. Every move is checked against them and the table along its whole path; direct joint moves
  ignore Viam's obstacles.
- A cloth where the ball lands, so it stops there, inside the overhead camera's view (or `--manual`: a tape
  measure from the target mark). Nobody in the throw path; someone at armfarm05's E-stop.

| # | Command | Tests | Pass when |
|---|---|---|---|
| 1 | `python toss.py probe` | joints, the arm's own kinematics (saved), our model vs. the arm, which settings it takes (no motion) | model within 2 mm and 1° |
| 2 | `python toss.py plan --design --distance 500` | offline: searches a wind-up pose and swing; joint speed, release point, margins, how forgiving | a plan |
| 3 | `python toss.py windup --distance 500` | the slow move to the wind-up pose | the fingers close across the swing (left-right); if not, turn the last joint 90° in `start_deg` and re-plan |
| 4 | `python toss.py dry --scale 0.5 --distance 500`, then `--repeat 3` | the swing with no ball: real start delay, speed, acceleration, jitter | speed within 1%, acceleration as set |
| 5 | `python toss.py grip --teach`, then `grip --repeat 10` | the opening that just holds the ball; how long "open" takes | holds; delay steady |
| 6 | `python toss.py throw --repeat 5 --distance 500` | grip, swing, let go; the camera finds where it landed | the landings bunch within a few cm (the offset from the plan is what step 8 fixes) |
| 7 | `python toss.py report` | per setting: spread, 95% radius, offset from the plan, release timing | spread under the bowl's radius |
| 8 | throws at a second distance and base angle, then `python toss.py calibrate` | speed factor, release delay, where the arms stand | residuals about the spread |
| 9 | `python toss.py catch --teach`, `catch --dry`, then `catch` | armfarm03 puts the bowl where the throw will land; the throw | caught |

The release isn't fixed at 45°: the planner picks the angle (25-60°) where a late release lands in the same
place (released later, the ball leaves further along the swing but heading differently, and the two cancel),
and prints how far the landing moves for 10 ms late, 1% faster, 1° higher. `trigger` in the config is
`predict` (from joint readings during the swing) or `timed` (from the start delay `dry` measured); `dry`
prints which is steadier. If the release timing still varies more than ~15 ms (`report`), the fix is a trigger
on the robot itself (Viam Control Code, as `catch-thrown-object-module/` was deployed); not built. Speeds:
0.8 m needs roughly 110-130°/s, above the 15°/s cap in Rules: get the staff OK.

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

| Package | Code | Its test | Owner |
|---|---|---|---|
| Find and pick up any block | `worldarm/blocks.py`, `act/grasp_plan.py`, `act/pick.py` | `drill.py scan`, `pick` | **Waj** |
| Open and close the drawers (75%, never further) | `see/organizer.py`, `act/drawer.py` | `drill.py drawer` | **Sandeep** |
| Sort the blocks into the drawers by color (dropping, taking out) | `see/contents.py`, `act/place.py` | `drill.py drop`, `unload` | **Riyaj** |
| World model, rules, integration, calibration, demo | `task/`, `cell/`, `drills.py`, `sort.py` | `drill.py policy`, `calibrate`, `touch`; `sort.py` | **Jithendra** |

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
