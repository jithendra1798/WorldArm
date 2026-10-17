# WorldArm

**Robot arms that model their world: they remember what they can't see, and predict where things will be.**

Our entry for Viam's *Code That Moves* hackathon (NYC, Oct 16–17 2026).

- **Chapter 1, memory:** an arm sorts colored blocks into a 3-drawer organizer, one color per drawer.
  - Closed drawers hide their contents, so the robot keeps a belief about where everything is.
  - It predicts what each action will do, and flags a surprise when what it sees disagrees.
- **Chapter 2, physics:** two arms face each other across one table, under an overhead depth camera, and pass a black rubber ball.
  - The ball goes into a drawer one arm holds open, or into a cup placed where a rolling ball will be.
  - The same world model now predicts motion, with the ball's physics learned from the robot's own trials.

**The plan, who does what, and how we measure results:** [PLAN.md](PLAN.md). **Machine configuration:** [docs/viam-config.md](docs/viam-config.md).

## How it works

```
 wrist cameras ──┐                                          ┌─► arm 1 (armfarm03): pick, drawers, hold a drawer open, catch
                 ├─► observations ─► world model ─► planner ┤
 overhead camera ┘    (blocks,       (belief +              └─► arm 2 (armfarm05): pick, drop, toss, roll
                       knobs, ball)   predictions)
                                          ▲
                                          └── compare predicted vs. observed: a mismatch is a surprise → replan
```

- **Perception:**
  - The wrist cameras see blocks, knobs and what's inside a drawer.
  - The overhead camera tracks the ball and judges every outcome.
  - Cameras are combined in the world model, not at the pixel level.
- **World model:** a belief about where every object is, a predicted effect for every action, and a surprise when a measurement disagrees. In chapter 2 the ball is one more object, and predictions include where it will land or stop.
- **Skills:** the team's Viam modules, controlled through their service commands (DoCommand). Our layer plans, checks and logs.

## Hardware

| | |
|---|---|
| **Arm 1: armfarm03** | UFactory xArm, two-finger gripper, wrist RealSense D435 (`cam`), the drawer organizer |
| **Arm 2: armfarm05** | UFactory xArm850, two-finger gripper, wrist RealSense (`cam`) |
| **Overhead camera** | `overview-cam` on armfarm03, an Orbbec Astra 2 (`viam:orbbec:astra2`). Color 1280×720, fx 834 px; the table is ≈ 1.31 m below it. For 3D, use its point cloud: the depth image comes back rounded to whole meters. |
| **Obstacles** | `table`, `wall-front`, `wall-side`, `ceiling` per machine (`erh:vmodutils:obstacle`) |

## Safety rules (read before moving an arm)

- **Someone at every arm that can move, with the E-stop in reach.** Two arms means two people.
- **Two arms share one table.** Each arm stays in its own half; the middle is for the ball only. Nobody stands in a toss path.
- **Speed caps:** 15°/s for the arm and 0.9 m/s for the ball, from the catcher module. Raise them only with Viam staff's OK.
- **Use the motion service for computed moves.** `MoveToPosition` and `MoveToJointPositions` go straight through obstacles.
- **Viam's table obstacle on armfarm03 sat 25 mm below the real surface** when measured Friday morning. Re-measure with `verify_camera.py` after an arm is moved.
- **The fingertips are about 60 mm past the `gripper` frame,** which is 105 mm from the arm flange.
- **Drawers go to 75% of their travel, never further:** a pulled-out drawer detaches.
- **Test every new motion slowly and high first.** The wrist RealSense can't measure depth closer than about 26 cm.

## Repository

| Path | What it is |
|---|---|
| `worldarm/`, `run_episode.py`, `try_skill.py`, `scoreboard.py` | World model, planner, episode loop, simulator, logging and scoreboard. `run_episode.py --fake` runs without an arm. |
| `sort.py` | **Moves the arm.** Chapter 1 on armfarm03: sorts the blocks into the drawers, one color per drawer, from whatever state the drawers are in (shut, partly open, open, mixed). `--dry-run` only looks and decides. |
| `drill.py` | One real test per piece of chapter 1: `check`, `view`, `scan`, `calibrate`, `touch`, `pick`, `drawer`, `drop`, `unload`, `policy` (PLAN.md, Stage 1). |
| `worldarm/cell/` | One arm's numbers (`config.py` + `config/armfarm03.json`), its hardware with safety checks (`rig.py`), the camera correction (`calibration.py`) |
| `worldarm/see/` | Perception, no motion: table blocks, the organizer (drawer numbers, how far each is out), what's in open drawers (`scene.py`) |
| `worldarm/act/` | Skills: grasp planning (`grasp_plan.py`), contact sensing (`watch.py`), `pick.py`, `place.py`, `drawer.py` |
| `worldarm/task/` | The world model (`belief.py`), the sorting rules (`policy.py`), the loop (`runner.py`) |
| `sort_bench.py` | **Moves the arm.** The previous chapter 1 benchmark on armfarm03 (kept as it was). Blocks are found anywhere on the table (no tape). Per block: drawer `open` → pick across the short side with a contact-checked descent → drop into the drawer → look in → `close`. Success is judged by the cameras; logged for the scoreboard. |
| `bench_skill.py` | **Moves the arm.** Benchmarks one skill as deployed: runs its service command N times, a person judges each try, and results go on the scoreboard |
| `modules/drawer_service.py` | Drawer service: finds the knobs from above, opens and closes a drawer |
| `pick_place_cubes.py`, `moving-blocks-module/`, `run_moving_blocks_viam.py` | Pick and place between two tape zones, several colors, with recovery |
| `pick-red-cube-module/`, `run_pick_module.py` | Pick service (commands `preview`, `pick`) |
| `catch-thrown-object-module/`, `run_catch_module.py`, `throw_catch_cube.py` | Soft catch service and one-arm self-toss, with hard speed caps |
| `workcell_obstacles.py` | Table and wall obstacles for motion planning |
| `teach.py` | Manual-mode helper: gripper open and grab, pose readout, captures. **Moves the gripper only.** |
| `check_connection.py`, `inspect_scene.py`, `verify_camera.py` | Read-only checks: connection, camera and frames, calibration |

The top-level `connection.py`, `perception.py` and `frames.py` are shims that forward to `worldarm/`; the team's scripts import them.

## Setup

Python 3.13. From the repo root:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest               # unit and simulation tests, no arm needed
python run_episode.py --fake   # a full simulated episode
cp .env.example .env           # armfarm03: fill in VIAM_API_KEY and VIAM_API_KEY_ID
python check_connection.py     # read-only smoke test
```

API keys are in the Viam app: **hackathons org → Arm Farm → the machine → CONNECT → API keys**. Never commit a filled-in env file.

**armfarm05:** `cp .env.example .env.arm2`, fill in its keys and address, then prefix commands with `WORLDARM_ENV=.env.arm2`.

## Calibration measured on armfarm03 (Friday morning)

Re-run `verify_camera.py` after moving an arm.

| Check | Result |
|---|---|
| Wrist camera intrinsics @1280×720 | fx 908.2, fy 907.8, cx 643.7, cy 368.9; no distortion |
| `cam` frame (parent `arm`) | translation (−83, 14, 18) mm, θ −97.7° |
| Our transform vs. Viam `transform_pose` | 0.000 mm max error |
| Table plane | z = +1.9 mm, tilt 0.16°, residual σ 2.2 mm |

## Team workflow

- `main` must always run. Work on `<name>/<feature>` branches and `git pull --rebase` before pushing.
- Keep secrets, `captures/`, `runs/` and `state/` out of git.
- Any script that moves an arm says so in the first line of its docstring.
