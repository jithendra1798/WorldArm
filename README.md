# WorldArm

**A robot arm that remembers what it can't see.**

Our entry for Viam's *Code That Moves* hackathon (NYC, Oct 16–17 2026).
The arm sorts three colored blocks into a 3-drawer organizer, one color per drawer: it pulls each drawer open, places the block, and closes it.
Once a drawer is closed, its contents are hidden, so the robot has to keep a **world model**: a belief about where every block is, including the ones it can't see. It predicts the effect of each action and checks the prediction against what the camera and gripper report.

The plan, roles and how we'll measure results are in **[PLAN.md](PLAN.md)**.

## The demo (three moments)

1. **Sort.** Three blocks sit anywhere on the table. The robot puts each into its color's drawer and closes the drawer.
2. **Object permanence.** A judge asks "where's the blue block?" The robot opens the right drawer and shows it.
3. **Surprise.** A judge secretly moves a block to a different drawer. When the robot opens it, what it sees doesn't match its prediction. It flags the mismatch, updates its belief, and recovers.

## How it works

```
 wrist RGB-D ──► perception ──► observations ──┐
                                               ▼
                       ┌──────── world model (belief state) ────────┐
                       │ blocks: where each one is (table / gripper /│
                       │         drawer k), plus last-seen pose      │
                       │ drawers: open or closed, what's inside      │
                       └──────┬───────────────────────────▲──────────┘
                              │ plan next action          │ compare predicted vs observed
                              ▼                           │ (mismatch = "surprise" → replan)
                     planner / sequencer ──► skills ──► verify
                                         pick · place · open · close
```

- **Perception:** find blocks by their height above the fitted table plane, classify color, and estimate position and angle in the world frame. Locate drawer handles from about 30 cm away (depth fails closer than that).
- **World model:** an explicit belief state. Each action has a *predicted effect* (for example, after `place(yellow, drawer 1)` the yellow block is in drawer 1). After every action we observe and compare. Any mismatch is logged and triggers a replan.
- **Skills:** pick, place, open drawer, close drawer. All motion goes through Viam's motion service, plus our own safety limits (see below).

## Hardware (per arm)

| | |
|---|---|
| Arm | UFactory xArm (6 joints), Viam component `arm` |
| Gripper | `viam:ufactory:gripper`, component `gripper`; supports `IsHoldingSomething` |
| Camera | Intel RealSense D435 on the wrist, component `cam`; aligned 1280×720 color and depth |
| Obstacles | `table`, `wall-front`, `wall-side`, `ceiling` (`erh:vmodutils:obstacle`; they show up as grippers in the SDK) |
| Motion | built-in motion service `builtin` |

## Safety rules (read before moving an arm)

- **Someone watches the arm with a hand near the E-stop** whenever code moves it.
- **Use the motion service, not direct moves.** `MoveToPosition` and `MoveToJointPositions` go *straight through* obstacles. Only `motion.move` plans around them.
- **Viam's table obstacle is wrong on armfarm03.** Its top is at world z = −23 mm, but the camera measures the real surface at **z ≈ +2 mm**. The planner would let the claws go about 25 mm into the table. Our code has to enforce its own minimum height. Measure each arm with `verify_camera.py`.
- **The claws reach about 50 mm past the `gripper` frame** (configured claw geometry; confirm with a ruler). The gripper frame is 105 mm from the arm flange. Fingertip height ≈ gripper-frame z − 50 mm.
- **`wall-front` occupies x = 690–790 mm.** Keep targets well short of it.
- **Test every new motion high first:** about 100 mm above the target, then lower it.
- The RealSense **can't measure depth closer than about 26 cm**. Perceive from a viewing pose, then move in without the camera.

## Setup

Python 3.13. From the repo root:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # fill in VIAM_API_KEY and VIAM_API_KEY_ID
python check_connection.py  # read-only smoke test
```

API keys are in the Viam app: **hackathons org → Arm Farm → your machine → CONNECT → API keys**. Never commit a filled-in env file (`.gitignore` already excludes them).

**Second arm:** `cp .env.example .env.arm2`, fill in that machine's keys and address, then prefix any command with `WORLDARM_ENV=.env.arm2`, for example `WORLDARM_ENV=.env.arm2 python verify_camera.py`.

## Scripts

| Script | What it does |
|---|---|
| `teach.py` | **Moves the gripper only, never the arm.** Use it in manual mode: open and grab the gripper, save named poses to `poses/<machine>.json`, save captures |
| `check_connection.py` | Read-only. Lists resources and reads the arm pose, a camera frame and the gripper state |
| `inspect_scene.py` | Read-only. Prints camera intrinsics and the frame system; saves color and depth images to `captures/` |
| `verify_camera.py` | Read-only. Calibration check: our pixel→world math vs. Viam's `transform_pose`, a table-plane fit (height, tilt, flatness) and the obstacle geometries |

Library modules:

| Module | What's in it |
|---|---|
| `connection.py` | `connect()` and `env()`; loads `.env` or the file named by `$WORLDARM_ENV` |
| `perception.py` | Decodes Viam color and depth images; `capture()`, `deproject()`, `pixel_to_cam()` |
| `frames.py` | `frame_to_world()`: camera→world rotation and translation built from Viam's own frame system; `to_world()` |

## Calibration measured on armfarm03 (2026-10-16)

Re-run `verify_camera.py` on each arm. These numbers are for **armfarm03** only.

| Check | Result |
|---|---|
| Intrinsics @1280×720 | fx 908.2, fy 907.8, cx 643.7, cy 368.9; distortion zero |
| `cam` frame (parent `arm`) | translation (−83, 14, 18) mm, θ −97.7° |
| Our transform vs. Viam `transform_pose` | 0.000 mm max error |
| Table plane | z = +1.9 mm, tilt 0.16°, residual σ 2.2 mm (34k points) |
| Visible table from the start pose (camera about 380 mm up) | x 317–731, y −535–12 mm |

## Team workflow

- `main` must always run. Work on branches named `<name>/<feature>`, keep commits small, and `git pull --rebase` before pushing.
- Keep secrets, `captures/` and `runs/` out of git. Pin any new dependency in `requirements.txt`.
- Any script that moves an arm says so in its first docstring line, in caps.
