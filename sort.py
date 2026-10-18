"""MOVES THE ARM. Sort the blocks on the table into the organizer's drawers, one color per drawer.

Every step starts with one look from the overview pose. From it the robot works out where the blocks
are, which drawers are shut, partly open or open, and what's inside the open ones; it remembers what's
in the shut ones (worldarm/task/belief.py). Then it follows the rules (worldarm/task/policy.py): one
drawer open at a time; a mixed drawer keeps the color it has most of and the rest come out; an open
drawer takes its color's blocks; a new color goes into an empty drawer; when nothing is left to do,
it stops.

  python sort.py --dry-run          one look and the first decision, no motion
  python sort.py                    the whole task (asks for the E-stop first)
  python sort.py --drawers service  open/close with the deployed drawer service instead

Drawer 3 stays free: the colors (up to two) go into drawers 1 and 2, and drawer 3 is opened at the end.
Then the arm goes back to y = 0 and waits for the other arm's ball; type 'close' and it comes back,
closes drawer 3 and goes back to y = 0 (--wait nudge: a nudge on the arm instead; --wait none leaves it open).

Each step is logged in runs/<stamp>-sort/: step-NN.jpg (what it saw and decided), step-NN.npz (the raw
look, replayable with `drill.py scan --frame`), episode.jsonl, summary.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

from worldarm import connection
from worldarm.act.celebrate import hurray
from worldarm.act.drawer import ClientDrawers, ServiceDrawers
from worldarm.cell.config import CellConfig
from worldarm.cell.geometry import to_pose
from worldarm.act.nudge import wait_for_nudge
from worldarm.cell.rig import Rig
from worldarm.see.scene import observe
from worldarm.logger import EpisodeLogger
from worldarm.task.runner import overview, run_task


GO_BACK_X_MM = (300.0, 350.0, 250.0, 400.0)       # out from the base: closer folds the forearm into wall-back
GO_BACK_THETAS = (None, 0.0, 180.0, 90.0, -90.0)   # None: the overview's wrist angle


async def go_back(rig, y_mm: float) -> bool:
    """End of the run: back to y = y_mm, gripper straight down, at least as high as the overview, with a
    planned move around the workcell obstacles. Tries a few reaches and wrist angles (near the base the arm
    folds into the wall-back obstacle and the claws into the upper arm); if Viam refuses them all, it says
    so and stays where it is instead of failing the run."""
    vp = rig.cfg.view_pose
    if vp is None:
        return False
    z = max(float(vp["z"]), 350.0)
    await rig.set_speed(rig.cfg.speeds.view_deg_s)
    errors = []
    for x in GO_BACK_X_MM:
        for th in GO_BACK_THETAS:
            pose = {"x": x, "y": float(y_mm), "z": z, "o_x": 0.0, "o_y": 0.0, "o_z": -1.0,
                    "theta": float(vp["theta"]) if th is None else th}
            try:
                await rig.move(to_pose(pose), label=f"back to y={y_mm:.0f}")
            except Exception as e:  # unreachable or in collision: try the next one
                errors.append(f"x={x:.0f} theta={pose['theta']:.0f}: {str(e)[:90]}")
                continue
            print(f"back at y = {y_mm:.0f} mm (x = {x:.0f}, z = {z:.0f}, gripper straight down)")
            return True
    print(f"couldn't go back to y = {y_mm:.0f} mm; staying here. Viam refused: " + " | ".join(errors[:3]))
    return False


def _left_open(result: dict) -> int | None:
    """The drawer a sorted run left open for the ball (drawer 3), if any."""
    for k, d in (result.get("final_belief") or {}).get("drawers", {}).items():
        if d.get("state") == "open":
            return int(k)
    return None


async def ball_and_close(rig, drawers, k: int, args) -> dict:
    """Drawer k is open for the other arm's ball. Wait at the original position until you type 'close'
    (or, with --wait nudge, until someone nudges the arm), then look again, close drawer k, and go back."""
    print(f"drawer {k} is open for the ball: the other arm can throw now")
    if args.wait == "nudge":
        nudge = await wait_for_nudge(rig, scale=args.nudge_scale)
        if not nudge.get("nudged"):
            print(f"no nudge ({nudge.get('why')}): leaving drawer {k} open")
            return {"nudge": nudge, "closed": False}
        print(f"nudged (joint {nudge['joint']}, {nudge['ratio']}x the limit): closing drawer {k}")
    else:
        nudge = {"command": True}
        while True:
            typed = (await asyncio.to_thread(input, f"type 'close' to close drawer {k} (or 'leave' to finish with it open): ")).strip().lower()
            if typed == "close":
                break
            if typed == "leave":
                print(f"leaving drawer {k} open")
                return {"nudge": nudge, "closed": False}
        print(f"closing drawer {k}")
    scene = observe(await overview(rig), rig.cfg)
    if scene.organizer is None:
        print("can't see the organizer to close the drawer; leaving it open")
        return {"nudge": nudge, "closed": False, "why": "organizer not seen"}
    r = await drawers.close(scene.organizer, k)
    print(f"drawer {k}: {'closed' if r.ok else 'NOT closed: ' + str(r.to_json().get('status'))}")
    await go_back(rig, args.end_y)
    return {"nudge": nudge, "closed": bool(r.ok), "drawer": r.to_json()}


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--machine", default="armfarm03")
    p.add_argument("--drawers", choices=["client", "service"], default="client")
    p.add_argument("--dry-run", action="store_true", help="look and decide once; no motion except to the overview")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--yes", action="store_true")
    p.add_argument("--end-y", type=float, default=0.0,
                   help="at the end, go back to this y (mm), gripper straight down")
    p.add_argument("--no-park", action="store_true", help="stay where the run ends instead of going back")
    p.add_argument("--wait", choices=["command", "nudge", "none"], default="command",
                   help="after opening drawer 3: wait for you to type 'close' (default), for a nudge on the arm, "
                        "or not at all (drawer 3 stays open)")
    p.add_argument("--nudge-scale", type=float, default=1.0,
                   help="how hard a nudge must be (x the contact limits; smaller = more sensitive)")
    args = p.parse_args()

    cfg = CellConfig.load(args.machine)
    address = connection.env("VIAM_MACHINE_ADDRESS").split(".")[0]
    if cfg.machine.replace("0", "") not in address.replace("0", ""):
        raise SystemExit(f"config is for {cfg.machine} but the env file points at {address}")
    if cfg.view_pose is None:
        raise SystemExit("No overview pose yet: put the arm there and run `python drill.py view --teach`.")
    if not cfg.camera.correction:
        print("Note: no camera correction saved yet (python drill.py calibrate --save); positions may be off.")
    if not args.yes and input("This MOVES the arm. Someone at the E-stop? Type 'go': ").strip() != "go":
        raise SystemExit("Cancelled.")

    logger = EpisodeLogger("runs", "sort", {"mode": "real", "goal": "sort", "blind": False, "machine": cfg.machine,
                                            "drawers": args.drawers, "args": vars(args), "config": cfg.path.name})
    rig = await Rig.connect(cfg)
    drawers = ClientDrawers(rig) if args.drawers == "client" else ServiceDrawers(rig.machine)
    t0 = time.time()
    try:
        result = await run_task(rig, drawers, logger, max_steps=args.max_steps, dry_run=args.dry_run)
        if result.get("success") and not args.dry_run:
            await hurray(rig)  # sorted: celebrate
        # only after a run that ended normally: never move again after an error, a collision or a stop
        if not args.dry_run and not args.no_park and result.get("status") not in ("aborted", "no progress"):
            await go_back(rig, args.end_y)
            if result.get("success") and _left_open(result) and args.wait != "none":
                result["ball"] = await ball_and_close(rig, drawers, _left_open(result), args)
    except BaseException as e:
        await rig.stop()
        result = {"status": "aborted", "success": False, "reason": f"{type(e).__name__}: {e}"}
        raise
    finally:
        logger.finish({"status": result.get("status"), "reason": result.get("reason"), "steps": result.get("steps"),
                       "surprises": result.get("surprises"), "duration_s": round(time.time() - t0, 1),
                       "belief_success": result.get("status") == "done", "truth_success": result.get("success"),
                       "belief_accuracy": None, "final_belief": result.get("final_belief")})
        await rig.close()
    print(f"\n{'SORTED' if result.get('success') else 'NOT SORTED'}: {result.get('reason')}  "
          f"({result.get('steps')} steps, {result.get('seconds', 0):.0f} s, {result.get('surprises', 0)} surprises)")
    print(f"log: {logger.dir}")
    print(json.dumps(result.get("final_belief", {}).get("drawers", {}), indent=1))


if __name__ == "__main__":
    asyncio.run(main())
