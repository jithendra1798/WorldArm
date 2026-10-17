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
from worldarm.cell.rig import Rig
from worldarm.logger import EpisodeLogger
from worldarm.task.runner import run_task


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--machine", default="armfarm03")
    p.add_argument("--drawers", choices=["client", "service"], default="client")
    p.add_argument("--dry-run", action="store_true", help="look and decide once; no motion except to the overview")
    p.add_argument("--max-steps", type=int)
    p.add_argument("--yes", action="store_true")
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
