"""Isolated real tests, one per piece of the sorting task (worldarm/drills.py). The first time, run them
in this order; after that, rerun only the piece that's failing.

  python drill.py check                          no motion: camera, depth, arm, gripper, loads, settings
  python drill.py view --teach [--force]         no motion: save where the arm is now as the overview pose
  python drill.py view                           go to the overview pose and look
  python drill.py scan [--here] [--times N]      look; report blocks, organizer, drawer states and contents
  python drill.py scan --frame runs/X/scan-1.npz offline: re-run perception on a saved look (no robot)
  python drill.py calibrate [--save]             fix the camera offset (wrist turned 180 deg, two heights)
  python drill.py touch [--save]                 measure the fingertip length on bare table
  python drill.py pick [--color C] [--times N]   pick a block, hold it, put it back where it was
  python drill.py drawer K --open|--close|--cycle [--times N] [--service]
  python drill.py drop K [--color C]             one table block into drawer K (opens it if needed)
  python drill.py unload K [--color C]           one block out of open drawer K onto the table
  python drill.py policy [--frame F.npz]         no motion: what the robot would do next, and why

Everything logs to runs/<stamp>-drill-<name>/ (annotated pictures, raw frames, summary.json);
`python scoreboard.py` shows each drill's pass rate over all runs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from worldarm import connection, drills
from worldarm.act.drawer import ClientDrawers, ServiceDrawers
from worldarm.cell.config import CellConfig
from worldarm.logger import EpisodeLogger
from worldarm.see.frame import Frame
from worldarm.see.scene import observe

MOVES = {"view", "scan", "calibrate", "touch", "pick", "drawer", "drop", "unload"}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--machine", default="armfarm03")
    p.add_argument("--yes", action="store_true", help="skip the 'someone at the E-stop?' question")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    v = sub.add_parser("view")
    v.add_argument("--teach", action="store_true")
    v.add_argument("--force", action="store_true", help="save even if the view has problems")
    s = sub.add_parser("scan")
    s.add_argument("--here", action="store_true", help="look from where the arm is now (no motion)")
    s.add_argument("--times", type=int, default=1)
    s.add_argument("--frame", help="a saved .npz look to replay offline")
    c = sub.add_parser("calibrate")
    c.add_argument("--save", action="store_true")
    c.add_argument("--drop-mm", type=float, default=120.0)
    t = sub.add_parser("touch")
    t.add_argument("--save", action="store_true")
    k = sub.add_parser("pick")
    k.add_argument("--color")
    k.add_argument("--times", type=int, default=1)
    k.add_argument("--hold-s", type=float, default=2.0)
    d = sub.add_parser("drawer")
    d.add_argument("k", type=int)
    m = d.add_mutually_exclusive_group(required=True)
    m.add_argument("--open", action="store_true")
    m.add_argument("--close", action="store_true")
    m.add_argument("--cycle", action="store_true")
    d.add_argument("--times", type=int, default=1)
    d.add_argument("--service", action="store_true", help="use the deployed drawer service instead")
    for name in ("drop", "unload"):
        x = sub.add_parser(name)
        x.add_argument("k", type=int)
        x.add_argument("--color")
        x.add_argument("--service", action="store_true")
    pol = sub.add_parser("policy")
    pol.add_argument("--frame")
    return p


def print_result(res: dict) -> None:
    for line in res.get("lines", []):
        print(line)
    for key in ("next", "rate", "report", "tip_below_frame_mm", "repeatability_mm", "saved", "problems"):
        if key in res and res[key]:
            print(f"  {key}: {json.dumps(res[key]) if not isinstance(res[key], str) else res[key]}")
    for t in res.get("trials", []):
        print("   ", {k: v for k, v in t.items() if k in ("op", "status", "color", "opening_before", "opening_after",
                                                         "box_moved_mm", "firm")})
    print("PASS" if res.get("pass") else "FAIL")


async def offline(args, cfg: CellConfig) -> None:
    frame = Frame.load(args.frame)
    if args.cmd == "policy":
        print_result(await drills.policy_dry(frame=frame, cfg=cfg))
        return
    scene = observe(frame, cfg)
    print_result({"lines": drills.describe(scene, cfg), "pass": scene.organizer is not None})


async def main() -> None:
    args = parser().parse_args()
    cfg = CellConfig.load(args.machine)
    if getattr(args, "frame", None):
        await offline(args, cfg)
        return
    address = connection.env("VIAM_MACHINE_ADDRESS").split(".")[0]
    if cfg.machine.replace("0", "") not in address.replace("0", ""):
        raise SystemExit(f"config is for {cfg.machine} but the env file points at {address}")
    moves = args.cmd in MOVES and not (args.cmd == "scan" and args.here) and not (args.cmd == "view" and args.teach)
    if moves and not args.yes and input("This MOVES the arm. Someone at the E-stop? Type 'go': ").strip() != "go":
        raise SystemExit("Cancelled.")
    name = args.cmd + (f"-{args.k}" if hasattr(args, "k") else "")
    logger = EpisodeLogger("runs", f"drill-{name}", {"mode": "real", "goal": f"drill:{args.cmd}", "blind": False,
                                                     "machine": cfg.machine, "args": vars(args)})
    out = Path(logger.dir)
    from worldarm.cell.rig import Rig  # needs the robot SDK connection; offline runs don't
    rig = await Rig.connect(cfg)
    t0 = time.time()
    try:
        backend = (ServiceDrawers(rig.machine) if getattr(args, "service", False) else ClientDrawers(rig))
        if args.cmd == "check":
            res = await drills.check(rig, out)
        elif args.cmd == "view":
            if args.teach:
                res = await drills.teach_view(rig, args.force, out)
            else:
                res = await drills.scan(rig, out=out)
        elif args.cmd == "scan":
            res = await drills.scan(rig, args.here, args.times, out)
        elif args.cmd == "calibrate":
            res = await drills.calibrate(rig, args.drop_mm, args.save, out)
        elif args.cmd == "touch":
            res = await drills.touch(rig, args.save, out)
        elif args.cmd == "pick":
            res = await drills.pick_test(rig, args.color, args.times, args.hold_s, out)
        elif args.cmd == "drawer":
            mode = "open" if args.open else "close" if args.close else "cycle"
            res = await drills.drawer_test(rig, backend, args.k, mode, args.times, out)
        elif args.cmd == "drop":
            res = await drills.drop_test(rig, backend, args.k, args.color, out)
        elif args.cmd == "unload":
            res = await drills.unload_test(rig, backend, args.k, args.color, out)
        else:
            res = await drills.policy_dry(rig, cfg=cfg, out=out)
    except BaseException as e:
        await rig.stop()
        res = {"pass": False, "problems": [f"{type(e).__name__}: {e}"]}
        raise
    finally:
        logger.finish({"status": "pass" if res.get("pass") else "fail", "truth_success": bool(res.get("pass")),
                       "duration_s": round(time.time() - t0, 1), "belief_accuracy": None, "result": res})
        await rig.close()
    print_result(res)
    print(f"  log: {out}")


if __name__ == "__main__":
    asyncio.run(main())
