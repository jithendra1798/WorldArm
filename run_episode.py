"""Run one episode and log it to runs/<timestamp>-<mode>/.

  python run_episode.py --fake                                 # simulated robot, no arm needed
  python run_episode.py --fake --p-fail 0.2 --seed 3           # with injected failures
  python run_episode.py --fake --like-real-v0 --p-fail 0.2     # only the checks real v0 has
  python run_episode.py --fake --goal find:blue --start sorted --move blue:drawer:3   # surprise demo
  python run_episode.py --dry-run          # real detections from the CURRENT pose, simulated actions; no motion
  python run_episode.py                    # REAL ARM: MOVES IT (asks first)

`python scoreboard.py` summarizes all logged episodes.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

from worldarm.config import load_config
from worldarm.episode import belief_accuracy, run_episode
from worldarm.logger import EpisodeLogger
from worldarm.planner import Goal, satisfied
from worldarm.schema import TABLE, UNKNOWN, drawer
from worldarm.sim import OFFSTAGE, FakeRobot, SimConfig
from worldarm.world_model import Belief

STATE_DIR = Path("state")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--fake", action="store_true", help="simulated robot")
    mode.add_argument("--dry-run", action="store_true", help="real detections, simulated actions, no motion")
    p.add_argument("--goal", default="sort", help="sort | find:<color>")
    p.add_argument("--blind", action="store_true", help="ablation: ignore post-action measurements")
    p.add_argument("--max-steps", type=int, default=40)
    sim = p.add_argument_group("simulation (--fake)")
    sim.add_argument("--seed", type=int, default=0)
    sim.add_argument("--p-fail", type=float, default=0.0, help="failure probability for every skill")
    sim.add_argument("--like-real-v0", action="store_true", help="sim measures only what real v0 measures")
    sim.add_argument("--start", choices=["table", "sorted"], default=None,
                     help="true starting world (default: table for sort, sorted for find)")
    sim.add_argument("--move", action="append", default=[], metavar="COLOR:LOCATION",
                     help="hidden hand before the episode, e.g. blue:drawer:3 or blue:table")
    real = p.add_argument_group("real arm")
    real.add_argument("--yes", action="store_true", help="skip the motion confirmation")
    real.add_argument("--ask-truth", action="store_true", help="ask where each block really is at the end")
    return p.parse_args()


def ask_truth(colors, n_drawers) -> dict:
    """Ground truth from the operator, so success is judged against reality, not the belief."""
    print("\nGround truth (look at the table and drawers):")
    blocks = {}
    for c in colors:
        answer = input(f"  where is {c}? [1-{n_drawers} = drawer, t = table, ? = don't know] ").strip()
        blocks[c] = drawer(int(answer)) if answer.isdigit() else TABLE if answer == "t" else UNKNOWN
    open_k = input("  which drawer is open? [number, or Enter for none] ").strip()
    return {"blocks": blocks, "drawers": {str(k): str(k) == open_k for k in range(1, n_drawers + 1)}}


async def main() -> None:
    args = parse_args()
    cfg = load_config()
    goal = Goal.parse(args.goal)
    mode = "fake" if args.fake else "dry" if args.dry_run else "real"
    state_file = STATE_DIR / f"{'sim' if args.fake else cfg.machine}-belief.json"
    machine = None

    # Starting belief: a sort starts from nothing; a find starts from what the robot remembers.
    belief = Belief.initial(cfg.colors, cfg.n_drawers)
    if goal.kind == "find" and not args.fake:
        if not state_file.exists():
            raise SystemExit(f"No remembered belief in {state_file}; run a sort episode first.")
        belief = Belief.from_json(json.loads(state_file.read_text()))

    if args.fake:
        start = args.start or ("table" if goal.kind == "sort" else "sorted")
        world = {c: TABLE if start == "table" else drawer(cfg.assignment[c]) for c in cfg.colors}
        if goal.kind == "find":
            belief = Belief(dict(world), {k: False for k in range(1, cfg.n_drawers + 1)})  # remembers the start
        sim_cfg = (SimConfig.like_real_v0 if args.like_real_v0 else SimConfig)(
            p_pick_fail=args.p_fail, p_place_miss=args.p_fail, p_open_fail=args.p_fail,
            p_close_fail=args.p_fail, seed=args.seed)
        robot = FakeRobot(world, cfg.n_drawers, sim_cfg)
        for move in args.move:
            color, location = move.split(":", 1)
            robot.move_block(color, location)
    else:
        from worldarm.connection import connect
        from worldarm.robot import ViamRobot
        if mode == "real" and not args.yes:
            answer = input(f"This MOVES the arm on {cfg.machine}. Someone at the E-stop? Type 'go': ")
            if answer.strip() != "go":
                raise SystemExit("Cancelled.")
        machine = await connect()
        try:
            robot = ViamRobot(machine, cfg)
        except Exception as e:
            await machine.close()
            raise SystemExit(f"{cfg.machine} is missing a resource ({e}). "
                             "Configure it from docs/viam-config.md.") from None
        if mode == "dry":
            seen = {b["color"] for b in await robot.detect_blocks(move=False)}
            print(f"Detected from the current pose: {sorted(seen) or 'nothing'}")
            robot = FakeRobot({c: TABLE if c in seen else OFFSTAGE for c in cfg.colors}, cfg.n_drawers)

    logger = EpisodeLogger("runs", mode, {
        "mode": mode, "goal": str(goal), "blind": args.blind, "args": vars(args), "machine": cfg.machine,
        "assignment": cfg.assignment, "limits": cfg.limits, "belief_start": belief,
        "truth_start": robot.truth(),
    })
    print(f"Episode: {mode}, goal {goal}{', BLIND' if args.blind else ''}  → {logger.dir}")
    t0 = time.time()
    try:
        result = await run_episode(robot, belief, goal, cfg.assignment, logger,
                                   blind=args.blind, max_steps=args.max_steps)
    finally:
        if machine is not None:
            await machine.close()

    truth = result.truth
    if truth is None and args.ask_truth:
        truth = ask_truth(cfg.colors, cfg.n_drawers)
    truth_success = satisfied(Belief.from_json(truth), goal, cfg.assignment) if truth else None
    summary = {
        "status": result.status, "reason": result.reason, "steps": result.steps,
        "surprises": result.surprises, "duration_s": round(time.time() - t0, 2),
        "belief_success": result.status == "done", "truth_success": truth_success,
        "belief_accuracy": belief_accuracy(result.belief, truth),
        "belief_final": result.belief, "truth_final": truth,
    }
    logger.finish(summary)
    if mode != "dry":
        STATE_DIR.mkdir(exist_ok=True)
        state_file.write_text(json.dumps(result.belief.to_json(), indent=2) + "\n")

    print(f"\n{result.status}: {result.reason}  ({result.steps} steps, {result.surprises} surprises)")
    if truth is not None:
        print(f"truth success: {truth_success}   belief accuracy: {summary['belief_accuracy']:.0%}")
    print(f"belief: {result.belief.blocks}")


if __name__ == "__main__":
    asyncio.run(main())
