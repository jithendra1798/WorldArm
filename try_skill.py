"""MOVES THE ARM. Run a short script of skills on a real arm; the world model checks each one.

  python try_skill.py observe
  python try_skill.py pick blue                        # looks at the table first
  python try_skill.py open 1 pick blue place 1 close 1
  python try_skill.py --holding blue place 2           # you put blue in the gripper by hand
  python try_skill.py --drawer-open 2 close 2          # drawer 2 is already open
  python try_skill.py --fake open 1 pick blue          # try the script in simulation first

Actions: observe | pick <color> | place <k> | open <k> | close <k>     (drawer 1 = top)
Every run is logged to runs/ and counts toward the per-action numbers in scoreboard.py.
"""

import argparse
import asyncio
import time

from worldarm.config import load_config
from worldarm.episode import run_episode, summarize
from worldarm.logger import EpisodeLogger
from worldarm.planner import Goal
from worldarm.schema import GRIPPER, TABLE, Act, Action
from worldarm.sim import FakeRobot
from worldarm.world_model import Belief


def parse_script(tokens: list[str], colors: tuple[str, ...]) -> list[Action]:
    actions, i = [], 0
    while i < len(tokens):
        word = tokens[i]
        if word == "observe":
            actions.append(Action(Act.OBSERVE_TABLE))
            i += 1
            continue
        if i + 1 >= len(tokens):
            raise SystemExit(f"'{word}' needs an argument")
        arg = tokens[i + 1]
        if word == "pick":
            if arg not in colors:
                raise SystemExit(f"unknown color {arg!r}; use one of: {', '.join(colors)}")
            actions.append(Action(Act.PICK, color=arg))
        elif word in ("place", "open", "close"):
            if not arg.isdigit():
                raise SystemExit(f"'{word}' needs a drawer number, got {arg!r}")
            actions.append(Action(Act(word), drawer=int(arg)))
        else:
            raise SystemExit(f"unknown action {word!r}; use observe, pick, place, open or close")
        i += 2
    # A pick needs the belief to know the block is on the table, so look first unless the script does.
    kinds = [a.kind for a in actions]
    if Act.PICK in kinds and (Act.OBSERVE_TABLE not in kinds or kinds.index(Act.OBSERVE_TABLE) > kinds.index(Act.PICK)):
        actions.insert(0, Action(Act.OBSERVE_TABLE))
    return actions


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("script", nargs="+", help="actions, e.g. open 1 pick blue place 1 close 1")
    p.add_argument("--fake", action="store_true", help="run against the simulated robot")
    p.add_argument("--holding", metavar="COLOR", help="the gripper already holds this block")
    p.add_argument("--drawer-open", type=int, action="append", default=[], metavar="K",
                   help="drawer K is already open (repeatable)")
    p.add_argument("--yes", action="store_true", help="skip the motion confirmation")
    args = p.parse_args()

    cfg = load_config()
    script = parse_script(args.script, cfg.colors)
    belief = Belief.initial(cfg.colors, cfg.n_drawers)
    if args.holding:
        belief.blocks[args.holding] = GRIPPER
    for k in args.drawer_open:
        belief.drawers[k] = True

    machine = None
    if args.fake:
        world = {c: GRIPPER if c == args.holding else TABLE for c in cfg.colors}
        robot = FakeRobot(world, cfg.n_drawers)
        for k in args.drawer_open:
            robot.drawers[k] = True
    else:
        from worldarm.connection import connect
        from worldarm.robot import ViamRobot
        print("Script: " + " → ".join(str(a) for a in script))
        if not args.yes and input(f"This MOVES the arm on {cfg.machine}. Someone at the E-stop? Type 'go': ").strip() != "go":
            raise SystemExit("Cancelled.")
        machine = await connect()
        try:
            robot = ViamRobot(machine, cfg)
        except Exception as e:
            await machine.close()
            raise SystemExit(f"{cfg.machine} is missing a resource ({e}). "
                             "Configure it from docs/viam-config.md.") from None
        robot.skills.held = args.holding

    mode = "fake" if args.fake else "real"
    logger = EpisodeLogger("runs", f"skill-{mode}", {
        "mode": mode, "goal": "script", "script": " ".join(args.script), "blind": False,
        "args": vars(args), "machine": cfg.machine, "containers": cfg.containers,
        "belief_start": belief, "truth_start": robot.truth(),
    })
    t0 = time.time()
    try:
        result = await run_episode(robot, belief, Goal("sort"), cfg.assignment, logger,
                                   script=script, max_steps=len(script) + 1)
    finally:
        if machine is not None:
            await machine.close()
    logger.finish(summarize(result, None, cfg.assignment, time.time() - t0, result.truth))
    print(f"\n{result.status}: {result.reason}  ({result.steps} steps, {result.surprises} surprises)")
    print(f"belief: {result.belief.blocks}  open drawers: {result.belief.open_drawers()}")
    print(f"log: {logger.dir}")


if __name__ == "__main__":
    asyncio.run(main())
