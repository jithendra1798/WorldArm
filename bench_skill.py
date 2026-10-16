"""MOVES THE ARM. Benchmark one skill as deployed: run its service command N times; a person judges each try.

  python bench_skill.py --service PickBottle --command '{"action": "pick"}' --label pick
  python bench_skill.py --service drawers --command '{"action": "drawer_run", "drawer": 2}' --label drawer-2
  WORLDARM_ENV=.env.arm2 python bench_skill.py --service pick-red-cube --command '{"action": "pick"}'
  python bench_skill.py --fake --label pick          # try the flow without an arm

Before each try: reset the scene, press Enter. The command runs, its reply is printed, and you judge:
y (worked) / n (failed; give a short reason) / s (skip: a bad reset, doesn't count).
--after runs a second command after each try (e.g. a release), logged but not judged.

Each run is logged to runs/<timestamp>-bench-<label>/ and appears in scoreboard.py.
The skill's own reply is logged next to your verdict, so we can also measure whether the skill
knows when it failed. Check what a command does before benchmarking it: some services move on {}.
"""

import argparse
import asyncio
import json
import random
import statistics
import time

from worldarm.config import load_config
from worldarm.logger import EpisodeLogger
from worldarm.metrics import rate, wilson


def short(obj, n: int = 300) -> str:
    text = json.dumps(obj, default=str)
    return text if len(text) <= n else text[:n] + " …"


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--service", help="generic service name on the machine")
    p.add_argument("--command", default="{}", help="DoCommand payload as JSON")
    p.add_argument("--after", help="optional DoCommand payload to run after each try (JSON)")
    p.add_argument("--label", help="name for this skill on the scoreboard (default: service name)")
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--fake", action="store_true", help="simulated replies, judged automatically")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--yes", action="store_true", help="skip the motion confirmation")
    args = p.parse_args()
    if not args.fake and not args.service:
        p.error("--service is required unless --fake")
    label = args.label or args.service
    command = json.loads(args.command)
    after = json.loads(args.after) if args.after else None
    cfg = load_config()

    machine, svc, rng = None, None, random.Random(args.seed)
    if not args.fake:
        from viam.services.generic import Generic

        from worldarm.connection import connect
        print(f"{args.trials} tries of {args.service}.do_command({short(command)}) on {cfg.machine}")
        if not args.yes and input("This MOVES the arm. Someone at the E-stop? Type 'go': ").strip() != "go":
            raise SystemExit("Cancelled.")
        machine = await connect()
        svc = Generic.from_robot(machine, args.service)

    mode = "fake" if args.fake else "real"
    logger = EpisodeLogger("runs", f"bench-{label}", {
        "mode": mode, "goal": f"bench:{label}", "blind": False, "machine": cfg.machine,
        "service": args.service, "command": command, "after": after, "trials": args.trials, "args": vars(args),
    })
    results, durations = [], []
    t_all = time.time()
    try:
        for i in range(1, args.trials + 1):
            if not args.fake and input(f"\n[{i}/{args.trials}] reset the scene, then Enter (q to stop): ").strip() == "q":
                break
            t0 = time.time()
            try:
                if args.fake:
                    reply = {"success": rng.random() < 0.8}
                else:
                    reply = dict(await svc.do_command(command))
            except Exception as e:  # a failed command is a failed try, not a crash
                reply = {"error": f"{type(e).__name__}: {e}"}
            duration = time.time() - t0
            after_reply = None
            if after is not None and svc is not None:
                try:
                    after_reply = dict(await svc.do_command(after))
                except Exception as e:
                    after_reply = {"error": f"{type(e).__name__}: {e}"}
            print(f"  reply ({duration:.1f} s): {short(reply)}")

            if args.fake:
                verdict, note = ("y" if reply["success"] else "n"), ""
            else:
                verdict = ""
                while verdict not in ("y", "n", "s"):
                    verdict = input("  worked? [y/n/s] ").strip().lower()
                note = input("  what went wrong? ").strip() if verdict == "n" else ""
            if verdict == "s":
                print("  skipped (not counted)")
                continue
            ok = verdict == "y"
            results.append(ok)
            durations.append(duration)
            logger.step({
                "step": i, "t_start": t0, "duration_s": round(duration, 3),
                "action": {"kind": label, "color": None, "drawer": None},
                "observation": {"pre_facts": {}, "facts": {"success": ok}, "detail": {"reply": reply, "after_reply": after_reply}},
                "surprises": [] if ok else [{"key": "success", "expected": True, "observed": False, "note": note}],
                "ignored_surprises": [],
            })
            print(f"  so far: {rate(sum(results), len(results))}")
    finally:
        if machine is not None:
            await machine.close()

    k, n = sum(results), len(results)
    lo, hi = wilson(k, n)
    summary = {
        "status": "done", "reason": "benchmark", "steps": n, "surprises": n - k,
        "duration_s": round(time.time() - t_all, 2), "belief_success": None, "truth_success": None,
        "belief_accuracy": None, "trials": n, "successes": k,
        "success_rate": round(k / n, 3) if n else None, "wilson95": [round(lo, 3), round(hi, 3)],
        "median_try_s": round(statistics.median(durations), 2) if durations else None,
    }
    logger.finish(summary)
    print(f"\n{label}: {rate(k, n)}   median {summary['median_try_s']} s per try   log: {logger.dir}")


if __name__ == "__main__":
    asyncio.run(main())
