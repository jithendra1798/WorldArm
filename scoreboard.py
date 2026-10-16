"""Print the scoreboard for all logged episodes: python scoreboard.py [--runs runs] [--mode fake|dry|real]"""

import argparse

from worldarm.metrics import load_runs, scoreboard

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", default="runs")
    p.add_argument("--mode", choices=["fake", "dry", "real"])
    args = p.parse_args()
    runs = [r for r in load_runs(args.runs) if args.mode is None or r["meta"]["mode"] == args.mode]
    print(scoreboard(runs))
