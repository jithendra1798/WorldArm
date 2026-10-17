"""Task B overnight simulation study: the numbers and charts for the H4 / H5 slides. No robot; one process.

  .venv/bin/python ball_study.py --quick                    # under a minute, small n: check it works
  .venv/bin/python ball_study.py                            # the full run (~50 min), progress + ETA every 15 s
  .venv/bin/python ball_study.py --studies h4,robust        # just some of: h4, h5, needs, robust
  .venv/bin/python ball_study.py --report-only runs/<stamp>-ball-study   # redraw summary.md + charts from the CSVs

Writes runs/<stamp>-ball-study/: one CSV per study (as rows finish), summary.md with the headline numbers
(n and 95% intervals), meta.json (plan, seeds, git commit, code hashes), and PNG charts when matplotlib is
installed. Ctrl-C stops cleanly: what finished is kept and summarised.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

STUDY_NAMES = ("h4", "h5", "needs", "robust")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true", help="small grids and n (under a minute)")
    p.add_argument("--studies", default=",".join(STUDY_NAMES), help="comma-separated: h4,h5,needs,robust")
    p.add_argument("--out", type=Path, help="run folder (default runs/<stamp>-ball-study[-quick])")
    p.add_argument("--seed", type=int, default=2026, help="base seed (each study derives its own from it)")
    p.add_argument("--decel", type=float, help="the table's slowdown in mm/s^2 (default: SimCfg's 120); e.g. the "
                                               "value `ball.py fit` learned")
    p.add_argument("--no-charts", action="store_true", help="CSV + summary.md only")
    p.add_argument("--report-only", type=Path, metavar="DIR",
                   help="no simulation: rebuild summary.md and the charts from the CSVs in DIR")
    args = p.parse_args()

    from worldarm.ball import study, study_report

    if args.report_only:
        if not args.report_only.is_dir():
            p.error(f"{args.report_only} is not a folder")
        study_report.report(args.report_only, charts=not args.no_charts)
        return 0

    studies = [s.strip() for s in args.studies.split(",") if s.strip()]
    unknown = sorted(set(studies) - set(STUDY_NAMES))
    if unknown or not studies:
        p.error(f"unknown studies {unknown}; pick from {','.join(STUDY_NAMES)}")
    if args.decel is not None and args.decel <= 0:
        p.error("--decel must be > 0 (a rolling ball slows down)")
    plan = study.QUICK if args.quick else study.FULL
    base = study.Baseline(seed=args.seed, **({"decel": args.decel} if args.decel else {}))
    out = args.out or Path("runs") / f"{time.strftime('%Y%m%d-%H%M%S')}-ball-study{'-quick' if args.quick else ''}"
    t0 = time.monotonic()
    try:
        study.run(plan, base, studies, out, charts=not args.no_charts)
    except KeyboardInterrupt:
        print(f"\nstopped after {time.monotonic() - t0:.0f} s; finished rows are kept: {out / 'summary.md'}")
        return 130
    print(f"done in {time.monotonic() - t0:.0f} s: {out / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
