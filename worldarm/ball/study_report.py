"""summary.md for the task B study, built only from the CSVs (and meta.json) in a run folder, so it can be
rebuilt at any time; report() also redraws the charts (study_charts.py) when matplotlib is installed."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

from worldarm.ball.sim import SimCfg
from worldarm.ball.study import FACTORS, GOAL, MIS_HORIZON_S, MIS_WATCHED_S, Baseline, hms, read_rows, write_atomic


def report(out: Path, charts: bool = True, say: Callable[[str], None] = print) -> Path:
    """Rebuild summary.md (and the charts, when matplotlib is installed) from the CSVs in `out`. A failure here is
    noted, never raised: the CSVs are the results, and a long run must not stop over a chart."""
    out = Path(out)
    note = "not drawn (charts=False)"
    if charts:
        try:
            from worldarm.ball import study_charts
            made, failed = study_charts.render(out)
            note = ", ".join([p.name for p in made] + [f"{name} FAILED ({err})" for name, err in failed]) or "none yet"
        except ImportError as e:
            note = (f"not drawn: {e.name or 'matplotlib'} isn't installed in this Python. Draw them from the CSVs "
                    f"with a Python that has matplotlib: `python ball_study.py --report-only {out}`")
    path = out / "summary.md"
    try:
        text = summary_md(out, note)
    except Exception as e:
        text = (f"# Task B in simulation\n\nThe summary couldn't be built ({e!r}); the CSVs here are complete. "
                f"Rebuild it with `python ball_study.py --report-only {out}`.\n\nCharts: {note}\n")
    write_atomic(path, text)
    say(f"  summary: {path}   charts: {note}")
    return path


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%" if math.isfinite(x) else "n/a"


def _rate(r: dict) -> str:
    return f"{_pct(r['rate'])} [{100 * r['lo95']:.0f}, {100 * r['hi95']:.0f}] ({r['defended']:.0f}/{r['counted']:.0f})"


def _gain(r: dict) -> str:
    if not math.isfinite(r.get("predict_gain", math.nan)):
        return "n/a"
    paired = "" if r.get("paired", 1) == 1 else ", unpaired"
    return (f"{100 * r['predict_gain']:+.0f} points [{100 * r['gain_lo95']:+.0f}, {100 * r['gain_hi95']:+.0f}]"
            f"{paired}")


def _ms(med: float, lo: float, hi: float) -> str:
    f = lambda v: "inf" if v == math.inf else f"{v:.0f}"
    return "n/a" if not math.isfinite(med) and med != math.inf else f"{f(med)} ms [{f(lo)}, {f(hi)}]"


def nearest(values, target: float) -> float:
    values = sorted(set(values))
    return min(values, key=lambda v: (abs(v - target), v)) if values else math.nan


def _speed(r: dict) -> str:
    return f"{r['ball_min_mm_s']:.0f}-{r['ball_max_mm_s']:.0f} mm/s"


def summary_md(out: Path, charts_note: str = "") -> str:
    """The headline numbers, from whatever CSVs are in `out` (n and 95% intervals everywhere)."""
    out = Path(out)
    meta = json.loads((out / "meta.json").read_text()) if (out / "meta.json").exists() else {}
    base = Baseline(**meta["baseline"]) if "baseline" in meta else Baseline()
    h5, needs = read_rows(out / "h5_sweep.csv"), read_rows(out / "needs.csv")
    arrival, stop = read_rows(out / "h4_arrival.csv"), read_rows(out / "h4_stop.csv")
    mis, robust = read_rows(out / "h4_misdecel.csv"), read_rows(out / "robust.csv")
    statuses = meta.get("studies", {})
    status = ", ".join(f"{s} {v['status']}" + (f" ({hms(v['seconds'])}, {v['plan']} plan)" if "seconds" in v else "")
                       for s, v in statuses.items()) or "no studies recorded"
    lines = ["# Task B in simulation: the rolling-ball goalie", "",
             f"Run `{out.name}`: {status}.", ""]
    sentence = needs_sentence(needs, base)
    if sentence:
        lines += [f"**For the slide:** {sentence}", ""]
    if h5:
        lines += _h5_md(h5, base)
    if arrival or stop or mis:
        lines += _h4_md(arrival, stop, mis, base)
    if robust:
        lines += _robust_md(robust)
    if needs:
        lines += _needs_md(needs, base)
    lines += [f"Charts: {charts_note}", ""] if charts_note else []
    lines += _setup_md(meta, base)
    return "\n".join(lines) + "\n"


def _h5_md(rows: list[dict], base: Baseline) -> list[str]:
    ranges = list(dict.fromkeys((r["ball_min_mm_s"], r["ball_max_mm_s"]) for r in rows))
    out = ["## H5: go to the predicted crossing vs chase the ball", ""]
    for i, rng in enumerate(ranges):
        rr = [r for r in rows if (r["ball_min_mm_s"], r["ball_max_mm_s"]) == rng]
        spd = nearest([r["lane_speed_mm_s"] for r in rr], base.lane_speed_mm_s)
        lat = nearest([r["latency_s"] for r in rr], base.latency_s)
        hw = nearest([r["half_width_mm"] for r in rr], base.half_width_mm)
        cell = {r["goalie"]: r for r in rr if (r["lane_speed_mm_s"], r["latency_s"], r["half_width_mm"]) == (spd, lat, hw)}
        if "predict" not in cell:
            continue
        out.append(f"Arm {spd:.0f} mm/s along the lane with {lat:.2f} s command latency, paddle half-width {hw:.0f} mm, "
                   f"balls {rng[0]:.0f}-{rng[1]:.0f} mm/s (rolls that reached the lane within reach):")
        cmds = lambda g: f"{cell[g]['mean_commands']:.1f} paddle commands per roll"
        out.append(f"- **predict: {_rate(cell['predict'])}**; {cmds('predict')}")
        if "chase" in cell:
            out.append(f"- chase: {_rate(cell['chase'])}; {cmds('chase')}. Predict {_gain(cell['chase'])} "
                       "(paired, same rolls)")
        if "chase-sparse" in cell:
            out.append(f"- chase that re-aims only every half-width (fewer interruptions): {_rate(cell['chase-sparse'])};"
                       f" {cmds('chase-sparse')}. Predict {_gain(cell['chase-sparse'])}")
        out.append("")
        if i == 0 and "chase" in cell:
            note = ("Every new command stops the paddle for a full command latency before it moves again (as "
                    "`LaneArm.go` does on the robot), so a goalie that re-aims often loses time, and part of "
                    "predict's lead is that it re-aims less.")
            sp = cell.get("chase-sparse")
            if sp is not None and math.isfinite(sp["gain_lo95"]):
                note += (" Against the chase that re-aims less, predict still leads for sure here (paired interval above "
                         "zero), so the result doesn't rest on a weak chase." if sp["gain_lo95"] > 0 else
                         " Against the chase that re-aims less, predict's lead here is not certain (the paired interval "
                         "includes zero): say so if the slide compares with chasing.")
            out += [note, ""]
    chase = [r for r in rows if r["goalie"] == "chase" and math.isfinite(r["predict_gain"])]
    if chase:
        ahead = sum(r["predict_gain"] > 0 for r in chase)
        sure = sum(r["gain_lo95"] > 0 for r in chase)
        behind = sum(r["gain_hi95"] < 0 for r in chase)
        big = max(chase, key=lambda r: r["predict_gain"])
        small = min(chase, key=lambda r: r["predict_gain"])
        where = lambda r: (f"{r['lane_speed_mm_s']:.0f} mm/s, {r['latency_s']:.2f} s, half-width "
                           f"{r['half_width_mm']:.0f} mm, balls {_speed(r)}")
        out += [f"Across all {len(chase)} conditions (lane speed x latency x half-width x ball speed), predict blocked "
                f"more than chase in {ahead}, with the paired 95% interval above zero in {sure}; chase was ahead "
                f"for sure in {behind}. Largest gap: {_gain(big)} at {where(big)}. Smallest: {_gain(small)} at "
                f"{where(small)}.", ""]
    return out


def _h4_md(arrival: list[dict], stop: list[dict], mis: list[dict], base: Baseline) -> list[str]:
    out = ["## H4: the learned slowdown vs \"it keeps its current velocity\"", ""]
    if arrival:
        hs = sorted({r["horizon_s"] for r in arrival})
        show = sorted({nearest(hs, h) for h in (0.3, 0.5, 1.0, 1.5, 2.0)})
        by = {(r["model"], r["horizon_s"]): r for r in arrival}
        out += [f"**When will it reach the lane?** Median arrival-time error [95% interval], straight rolls of "
                f"{SimCfg().speed[0]:.0f}-{SimCfg().speed[1]:.0f} mm/s that slow at {base.decel:.0f} mm/s^2. "
                "A learned-model prediction that the ball stops short, when it arrives, counts as the worst error.", "",
                "| time before arrival | learned slowdown | constant velocity | learned closer | rolls |",
                "|---|---|---|---|---|"]
        for h in show:
            le, cv = by.get(("learned", h)), by.get(("constant velocity", h))
            if not le or not cv:
                continue
            closer = (f"{_pct(le['closer_share'])} [{100 * le['closer_lo95']:.0f}, {100 * le['closer_hi95']:.0f}]"
                      if math.isfinite(le["closer_share"]) else "n/a")
            out.append(f"| {h:.1f} s | {_ms(le['median_err_ms'], le['lo95_ms'], le['hi95_ms'])} | "
                       f"{_ms(cv['median_err_ms'], cv['lo95_ms'], cv['hi95_ms'])} | {closer} | {le['n']:.0f} |")
        h1 = nearest(hs, 1.0)
        le, cv = by.get(("learned", h1)), by.get(("constant velocity", h1))
        if le and cv:
            out += ["", f"Where it crosses the lane, {h1:.1f} s before: {le['median_where_mm']:.1f} mm off (learned) "
                    f"vs {cv['median_where_mm']:.1f} mm (constant velocity). The slowdown changes when a straight roll "
                    f"arrives, not where. The learned model said \"stops short\" for {le['no_cross']:.0f} of "
                    f"{le['n']:.0f} rolls that arrived. At longer horizons only slower rolls qualify, since they are "
                    "the ones still that far out."]
        out.append("")
    if stop:
        out += ["**Where will it stop?** Slow rolls (150-450 mm/s), still rolling when predicted. Constant velocity "
                "can't answer: it never stops. \"Still to go\" is the error of guessing it stops right where it is.", "",
                "| watched for | stop-point error | still to go | rolls |", "|---|---|---|---|"]
        ws = sorted({r["watched_s"] for r in stop})
        for w in sorted({nearest(ws, w) for w in (0.2, 0.5, 1.0, 1.5, 2.0)}):
            r = next(r for r in stop if r["watched_s"] == w)
            out.append(f"| {w:.1f} s | {r['median_mm']:.0f} mm [{r['lo95_mm']:.0f}, {r['hi95_mm']:.0f}] | "
                       f"{r['median_to_go_mm']:.0f} mm | {r['n_moving']:.0f} |")
        out.append("")
    if mis:
        out += [f"**A mis-learned slowdown** (the goalie believes the wrong value; the table's is {base.decel:.0f} "
                f"mm/s^2). Arrival error {MIS_HORIZON_S:.1f} s before arrival, stop error after watching "
                f"{MIS_WATCHED_S:.1f} s, and the predict goalie's block rate at the baseline arm (change: paired, "
                "same rolls).", "",
                "| believed | arrival error | said \"stops short\" | stop error | predict blocks | change |",
                "|---|---|---|---|---|---|"]
        for r in sorted(mis, key=lambda r: r["decel_error_pct"]):
            name = ("constant velocity (0)" if r["decel_error_pct"] == -100
                    else f"{r['decel_error_pct']:+.0f}% ({r['believed_decel']:.0f})")
            stop_s = (f"{r['stop_median_mm']:.0f} mm [{r['stop_lo95_mm']:.0f}, {r['stop_hi95_mm']:.0f}]"
                      if math.isfinite(r["stop_median_mm"]) else "never stops")
            change = (f"{100 * r['rate_change']:+.0f} [{100 * r['change_lo95']:+.0f}, {100 * r['change_hi95']:+.0f}]"
                      if r["decel_error_pct"] != 0 else "reference")
            out.append(f"| {name} | {_ms(r['arrival_median_ms'], r['arrival_lo95_ms'], r['arrival_hi95_ms'])} | "
                       f"{r['arrival_no_cross']:.0f}/{r['arrival_n']:.0f} | {stop_s} | {_rate(r)} | {change} |")
        out += ["", "For blocking, the slowdown matters only through \"will it reach the lane at all\": the goalie "
                "aims at the crossing point, which a straight roll's slowdown doesn't move.", ""]
    return out


def _robust_md(rows: list[dict]) -> list[str]:
    out = ["## Robustness of the predict goalie", "",
           "One setting made worse at a time from the baseline (the baseline row of each is marked). Block rate "
           "[95% interval] (blocked/counted).", "",
           "| setting | level | predict | chase | predict gain | predict commands per roll |", "|---|---|---|---|---|---|"]
    worst = None
    for factor in dict.fromkeys(r["factor"] for r in rows):
        for level in sorted({r["level"] for r in rows if r["factor"] == factor}):
            cell = {r["goalie"]: r for r in rows if r["factor"] == factor and r["level"] == level}
            p, c = cell.get("predict"), cell.get("chase")
            if p is None:
                continue
            mark = " (baseline)" if p["is_baseline"] == 1 else ""
            out.append(f"| {FACTORS.get(factor, factor)} | {level:g}{mark} | {_rate(p)} | "
                       f"{_rate(c) if c else 'n/a'} | {_gain(c) if c else 'n/a'} | {p['mean_commands']:.1f} |")
            if worst is None or p["rate"] < worst[2]["rate"]:
                worst = (factor, level, p)
    if worst:
        out += ["", f"Lowest predict rate in this sweep: {_rate(worst[2])} with "
                f"{FACTORS.get(worst[0], worst[0])} = {worst[1]:g}. Where predict drops, look at its commands per "
                "roll: noisier early estimates move the predicted crossing, each re-aim stops the paddle for a full "
                "command latency."]
    return out + [""]


def needs_sentence(needs: list[dict], base: Baseline) -> str:
    """One sentence for the slide: what the arm needs to block >= 90% at the baseline paddle and ball speeds."""
    if not needs:
        return ""
    rng = (needs[0]["ball_min_mm_s"], needs[0]["ball_max_mm_s"])
    rows = [r for r in needs if (r["ball_min_mm_s"], r["ball_max_mm_s"]) == rng]
    hw = nearest([r["half_width_mm"] for r in rows], base.half_width_mm)
    rows = [r for r in rows if r["half_width_mm"] == hw]
    speeds = sorted(r["given_value"] for r in rows if r["given"] == "lane_speed_mm_s")
    lats = sorted(r["given_value"] for r in rows if r["given"] == "latency_s")
    if not speeds or not lats:
        return ""
    lat, spd = nearest(lats, base.latency_s), nearest(speeds, base.lane_speed_mm_s)
    a = next(r for r in rows if r["given"] == "latency_s" and r["given_value"] == lat)
    b = next(r for r in rows if r["given"] == "lane_speed_mm_s" and r["given_value"] == spd)
    est, low = a["by_estimate"], a["by_lower_bound"]
    if not math.isfinite(est):
        s1 = f"no tested paddle speed (up to {speeds[-1]:.0f} mm/s) is enough at {lat:.2f} s command latency"
    elif est == speeds[0]:
        s1 = f"even the slowest tested paddle speed, {est:.0f} mm/s along the lane, is enough at {lat:.2f} s latency"
    else:
        s1 = f"the paddle must move at least {est:.0f} mm/s along the lane at {lat:.2f} s command latency"
    s1 += (" (no tested speed clears it by the 95% lower bound)" if not math.isfinite(low) else
           " (the slowest tested speed clears it even by the 95% lower bound)" if low == speeds[0] else
           f" ({low:.0f} mm/s for the 95% lower bound to clear it too)")
    est, low = b["by_estimate"], b["by_lower_bound"]
    if not math.isfinite(est):
        s2 = f"at {spd:.0f} mm/s even the shortest tested latency, {lats[0]:.2f} s, is too long"
    elif est == lats[-1]:
        s2 = f"at {spd:.0f} mm/s every tested command latency works, up to {est:.2f} s"
    else:
        s2 = f"at {spd:.0f} mm/s the command latency can be up to {est:.2f} s"
    s2 += (" (none by the 95% lower bound)" if not math.isfinite(low) else f" ({low:.2f} s by the 95% lower bound)")
    return (f"In simulation, to block at least {GOAL:.0%} of rolls ({rng[0]:.0f}-{rng[1]:.0f} mm/s, paddle half-width "
            f"{hw:.0f} mm), {s1}; {s2}.")


def _needs_md(needs: list[dict], base: Baseline) -> list[str]:
    out = [f"## What the arm needs (predict goalie, >= {GOAL:.0%} blocked)", "",
           "Slowest tested lane speed that reaches it at each latency (by the estimate / by its 95% lower bound; "
           "\"-\" = none of the tested speeds).", ""]
    fmt = lambda v: f"{v:.0f}" if math.isfinite(v) else "-"
    for rng in dict.fromkeys((r["ball_min_mm_s"], r["ball_max_mm_s"]) for r in needs):
        rows = [r for r in needs if (r["ball_min_mm_s"], r["ball_max_mm_s"]) == rng
                and r["requirement"] == "min_lane_speed_mm_s"]
        widths = sorted({r["half_width_mm"] for r in rows})
        out += [f"Balls {rng[0]:.0f}-{rng[1]:.0f} mm/s, min lane speed in mm/s:", "",
                "| latency | " + " | ".join(f"half-width {w:.0f} mm" for w in widths) + " |",
                "|---|" + "---|" * len(widths)]
        for lat in sorted({r["given_value"] for r in rows}):
            cells = {r["half_width_mm"]: r for r in rows if r["given_value"] == lat}
            out.append(f"| {lat:.2f} s | " + " | ".join(
                f"{fmt(cells[w]['by_estimate'])} / {fmt(cells[w]['by_lower_bound'])}" if w in cells else "n/a"
                for w in widths) + " |")
        out.append("")
    return out


def _setup_md(meta: dict, base: Baseline) -> list[str]:
    c = SimCfg()
    inv = meta.get("invocations", [{}])[-1]
    git = inv.get("git", {})
    plan = meta.get("plan", {})
    return [
        "## Setup", "",
        f"- Lane {base.lane_length_mm:.0f} mm. Rolls start {c.start_distance[0] / 1000:.1f}-{c.start_distance[1] / 1000:.1f} m out, "
        f"aimed up to {c.aim_spread_mm:.0f} mm past either lane end, starting up to {c.side_mm:.0f} mm to the side "
        f"(diagonal rolls). The ball slows at {base.decel:.0f} mm/s^2, and the goalie knows that (except in the "
        "mis-learned check).",
        f"- Camera {c.fps:.0f} fps, {c.noise_mm:.0f} mm noise, {c.miss_prob:.0%} missed frames, "
        f"{1000 * c.camera_latency_s:.0f} ms latency. The paddle covers the ball within half-width + ball radius "
        f"({c.ball_radius_mm:.0f} mm) - 5 mm of its centre.",
        "- The arm: each command starts after the latency, then moves at the lane speed. A new command stops the "
        "paddle and waits a full latency again, as `LaneArm.go` does on the robot (stop, then a new move). "
        "Both goalies re-aim only when their target moved by more than a quarter of the paddle width (the "
        "Goalie default): predict's target is the predicted crossing, chase's is where the ball is now.",
        "- Rates count the rolls that reached the lane within reach of the paddle; rolls that stopped short are "
        "left out. Intervals: Wilson 95% for rates, Newcombe's paired 95% interval for predict minus chase (same "
        "rolls), order-statistic 95% for medians.",
        f"- Plan `{plan.get('name', '?')}`: h5 {plan.get('h5_n', '?')} rolls per cell, h4 {plan.get('h4_n', '?')} "
        f"rolls per point, robust {plan.get('robust_n', '?')} rolls per level. Seeds: {meta.get('seeds', {})} "
        "(roll i of a study is drawn from generator (seed, i)).",
        f"- Code: git {git.get('commit', '?')}{' (with uncommitted changes)' if git.get('dirty') else ''}; the ball "
        "code's file hashes are in meta.json (it isn't all committed yet).",
        "",
        "## What the simulation leaves out", "",
        "- Straight rolls only: no curve, spin or table tilt. The table is flat to 0.16 degrees, which can push a "
        "slow roll about 4 cm sideways over 2 s; the sim can't show that.",
        "- No false detections (a hand, the paddle taken for the ball), and missed frames are independent (no "
        "occlusion bursts). Frame timing and arm latency never jitter; the paddle reaches full speed at once.",
        "- The goalie never declines a target it can't reach in time (in_time is computed but not used).",
        "- The H4 timing and stop predictions see every frame with no camera latency, like sim.prediction_error.",
        "- Roll-to-roll slowdown variation is added by this study (lognormal around the learned value): SimCfg has "
        "a single slowdown.",
        "",
    ]
