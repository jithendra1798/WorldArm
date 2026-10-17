"""Charts for the task B study, drawn from the CSVs in a run folder, so they can be redrawn at any time
(`python ball_study.py --report-only <run folder>`, with any Python that has numpy and matplotlib).

matplotlib is imported only when drawing. Palette and marks follow the dataviz guide: fixed series colors
(predict blue, chase orange, the stronger chase aqua, each with its own marker shape), 95% intervals as light
bands, hairline grid, ink-colored text, one y-axis per panel, sized for a 16:9 slide.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from worldarm.ball.study import FACTORS, GOAL, MIS_HORIZON_S, MIS_WATCHED_S, Baseline, read_rows
from worldarm.ball.study_report import nearest

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
             "#1c5cab", "#184f95", "#104281", "#0d366b"]
GOALIES = {  # color, marker, legend label
    "predict": (BLUE, "o", "Predict: go to the predicted crossing"),
    "chase": (ORANGE, "s", "Chase: follow the ball"),
    "chase-sparse": (AQUA, "D", "Chase, re-aiming only every half-width"),
}
SIZE = (13.33, 7.5)          # inches: a 16:9 slide at 150 dpi
DPI = 150


def render(out: Path) -> tuple[list[Path], list[tuple[str, str]]]:
    """Draw every chart whose CSV is in `out`. Returns (PNGs written, [(chart, error)] for any that failed: one
    bad chart doesn't stop the others). Raises ImportError without matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out)
    base = _baseline(out)
    made, failed = [], []
    with plt.rc_context(_rc()):
        for draw in (_h5_sweep, _h5_needs, _h4_arrival, _h4_stop, _h4_misdecel, _robust):
            try:
                fig_path = draw(plt, out, base)
            except Exception as e:
                plt.close("all")
                failed.append((draw.__name__.lstrip("_"), repr(e)))
                continue
            if fig_path is not None:
                made.append(fig_path)
    return made, failed


def _baseline(out: Path) -> Baseline:
    meta = out / "meta.json"
    if meta.exists():
        b = json.loads(meta.read_text()).get("baseline")
        if b:
            return Baseline(**b)
    return Baseline()


def _rc() -> dict:
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    sans = [f for f in ("Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans") if f in have] or ["sans-serif"]
    return {"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
            "font.family": "sans-serif", "font.sans-serif": sans, "font.size": 13,
            "axes.edgecolor": AXIS, "axes.linewidth": 1.0, "axes.labelcolor": INK2, "axes.labelsize": 13,
            "axes.titlecolor": INK, "axes.titlesize": 14, "axes.titleweight": "semibold", "axes.titlelocation": "left",
            "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
            "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
            "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
            "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.frameon": False, "legend.fontsize": 12.5,
            "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round"}


def _head(fig, title: str, subtitle: str) -> None:
    fig.text(0.012, 0.975, title, ha="left", va="top", fontsize=19, fontweight="semibold", color=INK)
    fig.text(0.012, 0.922, subtitle, ha="left", va="top", fontsize=12.5, color=INK2, linespacing=1.4)


def _save(fig, path: Path) -> Path:
    fig.savefig(path, dpi=DPI)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def _finite(v) -> np.ndarray:
    v = np.asarray(v, float)
    return np.where(np.isfinite(v), v, np.nan)


def _series(ax, x, y, lo, hi, color: str, marker: str, label: str | None) -> None:
    """A line (thick enough for a projector) with ringed markers over its 95% interval band; a single point gets
    an error bar instead."""
    x, y, lo, hi = np.asarray(x, float), _finite(y), _finite(lo), _finite(hi)
    if len(x) > 1:
        ax.fill_between(x, lo, hi, color=color, alpha=0.14, linewidth=0, zorder=1)
    else:
        ax.errorbar(x, y, yerr=[y - lo, hi - y], color=color, lw=1.6, capsize=4, zorder=2)
    ax.plot(x, y, color=color, lw=2.4, marker=marker, ms=7.5, mec=SURFACE, mew=1.6, label=label, zorder=3)


def _percent(ax) -> None:
    from matplotlib.ticker import PercentFormatter
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))


def _goal_line(ax, label: bool) -> None:
    ax.axhline(GOAL, color=INK2, lw=1.1, ls=(0, (4, 3)), zorder=2)
    if label:
        ax.annotate(f"{GOAL:.0%} goal", (0.0, GOAL), xycoords=("axes fraction", "data"), xytext=(4, 4),
                    textcoords="offset points", fontsize=11, color=INK2, va="bottom")


def _mark_baseline(ax, x: float) -> None:
    ax.axvline(x, color=AXIS, lw=1.0, zorder=0)
    ax.annotate("baseline", (x, 0.0), xycoords=("data", "axes fraction"), xytext=(3, 4),
                textcoords="offset points", fontsize=10.5, color=MUTED, va="bottom")


def _ylim_rate(ax, rows: list[dict]) -> None:
    lows = [r["lo95"] for r in rows if math.isfinite(r["lo95"])]
    bottom = max(0.0, math.floor((min(lows) if lows else 0.0) * 10 - 0.5) / 10)
    ax.set_ylim(min(bottom, GOAL - 0.1), 1.02)


def _n(values) -> str:
    """'32' or '31-33': how many rolls a point is out of."""
    v = [x for x in values if math.isfinite(x)]
    if not v:
        return "?"
    return f"{min(v):.0f}" if min(v) == max(v) else f"{min(v):.0f}-{max(v):.0f}"


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _ranges(rows: list[dict]) -> list[tuple[float, float]]:
    return list(dict.fromkeys((r["ball_min_mm_s"], r["ball_max_mm_s"]) for r in rows))


def _h5_sweep(plt, out: Path, base: Baseline) -> Path | None:
    rows = read_rows(out / "h5_sweep.csv")
    if not rows:
        return None
    ranges = _ranges(rows)
    panels = [("lane_speed_mm_s", "Arm speed along the lane (mm/s)", ("latency_s", "half_width_mm")),
              ("latency_s", "Command latency (s)", ("lane_speed_mm_s", "half_width_mm")),
              ("half_width_mm", "Paddle half-width (mm)", ("lane_speed_mm_s", "latency_s"))]
    panels = [p for p in panels if len({r[p[0]] for r in rows}) > 1] or panels[:1]
    want = {"lane_speed_mm_s": base.lane_speed_mm_s, "latency_s": base.latency_s, "half_width_mm": base.half_width_mm}
    unit = {"lane_speed_mm_s": "{:.0f} mm/s", "latency_s": "{:.2f} s latency", "half_width_mm": "half-width {:.0f} mm"}
    fig, axes = plt.subplots(len(ranges), len(panels), figsize=SIZE, sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.79 if len(ranges) > 1 else 0.77, bottom=0.11,
                        wspace=0.08, hspace=0.62)
    n_counted = []
    for i, rng in enumerate(ranges):
        rr = [r for r in rows if (r["ball_min_mm_s"], r["ball_max_mm_s"]) == rng]
        for j, (xkey, xlabel, fixed) in enumerate(panels):
            ax = axes[i, j]
            hold = {k: nearest([r[k] for r in rr], want[k]) for k in fixed}
            sel = [r for r in rr if all(r[k] == v for k, v in hold.items())]
            for g, (color, marker, label) in GOALIES.items():
                s = sorted((r for r in sel if r["goalie"] == g), key=lambda r: r[xkey])
                if len(s) < 2 and g == "chase-sparse":
                    continue                          # only run at the baseline half-width: a lone point says little
                if s:
                    _series(ax, [r[xkey] for r in s], [r["rate"] for r in s], [r["lo95"] for r in s],
                            [r["hi95"] for r in s], color, marker, label if (i, j) == (0, 0) else None)
                    n_counted += [r["counted"] for r in s]
            _goal_line(ax, j == 0)
            _mark_baseline(ax, nearest([r[xkey] for r in rr], want[xkey]))
            _percent(ax)
            held = ", ".join(unit[k].format(v) for k, v in hold.items())
            ax.set_xlabel(f"{xlabel}\n(at {held})", fontsize=12)
            if j == 0:
                ax.set_ylabel(f"Rolls blocked\nballs {rng[0]:.0f}-{rng[1]:.0f} mm/s")
        _ylim_rate(axes[i, 0], rr)
    fig.legend(loc="upper left", bbox_to_anchor=(0.008, 0.875), ncol=3, handlelength=2.6, columnspacing=1.8)
    _head(fig, "H5 in simulation: predict vs chase",
          f"Share of rolls blocked, with 95% intervals (Wilson); {_n(n_counted)} rolls reached the lane per point. "
          "One setting varies per panel; the others stay at the baseline.")
    return _save(fig, out / "h5_sweep.png")


def _h5_needs(plt, out: Path, base: Baseline) -> Path | None:
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.ticker import PercentFormatter
    rows = [r for r in read_rows(out / "h5_sweep.csv") if r["goalie"] == "predict"]
    if not rows:
        return None
    hw = nearest([r["half_width_mm"] for r in rows], base.half_width_mm)
    rows = [r for r in rows if r["half_width_mm"] == hw]
    ranges = _ranges(rows)
    lowest = min((r["rate"] for r in rows if math.isfinite(r["rate"])), default=0.5)
    norm = Normalize(vmin=min(math.floor(lowest * 10) / 10, GOAL - 0.1), vmax=1.0)
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP[1:-1])
    fig, axes = plt.subplots(1, len(ranges), figsize=SIZE, squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.90, top=0.80, bottom=0.12, wspace=0.18)
    mesh = None
    for ax, rng in zip(axes[0], ranges):
        rr = [r for r in rows if (r["ball_min_mm_s"], r["ball_max_mm_s"]) == rng]
        speeds = sorted({r["lane_speed_mm_s"] for r in rr})
        lats = sorted({r["latency_s"] for r in rr})
        grid = np.full((len(lats), len(speeds)), np.nan)
        for r in rr:
            grid[lats.index(r["latency_s"]), speeds.index(r["lane_speed_mm_s"])] = r["rate"]
        mesh = ax.pcolormesh(np.arange(len(speeds) + 1) - 0.5, np.arange(len(lats) + 1) - 0.5,
                             np.ma.masked_invalid(grid), cmap=cmap, norm=norm, edgecolors=SURFACE, linewidth=2)
        bs, bl = nearest(speeds, base.lane_speed_mm_s), nearest(lats, base.latency_s)
        for i, lat in enumerate(lats):
            for j, spd in enumerate(speeds):
                v = grid[i, j]
                if not math.isfinite(v):
                    continue
                r_, g_, b_, _ = cmap(norm(v))
                dark = 0.2126 * r_ + 0.7152 * g_ + 0.0722 * b_ < 0.5
                is_base = (spd, lat) == (bs, bl)
                ax.text(j, i + (0.1 if is_base else 0.0), f"{100 * v:.0f}", ha="center", va="center",
                        fontsize=12.5, color="white" if dark else INK)
                if is_base:
                    ax.text(j, i - 0.22, "baseline", ha="center", va="center", fontsize=9.5,
                            color="white" if dark else INK2)
        ok = grid >= GOAL                                    # outline the region that blocks >= 90%
        for i in range(len(lats)):
            for j in range(len(speeds)):
                if not ok[i, j]:
                    continue
                if j > 0 and not ok[i, j - 1]:
                    ax.plot([j - 0.5, j - 0.5], [i - 0.5, i + 0.5], color=INK, lw=3.2, solid_capstyle="projecting")
                if j < len(speeds) - 1 and not ok[i, j + 1]:
                    ax.plot([j + 0.5, j + 0.5], [i - 0.5, i + 0.5], color=INK, lw=3.2, solid_capstyle="projecting")
                if i > 0 and not ok[i - 1, j]:
                    ax.plot([j - 0.5, j + 0.5], [i - 0.5, i - 0.5], color=INK, lw=3.2, solid_capstyle="projecting")
                if i < len(lats) - 1 and not ok[i + 1, j]:
                    ax.plot([j - 0.5, j + 0.5], [i + 0.5, i + 0.5], color=INK, lw=3.2, solid_capstyle="projecting")
        ax.set_xticks(range(len(speeds)), [f"{s:.0f}" for s in speeds])
        ax.set_yticks(range(len(lats)), [f"{lt:.2f}" for lt in lats])
        ax.set_xlabel("Arm speed along the lane (mm/s)")
        ax.set_ylabel("Command latency (s)")
        ax.set_title(f"Balls {rng[0]:.0f}-{rng[1]:.0f} mm/s", pad=10)
        ax.grid(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=0)
    cax = fig.add_axes((0.915, 0.12, 0.016, 0.68))
    cb = fig.colorbar(mesh, cax=cax)
    cb.ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    cb.outline.set_visible(False)
    cb.set_label("Rolls blocked by the predict goalie", color=INK2)
    _head(fig, "What the arm needs: share of rolls the predict goalie blocks (%)",
          f"Paddle half-width {hw:.0f} mm; {_n([r['counted'] for r in rows])} rolls per cell. The black line "
          f"encloses the arms that block at least {GOAL:.0%}.")
    return _save(fig, out / "h5_needs.png")


def _end_label(ax, x: float, y: float, text: str) -> None:
    if math.isfinite(y):
        ax.annotate(text, (x, y), xytext=(8, 0), textcoords="offset points", va="center", fontsize=12.5, color=INK)


def _h4_arrival(plt, out: Path, base: Baseline) -> Path | None:
    rows = read_rows(out / "h4_arrival.csv")
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=SIZE)
    fig.subplots_adjust(left=0.085, right=0.83, top=0.78, bottom=0.11)
    for model, color, marker, label in (("learned", BLUE, "o", f"Learned slowdown ({base.decel:.0f} mm/s²)"),
                                        ("constant velocity", ORANGE, "s", "Constant velocity")):
        s = sorted((r for r in rows if r["model"] == model), key=lambda r: r["horizon_s"])
        if not s:
            continue
        x = [r["horizon_s"] for r in s]
        _series(ax, x, [r["median_err_ms"] for r in s], [r["lo95_ms"] for r in s], [r["hi95_ms"] for r in s],
                color, marker, label)
        _end_label(ax, x[-1], s[-1]["median_err_ms"], f"{model}: {s[-1]['median_err_ms']:.0f} ms")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Time before the ball reaches the lane (s)")
    ax.set_ylabel("Arrival-time error, median (ms)")
    ax.legend(loc="upper left")
    _head(fig, "H4: when will the ball reach the lane?",
          f"Median error of the predicted arrival time, with 95% interval; {_n([r['n'] for r in rows])} rolls per point "
          "(fewer far out: only slower rolls are still that far away).\nWhere it crosses doesn't depend on the "
          "slowdown (both models agree); when it gets there does.")
    return _save(fig, out / "h4_arrival.png")


def _h4_stop(plt, out: Path, base: Baseline) -> Path | None:
    rows = sorted(read_rows(out / "h4_stop.csv"), key=lambda r: r["watched_s"])
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=SIZE)
    fig.subplots_adjust(left=0.085, right=0.80, top=0.78, bottom=0.11)
    x = [r["watched_s"] for r in rows]
    _series(ax, x, [r["median_mm"] for r in rows], [r["lo95_mm"] for r in rows], [r["hi95_mm"] for r in rows],
            BLUE, "o", f"Learned slowdown ({base.decel:.0f} mm/s²)")
    _end_label(ax, x[-1], rows[-1]["median_mm"], f"learned: {rows[-1]['median_mm']:.0f} mm")
    ax.plot(x, _finite([r["median_to_go_mm"] for r in rows]), color=MUTED, lw=2.0, zorder=2,
            label="Guess it stops where it is now (distance still to go)")
    _end_label(ax, x[-1], rows[-1]["median_to_go_mm"], f"still to go: {rows[-1]['median_to_go_mm']:.0f} mm")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("How long the roll has been watched (s)")
    ax.set_ylabel("Error in where it stops, median (mm)")
    ax.legend(loc="upper right")
    ax.annotate("Constant velocity can't predict a stop: it says the ball never stops.", (0.99, 0.80),
                xycoords="axes fraction", ha="right", fontsize=12.5, color=INK2)
    _head(fig, "H4: where will a slow roll stop?",
          f"Rolls of 150-450 mm/s, predicted while still rolling; median with 95% interval; "
          f"{_n([r['n_moving'] for r in rows])} rolls per point (fewer later on: the slowest have stopped).\n"
          "The tracker fits the last 0.6 s of sightings, so past 0.6 s the gain comes from being closer to the stop.")
    return _save(fig, out / "h4_stop.png")


def _h4_misdecel(plt, out: Path, base: Baseline) -> Path | None:
    rows = sorted(read_rows(out / "h4_misdecel.csv"), key=lambda r: r["decel_error_pct"])
    if not rows:
        return None
    cv = next((r for r in rows if r["decel_error_pct"] == -100), None)
    mis = [r for r in rows if r["decel_error_pct"] != -100]
    fig, axes = plt.subplots(1, 3, figsize=SIZE)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.71, bottom=0.13, wspace=0.28)
    x = [r["decel_error_pct"] for r in mis]
    a, b, c = axes
    _series(a, x, [r["arrival_median_ms"] for r in mis], [r["arrival_lo95_ms"] for r in mis],
            [r["arrival_hi95_ms"] for r in mis], BLUE, "o", "Learned model, believing this slowdown")
    _series(b, x, [r["stop_median_mm"] for r in mis], [r["stop_lo95_mm"] for r in mis],
            [r["stop_hi95_mm"] for r in mis], BLUE, "o", None)
    _series(c, x, [r["rate"] for r in mis], [r["lo95"] for r in mis], [r["hi95"] for r in mis], BLUE, "o", None)
    if cv is not None:
        a.axhline(cv["arrival_median_ms"], color=ORANGE, lw=2.0, ls=(0, (5, 3)), label="Constant velocity")
        c.axhline(cv["rate"], color=ORANGE, lw=2.0, ls=(0, (5, 3)))
        _note(a, f"constant velocity: {cv['arrival_median_ms']:.0f} ms", cv["arrival_median_ms"])
        _note(c, f"constant velocity: {100 * cv['rate']:.0f}%", cv["rate"])
    b.annotate("constant velocity: never stops", (0.97, 0.95), xycoords="axes fraction", fontsize=11.5, color=INK2,
               va="top", ha="right")
    a.set_title(f"When it arrives ({MIS_HORIZON_S:.1f} s before)", pad=12)
    b.set_title(f"Where a slow roll stops ({MIS_WATCHED_S:.1f} s in)", pad=12)
    c.set_title("Rolls the predict goalie blocks", pad=12)
    a.set_ylabel("Arrival-time error, median (ms)")
    b.set_ylabel("Stop-point error, median (mm)")
    c.set_ylabel("Rolls blocked")
    _percent(c)
    _goal_line(c, True)
    _ylim_rate(c, mis + ([cv] if cv else []))
    for ax in axes:
        ax.axvline(0, color=AXIS, lw=1.0, zorder=0)
        ax.set_xlabel("Believed slowdown, % off the true value")
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:+.0f}%" if v else "0%")
    a.set_ylim(bottom=0)
    b.set_ylim(bottom=0)
    fig.legend(loc="upper left", bbox_to_anchor=(0.008, 0.865), ncol=2, handlelength=2.6)
    _head(fig, "H4: what a mis-learned slowdown costs",
          f"The table slows the ball at {base.decel:.0f} mm/s²; the model believes 50% less to 50% more. Medians and "
          f"rates with 95% intervals; {_n([r['arrival_n'] for r in mis])} rolls (timing), "
          f"{_n([r['counted'] for r in mis])} (blocking).")
    return _save(fig, out / "h4_misdecel.png")


def _note(ax, text: str, y: float) -> None:
    if math.isfinite(y):
        ax.annotate(text, (1.0, y), xycoords=("axes fraction", "data"), xytext=(-4, 5), textcoords="offset points",
                    ha="right", va="bottom", fontsize=11, color=INK2)


def _robust(plt, out: Path, base: Baseline) -> Path | None:
    rows = read_rows(out / "robust.csv")
    if not rows:
        return None
    factors = list(dict.fromkeys(r["factor"] for r in rows))
    fig, axes = plt.subplots(2, 3, figsize=SIZE, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.80, bottom=0.10, wspace=0.10, hspace=0.55)
    flat = axes.ravel()
    for ax, factor in zip(flat, factors):
        fr = [r for r in rows if r["factor"] == factor]
        for g in ("predict", "chase"):
            color, marker, label = GOALIES[g]
            s = sorted((r for r in fr if r["goalie"] == g), key=lambda r: r["level"])
            if s:
                _series(ax, [r["level"] for r in s], [r["rate"] for r in s], [r["lo95"] for r in s],
                        [r["hi95"] for r in s], color, marker, label if ax is flat[0] else None)
        base_level = next((r["level"] for r in fr if r["is_baseline"] == 1), None)
        if base_level is not None:
            _mark_baseline(ax, base_level)
        _goal_line(ax, ax is flat[0])
        _percent(ax)
        ax.set_xlabel(_cap(FACTORS.get(factor, factor)), fontsize=12)
    for ax in axes[:, 0]:
        ax.set_ylabel("Rolls blocked")
    _ylim_rate(flat[0], rows)
    for ax in flat[len(factors):]:
        ax.set_visible(False)
    fig.legend(loc="upper left", bbox_to_anchor=(0.008, 0.875), ncol=2, handlelength=2.6)
    _head(fig, "Robustness: one camera or roll setting made worse at a time",
          f"Baseline arm ({base.lane_speed_mm_s:.0f} mm/s, {base.latency_s:.2f} s), half-width {base.half_width_mm:.0f} "
          f"mm; 95% intervals (Wilson); {_n([r['counted'] for r in rows])} rolls reached the lane per point.")
    return _save(fig, out / "robust.png")
