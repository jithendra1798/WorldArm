"""Task B in simulation: the numbers behind H4 and H5 for the slides (summary.md: study_report.py; charts:
study_charts.py).

  h4      When will it arrive? Arrival-time error vs time before arrival, learned slowdown vs constant velocity.
          Where will it stop? Error vs how long the roll was watched. What does a mis-learned slowdown cost?
  h5      Intercept rate, predict vs chase, over arm lane speed x command latency x paddle half-width x ball speed.
  needs   From the h5 sweep: the slowest lane speed and the longest latency that still block >= 90%.
  robust  The predict goalie with a worse camera (noise, missed frames, latency, fps) and uneven rolls.

One process by design (no multiprocessing). Rows go to CSV as they finish; summary.md and the charts are
rebuilt from the CSVs in the run folder after each study, so a partial run is still useful.

Roll i of a study is drawn from its own generator (seed, i), so every arm, goalie and camera setting sees the
same rolls: predict and chase are compared on identical rolls (paired), and sweeps move smoothly.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from worldarm.ball.intercept import ArmModel, Goalie, Lane
from worldarm.ball.model import Roll, RollModel
from worldarm.ball.sim import SimCfg, roll_once
from worldarm.ball.track import Estimate, Tracker
from worldarm.metrics import wilson

STUDIES = ("h4", "h5", "needs", "robust")           # the order they run in
GOAL = 0.90                                          # "blocks at least 90% of rolls"
MIS_HORIZON_S = 1.0                                  # mis-learned slowdown: arrival error this long before arrival,
MIS_WATCHED_S = 0.5                                  # and stop error after watching the roll this long
SLOW_ROLLS = (150.0, 450.0)                          # mm/s: the slow rolls of sim.stop_error (mirrored)
ROLL_S = 0.0044                                      # s of compute per baseline roll on the dev laptop (for estimates)
FIT_COST = 0.01                                      # one tracker fit, in baseline rolls (for the ETA)
CSVS = {"h4": ("h4_arrival.csv", "h4_stop.csv", "h4_misdecel.csv"), "h5": ("h5_sweep.csv",),
        "needs": ("needs.csv",), "robust": ("robust.csv",)}
FACTORS = {                                          # robust: what each factor means
    "noise_mm": "camera noise (mm, 1 sd)",
    "miss_prob": "missed frames (share)",
    "camera_latency_s": "camera latency (s)",
    "fps": "camera frame rate (fps)",
    "decel_spread": "slowdown varies roll to roll (sd, share of learned)",
}


@dataclass(frozen=True)
class Plan:
    """What to simulate, and how many rolls per point."""
    name: str
    lane_speeds: tuple[float, ...]                   # h5: paddle speed along the lane, mm/s
    latencies: tuple[float, ...]                     # h5: command -> paddle starts moving, s
    half_widths: tuple[float, ...]                   # h5: paddle half-width, mm
    ball_speeds: tuple[tuple[float, float], ...]     # h5: starting-speed ranges, mm/s
    h5_n: int                                        # h5: rolls per cell
    horizons: tuple[float, ...]                      # h4: s before arrival
    watched: tuple[float, ...]                       # h4: s of the roll seen before predicting where it stops
    decel_errors: tuple[float, ...]                  # h4: believed slowdown off by this share (-0.5 = half)
    h4_n: int                                        # h4: rolls per point
    h4_rate_n: int                                   # h4: rolls per intercept rate, mis-learned slowdown
    robust: dict                                     # robust: factor -> levels
    robust_n: int                                    # robust: rolls per level


def _steps(a: float, b: float, step: float) -> tuple[float, ...]:
    return tuple(round(a + i * step, 3) for i in range(int(round((b - a) / step)) + 1))


FULL = Plan("full", lane_speeds=(100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 500.0),
            latencies=(0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.6), half_widths=(40.0, 60.0, 80.0, 100.0),
            ball_speeds=((350.0, 900.0), (700.0, 1200.0)), h5_n=700,
            horizons=_steps(0.2, 2.0, 0.1), watched=_steps(0.2, 2.0, 0.1), decel_errors=_steps(-0.5, 0.5, 0.1),
            h4_n=4000, h4_rate_n=1500,
            robust={"noise_mm": (2.0, 4.0, 6.0, 8.0, 10.0), "miss_prob": (0.0, 0.05, 0.1, 0.2, 0.3),
                    "camera_latency_s": (0.03, 0.06, 0.1, 0.15, 0.2), "fps": (10.0, 15.0, 20.0, 23.0, 30.0),
                    "decel_spread": (0.0, 0.1, 0.2, 0.3, 0.4)}, robust_n=1500)

QUICK = Plan("quick", lane_speeds=(100.0, 175.0, 250.0, 400.0), latencies=(0.1, 0.3, 0.45, 0.6), half_widths=(60.0,),
             ball_speeds=((350.0, 900.0),), h5_n=50,
             horizons=(0.2, 0.5, 1.0, 1.5, 2.0), watched=(0.2, 0.5, 1.0, 1.5, 2.0),
             decel_errors=(-0.5, -0.25, 0.0, 0.25, 0.5), h4_n=300, h4_rate_n=60,
             robust={"noise_mm": (2.0, 4.0, 10.0), "miss_prob": (0.0, 0.05, 0.3), "camera_latency_s": (0.03, 0.06, 0.2),
                     "fps": (10.0, 23.0, 30.0), "decel_spread": (0.0, 0.4)}, robust_n=50)


@dataclass(frozen=True)
class Baseline:
    """The setup every sweep varies around: the 'realistic arm' of tests/test_ball.py on a 50 cm lane."""
    lane_length_mm: float = 500.0
    lane_speed_mm_s: float = 250.0
    latency_s: float = 0.45
    half_width_mm: float = 60.0                      # BallCfg.paddle_half_width_mm
    decel: float = SimCfg().true_decel               # mm/s^2: the table's slowdown, and what the goalie learned
    seed: int = 2026

    def lane(self) -> Lane:
        h = self.lane_length_mm / 2.0
        return Lane(np.array([450.0, -h]), np.array([450.0, h]), 120.0, 0.0, home=np.zeros(2))

    def arm(self) -> ArmModel:
        return ArmModel(latency_s=self.latency_s, speed_mm_s=self.lane_speed_mm_s)

    def sim(self, **changes) -> SimCfg:
        return dataclasses.replace(SimCfg(true_decel=self.decel), **changes)


def seeds(base: Baseline) -> dict[str, int]:
    """Every generator seed a study uses (recorded in meta.json and in each CSV row)."""
    s = base.seed
    return {"h4_arrival": s + 41, "h4_stop": s + 42, "h4_misdecel_arrival": s + 43, "h4_misdecel_stop": s + 44,
            "h4_misdecel_rate": s + 45, "h5": s + 50, "robust": s + 60}


# ---------------------------------------------------------------------------------------------- simulation

@dataclass
class Rolls:
    """One goalie's per-roll outcomes on one set of simulated rolls."""
    counted: np.ndarray      # reached the lane within the paddle's reach: the rolls a rate is out of
    defended: np.ndarray
    s_cross: np.ndarray      # mm along the lane where it crossed (nan: stopped short)
    lead_s: np.ndarray       # first paddle command, s before the crossing (nan: none)
    commands: np.ndarray     # paddle commands sent

    @property
    def n(self) -> int:
        return int(self.counted.sum())

    @property
    def k(self) -> int:
        return int((self.defended & self.counted).sum())


def simulate_rolls(lane: Lane, mode: str, believed_decel: float, arm: ArmModel, cfg: SimCfg, n: int, seed: int,
                   half_width_mm: float, retarget_mm: float | None = None, decel_spread: float = 0.0,
                   per_roll: bool = True) -> Rolls:
    """n rolls through sim.roll_once against one goalie, with per-roll outcomes.

    Roll i is drawn from its own generator (seed, i), so every setting sees the same rolls. per_roll=False shares
    one generator across rolls exactly as sim.intercept_rate does (same seed, same numbers). decel_spread > 0
    makes the true slowdown vary roll to roll (lognormal, this sd in log terms, median cfg.true_decel) while the
    goalie keeps believing believed_decel. (SimCfg.decel_spread would draw it from the roll's own generator and
    shift every later draw; drawn here from a separate one, the rolls stay the same at every spread.)
    """
    goalie = Goalie(lane, RollModel(believed_decel), arm, half_width_mm, mode, retarget_mm)
    tracker = Tracker(decel=believed_decel)
    shared = None if per_roll else np.random.default_rng(seed)
    cols: dict[str, list] = {"counted": [], "defended": [], "s_cross": [], "lead_s": [], "commands": []}
    for i in range(n):
        rng = np.random.default_rng([seed, i]) if per_roll else shared
        c = cfg
        if decel_spread > 0:
            z = float(np.random.default_rng([seed, i, 1]).standard_normal())
            c = dataclasses.replace(cfg, true_decel=cfg.true_decel * math.exp(decel_spread * z))
        o = roll_once(lane, goalie, arm, tracker, c, rng)
        cols["counted"].append(o.reached and o.s_cross is not None
                               and -half_width_mm <= o.s_cross <= lane.length + half_width_mm)  # as intercept_rate
        cols["defended"].append(bool(o.defended))
        cols["s_cross"].append(math.nan if o.s_cross is None else o.s_cross)
        cols["lead_s"].append(math.nan if o.first_command_s is None else o.first_command_s)
        cols["commands"].append(o.commands)
    return Rolls(np.array(cols["counted"], bool), np.array(cols["defended"], bool), np.array(cols["s_cross"], float),
                 np.array(cols["lead_s"], float), np.array(cols["commands"], int))


def _sightings(true_model: RollModel, roll: Roll, t_until: float, cfg: SimCfg,
               rng: np.random.Generator) -> list[tuple[float, np.ndarray]] | None:
    """0.6 s of noisy sightings ending at t_until, drawn exactly as sim._observe draws them (None: too early)."""
    if t_until < 0.15:
        return None
    return [(tf, true_model.at(roll, tf) + rng.normal(0, cfg.noise_mm, 2))
            for tf in np.arange(max(0.0, t_until - 0.6), t_until, 1.0 / cfg.fps)]


def _fit(seen: list[tuple[float, np.ndarray]], decel: float) -> Estimate | None:
    tracker = Tracker(decel=decel)
    for tf, xy in seen:
        tracker.add(tf, xy)
    return tracker.estimate()


def arrival_errors(cfg: SimCfg, lane: Lane, decels: Sequence[float], n: int, seed: int,
                   horizons: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """How far off the predicted arrival time at the lane is, for each believed slowdown, by how long before the
    arrival the prediction is made. Same rolls and camera noise as sim.prediction_error(seed=seed); every believed
    slowdown sees the same sightings (paired).

    Returns (err_s, where_mm), each [slowdown, roll, horizon]. err nan: too early to have seen enough (or the
    ball stopped short); inf: it predicted the ball would stop short, but it arrived (a failure, as in
    sim.prediction_error). where: how far off the predicted crossing point is.
    """
    rng = np.random.default_rng(seed)
    true_model = RollModel(cfg.true_decel)
    models = [RollModel(d) for d in decels]
    err = np.full((len(decels), n, len(horizons)), np.nan)
    where = np.full_like(err, np.nan)
    for i in range(n):
        aim = lane.point(rng.uniform(0, lane.length))
        start = aim - lane.normal * rng.uniform(*cfg.start_distance)
        u = (aim - start) / np.linalg.norm(aim - start)
        roll = Roll(0.0, start, u, rng.uniform(*cfg.speed))
        hit = true_model.crossing(roll, lane.a, lane.normal)
        if hit is None:
            continue
        for j, h in enumerate(horizons):
            seen = _sightings(true_model, roll, hit[1] - h, cfg, rng)
            if seen is None:
                continue
            for k, m in enumerate(models):
                est = _fit(seen, m.decel)
                if est is None:
                    continue
                pred = m.crossing(est.roll, lane.a, lane.normal)
                if pred is None:
                    err[k, i, j] = math.inf
                else:
                    err[k, i, j] = abs(pred[1] - hit[1])
                    where[k, i, j] = abs(lane.along(pred[0]) - lane.along(hit[0]))
    return err, where


def stop_errors(cfg: SimCfg, decels: Sequence[float], n: int, seed: int,
                watched: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where a slow roll will stop, predicted after watching it for w seconds, for each believed slowdown (> 0).
    For each w: the same rolls and noise as sim.stop_error(after_s=w, seed=seed).

    Returns err_mm [slowdown, w, roll] (nan: no estimate), moving [w, roll] (still rolling at w: predicting
    where a stopped ball stops is trivial) and to_go_mm [w, roll] (how far it still had to roll).
    """
    true_model = RollModel(cfg.true_decel)
    models = [RollModel(d) for d in decels]
    err = np.full((len(decels), len(watched), n), np.nan)
    moving = np.zeros((len(watched), n), bool)
    to_go = np.full((len(watched), n), np.nan)
    for j, w in enumerate(watched):
        rng = np.random.default_rng(seed)
        for i in range(n):
            a = rng.uniform(-0.6, 0.6)
            roll = Roll(0.0, np.zeros(2), np.array([np.cos(a), np.sin(a)]), rng.uniform(*SLOW_ROLLS))
            seen = _sightings(true_model, roll, w, cfg, rng)
            if seen is None:
                continue
            stop = true_model.stop_point(roll)
            moving[j, i] = w < true_model.stop_time(roll)
            to_go[j, i] = float(np.linalg.norm(stop - true_model.at(roll, w)))
            for k, m in enumerate(models):
                est = _fit(seen, m.decel)
                if est is not None:
                    err[k, j, i] = float(np.linalg.norm(m.stop_point(est.roll) - stop))
    return err, moving, to_go


# ----------------------------------------------------------------------------------------------- statistics

def median_ci(x, z: float = 1.96) -> tuple[float, float, float]:
    """Median with a distribution-free 95% interval (order statistics). nan values are dropped; inf values
    (failed predictions) stay in as the worst errors."""
    x = np.sort(np.asarray(x, float))
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return math.nan, math.nan, math.nan
    lo = max(int(math.floor((n - z * math.sqrt(n)) / 2)), 1)
    hi = min(int(math.ceil(1 + (n + z * math.sqrt(n)) / 2)), n)
    return float(np.median(x)), float(x[lo - 1]), float(x[hi - 1])


def gain_ci(a, b, paired: bool = True) -> tuple[float, float, float]:
    """Success rate of a minus that of b, with Newcombe's 95% interval. Paired (the same rolls, in order) uses
    their correlation (Newcombe 1998, method 10); unpaired treats them as independent samples."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if len(a) == 0 or len(b) == 0:
        return math.nan, math.nan, math.nan
    p1, p2 = float(a.mean()), float(b.mean())
    l1, u1 = wilson(int(a.sum()), len(a))
    l2, u2 = wilson(int(b.sum()), len(b))
    phi = 0.0
    if paired:
        both, only_a = int(np.sum(a & b)), int(np.sum(a & ~b))
        only_b, neither = int(np.sum(~a & b)), int(np.sum(~a & ~b))
        den = math.sqrt((both + only_a) * (only_b + neither) * (both + only_b) * (only_a + neither))
        phi = (both * neither - only_a * only_b) / den if den > 0 else 0.0
    d = p1 - p2
    lo = d - math.sqrt(max((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2, 0.0))
    hi = d + math.sqrt(max((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2, 0.0))
    return d, max(-1.0, lo), min(1.0, hi)


def _nanmedian(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else math.nan


def _p90(x) -> float:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return float(np.percentile(x, 90, method="higher")) if len(x) else math.nan


def rate_row(r: Rolls) -> dict:
    lo, hi = wilson(r.k, r.n)
    return {"rolls": len(r.counted), "counted": r.n, "defended": r.k, "rate": r.k / r.n if r.n else math.nan,
            "lo95": lo, "hi95": hi, "median_lead_s": _nanmedian(r.lead_s[r.counted]),
            "mean_commands": float(r.commands[r.counted].mean()) if r.n else math.nan}


def gain_row(pred: Rolls, other: Rolls) -> dict:
    """How much more the predict goalie blocked than `other`, paired when they faced the same counted rolls."""
    paired = bool(np.array_equal(pred.counted, other.counted))
    g, lo, hi = gain_ci(pred.defended[pred.counted], other.defended[other.counted], paired)
    return {"predict_gain": g, "gain_lo95": lo, "gain_hi95": hi, "paired": paired}


# --------------------------------------------------------------------------------------------------- output

class Table:
    """A CSV written a row at a time and flushed, so an interrupted run keeps every finished row."""

    def __init__(self, path: Path, columns: Sequence[str]):
        self.columns = list(columns)
        self._f = open(path, "w", newline="")
        self._w = csv.DictWriter(self._f, self.columns)
        self._w.writeheader()
        self._f.flush()

    def add(self, row: dict) -> None:
        self._w.writerow({c: _fmt(row.get(c)) for c in self.columns})
        self._f.flush()

    def __enter__(self) -> Table:
        return self

    def __exit__(self, *exc) -> None:
        self._f.close()


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "1" if v else "0"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return "" if math.isnan(v) else ("inf" if v == math.inf else f"{v:.6g}")
    return str(v)


def read_rows(path: Path) -> list[dict]:
    """A study CSV back as dicts: numbers as floats (blank = nan), words as str."""
    if not Path(path).exists():
        return []
    with open(path, newline="") as f:
        return [{k: _parse(v) for k, v in row.items()} for row in csv.DictReader(f)]


def _parse(v: str):
    if v == "":
        return math.nan
    try:
        return float(v)
    except ValueError:
        return v


class Progress:
    """Prints the share done, elapsed time and an ETA, at most every `every_s` seconds. Work is counted in
    baseline rolls (a rough cost model), and the ETA uses the speed measured so far."""

    def __init__(self, total: float, say: Callable[[str], None] = print, every_s: float = 15.0):
        self.total, self.done, self.say, self.every = max(total, 1e-9), 0.0, say, every_s
        self.t0, self.last = time.monotonic(), time.monotonic()

    def add(self, units: float, what: str) -> None:
        self.done += units
        now = time.monotonic()
        if now - self.last >= self.every:
            self.last = now
            spent = now - self.t0
            eta = spent / self.done * max(self.total - self.done, 0.0) if self.done > 0 else math.nan
            self.say(f"  {min(self.done / self.total, 1.0):6.1%}  elapsed {hms(spent)}  ETA {hms(eta)}  ({what})")


def hms(s: float) -> str:
    if not math.isfinite(s):
        return "?"
    s = int(round(s))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _sim_seconds(cfg: SimCfg) -> float:
    """Average simulated seconds per roll (straight rolls; good enough for an ETA)."""
    m, ts = RollModel(cfg.true_decel), []
    for v in np.linspace(*cfg.speed, 9):
        for d in np.linspace(*cfg.start_distance, 5):
            r = Roll(0.0, np.zeros(2), np.array([1.0, 0.0]), float(v))
            hit = m.crossing(r, np.array([d, 0.0]), np.array([1.0, 0.0]))
            ts.append(hit[1] + 0.8 if hit else m.stop_time(r) + 0.3)
    return float(np.mean(ts))


def roll_cost(cfg: SimCfg) -> float:
    """Compute cost of one simulated roll, in baseline rolls: mostly camera frames, so time rolled x fps."""
    return _sim_seconds(cfg) / _sim_seconds(SimCfg()) * (0.12 + 0.88 * cfg.fps / 23.0)


def planned_cost(plan: Plan, base: Baseline, study: str) -> float:
    """How much work a study is, in baseline rolls."""
    if study == "h5":
        cells = len(plan.lane_speeds) * len(plan.latencies)
        cost = sum(roll_cost(base.sim(speed=tuple(s))) for s in plan.ball_speeds) * cells * len(plan.half_widths) * 2
        if base.half_width_mm in plan.half_widths and plan.ball_speeds:   # the extra chase variant
            cost += roll_cost(base.sim(speed=tuple(plan.ball_speeds[0]))) * cells
        return cost * plan.h5_n
    if study == "h4":
        k = len(set(plan.decel_errors) | {0.0})
        fits = plan.h4_n * (len(plan.horizons) * 3 + len(plan.watched) * 2 + (k + 1) * 1.5 + k * 1.5)
        return fits * FIT_COST + (k + 1) * plan.h4_rate_n * roll_cost(base.sim())
    if study == "robust":
        cfgs = {key: cfg for _, _, cfg, _, key in robust_settings(plan, base)}
        return sum(2 * plan.robust_n * roll_cost(cfg) for cfg in cfgs.values())
    return 0.0


def robust_settings(plan: Plan, base: Baseline):
    """(factor, level, SimCfg, slowdown spread, key) for each robustness level; equal keys are the same run (each
    factor's baseline level is the baseline itself)."""
    ref = base.sim()
    for factor, levels in plan.robust.items():
        for level in levels:
            spread = float(level) if factor == "decel_spread" else 0.0
            cfg = ref if factor == "decel_spread" else dataclasses.replace(ref, **{factor: level})
            yield factor, level, cfg, spread, (dataclasses.astuple(cfg), spread)


# --------------------------------------------------------------------------------------------------- studies

H5_COLS = ["ball_min_mm_s", "ball_max_mm_s", "lane_speed_mm_s", "latency_s", "half_width_mm", "goalie", "rolls",
           "counted", "defended", "rate", "lo95", "hi95", "predict_gain", "gain_lo95", "gain_hi95", "paired",
           "median_lead_s", "mean_commands", "seed"]
ARRIVAL_COLS = ["model", "believed_decel", "horizon_s", "n", "no_cross", "median_err_ms", "lo95_ms", "hi95_ms",
                "p90_err_ms", "median_where_mm", "closer_than_cv", "closer_n", "closer_share", "closer_lo95",
                "closer_hi95", "seed"]
STOP_COLS = ["watched_s", "believed_decel", "n_all", "median_all_mm", "n_moving", "median_mm", "lo95_mm", "hi95_mm",
             "p90_mm", "median_to_go_mm", "seed"]
MIS_COLS = ["decel_error_pct", "believed_decel", "arrival_n", "arrival_median_ms", "arrival_lo95_ms",
            "arrival_hi95_ms", "arrival_no_cross", "stop_n", "stop_median_mm", "stop_lo95_mm", "stop_hi95_mm",
            "counted", "defended", "rate", "lo95", "hi95", "rate_change", "change_lo95", "change_hi95", "seed"]
ROBUST_COLS = ["factor", "level", "is_baseline", "goalie", "rolls", "counted", "defended", "rate", "lo95", "hi95",
               "predict_gain", "gain_lo95", "gain_hi95", "paired", "median_lead_s", "mean_commands", "seed"]
NEEDS_COLS = ["ball_min_mm_s", "ball_max_mm_s", "half_width_mm", "given", "given_value", "requirement",
              "by_estimate", "by_lower_bound"]


def run_h4(plan: Plan, base: Baseline, out: Path, progress: Progress) -> None:
    """H4: arrival time (learned slowdown vs constant velocity), where it stops, and a mis-learned slowdown."""
    lane, cfg, d, sd = base.lane(), base.sim(), base.decel, seeds(base)

    err, where = arrival_errors(cfg, lane, (d, 0.0), plan.h4_n, sd["h4_arrival"], plan.horizons)
    with Table(out / "h4_arrival.csv", ARRIVAL_COLS) as t:
        for k, (model, believed) in enumerate((("learned", d), ("constant velocity", 0.0))):
            for j, h in enumerate(plan.horizons):
                e = err[k, :, j]
                seen = ~np.isnan(e)
                med, lo, hi = median_ci(e[seen])
                row = {"model": model, "believed_decel": believed, "horizon_s": h, "n": int(seen.sum()),
                       "no_cross": int(np.isinf(e).sum()), "median_err_ms": 1000 * med, "lo95_ms": 1000 * lo,
                       "hi95_ms": 1000 * hi, "p90_err_ms": 1000 * _p90(e[seen]),
                       "median_where_mm": _nanmedian(where[k, :, j]), "seed": sd["h4_arrival"]}
                if k == 0:                                   # paired with constant velocity: same sightings
                    both = seen & ~np.isnan(err[1, :, j])
                    closer, m = int(np.sum(e[both] < err[1, both, j])), int(both.sum())
                    c_lo, c_hi = wilson(closer, m)
                    row |= {"closer_than_cv": closer, "closer_n": m, "closer_share": closer / m if m else math.nan,
                            "closer_lo95": c_lo, "closer_hi95": c_hi}
                t.add(row)
    progress.add(plan.h4_n * len(plan.horizons) * 3 * FIT_COST, "h4: arrival time")

    err_s, moving, to_go = stop_errors(cfg, (d,), plan.h4_n, sd["h4_stop"], plan.watched)
    with Table(out / "h4_stop.csv", STOP_COLS) as t:
        for j, w in enumerate(plan.watched):
            e = err_s[0, j]
            mv = ~np.isnan(e) & moving[j]
            med, lo, hi = median_ci(e[mv])
            t.add({"watched_s": w, "believed_decel": d, "n_all": int((~np.isnan(e)).sum()),
                   "median_all_mm": _nanmedian(e), "n_moving": int(mv.sum()), "median_mm": med, "lo95_mm": lo,
                   "hi95_mm": hi, "p90_mm": _p90(e[mv]), "median_to_go_mm": _nanmedian(to_go[j][mv]),
                   "seed": sd["h4_stop"]})
    progress.add(plan.h4_n * len(plan.watched) * 2 * FIT_COST, "h4: where it stops")

    errors = sorted(set(plan.decel_errors) | {0.0})
    believed = [d * (1 + e) for e in errors]
    err_m, _ = arrival_errors(cfg, lane, believed + [0.0], plan.h4_n, sd["h4_misdecel_arrival"], (MIS_HORIZON_S,))
    err_st, moving_st, _ = stop_errors(cfg, [b for b in believed if b > 0], plan.h4_n, sd["h4_misdecel_stop"],
                                       (MIS_WATCHED_S,))
    progress.add(plan.h4_n * (len(believed) * 3 + 1.5) * FIT_COST, "h4: mis-learned slowdown")
    rates = []
    for b in believed + [0.0]:
        rates.append(simulate_rolls(lane, "predict", b, base.arm(), cfg, plan.h4_rate_n, sd["h4_misdecel_rate"],
                                    base.half_width_mm))
        progress.add(plan.h4_rate_n * roll_cost(cfg), f"h4: goalie believing {b:.0f} mm/s^2")
    ref = rates[errors.index(0.0)]
    stop_k = 0
    with Table(out / "h4_misdecel.csv", MIS_COLS) as t:
        for k, (e_pct, b) in enumerate(zip([100 * e for e in errors] + [-100.0], believed + [0.0])):
            a = err_m[k, :, 0]
            a_med, a_lo, a_hi = median_ci(a)
            row = {"decel_error_pct": e_pct, "believed_decel": b, "arrival_n": int((~np.isnan(a)).sum()),
                   "arrival_median_ms": 1000 * a_med, "arrival_lo95_ms": 1000 * a_lo, "arrival_hi95_ms": 1000 * a_hi,
                   "arrival_no_cross": int(np.isinf(a).sum()), "seed": sd["h4_misdecel_rate"]}
            if b > 0:                                        # constant velocity never predicts a stop
                s = err_st[stop_k, 0][~np.isnan(err_st[stop_k, 0]) & moving_st[0]]
                stop_k += 1
                s_med, s_lo, s_hi = median_ci(s)
                row |= {"stop_n": len(s), "stop_median_mm": s_med, "stop_lo95_mm": s_lo, "stop_hi95_mm": s_hi}
            r = rates[k]
            g = gain_row(r, ref)
            row |= {k2: v for k2, v in rate_row(r).items() if k2 in ("counted", "defended", "rate", "lo95", "hi95")}
            row |= {"rate_change": g["predict_gain"], "change_lo95": g["gain_lo95"], "change_hi95": g["gain_hi95"]}
            t.add(row)


def run_h5(plan: Plan, base: Baseline, out: Path, progress: Progress) -> None:
    """H5: predict vs chase over lane speed x latency x half-width x ball speed. The baseline ball speeds and
    half-width go first, so an interrupted run still has the main slice. There, a stronger chase that re-aims only
    after the ball moved a full half-width (fewer interruptions of the arm) is run too, to check the baseline."""
    lane, sd = base.lane(), seeds(base)["h5"]
    widths = sorted(plan.half_widths, key=lambda w: (w != base.half_width_mm, w))
    with Table(out / "h5_sweep.csv", H5_COLS) as t:
        for bi, speed in enumerate(plan.ball_speeds):
            cfg = base.sim(speed=tuple(speed))
            cost = roll_cost(cfg) * plan.h5_n
            for hw in widths:
                for lat in plan.latencies:
                    for spd in plan.lane_speeds:
                        arm = ArmModel(latency_s=lat, speed_mm_s=spd)
                        goalies: dict[str, tuple[str, float | None]] = {"predict": ("predict", None),
                                                                        "chase": ("chase", None)}
                        if bi == 0 and hw == base.half_width_mm:
                            goalies["chase-sparse"] = ("chase", hw)
                        res = {g: simulate_rolls(lane, mode, base.decel, arm, cfg, plan.h5_n, sd, hw, retarget_mm=rt)
                               for g, (mode, rt) in goalies.items()}
                        cell = {"ball_min_mm_s": speed[0], "ball_max_mm_s": speed[1], "lane_speed_mm_s": spd,
                                "latency_s": lat, "half_width_mm": hw, "seed": sd}
                        for g, r in res.items():
                            t.add(cell | {"goalie": g} | rate_row(r) | ({} if g == "predict" else gain_row(res["predict"], r)))
                        progress.add(cost * len(res), f"h5: balls {speed[0]:.0f}-{speed[1]:.0f} mm/s, half-width "
                                                      f"{hw:.0f} mm, latency {lat:.2f} s, lane {spd:.0f} mm/s")


def run_robust(plan: Plan, base: Baseline, out: Path, progress: Progress) -> None:
    """The predict goalie (and chase, for reference) with one camera or roll setting made worse at a time."""
    lane, arm, hw, sd, ref = base.lane(), base.arm(), base.half_width_mm, seeds(base)["robust"], base.sim()
    done: dict = {}
    with Table(out / "robust.csv", ROBUST_COLS) as t:
        for factor, level, cfg, spread, key in robust_settings(plan, base):
            if key not in done:
                done[key] = {m: simulate_rolls(lane, m, base.decel, arm, cfg, plan.robust_n, sd, hw,
                                               decel_spread=spread) for m in ("predict", "chase")}
                progress.add(2 * plan.robust_n * roll_cost(cfg), f"robust: {factor} = {level:g}")
            res = done[key]
            is_base = level == (0.0 if factor == "decel_spread" else getattr(ref, factor))
            for g, r in res.items():
                t.add({"factor": factor, "level": level, "is_baseline": is_base, "goalie": g, "seed": sd}
                      | rate_row(r) | ({} if g == "predict" else gain_row(res["predict"], r)))


def arm_needs(h5_rows: list[dict], goal: float = GOAL) -> list[dict]:
    """From the h5 predict rows, for each ball-speed range and paddle half-width: the slowest tested lane speed that
    blocks >= goal at each latency (every faster tested speed must too), and the longest tested latency that does
    at each lane speed (every shorter one must too). By the estimate, and stricter, by its 95% lower bound."""
    pred = [r for r in h5_rows if r["goalie"] == "predict"]
    out = []
    for key in sorted({(r["ball_min_mm_s"], r["ball_max_mm_s"], r["half_width_mm"]) for r in pred}):
        cells = {(r["lane_speed_mm_s"], r["latency_s"]): r for r in pred
                 if (r["ball_min_mm_s"], r["ball_max_mm_s"], r["half_width_mm"]) == key}
        speeds, lats = sorted({s for s, _ in cells}), sorted({lt for _, lt in cells})
        given = [("latency_s", lt, "min_lane_speed_mm_s", [(s, cells.get((s, lt))) for s in reversed(speeds)])
                 for lt in lats]
        given += [("lane_speed_mm_s", s, "max_latency_s", [(lt, cells.get((s, lt))) for lt in lats]) for s in speeds]
        for g, gv, req, line in given:
            row = {"ball_min_mm_s": key[0], "ball_max_mm_s": key[1], "half_width_mm": key[2], "given": g,
                   "given_value": gv, "requirement": req}
            for col, stat in (("by_estimate", "rate"), ("by_lower_bound", "lo95")):
                best = math.nan
                for v, r in line:                            # easiest setting first; stop at the first miss
                    if r is None or not r[stat] >= goal:
                        break
                    best = v
                row[col] = best
            out.append(row)
    return out


def run_needs(plan: Plan, base: Baseline, out: Path, progress: Progress) -> None:
    """What the arm needs, from h5_sweep.csv in the run folder."""
    with Table(out / "needs.csv", NEEDS_COLS) as t:
        for row in arm_needs(read_rows(out / "h5_sweep.csv")):
            t.add(row)


RUNNERS: dict[str, Callable[[Plan, Baseline, Path, Progress], None]] = {
    "h4": run_h4, "h5": run_h5, "needs": run_needs, "robust": run_robust}


# ------------------------------------------------------------------------------------------------ the run

def source_hashes() -> dict[str, str]:
    """sha256 (12 hex) of the ball code the numbers come from: it changes fast and isn't all committed."""
    here = Path(__file__).resolve().parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:12] for p in sorted(here.glob("*.py"))}


def _git() -> dict:
    try:
        from worldarm.logger import git_info                 # needs protobuf; only the simulation path asks
        return git_info()
    except Exception as e:                                   # never lose a night's run over provenance
        return {"commit": "", "dirty": None, "error": repr(e)}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def run(plan: Plan, base: Baseline, studies: Sequence[str], out: Path, charts: bool = True,
        say: Callable[[str], None] = print) -> Path:
    """Run the chosen studies into `out` (created if needed), refreshing summary.md (and the charts) after each.
    needs is computed from h5_sweep.csv, so h5 is added when it isn't there yet."""
    out = Path(out)
    todo = [s for s in STUDIES if s in studies]
    if "needs" in todo and "h5" not in todo and not (out / "h5_sweep.csv").exists():
        say("needs comes from the h5 sweep, which isn't in this folder yet: running h5 too")
        todo.insert(todo.index("needs"), "h5")
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"studies": {}, "invocations": []}
    meta |= {"plan": dataclasses.asdict(plan), "baseline": dataclasses.asdict(base), "seeds": seeds(base),
             "goal": GOAL, "single_process": True}
    meta["invocations"].append({"argv": sys.argv, "started_at": _now(), "plan": plan.name, "studies": todo,
                                "git": _git(), "sources": source_hashes(), "python": platform.python_version(),
                                "numpy": np.__version__, "platform": platform.platform()})
    costs = {s: planned_cost(plan, base, s) for s in todo}
    progress = Progress(sum(costs.values()), say)
    say(f"plan '{plan.name}': {', '.join(todo)} -> {out}")
    say(f"  about {hms(sum(costs.values()) * ROLL_S)} at the dev laptop's speed; progress every 15 s")
    for s in todo:
        meta["studies"][s] = {"status": "running", "plan": plan.name, "started_at": _now()}
        write_atomic(meta_path, json.dumps(meta, indent=2) + "\n")
        t0, status = time.monotonic(), "done"
        try:
            RUNNERS[s](plan, base, out, progress)
        except KeyboardInterrupt:
            status = "interrupted"
        meta["studies"][s] |= {"status": status, "seconds": round(time.monotonic() - t0, 1), "ended_at": _now()}
        write_atomic(meta_path, json.dumps(meta, indent=2) + "\n")
        say(f"{s} {status} in {hms(time.monotonic() - t0)}: {', '.join(CSVS[s])}")
        from worldarm.ball.study_report import report   # summary.md and charts, from the CSVs
        report(out, charts, say)
        if status == "interrupted":
            raise KeyboardInterrupt
    return out
