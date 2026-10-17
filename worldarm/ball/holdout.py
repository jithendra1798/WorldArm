"""H4 on real rolls: does the learned slowdown predict better than "it keeps its speed"? Measured honestly.

Each recorded roll is predicted with the slowdown learned from the OTHER rolls (leave-one-out), so no roll
is judged by a model that has seen it. Two questions, both from what the camera saw:

  arrival   When does it reach a finish line? The line is across the roll's path, where the ball had covered
            70% of its observed path (still rolling at about half its speed, as a ball reaching the lane is);
            the prediction is made h seconds before it got there, from the sightings up to then (the goalie's
            tracker), with the learned slowdown and with none.
  stop      Where does it come to rest? Predicted after watching the first `watch_s` seconds; only rolls that
            ended by stopping count (not ones lost from view or turned back by something).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from worldarm.ball.model import RollModel, fit_deceleration
from worldarm.ball.track import Tracker

LINE_AT = 0.7               # the finish line: where the ball had covered this share of its observed path
STOPPED_MM_S = 100.0        # a roll ending slower than this came to rest (else it was lost or hit something)
MIN_SEEN = 6                # sightings needed before predicting (~0.25 s)


@dataclass
class Holdout:
    horizons: tuple
    arrival_learned: np.ndarray    # [roll, horizon] s; nan: too little seen; inf: said it stops short, it arrived
    arrival_constant: np.ndarray   # same, constant velocity
    stop_learned: np.ndarray       # [roll] mm; nan: the roll didn't end at rest, or too little seen
    decels: np.ndarray             # [roll] the slowdown each roll was predicted with (learned without it)

    def summary(self) -> dict:
        """Medians per horizon (inf counts as a failure, so it stays in), how many rolls each, and the stop error."""
        out = {"rolls": int(len(self.decels)), "arrival": {}, "stop": _med(self.stop_learned)}
        for j, h in enumerate(self.horizons):
            out["arrival"][h] = {"learned": _med(self.arrival_learned[:, j]),
                                 "constant": _med(self.arrival_constant[:, j]),
                                 "learned_closer": _closer(self.arrival_learned[:, j], self.arrival_constant[:, j])}
        return out


def holdout_errors(tracks: list[tuple[np.ndarray, np.ndarray]], horizons=(1.0, 0.6, 0.3),
                   watch_s: float = 0.5) -> Holdout:
    """tracks: the free-rolling part of each recorded roll, (t (N,), xy (N, 2)) (ball.py load_rolls)."""
    tracks = [(np.asarray(t, float), np.asarray(xy, float)) for t, xy in tracks]
    n, k = len(tracks), len(horizons)
    learned, constant = np.full((n, k), np.nan), np.full((n, k), np.nan)
    stop, decels = np.full(n, np.nan), np.full(n, np.nan)
    for i, (t, xy) in enumerate(tracks):
        a, _, used = fit_deceleration(tracks[:i] + tracks[i + 1:])
        if used == 0 or len(t) < MIN_SEEN + 2:
            continue
        decels[i] = a
        u = _direction(xy)
        s = (xy - xy[0]) @ u
        t_line = float(t[int(np.argmax(s >= LINE_AT * s[-1]))])
        s_line = _smooth_at(t, s, t_line)       # where it was then: the line, and t_line is the true arrival
        line_point = xy[0] + u * s_line
        for j, h in enumerate(horizons):
            seen = t <= t_line - h
            if seen.sum() < MIN_SEEN:
                continue
            for decel, out in ((a, learned), (0.0, constant)):
                est = _estimate(t[seen], xy[seen], decel)
                if est is None:
                    continue
                hit = RollModel(decel).crossing(est.roll, line_point, u)
                out[i, j] = math.inf if hit is None else abs(hit[1] - t_line)
        if a > 0 and _end_speed(t, s) < STOPPED_MM_S:
            seen = t <= t[0] + watch_s
            est = _estimate(t[seen], xy[seen], a) if seen.sum() >= MIN_SEEN else None
            if est is not None:
                stop[i] = float(np.linalg.norm(RollModel(a).stop_point(est.roll) - xy[-1]))
    return Holdout(tuple(horizons), learned, constant, stop, decels)


def _estimate(t, xy, decel):
    """The goalie's view: the tracker's fit to the latest sightings (its own window)."""
    tracker = Tracker(decel=decel)
    for tt, p in zip(t, xy):
        tracker.add(float(tt), p)
    return tracker.estimate()


def _direction(xy: np.ndarray) -> np.ndarray:
    u = np.linalg.svd(xy - xy.mean(axis=0))[2][0]
    return u if float((xy[-1] - xy[0]) @ u) >= 0 else -u


def _smooth_at(t: np.ndarray, s: np.ndarray, t0: float, half_s: float = 0.25) -> float:
    """s at t0 from a local quadratic fit (the raw sighting has the camera's few mm of noise)."""
    near = np.abs(t - t0) <= half_s
    if near.sum() < 4:
        return float(np.interp(t0, t, s))
    c = np.polyfit(t[near] - t0, s[near], 2)
    return float(c[-1])


def _end_speed(t: np.ndarray, s: np.ndarray, last_s: float = 0.5) -> float:
    """Mean speed over the track's last last_s seconds (3-sighting averages at both ends, for the camera's noise).
    A ball coming to rest averages well under 100 mm/s there; one lost from view while rolling, far more."""
    i = int(np.searchsorted(t, t[-1] - last_s))
    if len(t) - i < 6:
        return math.inf
    return abs(float(s[-3:].mean() - s[i:i + 3].mean())) / max(float(t[-3:].mean() - t[i:i + 3].mean()), 1e-6)


def _med(x) -> dict:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return {"median": float(np.median(x)) if len(x) else None, "n": int(len(x)),
            "failed": int(np.isinf(x).sum())}


def _closer(a, b) -> dict:
    """Paired: on how many rolls the learned prediction was closer than constant velocity."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    both = ~np.isnan(a) & ~np.isnan(b)
    return {"closer": int((a[both] < b[both]).sum()), "n": int(both.sum())}
