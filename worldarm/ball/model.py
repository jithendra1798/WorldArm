"""The world model of a rolling ball: it goes straight and slows at a steady rate until it stops.

    position(t) = p0 + u * s(t),   s(t) = v0 t - a t^2 / 2   (until v0 - a t = 0, then it stays put)

`a` (mm/s^2) is learned from our own recorded rolls (fit_deceleration). The baseline for H4 is the same
with a = 0: "it keeps its current velocity".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Roll:
    """A ball's state at time t0: where, which way, how fast."""
    t0: float
    p0: np.ndarray        # (2,) mm
    u: np.ndarray         # (2,) unit direction of travel
    v0: float             # mm/s along u

    def velocity(self) -> np.ndarray:
        return self.u * self.v0


@dataclass(frozen=True)
class RollModel:
    decel: float = 0.0    # mm/s^2; 0 = constant velocity (the H4 baseline)

    def stop_time(self, roll: Roll) -> float:
        """Seconds after t0 until it stops (inf if it never does)."""
        return math.inf if self.decel <= 0 else max(0.0, roll.v0 / self.decel)

    def distance(self, roll: Roll, dt: float) -> float:
        """Distance travelled dt seconds after t0."""
        dt = min(max(dt, 0.0), self.stop_time(roll))
        return roll.v0 * dt - 0.5 * self.decel * dt * dt

    def at(self, roll: Roll, t: float) -> np.ndarray:
        return roll.p0 + roll.u * self.distance(roll, t - roll.t0)

    def stop_point(self, roll: Roll) -> np.ndarray | None:
        return None if self.decel <= 0 else roll.p0 + roll.u * (roll.v0 ** 2 / (2 * self.decel))

    def crossing(self, roll: Roll, line_point: np.ndarray, line_normal: np.ndarray) -> tuple[np.ndarray, float] | None:
        """Where and when the ball crosses the line {x : (x - line_point) . n = 0}, or None if it won't
        (moving away, parallel, or stopping short)."""
        n = np.asarray(line_normal, float)
        gap = float((np.asarray(line_point, float) - roll.p0) @ n)   # distance to go, along n
        closing = float(roll.u @ n) * roll.v0                          # speed toward the line
        if closing <= 1e-6 or gap < 0:
            return None
        along = gap / float(roll.u @ n)                                # path length to the line
        if self.decel > 0:
            reach = roll.v0 ** 2 / (2 * self.decel)
            if along > reach:
                return None                                            # stops before the line
            disc = roll.v0 ** 2 - 2 * self.decel * along
            dt = (roll.v0 - math.sqrt(max(disc, 0.0))) / self.decel
        else:
            dt = along / roll.v0
        return roll.p0 + roll.u * along, roll.t0 + dt


def fit_deceleration(tracks: list[tuple[np.ndarray, np.ndarray]], min_points: int = 8,
                     min_duration_s: float = 0.6) -> tuple[float, float, int]:
    """Learn the slowdown from recorded rolls. Each track is (t (N,), xy (N, 2)) of one roll.

    For each roll, the distance travelled along its path is fitted with s(t) = s0 + v0 t - a t^2 / 2 while
    it is still moving; a is the roll's deceleration. Rolls watched for less than min_duration_s are left out
    (over a short stretch a few mm of camera noise swamp the curvature). Returns (median a, spread (IQR/1.35),
    rolls used).
    """
    found = []
    for t, xy in tracks:
        t, xy = np.asarray(t, float), np.asarray(xy, float)
        if len(t) < min_points:
            continue
        d = xy - xy[0]
        u = np.linalg.svd(d - d.mean(axis=0))[2][0]
        s = d @ u
        if s[-1] < 0:
            s = -s
        speed = np.abs(np.gradient(s, t))
        moving = speed > 30.0                    # stopped, or picked up: not rolling any more
        if moving.sum() < min_points:
            continue
        tt, ss = t[moving] - t[moving][0], s[moving]
        if tt[-1] < min_duration_s:
            continue
        coef = _fit_roll(tt, ss)
        if coef[1] > 0:
            found.append(float(coef[2]))
    if not found:
        return 0.0, math.inf, 0
    a = np.array(found)
    q1, q3 = np.percentile(a, [25, 75])
    return float(max(0.0, np.median(a))), float((q3 - q1) / 1.35), len(found)


def _fit_roll(tt: np.ndarray, ss: np.ndarray) -> np.ndarray:
    """(s0, v0, a) by least squares, refitted once without the outliers (a misdetection or two must not bend
    the slowdown): residuals beyond 3 robust sd, and at least 3 mm."""
    A = np.c_[np.ones_like(tt), tt, -0.5 * tt * tt]
    coef, *_ = np.linalg.lstsq(A, ss, rcond=None)
    r = ss - A @ coef
    keep = np.abs(r - np.median(r)) <= max(3.0 * 1.4826 * float(np.median(np.abs(r - np.median(r)))), 3.0)
    if keep.sum() >= 6 and not keep.all():
        coef, *_ = np.linalg.lstsq(A[keep], ss[keep], rcond=None)
    return coef
