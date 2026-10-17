"""Fit the rolling model to the latest sightings of the ball.

Sightings are noisy (a few mm) and arrive ~23 per second, some frames missing. Over a short window the
ball's path is a straight line: its direction comes from the spread of the points, and with the learned
deceleration known, start position and speed come from a linear least-squares fit - robust with only a
handful of points, which is what lets the arm commit early.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from worldarm.ball.model import Roll


@dataclass
class Estimate:
    roll: Roll
    n: int                 # sightings used
    rms_mm: float          # fit residual
    moving: bool
    span_mm: float = 0.0   # how far apart the sightings are along the path (more = surer of the direction)


class Tracker:
    def __init__(self, decel: float = 0.0, window_s: float = 0.6, min_points: int = 4, max_gap_s: float = 0.3,
                 jump_mm: float = 120.0):
        self.decel, self.window_s, self.min_points = decel, window_s, min_points
        self.max_gap_s, self.jump_mm = max_gap_s, jump_mm
        self.seen: deque[tuple[float, np.ndarray]] = deque()

    def reset(self) -> None:
        self.seen.clear()

    def add(self, t: float, xy) -> None:
        xy = np.asarray(xy, float)
        if not (np.all(np.isfinite(xy)) and np.isfinite(t)):
            return                                      # a bad table-map point: never let it reach the arm
        if self.seen:
            t_last, p_last = self.seen[-1]
            # a long gap or a jump means a new roll (picked up, or a different ball): start over
            if t - t_last > self.max_gap_s or float(np.linalg.norm(xy - p_last)) > self.jump_mm + 2000.0 * (t - t_last):
                self.seen.clear()
        self.seen.append((t, xy))
        while self.seen and t - self.seen[0][0] > self.window_s:
            self.seen.popleft()

    def estimate(self) -> Estimate | None:
        if len(self.seen) < self.min_points:
            return None
        t = np.array([s[0] for s in self.seen])
        p = np.array([s[1] for s in self.seen])
        tr = t - t[-1]                                  # times relative to the newest sighting (<= 0)
        c = p.mean(axis=0)
        u = np.linalg.svd(p - c)[2][0]
        if float((p[-1] - p[0]) @ u) < 0:
            u = -u                                      # point it the way the ball is going
        across = np.array([-u[1], u[0]])
        s = (p - c) @ u
        q = (p - c) @ across
        # s(tr) = s1 + v1 tr - a tr^2 / 2 around the newest sighting (v1 = speed now); the deceleration is known
        A = np.c_[np.ones_like(tr), tr]
        coef, *_ = np.linalg.lstsq(A, s + 0.5 * self.decel * tr * tr, rcond=None)
        s1, v1 = float(coef[0]), float(coef[1])
        q1 = float(np.mean(q))
        resid = s - (s1 + v1 * tr - 0.5 * self.decel * tr * tr)
        rms = float(np.sqrt(np.mean(resid ** 2 + (q - q1) ** 2)))
        moving = v1 > 40.0
        p_now = c + u * s1 + across * q1
        return Estimate(Roll(float(t[-1]), p_now, u, max(v1, 0.0)), len(t), rms, moving, float(s.max() - s.min()))
