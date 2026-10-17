"""The swing's speed profile, and its fit to a real swing.

The arm turns the swinging joints with a trapezoid in speed: up to `w` at `a`, cruise, down at `a` (a
triangle if the swing is too short to reach `w`). "Progress" is how far the swinging joints have turned
from the wind-up pose (deg). The release must happen in the cruise, where the ball's speed is steady.

toss.py dry reads the joints as fast as the arm answers during a swing and fits this model to the readings:
when the arm really started (after the command), how fast it really cruised, how hard it accelerated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from worldarm.toss.lsq import least_squares


@dataclass(frozen=True)
class Trapezoid:
    amplitude: float        # deg the swinging joints turn
    w: float                # deg/s cruise speed asked for
    a: float                # deg/s^2 speed-up and slow-down
    t0: float = 0.0         # when it starts moving

    @property
    def peak(self) -> float:
        """The top speed actually reached (less than w if the swing is too short: a triangle)."""
        return min(self.w, math.sqrt(self.a * self.amplitude))

    @property
    def ramp_deg(self) -> float:
        """Progress spent speeding up (and again slowing down)."""
        return self.peak ** 2 / (2.0 * self.a)

    @property
    def cruise(self) -> tuple[float, float]:
        """Progress range at full speed."""
        return self.ramp_deg, self.amplitude - self.ramp_deg

    @property
    def duration(self) -> float:
        return 2.0 * self.peak / self.a + (self.amplitude - 2.0 * self.ramp_deg) / self.peak

    def progress(self, t: float) -> float:
        tt = t - self.t0
        if tt <= 0.0:
            return 0.0
        v, a, r = self.peak, self.a, self.ramp_deg
        t_up = v / a
        if tt < t_up:
            return 0.5 * a * tt * tt
        t_cruise = (self.amplitude - 2.0 * r) / v
        if tt < t_up + t_cruise:
            return r + v * (tt - t_up)
        td = tt - t_up - t_cruise
        if td < t_up:
            return self.amplitude - r + v * td - 0.5 * a * td * td
        return self.amplitude

    def time_at(self, progress: float) -> float:
        """When it reaches this progress (inverse of progress())."""
        p = min(max(progress, 0.0), self.amplitude)
        v, a, r = self.peak, self.a, self.ramp_deg
        t_up = v / a
        if p <= r:
            return self.t0 + math.sqrt(2.0 * p / a)
        if p <= self.amplitude - r:
            return self.t0 + t_up + (p - r) / v
        left = self.amplitude - p                          # mirror of the speed-up
        return self.t0 + self.duration - math.sqrt(2.0 * left / a)

    def in_cruise(self, progress: float, margin_deg: float = 0.0) -> bool:
        lo, hi = self.cruise
        return lo + margin_deg <= progress <= hi - margin_deg


def fit_trapezoid(t, progress, amplitude: float, w_guess: float, a_guess: float) -> tuple[Trapezoid, np.ndarray, float]:
    """Fit (start time, cruise speed, acceleration) to readings taken during a swing.
    Returns (profile, standard deviations of (t0, w, a), rms deg)."""
    t, p = np.asarray(t, float), np.asarray(progress, float)
    moving = (p > 0.5) & (p < amplitude - 0.5)              # resting readings say little but "not yet / done"
    mid = (p > 0.25 * amplitude) & (p < 0.75 * amplitude)
    w0 = float(np.polyfit(t[mid], p[mid], 1)[0]) if mid.sum() >= 2 else w_guess
    w0 = w0 if w0 > 0 else w_guess
    first = t[moving][0] if moving.any() else t[0]
    t00 = first - math.sqrt(2.0 * max(p[moving][0] if moving.any() else 0.0, 0.0) / a_guess)

    def resid(x):
        prof = Trapezoid(amplitude, x[1], math.exp(x[2]), x[0])
        return np.array([prof.progress(ti) for ti in t]) - p

    fit = least_squares(resid, [t00, w0, math.log(a_guess)])
    t0, w, la = fit.x
    a = math.exp(la)
    sd = np.array([fit.sd[0], fit.sd[1], a * fit.sd[2]])
    return Trapezoid(amplitude, float(w), float(a), float(t0)), sd, fit.rms
