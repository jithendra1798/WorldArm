"""Ballistics of the throw, in SI units (m, s, rad).

A ball leaves the gripper at point p with velocity v and flies under gravity (and, optionally, a little
air drag: ~2% of the range for a 5 cm rubber ball at 4 m/s). It "lands" when it comes down through the
catch height. 45 degrees is the forgiving angle: the range barely changes with small angle errors, so the
range is set by the speed alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

G = 9.81


@dataclass(frozen=True)
class Release:
    p: np.ndarray        # (3,) m, where the ball leaves the fingers
    v: np.ndarray        # (3,) m/s, its velocity then

    @classmethod
    def aimed(cls, p, speed: float, elevation_deg: float, azimuth_deg: float) -> Release:
        el, az = math.radians(elevation_deg), math.radians(azimuth_deg)
        v = speed * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        return cls(np.asarray(p, float), v)

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.v))

    @property
    def elevation_deg(self) -> float:
        return math.degrees(math.atan2(self.v[2], float(np.hypot(self.v[0], self.v[1]))))


@dataclass(frozen=True)
class Landing:
    xy: np.ndarray       # (2,) m, where it comes down through the catch height
    t: float             # s of flight
    v: np.ndarray        # (3,) m/s on arrival
    apex_z: float        # m, highest point


def drag_k(diameter_m: float = 0.05, mass_kg: float = 0.05, cd: float = 0.47, rho: float = 1.2) -> float:
    """Quadratic drag constant k (1/m) for a = -k |v| v."""
    area = math.pi * (diameter_m / 2.0) ** 2
    return 0.5 * rho * cd * area / mass_kg


def fly(rel: Release, catch_z: float, k: float = 0.0, dt: float = 2e-4, max_t: float = 5.0) -> Landing | None:
    """Where and when the ball comes down through catch_z (None if it never gets back down to it)."""
    p, v = rel.p.astype(float), rel.v.astype(float)
    if k == 0.0:
        a = -0.5 * G
        b, c = v[2], p[2] - catch_z
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        t = (-b - math.sqrt(disc)) / (2 * a)
        if t <= 0:
            return None
        t_apex = max(0.0, v[2] / G)
        apex = p[2] + v[2] * t_apex - 0.5 * G * t_apex ** 2
        return Landing(p[:2] + v[:2] * t, t, np.array([v[0], v[1], v[2] - G * t]), apex)
    t, apex = 0.0, p[2]

    def acc(vel):
        return np.array([0.0, 0.0, -G]) - k * float(np.linalg.norm(vel)) * vel
    while t < max_t:
        # RK4 on (p, v)
        k1v, k1p = acc(v), v
        k2v, k2p = acc(v + 0.5 * dt * k1v), v + 0.5 * dt * k1v
        k3v, k3p = acc(v + 0.5 * dt * k2v), v + 0.5 * dt * k2v
        k4v, k4p = acc(v + dt * k3v), v + dt * k3v
        p_new = p + dt / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        v_new = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        apex = max(apex, p_new[2])
        if p_new[2] <= catch_z and v_new[2] < 0:
            f = (p[2] - catch_z) / (p[2] - p_new[2])          # interpolate the crossing
            return Landing(p[:2] + f * (p_new[:2] - p[:2]), t + f * dt, v + f * (v_new - v), apex)
        p, v, t = p_new, v_new, t + dt
    return None


def speed_for(distance: float, elevation_deg: float, drop: float = 0.0) -> float:
    """Release speed that carries the ball `distance` horizontally at this elevation, landing `drop` m below
    the release height (no drag). From x = v cos(e) t, z = v sin(e) t - g t^2 / 2 = -drop."""
    e = math.radians(elevation_deg)
    denom = 2 * math.cos(e) ** 2 * (distance * math.tan(e) + drop)
    if denom <= 0:
        raise ValueError("can't reach that distance at this elevation (it would have to go below the catch point)")
    return math.sqrt(G * distance * distance / denom)


def speed_for_drag(distance: float, elevation_deg: float, drop: float, k: float, iters: int = 24) -> float:
    """The same with drag, by bisection on the drag-free answer."""
    lo, hi = speed_for(distance, elevation_deg, drop), 1.5 * speed_for(distance, elevation_deg, drop)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        land = fly(Release.aimed([0, 0, drop], mid, elevation_deg, 0.0), 0.0, k, dt=1e-3)
        if land is None or land.xy[0] < distance:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class Sensitivity:
    per_pct_speed_m: float        # landing shift for 1% more speed
    per_deg_elevation_m: float    # ... for 1 degree more elevation
    per_ms_late_m: float          # ... for releasing 1 ms late during a swing turning at end_rate


def sensitivity(speed: float, elevation_deg: float, drop: float, end_rate_deg_s: float, k: float = 0.0) -> Sensitivity:
    """How far the landing moves per small error. A late release both turns the ball's direction (the end
    of the arm keeps rotating at end_rate) and lets it travel further along the swing."""
    def land(s, e, dz=0.0, dx=0.0):
        r = fly(Release.aimed([dx, 0, drop + dz], s, e, 0.0), 0.0, k)
        return r.xy[0] if r else float("nan")
    base = land(speed, elevation_deg)
    d_speed = land(speed * 1.01, elevation_deg) - base
    d_el = land(speed, elevation_deg + 1.0) - base
    dt = 0.001
    e2 = elevation_deg + end_rate_deg_s * dt            # the swing's direction keeps turning upward
    e_r = math.radians(elevation_deg)
    late = land(speed, e2, dz=speed * dt * math.sin(e_r), dx=speed * dt * math.cos(e_r)) - base
    return Sensitivity(d_speed, d_el, late)
