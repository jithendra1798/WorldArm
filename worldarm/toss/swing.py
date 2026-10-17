"""Plan a throw: one synchronized swing of the arm's pitch joints, released at full speed at 45 degrees.

The swing turns the chosen joints by the same angle at the same rate (accelerate, cruise, decelerate). While
cruising, the ball's direction of travel depends only on the arm's shape, not on how fast it turns, so:

  release point   where the ball's velocity points `elevation` above horizontal - fixed by the geometry
  distance        set by the joint speed alone (ball speed is proportional to it)
  direction       set by the base joint alone (turning it turns the whole throw about the base)

The release must fall in the cruise, after the speed-up and before the slow-down; the gripper is told to open
`release_delay_s` early (calibrated from throws), and the trigger is a joint angle, so it's independent of
when the command started.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from worldarm.toss.kinematics import Chain
from worldarm.toss.physics import G, Landing, Release, fly, speed_for


@dataclass
class SwingSpec:
    start_deg: list[float]            # wind-up pose, all joints
    joints: list[int]                 # which joints swing (0-based), all by the same angle
    signs: list[int]                  # direction of each (+1 / -1)
    amplitude_deg: float              # how far each swinging joint turns
    accel_deg_s2: float               # joint acceleration and deceleration
    tool_mm: tuple = (0.0, 0.0, 155.0)  # the ball's centre in the flange frame
    table_z_mm: float = 0.0           # in the thrower's base frame
    clearance_mm: float = 80.0        # the ball stays this far above the table all swing


@dataclass
class ThrowPlan:
    ok: bool
    reason: str
    base_deg: float = 0.0             # base joint angle (aims the throw)
    speed_deg_s: float = 0.0          # swing joint speed
    release_progress_deg: float = 0.0  # how far into the swing the ball leaves
    trigger_progress_deg: float = 0.0  # when to tell the gripper to open (earlier by the delay)
    release_time_s: float = 0.0       # after the swing starts moving
    release_mm: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity_m_s: np.ndarray = field(default_factory=lambda: np.zeros(3))
    landing: Landing | None = None    # in the thrower's base frame, metres
    cruise_deg: tuple = (0.0, 0.0)    # progress range at full speed

    def to_json(self) -> dict:
        return {"ok": self.ok, "reason": self.reason, "base_deg": round(self.base_deg, 3),
                "speed_deg_s": round(self.speed_deg_s, 2), "release_progress_deg": round(self.release_progress_deg, 3),
                "trigger_progress_deg": round(self.trigger_progress_deg, 3), "release_time_s": round(self.release_time_s, 4),
                "release_mm": np.round(self.release_mm, 1).tolist(), "velocity_m_s": np.round(self.velocity_m_s, 3).tolist(),
                "landing_m": None if self.landing is None else np.round(self.landing.xy, 4).tolist(),
                "flight_s": None if self.landing is None else round(self.landing.t, 3),
                "cruise_deg": [round(c, 2) for c in self.cruise_deg]}


def pose_at(spec: SwingSpec, progress_deg: float, base_deg: float) -> np.ndarray:
    q = np.array(spec.start_deg, float)
    q[0] = base_deg
    for j, s in zip(spec.joints, spec.signs):
        q[j] += s * progress_deg
    return q


def unit_velocity(chain: Chain, spec: SwingSpec, progress_deg: float, base_deg: float) -> np.ndarray:
    """Ball velocity (mm/s) per 1 deg/s of swing speed, at this point of the swing."""
    qd = np.zeros(chain.n)
    for j, s in zip(spec.joints, spec.signs):
        qd[j] = s
    return chain.velocity(pose_at(spec, progress_deg, base_deg), qd, spec.tool_mm)


def elevation_deg(v: np.ndarray) -> float:
    return math.degrees(math.atan2(v[2], float(np.hypot(v[0], v[1]))))


def find_release(chain: Chain, spec: SwingSpec, elevation: float, base_deg: float = 0.0, step: float = 0.25) -> float | None:
    """Progress (deg) where the ball's velocity points `elevation` above horizontal, rising through it."""
    prev_p, prev_e = None, None
    for p in np.arange(0.0, spec.amplitude_deg + 1e-9, step):
        e = elevation_deg(unit_velocity(chain, spec, p, base_deg))
        if prev_e is not None and (prev_e - elevation) * (e - elevation) <= 0 and prev_e != e:
            lo, hi = prev_p, p                       # refine by bisection
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                em = elevation_deg(unit_velocity(chain, spec, mid, base_deg))
                if (prev_e - elevation) * (em - elevation) <= 0:
                    hi = mid
                else:
                    lo = mid
            return 0.5 * (lo + hi)
        prev_p, prev_e = p, e
    return None


def plan_throw(chain: Chain, spec: SwingSpec, target_xy_mm, catch_z_mm: float, elevation: float = 45.0,
               max_speed_deg_s: float = 180.0, release_delay_s: float = 0.0, speed_factor: float = 1.0) -> ThrowPlan:
    """The base angle and swing speed that land the ball at target_xy (thrower's base frame, mm) at catch_z.

    speed_factor and release_delay_s are the two calibrated constants (from a few measured throws): the arm's
    real speed vs. commanded, and how much earlier than the release angle the gripper must be told to open.
    """
    target = np.asarray(target_xy_mm, float)
    want_az = math.degrees(math.atan2(target[1], target[0]))
    base = want_az
    rel_p = None
    for _ in range(4):                               # the release point sits off the base axis: settle the aim
        rel_p = find_release(chain, spec, elevation, base)
        if rel_p is None:
            return ThrowPlan(False, f"the ball's direction never reaches {elevation:.0f} deg in this swing")
        u = unit_velocity(chain, spec, rel_p, base)
        p0 = chain.point(pose_at(spec, rel_p, base), spec.tool_mm)
        # where a ball released here would land, as a ray from the release point along u's horizontal part
        az_ball = math.degrees(math.atan2(u[1], u[0]))
        to_target = target - p0[:2]
        base += math.degrees(math.atan2(to_target[1], to_target[0])) - az_ball
    u = unit_velocity(chain, spec, rel_p, base)
    p0 = chain.point(pose_at(spec, rel_p, base), spec.tool_mm)
    dist_m = float(np.linalg.norm(target - p0[:2])) / 1000.0
    drop_m = (p0[2] - catch_z_mm) / 1000.0
    try:
        v_need = speed_for(dist_m, elevation, drop_m)
    except ValueError as e:
        return ThrowPlan(False, str(e))
    per_deg_s = float(np.linalg.norm(u)) / 1000.0 * speed_factor     # m/s of ball speed per deg/s
    w = v_need / per_deg_s
    a = spec.accel_deg_s2
    acc_deg = w * w / (2 * a)
    cruise = (acc_deg, spec.amplitude_deg - acc_deg)
    trigger = rel_p - w * release_delay_s
    plan = ThrowPlan(True, "", base, w, rel_p, trigger, w / a + (rel_p - acc_deg) / w, p0,
                     u / 1000.0 * w * speed_factor, None, cruise)
    plan.landing = fly(Release(p0 / 1000.0, plan.velocity_m_s), catch_z_mm / 1000.0)
    problems = []
    if w > max_speed_deg_s:
        problems.append(f"needs {w:.0f} deg/s, more than the {max_speed_deg_s:.0f} allowed")
    if not (cruise[0] + 2.0 <= trigger and rel_p <= cruise[1] - 2.0):
        problems.append(f"release at {rel_p:.1f} deg (gripper told at {trigger:.1f}) isn't inside the full-speed part "
                        f"{cruise[0]:.1f}-{cruise[1]:.1f} deg: start the swing earlier, swing further, or accelerate harder")
    lowest = min(chain.point(pose_at(spec, p, base), spec.tool_mm)[2] for p in np.linspace(0, spec.amplitude_deg, 61))
    if lowest < spec.table_z_mm + spec.clearance_mm:
        problems.append(f"the ball comes within {lowest - spec.table_z_mm:.0f} mm of the table during the swing")
    if not all(chain.within_limits(pose_at(spec, p, base)) for p in np.linspace(0, spec.amplitude_deg, 31)):
        problems.append("the swing goes past a joint limit")
    if problems:
        plan.ok, plan.reason = False, "; ".join(problems)
    return plan
