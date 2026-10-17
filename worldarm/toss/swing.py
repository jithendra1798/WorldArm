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
               max_speed_deg_s: float = 180.0, release_delay_s: float = 0.0, speed_factor: float = 1.0,
               release_step_deg: float = 1.0) -> ThrowPlan:
    """The base angle and swing speed that land the ball at target_xy (thrower's base frame, mm) at catch_z.

    speed_factor and release_delay_s are the two calibrated constants (from a few measured throws): the arm's
    real speed vs. commanded, and how much earlier than the release angle the gripper must be told to open.
    """
    target = np.asarray(target_xy_mm, float)
    want_az = math.degrees(math.atan2(target[1], target[0]))
    rel_p = find_release(chain, spec, elevation, 0.0, release_step_deg)
    if rel_p is None:
        return ThrowPlan(False, f"the ball's direction never reaches {elevation:.0f} deg in this swing")
    u0 = unit_velocity(chain, spec, rel_p, 0.0)
    base = want_az - math.degrees(math.atan2(u0[1], u0[0]))   # first guess: the ball heads the target's way
    for _ in range(4):                               # the release point sits off the base axis: settle the aim
        rel_p = find_release(chain, spec, elevation, base, release_step_deg)
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
    if float(u[:2] @ (target - p0[:2])) <= 0:
        return ThrowPlan(False, "the target is behind the release point for this swing")
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


# ---- how forgiving a throw is, and choosing the swing ----------------------------------------------------

def end_pose(spec: SwingSpec, base_deg: float) -> np.ndarray:
    return pose_at(spec, spec.amplitude_deg, base_deg)


def landing_for(chain: Chain, spec: SwingSpec, base_deg: float, progress_deg: float, speed_deg_s: float,
                catch_z_mm: float, speed_factor: float = 1.0) -> np.ndarray | None:
    """Where the ball comes down (thrower's base frame, mm) if it leaves at this point of the swing at this
    joint speed."""
    p0 = chain.point(pose_at(spec, progress_deg, base_deg), spec.tool_mm)
    v = unit_velocity(chain, spec, progress_deg, base_deg) * (speed_deg_s * speed_factor / 1000.0)
    land = fly(Release(p0 / 1000.0, v), catch_z_mm / 1000.0)
    return None if land is None else land.xy * 1000.0


@dataclass
class Forgiveness:
    late_10ms_mm: float           # landing shift (along the throw) if the ball leaves 10 ms late
    speed_1pct_mm: float          # ... if the swing is 1% faster
    late_sideways_mm: float       # sideways part of the late shift
    elevation_1deg_mm: float = 0.0  # ... if it heads 1 degree higher than the arm's model says

    def to_json(self) -> dict:
        return {k: round(v, 1) for k, v in self.__dict__.items()}


def forgiveness(chain: Chain, spec: SwingSpec, plan: ThrowPlan, catch_z_mm: float, speed_factor: float = 1.0) -> Forgiveness:
    """How far the landing moves for the two errors that matter, computed on the real arm: the release a
    little late (it leaves further along the swing, heading a little differently) and the speed a little off."""
    args = (chain, spec, plan.base_deg)
    at = landing_for(*args, plan.release_progress_deg, plan.speed_deg_s, catch_z_mm, speed_factor)
    late = landing_for(*args, plan.release_progress_deg + plan.speed_deg_s * 0.010, plan.speed_deg_s, catch_z_mm,
                       speed_factor)
    fast = landing_for(*args, plan.release_progress_deg, plan.speed_deg_s * 1.01, catch_z_mm, speed_factor)
    v = plan.velocity_m_s
    h = float(np.hypot(v[0], v[1]))
    el = math.atan2(v[2], h) + math.radians(1.0)
    v_up = np.r_[v[:2] / (h or 1.0) * math.cos(el), math.sin(el)] * float(np.linalg.norm(v))
    up = fly(Release(plan.release_mm / 1000.0, v_up), catch_z_mm / 1000.0)
    if at is None or late is None or fast is None or up is None:
        return Forgiveness(math.inf, math.inf, math.inf, math.inf)
    u = at - plan.release_mm[:2]
    u = u / (np.linalg.norm(u) or 1.0)
    side = np.array([-u[1], u[0]])
    return Forgiveness(float((late - at) @ u), float((fast - at) @ u), float((late - at) @ side),
                       float((up.xy * 1000.0 - at) @ u))


def choose_elevation(chain: Chain, spec: SwingSpec, target_xy_mm, catch_z_mm: float, max_speed_deg_s: float,
                     release_delay_s: float = 0.0, speed_factor: float = 1.0,
                     elevations=range(25, 61)) -> tuple[ThrowPlan | None, Forgiveness | None]:
    """The release elevation (near 45 degrees) where a late release moves the landing least.

    Releasing late, the ball leaves further along the swing (lands further) but heading higher; above the
    catch point, past the elevation of greatest range, heading higher lands shorter. Near the right elevation
    the two cancel, and the laptop's timing jitter stops mattering."""
    elevations = list(elevations)
    tried: dict[float, tuple | None] = {}

    def at(e):
        if e not in tried:
            plan = plan_throw(chain, spec, target_xy_mm, catch_z_mm, float(e), max_speed_deg_s, release_delay_s,
                              speed_factor)
            tried[e] = (plan, forgiveness(chain, spec, plan, catch_z_mm, speed_factor)) if plan.ok else None
        return tried[e]

    def key(e):
        r = at(e)
        return (math.inf, 0.0) if r is None else (round(abs(r[1].late_10ms_mm), 0), abs(e - 45))   # mm, then near 45

    coarse = elevations[::5] + [elevations[-1]]
    best = min(coarse, key=key)
    near = [e for e in elevations if abs(e - best) < 5]                      # then every degree around the best
    best = min(near + [best], key=key)
    return at(best) if at(best) is not None else (None, None)


def design_swing(chain: Chain, joints, amplitude_deg: float, accel_deg_s2: float, target_xy_mm, catch_z_mm: float,
                 keep, rest_deg, tool_mm=(0.0, 0.0, 155.0), max_speed_deg_s: float = 180.0, step_deg: float = 15.0,
                 elevation: float = 45.0, say=None) -> list[tuple[float, SwingSpec, ThrowPlan, Forgiveness]]:
    """Search wind-up poses for the swinging joints (all turning the same way, every step_deg within their
    limits; the other joints as in rest_deg, the base aimed by the planner) for swings that land the ball on
    target_xy_mm (thrower's base frame) with the release in the cruise and the whole swing clear of the table,
    ceiling and walls. Returns the good ones, best first: the least joint speed plus twice the landing shift
    (mm) of a 10 ms late release, which is what the laptop's timing costs. (score, spec, plan, forgiveness)."""
    from worldarm.toss.sweep import motion_problem, pose_problem
    ranges = [np.arange(chain.limits_deg[j][0] + 5.0, chain.limits_deg[j][1] - 5.0 + 1e-9, step_deg) for j in joints]
    grid = np.stack(np.meshgrid(*ranges, indexing="ij"), -1).reshape(-1, len(joints))
    found = []
    for k, start in enumerate(grid):
        if say and k % 500 == 0:
            say(f"  {k}/{len(grid)} wind-up poses tried, {len(found)} good so far")
        for sign in (1, -1):
            q0 = np.array(rest_deg, float)
            q0[list(joints)] = start
            spec = SwingSpec(q0.tolist(), list(joints), [sign] * len(joints), amplitude_deg, accel_deg_s2,
                             tuple(tool_mm), keep.table_z_mm, keep.clearance_mm)
            if pose_problem(chain, q0, tool_mm, keep) or pose_problem(chain, end_pose(spec, q0[0]), tool_mm, keep):
                continue                               # cheap checks first (the base turns both the same way)
            plan = plan_throw(chain, spec, target_xy_mm, catch_z_mm, elevation, max_speed_deg_s, release_step_deg=1.0)
            if not plan.ok:
                continue
            spec.start_deg[0] = plan.base_deg
            if motion_problem(chain, spec.start_deg, end_pose(spec, plan.base_deg), tool_mm, keep, 2.0):
                continue
            f = forgiveness(chain, spec, plan, catch_z_mm)
            found.append((plan.speed_deg_s + 2.0 * abs(f.late_10ms_mm), spec, plan, f))
    found.sort(key=lambda r: r[0])
    return found
