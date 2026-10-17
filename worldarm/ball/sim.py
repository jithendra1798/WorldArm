"""Rolls, the overhead camera and the arm, simulated - to measure intercept rates before using the robot.

A ball starts at the far end, rolls toward the lane and slows at the true rate. The camera sees it ~23
times a second, a little late, with a few mm of noise and the odd missed frame. The goalie decides from
what the camera saw; the paddle starts moving after the command latency and moves at the lane speed;
a new command interrupts the old one. A roll is defended if the paddle covers the ball when it crosses.

SimCfg's defaults are the setting every test and the study were run with. SimCfg.realistic() adds what the
robot will really do on top (the paddle speeding up and slowing down, a camera that sees only part of the
table, jittery timestamps, uneven rolls, a ball pushed from rest); its numbers are guesses until measured.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from worldarm.ball.intercept import ArmModel, Goalie, Lane
from worldarm.ball.model import Roll, RollModel
from worldarm.ball.session import RollSession
from worldarm.ball.track import Tracker


@dataclass
class SimCfg:
    fps: float = 23.0
    camera_latency_s: float = 0.06
    noise_mm: float = 4.0
    miss_prob: float = 0.05
    ball_radius_mm: float = 25.0
    true_decel: float = 120.0               # mm/s^2, the table's real slowdown
    speed: tuple = (350.0, 900.0)           # mm/s at the start
    start_distance: tuple = (900.0, 1300.0)  # mm from the lane
    aim_spread_mm: float = 60.0             # aimed this far past either lane end at most
    side_mm: float = 400.0                  # starts up to this far to the side of where it's aimed: diagonal rolls
    dt: float = 0.01
    # off by default (see realistic()):
    arm_accel_mm_s2: float = math.inf       # the paddle speeds up and slows down this fast (inf: at once)
    visible_mm: float = math.inf            # the camera sees the ball only this close to the lane
    time_jitter_s: float = 0.0              # frame timestamps are off by this much (1 sd)
    decel_spread: float = 0.0               # the slowdown varies from roll to roll (lognormal, sd in log terms)
    push_s: float = 0.0                     # the ball starts at rest and is pushed up to speed over this long

    @classmethod
    def realistic(cls, **changes) -> "SimCfg":
        """The robot as we expect it, harsher than the defaults. Guesses until measured: arm acceleration
        (lane-test), latency and jitter (ball.py check), the slowdown's spread (ball.py fit)."""
        guess = dict(camera_latency_s=0.10, noise_mm=5.0, miss_prob=0.10, arm_accel_mm_s2=1000.0, visible_mm=900.0,
                     time_jitter_s=0.01, decel_spread=0.25, push_s=0.25)
        return cls(**{**guess, **changes})


@dataclass
class Outcome:
    defended: bool
    reached: bool                            # the ball got to the lane at all
    s_cross: float | None
    paddle_at_cross: float | None
    commands: int
    first_command_s: float | None            # seconds before the crossing when the first command went out
    judged: str | None = None                # what the session (the camera) called it: blocked / goal / short


def roll_once(lane: Lane, goalie: Goalie, true_arm: ArmModel, tracker: Tracker, cfg: SimCfg,
              rng: np.random.Generator) -> Outcome:
    aim_s = rng.uniform(-cfg.aim_spread_mm, lane.length + cfg.aim_spread_mm)
    aim = lane.point(aim_s)
    dist = rng.uniform(*cfg.start_distance)
    lateral = rng.uniform(-cfg.side_mm, cfg.side_mm)
    start = aim - lane.normal * dist + lane.unit * lateral
    u = (aim - start) / np.linalg.norm(aim - start)
    v0 = rng.uniform(*cfg.speed)
    decel = cfg.true_decel * (math.exp(cfg.decel_spread * rng.standard_normal()) if cfg.decel_spread > 0 else 1.0)
    true_model = RollModel(decel)
    roll = Roll(cfg.push_s, start + u * (0.5 * v0 * cfg.push_s), u, v0)   # rolling freely once the push is over
    hit = true_model.crossing(roll, lane.a, lane.normal)
    s_cross = lane.along(hit[0]) if hit else None
    t_end = (hit[1] + 0.8) if hit else roll.t0 + true_model.stop_time(roll) + 0.3

    goalie.reset()
    tracker.reset()
    session = RollSession(lane, goalie, tracker, cfg.ball_radius_mm)
    s_paddle, v_paddle = lane.length / 2.0, 0.0
    move_to, move_start = s_paddle, 0.0
    commands, first_cmd = 0, None
    covered, bounce, paddle_at_cross = None, None, None
    next_frame = 0.0
    pending: list[tuple[float, float, np.ndarray]] = []   # (deliver at, frame time, observed xy)
    t = 0.0
    while t <= t_end:
        if math.isinf(cfg.arm_accel_mm_s2):
            if t >= move_start:                              # the paddle slides toward its latest target
                step = true_arm.speed_mm_s * cfg.dt
                d = move_to - s_paddle
                s_paddle = move_to if abs(d) <= step else s_paddle + np.sign(d) * step
        else:                                                # it brakes while a new command is on its way
            s_paddle, v_paddle = paddle_step(s_paddle, v_paddle, move_to if t >= move_start else None,
                                             true_arm.speed_mm_s, cfg.arm_accel_mm_s2, cfg.dt)
        if hit is not None and covered is None and t >= hit[1]:
            covered = abs(s_paddle - s_cross) <= goalie.half + cfg.ball_radius_mm - 5.0
            paddle_at_cross = s_paddle
            if covered:                                      # it bounces back off the paddle
                v_hit = max(roll.v0 - decel * (hit[1] - roll.t0), 50.0)
                bounce = Roll(hit[1], hit[0], -roll.u, 0.5 * v_hit)
        if t >= next_frame:
            if bounce is not None:
                p = true_model.at(bounce, t)
            elif t < roll.t0:                                # still being pushed, speeding up from rest
                p = start + u * (0.5 * v0 / cfg.push_s * t * t)
            else:
                p = true_model.at(roll, t)
            in_view = float((lane.a - p) @ lane.normal) <= cfg.visible_mm
            if in_view and rng.random() > cfg.miss_prob:
                stamp = t + rng.normal(0, cfg.time_jitter_s) if cfg.time_jitter_s > 0 else t
                pending.append((t + cfg.camera_latency_s, stamp, p + rng.normal(0, cfg.noise_mm, 2)))
            next_frame += 1.0 / cfg.fps
        while pending and pending[0][0] <= t:
            _, tf, p = pending.pop(0)
            dec = session.on_sighting(tf, p, s_paddle)
            if dec is not None:
                commands += 1
                if first_cmd is None and hit:
                    first_cmd = hit[1] - t
                move_to, move_start = dec.s, t + true_arm.latency_s
        t += cfg.dt
    if hit is None:
        return Outcome(True, False, None, None, commands, first_cmd, judged=session.log.outcome)
    outside = s_cross < -goalie.half or s_cross > lane.length + goalie.half
    return Outcome(bool(covered) and not outside, True, s_cross, paddle_at_cross, commands, first_cmd,
                   judged=session.log.outcome)


def paddle_step(s: float, v: float, target: float | None, v_max: float, accel: float,
                dt: float) -> tuple[float, float]:
    """One time step of a paddle that can't change speed faster than accel (mm/s^2): with no target (a stop,
    while the next command is on its way) it brakes to rest; else it speeds up toward the target, up to v_max,
    and brakes to stop on it. Returns the new (position, velocity)."""
    dv = accel * dt
    if target is None:
        v = 0.0 if abs(v) <= dv else v - math.copysign(dv, v)
        return s + v * dt, v
    d = target - s
    if abs(d) <= 0.5 and abs(v) <= dv:
        return target, 0.0
    toward = math.copysign(1.0, d)
    if v * toward < 0:                                       # going the wrong way: brake first
        v += toward * dv
    elif v * v / (2.0 * accel) >= abs(d) - abs(v) * dt:      # it must brake now to stop on the target
        v = 0.0 if abs(v) <= dv else v - math.copysign(dv, v)
    else:
        v = toward * min(abs(v) + dv, v_max)
    s_new = s + v * dt
    if (target - s_new) * toward < 0:                        # it would pass the target: stop on it
        return target, 0.0
    return s_new, v


def intercept_rate(lane: Lane, mode: str, model_decel: float, true_arm: ArmModel, cfg: SimCfg, n: int = 300,
                   seed: int = 0, believed_arm: ArmModel | None = None, half_width_mm: float = 90.0) -> dict:
    """Share of rolls that reach the lane and are defended, for one goalie mode and one believed model."""
    rng = np.random.default_rng(seed)
    model = RollModel(model_decel)
    goalie = Goalie(lane, model, believed_arm or true_arm, half_width_mm, mode)
    tracker = Tracker(decel=model_decel)
    outs = [roll_once(lane, goalie, true_arm, tracker, cfg, rng) for _ in range(n)]
    reached = [o for o in outs if o.reached and o.s_cross is not None
               and -half_width_mm <= o.s_cross <= lane.length + half_width_mm]
    k = sum(o.defended for o in reached)
    lead = [o.first_command_s for o in reached if o.first_command_s is not None]
    return {"mode": mode, "decel": model_decel, "defended": k, "reached": len(reached),
            "rate": k / len(reached) if reached else 0.0,
            "median_lead_s": float(np.median(lead)) if lead else None}


def prediction_error(model_decel: float, cfg: SimCfg, lane: Lane, n: int = 200, seed: int = 0,
                     horizons=(1.5, 1.0, 0.6, 0.3)) -> dict:
    """H4, timing: how far off the predicted arrival time at the lane is (s), by how long before the
    arrival it was made. (Where a straight roll crosses doesn't depend on its slowdown; when it does.)
    A prediction that the ball stops short when it does arrive counts as an infinite error: it stays in the
    median as a failure. The sightings have no misses or latency (see _observe)."""
    rng = np.random.default_rng(seed)
    true_model, model = RollModel(cfg.true_decel), RollModel(model_decel)
    errs = {h: [] for h in horizons}
    for _ in range(n):
        aim = lane.point(rng.uniform(0, lane.length))
        start = aim - lane.normal * rng.uniform(*cfg.start_distance)
        u = (aim - start) / np.linalg.norm(aim - start)
        roll = Roll(0.0, start, u, rng.uniform(*cfg.speed))
        hit = true_model.crossing(roll, lane.a, lane.normal)
        if hit is None:
            continue
        for h in horizons:
            est = _observe(true_model, roll, hit[1] - h, model_decel, cfg, rng)
            if est is None:
                continue                                     # too early: not enough seen yet
            pred = model.crossing(est.roll, lane.a, lane.normal)
            errs[h].append(math.inf if pred is None else abs(pred[1] - hit[1]))
    return {h: (float(np.median(v)) if v else None, len(v)) for h, v in errs.items()}


def stop_error(model_decel: float, cfg: SimCfg, n: int = 200, seed: int = 0, after_s: float = 0.5) -> tuple[float, int]:
    """H4, where it stops: median error (mm) of the predicted resting point, after_s seconds into the roll,
    over the balls still rolling then (where a stopped ball stops is no prediction). A constant-velocity model
    can't predict a stop at all (returns inf)."""
    if model_decel <= 0:
        return float("inf"), 0
    rng = np.random.default_rng(seed)
    true_model, model = RollModel(cfg.true_decel), RollModel(model_decel)
    errs = []
    for _ in range(n):
        u = np.array([np.cos(a := rng.uniform(-0.6, 0.6)), np.sin(a)])
        roll = Roll(0.0, np.zeros(2), u, rng.uniform(150.0, 450.0))
        est = _observe(true_model, roll, after_s, model_decel, cfg, rng)
        if est is None or after_s >= true_model.stop_time(roll):
            continue
        p = model.stop_point(est.roll)
        errs.append(float(np.linalg.norm(p - true_model.stop_point(roll))))
    return (float(np.median(errs)) if errs else float("inf")), len(errs)


def _observe(true_model, roll, t_until: float, model_decel: float, cfg: SimCfg, rng) -> object:
    """The tracker's estimate from 0.6 s of noisy camera sightings ending at t_until (every frame, no latency:
    the H4 measures isolate the model; the robot's own camera is judged by H4 on real rolls, holdout.py)."""
    if t_until < 0.15:
        return None
    tracker = Tracker(decel=model_decel)
    for tf in np.arange(max(0.0, t_until - 0.6), t_until, 1.0 / cfg.fps):
        tracker.add(tf, true_model.at(roll, tf) + rng.normal(0, cfg.noise_mm, 2))
    return tracker.estimate()
