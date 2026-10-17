"""Where to put the paddle, and whether it can get there in time.

The paddle moves along a taught lane (segment A-B in the arm's world, at a fixed height and wrist). The
goalie predicts where the ball will cross the lane and when (the world model), and sends the paddle
there - re-targeting only when the prediction moves by a meaningful amount, because every new command
interrupts the arm. The "chase" mode, for the H5 comparison, follows where the ball is now instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from worldarm.ball.model import RollModel
from worldarm.ball.track import Estimate


@dataclass
class Lane:
    a: np.ndarray                   # (2,) one end, arm world mm
    b: np.ndarray                   # (2,) the other end
    z: float                        # flange height while defending (as taught)
    theta: float                    # flange wrist angle while defending (as taught)
    home: np.ndarray = field(default_factory=lambda: np.zeros(2))  # arm base: the side being defended
    orientation: tuple = (0.0, 0.0, -1.0)  # flange orientation vector as taught (pointing down)

    def __post_init__(self):
        self.a, self.b = np.asarray(self.a, float), np.asarray(self.b, float)
        self.orientation = tuple(float(v) for v in self.orientation)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.b - self.a))

    @property
    def unit(self) -> np.ndarray:
        return (self.b - self.a) / (self.length or 1.0)

    @property
    def normal(self) -> np.ndarray:
        """Unit normal pointing toward the defended side (the arm), i.e. the way an incoming ball travels."""
        n = np.array([-self.unit[1], self.unit[0]])
        return n if float((np.asarray(self.home, float) - self.a) @ n) > 0 else -n

    def along(self, xy) -> float:
        return float((np.asarray(xy, float) - self.a) @ self.unit)

    def point(self, s: float) -> np.ndarray:
        return self.a + self.unit * s

    def clamp(self, s: float) -> float:
        if not math.isfinite(s):
            raise ValueError("not a position on the lane")   # never send a NaN target to the arm
        return float(min(max(s, 0.0), self.length))

    def to_json(self) -> dict:
        return {"a": self.a.tolist(), "b": self.b.tolist(), "z": self.z, "theta": self.theta,
                "home": np.asarray(self.home, float).tolist(), "orientation": list(self.orientation)}

    @classmethod
    def from_json(cls, d: dict) -> Lane:
        return cls(np.array(d["a"]), np.array(d["b"]), float(d["z"]), float(d["theta"]), np.array(d.get("home", [0, 0])),
                   tuple(d.get("orientation", (0.0, 0.0, -1.0))))


@dataclass
class ArmModel:
    """How long the paddle takes to move along the lane: command latency + distance / speed (lane-test)."""
    latency_s: float = 0.25
    speed_mm_s: float = 250.0

    def time_to(self, distance_mm: float) -> float:
        return self.latency_s + abs(distance_mm) / max(self.speed_mm_s, 1.0)


@dataclass
class Decision:
    s: float                        # target along the lane
    t_cross: float | None           # predicted crossing time (predict mode)
    s_cross: float | None           # predicted crossing point along the lane (unclamped)
    in_time: bool | None            # can the paddle get there before the ball
    reason: str


class Goalie:
    """predict: commit once the fit is solid (or time is running out), then re-aim only when the prediction
    moves by more than the paddle's slack or twice its own uncertainty - every new command costs a stop and a
    restart. chase: follow the ball's current position along the lane (the H5 baseline)."""

    def __init__(self, lane: Lane, model: RollModel, arm: ArmModel, half_width_mm: float = 90.0,
                 mode: str = "predict", retarget_mm: float | None = None, ball_radius_mm: float = 25.0,
                 min_commit_points: int = 6, cautious_decel: float | None = None):
        if mode not in ("predict", "chase"):
            raise ValueError(mode)
        self.lane, self.model, self.arm, self.half = lane, model, arm, half_width_mm
        self.mode = mode
        self.retarget = retarget_mm if retarget_mm is not None else half_width_mm / 2.0
        self.reach = half_width_mm + ball_radius_mm      # the paddle covers a ball this far either side of its centre
        self.min_commit = min_commit_points
        # over-estimating the slowdown says "stops short" for balls that arrive (costly); under-estimating is harmless
        self.cautious = RollModel(cautious_decel if cautious_decel is not None else 0.7 * model.decel)
        self.target: float | None = None

    def decide(self, t_now: float, est: Estimate | None, s_now: float) -> Decision | None:
        """A new target for the paddle, or None to keep going where it's going."""
        if est is None or not est.moving or not _finite(est):
            return None
        if self.mode == "chase":
            s = self.lane.clamp(self.lane.along(est.roll.p0))
            return self._maybe(Decision(s, None, None, None, "follow the ball"))
        hit = (self.model.crossing(est.roll, self.lane.a, self.lane.normal)
               or self.cautious.crossing(est.roll, self.lane.a, self.lane.normal))
        if hit is None:
            return None                                     # moving away, or it will stop short even so
        xy, t_cross = hit
        s_cross = self.lane.along(xy)
        if s_cross < -self.reach or s_cross > self.lane.length + self.reach:
            return None                                     # it will miss the lane altogether
        s = self.lane.clamp(s_cross)
        # how sure: the direction is known to about rms/span, over the distance still to go
        to_go = max(0.0, float((self.lane.a - est.roll.p0) @ self.lane.normal))
        sigma = est.rms_mm * (1.0 + to_go / max(est.span_mm, 20.0))
        start = self.target if self.target is not None else s_now
        in_time = t_now + self.arm.time_to(s - start) <= t_cross
        if self.target is None:
            urgent = t_cross - t_now <= self.arm.time_to(s - s_now) + 0.15
            if not urgent and (est.n < self.min_commit or sigma > self.half):
                return None                                 # wait for a better fit while there's time
        return self._maybe(Decision(s, t_cross, s_cross, in_time, "predicted crossing"), 2.0 * sigma)

    def _maybe(self, d: Decision, noise_mm: float = 0.0) -> Decision | None:
        if self.target is not None and abs(d.s - self.target) < max(self.retarget, noise_mm):
            return None
        self.target = d.s
        return d

    def reset(self) -> None:
        self.target = None


def _finite(est: Estimate) -> bool:
    r = est.roll
    return bool(np.all(np.isfinite(r.p0)) and np.all(np.isfinite(r.u)) and math.isfinite(r.v0)
                and math.isfinite(est.rms_mm))
