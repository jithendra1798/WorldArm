"""One roll, frame by frame: the logic the simulator and the robot share.

Feed it every sighting (world x, y on the table, or None when the camera missed the ball) with the paddle's
position along the lane. It tracks the roll, returns a new paddle target when the goalie decides one, and
judges the outcome from the sightings themselves (not from the model fit, which mixes the way in with the
way back):

  blocked   it came near the lane and was then seen going back: further from the lane than its closest
            approach, twice in a row (also when it reappears after being hidden next to the paddle); or it
            came to rest against the paddle (or in front of it, after vanishing at it); or it vanished
            heading into the paddle, the paddle was there when it should have crossed, and it wasn't seen
            past the lane in the second after that
  goal      it was seen past the lane
  short     it came to rest anywhere else (not counted either way)

Nothing is called short before the ball has started rolling toward the lane: a ball still waiting at the
far end hasn't been rolled yet. A sighting too far from the last one to be the same ball is left out of
the judging and of the tracking. Live, the camera ignores the picture around the paddle, so the ball can
vanish before it reaches the lane; `near_mm` is set to match.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from worldarm.ball.intercept import Decision, Goalie, Lane
from worldarm.ball.model import RollModel
from worldarm.ball.track import Tracker

BACK_MM = 15.0        # going back: this much further from the lane than its closest approach
PAST_MM = 15.0        # a goal: its centre more than its radius + this past the lane
TOUCH_MM = 20.0       # at rest within its radius + this of the lane, in front of the paddle: the paddle stopped it
STILL_MM = 6.0        # at rest: it moved less than this ...
STILL_S = 0.4         # ... over this long
START_MM = 40.0       # rolling: this much closer to the lane than the furthest it was seen
REST_MM = 12.0        # sightings this close to where it waited are the wait, not the roll
LOST_S = 2.0          # rolling and then unseen this long: lost (stopped out of sight, or picked up)
HIDDEN_S = 1.0        # vanished heading into the paddle, unseen past the lane this long after it should have crossed
JUMP_MM = 120.0       # the same ball is never further than this + MAX_SPEED * dt from where it was last seen
MAX_SPEED = 2000.0    # mm/s


@dataclass
class RollLog:
    sightings: list = field(default_factory=list)   # (t, x, y)
    decisions: list = field(default_factory=list)   # (t, target_s, t_cross, s_cross, in_time)
    outcome: str | None = None
    s_cross: float | None = None                    # where it reached the lane, along it (goal, blocked)
    t_outcome: float | None = None
    t_start: float | None = None                    # when it started rolling toward the lane

    def to_json(self) -> dict:
        return {"outcome": self.outcome, "s_cross": self.s_cross, "t_outcome": self.t_outcome,
                "t_start": self.t_start,
                "sightings": [[round(a, 4), round(b, 1), round(c, 1)] for a, b, c in self.sightings],
                "decisions": [[round(t, 4), round(s, 1), None if tc is None else round(tc, 4),
                               None if sc is None else round(sc, 1), it] for t, s, tc, sc, it in self.decisions]}


class RollSession:
    def __init__(self, lane: Lane, goalie: Goalie, tracker: Tracker, ball_radius_mm: float = 25.0,
                 near_mm: float | None = None):
        self.lane, self.goalie, self.tracker, self.r = lane, goalie, tracker, ball_radius_mm
        self.near_mm = near_mm if near_mm is not None else ball_radius_mm + 60.0
        self.log = RollLog()
        self._recent: deque[tuple[float, np.ndarray]] = deque()   # judged sightings, the last STILL_S
        self._last: tuple[float, np.ndarray, float] | None = None  # the last judged sighting (t, xy, gap)
        self._closest, self._closest_xy = np.inf, None
        self._furthest, self._furthest_xy = -np.inf, None
        self._back = False                                         # the last judged sighting was going back
        self._entry: list | None = None    # heading in: [where along the lane, when] it will cross, paddle there?
        self._went_in = False              # it was out of sight when it should have been crossing (at the paddle)

    @property
    def t_start(self) -> float | None:
        return self.log.t_start

    def gap(self, xy) -> float:
        """Distance still to go to the lane, along its normal (negative once past it)."""
        return float((self.lane.a - np.asarray(xy, float)) @ self.lane.normal)

    def lost(self, t: float) -> bool:
        """It started rolling and hasn't been seen for LOST_S: past judging (a ball turned back by the paddle
        reappears well before that; a ball seen again only after someone picked it up proves nothing)."""
        return (self.log.t_start is not None and self._last is not None and t - self._last[0] > LOST_S
                and self._entry is None)

    def on_sighting(self, t: float, xy, s_paddle: float, paddle_known: bool = True) -> Decision | None:
        """One frame: the ball's table position (None if not seen), and where the paddle is along the lane
        (paddle_known: measured just now, not only where it was sent)."""
        if self.log.outcome is not None:
            return None
        if xy is None:
            self._unseen(t, s_paddle, paddle_known)
            return None
        xy = np.asarray(xy, float)
        self.log.sightings.append((t, float(xy[0]), float(xy[1])))
        waiting = self.log.t_start is None
        if not self._judge(t, xy, s_paddle):
            self._unseen(t, s_paddle, paddle_known)      # not the ball: as good as not seen
            return None
        if self.log.outcome is not None:
            return None
        if waiting and self.log.t_start is not None:    # it has just been rolled
            rest = [ts for ts, p in self._recent if float(np.linalg.norm(p - self._furthest_xy)) <= REST_MM]
            if len(rest) >= 3:                           # after waiting at rest: fit the roll, not the wait before it
                self.tracker.reset()
                for ts, p in list(self._recent)[:-1]:
                    if ts > max(rest):
                        self.tracker.add(ts, p)
        self.tracker.add(t, xy)
        est = self.tracker.estimate()
        self._heading_in(est)
        dec = self.goalie.decide(t, est, s_paddle)
        if dec is not None:
            self.log.decisions.append((t, dec.s, dec.t_cross, dec.s_cross, dec.in_time))
        return dec

    def _heading_in(self, est) -> None:
        """Close to the lane and heading for it: remember where and when it will cross (straight on: the
        slowdown hardly matters this close), in case it vanishes next to the paddle."""
        self._entry = None
        if est is None or not est.moving or self._last[2] > self.near_mm:
            return
        hit = RollModel(0.0).crossing(est.roll, self.lane.a, self.lane.normal)
        if hit is not None:
            self._entry = [self.lane.along(hit[0]), hit[1], None]

    def _unseen(self, t: float, s_paddle: float, paddle_known: bool) -> None:
        """A frame without the ball. If it vanished heading into the paddle, the paddle was there all the time
        from then until it should have crossed, and it isn't seen past the lane by HIDDEN_S after that: the
        paddle stopped it. (A paddle that got there only just after can't be told from one that was late.)"""
        e = self._entry
        if e is None:
            return
        if e[2] is None or t <= e[1]:                     # the paddle there from when it vanished till it crossed
            here = paddle_known and abs(s_paddle - e[0]) <= self.goalie.half + self.r / 2
            e[2] = here if e[2] is None else e[2] and here
        if t < e[1]:
            return
        self._went_in = True
        if t >= e[1] + HIDDEN_S:
            self._entry = None
            if e[2]:
                self._finish("blocked", t, self.lane.point(e[0]))

    def _judge(self, t: float, xy: np.ndarray, s_paddle: float) -> bool:
        """Judge one sighting; False if it can't be the ball (too far from where the ball just was)."""
        last = self._last
        if last is not None and float(np.linalg.norm(xy - last[1])) > JUMP_MM + MAX_SPEED * (t - last[0]):
            return False                                   # something else, or a glitch: not the ball
        d = self.gap(xy)
        self._last = (t, xy, d)
        self._recent.append((t, xy))
        while t - self._recent[0][0] > STILL_S:
            self._recent.popleft()
        if d < -(self.r + PAST_MM) and last is not None:
            self._finish("goal", t, self._crossing(last, xy, d))
            return True
        if d > self._furthest:
            self._furthest, self._furthest_xy = d, xy
        if self.log.t_start is None and self._furthest - d >= START_MM:
            self.log.t_start = t
        back = self._closest <= self.near_mm and d > self._closest + BACK_MM
        if back and self._back:
            self._finish("blocked", t, self._closest_xy)
            return True
        self._back = back
        if d < self._closest:
            self._closest, self._closest_xy = d, xy
        if self.log.t_start is not None and self._still():
            # at rest in front of the paddle: against it, or back out of sight of it (it went in and came out)
            touching = ((d <= self.r + TOUCH_MM or self._went_in)
                        and abs(self.lane.along(xy) - s_paddle) <= self.goalie.half + self.r)
            self._finish("blocked" if touching else "short", t, xy)
        return True

    def _still(self) -> bool:
        """At rest: over the last STILL_S, where it was at the start and at the end (averaged) agree."""
        seen = self._recent
        if len(seen) < 5 or seen[-1][0] - seen[0][0] < 0.75 * STILL_S:
            return False
        k = max(2, len(seen) // 3)
        first = np.mean([p for _, p in list(seen)[:k]], axis=0)
        end = np.mean([p for _, p in list(seen)[-k:]], axis=0)
        return float(np.linalg.norm(end - first)) < STILL_MM

    def _crossing(self, last: tuple[float, np.ndarray, float], xy: np.ndarray, d: float) -> np.ndarray:
        """Where it crossed the lane, between the last sighting before and the first one past it."""
        _, p0, d0 = last
        if d0 <= 0:
            return xy
        return p0 + (xy - p0) * (d0 / (d0 - d))

    def _finish(self, outcome: str, t: float, xy) -> None:
        self.log.outcome, self.log.t_outcome = outcome, t
        self.log.s_cross = self.lane.along(xy)
