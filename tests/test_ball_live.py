"""The goalie's robot loop (live.py) against a fake overhead camera and a fake arm, in real time: frames,
detection, table map, re-targeting that interrupts moves, ignoring the paddle, judging the roll."""

import asyncio
import time

import cv2
import numpy as np
import pytest
from viam.proto.common import Pose

from worldarm.ball.config import BallCfg
from worldarm.ball.intercept import ArmModel, Goalie, Lane
from worldarm.ball.live import LaneArm, defend, sightings
from worldarm.ball.model import Roll, RollModel
from worldarm.ball.table import TableMap

pytestmark = pytest.mark.sim  # real time: `pytest -m sim`

LANE = Lane(np.array([450.0, -250.0]), np.array([450.0, 250.0]), 200.0, 0.0, home=np.array([0.0, 0.0]))
TABLE = TableMap(np.array([[0.0, -2.0, 1300.0], [2.0, 0.0, -360.0], [0.0, 0.0, 1.0]]))  # 2 mm per pixel
DECEL, HALF, R = 120.0, 60.0, 25.0   # the ball's slowdown, the paddle's half width, the ball's radius


class FakeArm:
    """move_to_position with a start-up delay and a top speed; a cancelled move stops where it is."""

    def __init__(self, latency=0.2, speed=300.0):
        self.latency, self.speed = latency, speed
        self.p = LANE.point(LANE.length / 2)
        self.moves = 0

    async def do_command(self, cmd):
        return {}

    async def move_to_position(self, pose: Pose):
        self.moves += 1
        await asyncio.sleep(self.latency)
        goal = np.array([pose.x, pose.y])
        while np.linalg.norm(goal - self.p) > 1.0:
            await asyncio.sleep(0.01)
            step = goal - self.p
            self.p = self.p + step * min(1.0, self.speed * 0.01 / np.linalg.norm(step))

    async def get_end_position(self):
        return Pose(x=float(self.p[0]), y=float(self.p[1]), z=LANE.z, o_z=-1.0)

    async def stop(self):
        pass


class FakeOverhead:
    """~23 frames a second from wall-clock time: the ball (waiting `wait_s`, then rolling; it bounces back
    off the paddle if the paddle covers it where it crosses the lane) and the gripper above the paddle, a
    dark disc as round as the ball that only the paddle zone keeps from being taken for it."""

    def __init__(self, arm: FakeArm, start, aim, v0: float, wait_s: float = 0.0, bounce: float = 200.0):
        start, aim = np.asarray(start, float), np.asarray(aim, float)
        self.arm, self.model, self.bounce_speed = arm, RollModel(DECEL), bounce
        self.roll = Roll(time.monotonic() + wait_s, start, (aim - start) / np.linalg.norm(aim - start), v0)
        self.hit = self.model.crossing(self.roll, LANE.a, LANE.normal)
        self.covered, self.bounce = None, None

    def ball_xy(self, t):
        if t < self.roll.t0:
            return self.roll.p0
        if self.hit is not None and self.covered is None and t >= self.hit[1]:
            self.covered = abs(LANE.along(self.arm.p) - LANE.along(self.hit[0])) <= HALF + R - 5
            if self.covered:
                self.bounce = Roll(self.hit[1], self.hit[0], -self.roll.u, self.bounce_speed)
        return self.model.at(self.bounce if self.bounce is not None else self.roll, t)

    async def frame(self):
        await asyncio.sleep(1 / 23)
        t = time.monotonic()
        img = np.full((720, 1280, 3), 225, np.uint8)
        for xy in (self.arm.p, self.ball_xy(t)):
            u, v = TABLE.to_pixel(xy)
            cv2.circle(img, (int(u), int(v)), 13, (20, 20, 20), -1)
        return t, img


def run_roll(mode: str, start, aim, v0: float = 900.0, wait_s: float = 0.0, timeout_s: float = 3.0, arm=None,
             bounce: float = 200.0):
    """One roll through live.defend; returns (the roll's log, the fake arm, the fake camera)."""
    async def go():
        a = arm or FakeArm()
        cam = FakeOverhead(a, start, aim, v0, wait_s, bounce)
        cfg = BallCfg(decel_mm_s2=DECEL, camera_latency_s=0.0, paddle_half_width_mm=HALF, ball_radius_mm=R)
        avoid: list = []
        stream = sightings(cam, TABLE, cfg, avoid=avoid, lane=LANE)
        goalie = Goalie(LANE, RollModel(DECEL), ArmModel(0.2, 300.0), HALF, mode)
        log = await defend(stream, LANE, goalie, LaneArm(a, LANE, 30.0), cfg, timeout_s=timeout_s,
                           say=lambda *_: None, avoid=avoid, table=TABLE)
        return log, a, cam
    return asyncio.run(go())


def test_live_goalie_blocks_a_diagonal_roll():
    """The ball waits at the far end first (not judged short), then rolls; it disappears for a moment at the
    paddle (the camera ignores the paddle's surroundings) and is judged blocked when it's seen going back."""
    log, arm, cam = run_roll("predict", (1250.0, 230.0), (450.0, -170.0), wait_s=0.6)
    assert cam.covered and log.outcome == "blocked" and 1 <= arm.moves <= 3, (log.outcome, arm.moves)
    assert log.t_start is not None and log.t_start >= cam.roll.t0 - 0.05


def test_live_chase_follows_the_ball_and_blocks_a_straight_roll():
    log, arm, cam = run_roll("chase", (1250.0, -120.0), (450.0, -120.0))
    assert log.outcome == "blocked" and arm.moves >= 1, (log.outcome, arm.moves)
    assert abs(LANE.along(arm.p) - 130.0) < 35.0                       # followed it to where it was going


def test_live_slow_roll_stops_short_and_the_paddle_stays():
    log, arm, _ = run_roll("predict", (1250.0, 50.0), (450.0, 50.0), v0=240.0)
    assert log.outcome == "short" and arm.moves == 0, (log.outcome, arm.moves)
    assert abs(LANE.along(arm.p) - LANE.length / 2) < 1.0


def test_live_roll_aimed_past_the_lane_end_is_not_chased_or_blocked():
    log, arm, cam = run_roll("predict", (1250.0, 0.0), (450.0, 600.0), timeout_s=2.0)
    assert not cam.covered and arm.moves == 0 and log.outcome in ("goal", None), (log.outcome, arm.moves)


def test_live_ball_stopped_dead_by_the_paddle_is_blocked():
    """It doesn't bounce: it stays where the camera can't see it, against the paddle. Heading in, the paddle
    measured there, not seen past the lane a second later: blocked."""
    log, arm, cam = run_roll("predict", (1250.0, -50.0), (450.0, -50.0), v0=700.0, bounce=0.0)
    assert cam.covered and log.outcome == "blocked" and abs(log.s_cross - 200.0) < 30.0, (log.outcome, log.s_cross)


def test_live_goal_past_a_paddle_that_is_too_slow():
    log, arm, cam = run_roll("predict", (1250.0, -220.0), (450.0, -220.0), v0=900.0, arm=FakeArm(speed=100.0))
    assert cam.covered is False and arm.moves >= 1 and log.outcome == "goal", (log.outcome, arm.moves)
