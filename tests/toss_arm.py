"""Task C test kit: an xArm-like arm model, a simulated clock, and fakes of armfarm05 (arm + gripper) and the
overhead camera that follow the physics: the arm turns its joints with a trapezoid in speed after a start
delay; the gripper starts opening a moment after it's told; the ball comes free when the fingers have opened
a little, leaves with the arm's real velocity at that instant, and the camera sees it where it comes down."""

import asyncio
import json
import math
import selectors

import cv2
import numpy as np
from viam.proto.common import Pose
from viam.proto.component.arm import JointPositions

from worldarm.ball.table import TableMap
from worldarm.toss.kinematics import Chain
from worldarm.toss.landing import ArmsFrame
from worldarm.toss.physics import Release, fly
from worldarm.toss.profile import Trapezoid

# Shoulder, elbow and wrist pitch parallel (J2, J3, J5), rolls J4 and J6: an xArm850's layout, roughly its sizes.
# The flange frame isn't turned, so the gripper points along -z at the zero pose: the ball is at (0, 0, -155).
XARMISH = {
    "name": "xarmish", "kinematic_param_type": "SVA",
    "links": [
        {"id": "base", "parent": "world", "translation": {"x": 0, "y": 0, "z": 0}},
        {"id": "l1", "parent": "j1", "translation": {"x": 0, "y": 0, "z": 364}},
        {"id": "l2", "parent": "j2", "translation": {"x": 0, "y": 0, "z": 390}},
        {"id": "l3", "parent": "j3", "translation": {"x": 426, "y": 0, "z": 0}},
        {"id": "l4", "parent": "j4", "translation": {"x": 0, "y": 0, "z": 0}},
        {"id": "l5", "parent": "j5", "translation": {"x": 0, "y": 0, "z": -90}},
        {"id": "flange", "parent": "j6", "translation": {"x": 0, "y": 0, "z": 0}},
    ],
    "joints": [
        {"id": "j1", "type": "revolute", "parent": "base", "axis": {"x": 0, "y": 0, "z": 1}, "min": -360, "max": 360},
        {"id": "j2", "type": "revolute", "parent": "l1", "axis": {"x": 0, "y": 1, "z": 0}, "min": -118, "max": 120},
        {"id": "j3", "type": "revolute", "parent": "l2", "axis": {"x": 0, "y": 1, "z": 0}, "min": -225, "max": 11},
        {"id": "j4", "type": "revolute", "parent": "l3", "axis": {"x": 1, "y": 0, "z": 0}, "min": -360, "max": 360},
        {"id": "j5", "type": "revolute", "parent": "l4", "axis": {"x": 0, "y": 1, "z": 0}, "min": -97, "max": 180},
        {"id": "j6", "type": "revolute", "parent": "l5", "axis": {"x": 0, "y": 0, "z": -1}, "min": -360, "max": 360},
    ],
}
CHAIN = Chain.from_sva(XARMISH)
TOOL = (0.0, 0.0, -155.0)
# A swing the planner's search finds for this arm: wound back over the shoulder, thrown overhand.
SWING = {"start_deg": [0.0, -113.0, -80.0, 0.0, -32.0, 0.0], "joints": [1, 2, 4], "signs": [1, 1, 1],
         "amplitude_deg": 90.0, "accel_deg_s2": 600.0, "tool_mm": TOOL}
BALL_MM = 50.0
TABLE = TableMap(np.array([[0.0, -2.0, 1300.0], [2.0, 0.0, -360.0], [0.0, 0.0, 1.0]]))   # 2 mm per pixel
ARMS = ArmsFrame(1780.0, 0.0, 180.0)        # armfarm05 faces armfarm03 across the table


class SimClock(asyncio.SelectorEventLoop):
    """An event loop whose clock jumps straight to the next timer instead of waiting for it."""

    def __init__(self):
        self._t = 0.0
        sel = selectors.DefaultSelector()
        real = sel.select

        def select(timeout=None):
            ready = real(0)
            if ready or timeout == 0:
                return ready
            if timeout is None:
                return real(None)
            self._t += timeout
            return []

        sel.select = select
        super().__init__(sel)

    def time(self) -> float:
        return self._t


def run(coro):
    return asyncio.run(coro, loop_factory=SimClock)


def now() -> float:
    return asyncio.get_running_loop().time()


class FakeArm:
    """armfarm05's arm. A joint move starts start_latency after the command (plus the next of `jitter`), then
    turns all joints together along the straight line with a trapezoid of the set speed and acceleration
    (its own default acceleration if it won't take one). A joint reading takes rtt; the arm reads its joints
    halfway. stop() halts it where it is."""

    def __init__(self, q=None, start_latency=0.08, jitter=(0.0,), rtt=0.02, accel_default=900.0, takes_accel=True):
        self.q = np.array(q if q is not None else SWING["start_deg"], float)
        self.speed, self.accel, self.accel_default, self.takes_accel = 20.0, accel_default, accel_default, takes_accel
        self.start_latency, self.jitter, self.rtt = start_latency, list(jitter), rtt
        self.moves = 0
        self.calls: list = []
        self.motion = None                         # (t_start, q_from, q_to, Trapezoid) of the move under way
        self.stopped = False

    async def do_command(self, cmd):
        self.calls.append(("do_command", cmd))
        if "set_speed" in cmd:
            self.speed = float(cmd["set_speed"])
        if "set_acceleration" in cmd:
            if not self.takes_accel:
                raise RuntimeError("unknown command set_acceleration")
            self.accel = float(cmd["set_acceleration"])
        return {}

    def state(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Joints and joint speeds (deg, deg/s) at time t."""
        if self.motion is None:
            return self.q.copy(), np.zeros_like(self.q)
        t0, a, b, prof = self.motion
        d = b - a
        span = float(np.max(np.abs(d))) or 1.0
        p = prof.progress(t - t0)
        rate = (prof.progress(t - t0 + 1e-4) - prof.progress(t - t0 - 1e-4)) / 2e-4
        return a + d * (p / span), d / span * rate

    async def move_to_joint_positions(self, positions: JointPositions, **kw):
        self.calls.append(("move", list(positions.values)))
        lag = self.start_latency + self.jitter[self.moves % len(self.jitter)]
        self.moves += 1
        self.stopped = False
        await asyncio.sleep(lag)
        a, b = self.q.copy(), np.array(positions.values, float)
        span = float(np.max(np.abs(b - a)))
        if span < 1e-9:
            return
        prof = Trapezoid(span, self.speed, self.accel)
        self.motion = (now(), a, b, prof)
        try:
            while not self.stopped and now() - self.motion[0] < prof.duration:
                await asyncio.sleep(0.002)
                self.q = self.state(now())[0]
            if not self.stopped:
                self.q = b
        finally:
            self.q, self.motion = self.state(now())[0], None

    async def get_joint_positions(self, **kw):
        await asyncio.sleep(self.rtt / 2)
        q = self.state(now())[0] if self.motion else self.q.copy()
        await asyncio.sleep(self.rtt / 2)
        return JointPositions(values=[float(v) for v in q])

    async def get_end_position(self, **kw):
        M, _ = CHAIN.frames(self.q)
        z = M[:3, 2]
        return Pose(x=float(M[0, 3]), y=float(M[1, 3]), z=float(M[2, 3]), o_x=float(z[0]), o_y=float(z[1]),
                    o_z=float(z[2]), theta=0.0)

    async def get_kinematics(self, **kw):
        return 1, json.dumps(XARMISH).encode(), {}

    async def stop(self, **kw):
        self.calls.append(("stop", None))
        self.stopped = True


class FakeGripper:
    """armfarm05's gripper: a "set" starts moving the fingers open_delay after it arrives (rtt / 2), at
    speed_mm_s. Closed below the ball's size with a ball put between the fingers, it holds it (on_grip); opening
    free_mm past that, the ball comes free (on_free(t))."""

    def __init__(self, mm=84.0, open_delay=0.03, speed_mm_s=200.0, rtt=0.02, free_mm=2.0, on_free=None, on_grip=None):
        self.mm, self.target, self.open_delay, self.speed, self.rtt = mm, mm, open_delay, speed_mm_s, rtt
        self.free_mm, self.on_free, self.on_grip = free_mm, on_free, on_grip
        self.holding_at: float | None = None     # finger opening when the ball was gripped
        self.calls: list = []
        self._task = None

    async def _drive(self, target, start_at):
        await asyncio.sleep(max(0.0, start_at - now()))
        while abs(self.mm - target) > 1e-6:
            await asyncio.sleep(0.002)
            step = self.speed * 0.002
            self.mm = target if abs(target - self.mm) <= step else self.mm + math.copysign(step, target - self.mm)
            if self.holding_at is not None and target > self.holding_at and self.mm >= self.holding_at + self.free_mm:
                t, self.holding_at = now(), None
                if self.on_free:
                    self.on_free(t)

    async def do_command(self, cmd, **kw):
        self.calls.append(cmd)
        if "set" in cmd:
            target = float(cmd["set"]) / 850.0 * 84.0
            await asyncio.sleep(self.rtt / 2)
            if target < BALL_MM and self.holding_at is None:
                self.holding_at = target           # closing on a (new) ball
                if self.on_grip:
                    self.on_grip()
            if self._task and not self._task.done():
                self._task.cancel()
            self._task = asyncio.ensure_future(self._drive(target, now() + self.open_delay))
            await asyncio.sleep(self.rtt / 2)
            return {}
        if "get" in cmd:
            await asyncio.sleep(self.rtt / 2)
            pos = self.mm / 84.0 * 850.0
            await asyncio.sleep(self.rtt / 2)
            return {"pos": pos}
        return {}


class World:
    """The arm, the gripper and the ball together: when the ball comes free it flies from where the arm had
    it, with the arm's velocity there (times speed_factor: the ball rolling off the fingers), lands, and the
    overhead camera (frame()) sees it there in the catcher's frame (`arms`: where the thrower really is)."""

    def __init__(self, arm: FakeArm, speed_factor=1.0, arms: ArmsFrame = ARMS, catch_z_mm=25.0, table=TABLE):
        self.arm, self.k, self.arms, self.catch_z, self.table = arm, speed_factor, arms, catch_z_mm, table
        self.gripper = FakeGripper(on_free=self.free, on_grip=self.picked_up)
        self.ball_catcher_mm = None
        self.t_land = None
        self.released = []

    def picked_up(self) -> None:
        self.ball_catcher_mm = self.t_land = None   # the last ball was picked up to throw again

    def free(self, t: float) -> None:
        q, qd = self.arm.state(t)
        p = CHAIN.point(q, TOOL)
        v = CHAIN.velocity(q, qd, TOOL) * self.k
        land = fly(Release(p / 1000.0, v / 1000.0), self.catch_z / 1000.0)
        self.released.append((t, q, qd))
        if land is not None:
            self.ball_catcher_mm = self.arms.to_catcher(land.xy * 1000.0)
            self.t_land = t + land.t

    async def frame(self):
        await asyncio.sleep(1 / 23)
        img = np.full((720, 1280, 3), 235, np.uint8)
        if self.ball_catcher_mm is not None and now() >= self.t_land:
            u, v = self.table.to_pixel(self.ball_catcher_mm)
            cv2.circle(img, (int(round(u)), int(round(v))), 12, (25, 25, 25), -1)
        return now(), img
