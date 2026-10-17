"""Waj's throw (throw/real_throw_armfarm05.py), on our kinematics and without scipy or pybullet.

One smooth joint spline through poses taught on the Viam control page (start -> second -> last, no stop at
second), time-scaled so the joint speeds, joint accelerations and the tool speed (the flange and a point 175 mm
out, covering any controller TCP) stay under the limits: the xArm faults with "Linear speed exceeds limit in
ServoJ mode" above its tool ceiling (~1 m/s). Streamed to the arm; the gripper opens near peak ball speed from
its own thread and Viam connection, timed from the stream start (the stream's acks only come back at the end).
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from datetime import timedelta

import numpy as np

from worldarm.toss.kinematics import Chain

# Viam Control-page samples on armfarm05 (deg), Waj's
POSES = {
    "start": [4.33, -27.34, -29.06, 77.01, -29.26, -165.54],
    "second": [39.60, -27.35, -44.68, 122.56, -41.74, -158.10],
    "last": [53.96, -27.35, -78.44, 122.15, -52.40, -148.37],
}
MAX_SPEED, MAX_ACCEL = 180.0, 1145.0      # viam:ufactory documented ceilings
BALL_FROM_FLANGE_MM = 105.0 + 55.0 - 6.0  # gripper frame + TCP to tip - grasp lower: where the ball is held


def clamped_spline(knots, Q):
    """Cubic spline through (knots, Q) (Q: (n, joints)), zero velocity at both ends. Returns f(t, d) for the
    value (d=0), first or second derivative (d=1, 2)."""
    t, Q = np.asarray(knots, float), np.asarray(Q, float)
    n, h = len(t), np.diff(t)
    m = np.zeros_like(Q)                          # knot slopes; clamped: 0 at the ends
    if n > 2:
        A = np.zeros((n - 2, n - 2))
        rhs = np.zeros((n - 2, Q.shape[1]))
        for i in range(1, n - 1):
            k = i - 1
            if k > 0:
                A[k, k - 1] = h[i]
            A[k, k] = 2.0 * (h[i - 1] + h[i])
            if k < n - 3:
                A[k, k + 1] = h[i - 1]
            rhs[k] = 3.0 * (h[i] * (Q[i] - Q[i - 1]) / h[i - 1] + h[i - 1] * (Q[i + 1] - Q[i]) / h[i])
        m[1:-1] = np.linalg.solve(A, rhs)

    def f(tt, d: int = 0):
        tt = np.atleast_1d(np.asarray(tt, float))
        i = np.clip(np.searchsorted(t, tt, side="right") - 1, 0, n - 2)
        hi = h[i][:, None]
        s = ((tt - t[i]) / h[i])[:, None]
        if d == 0:
            b = (2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2)
            k = 1.0
        elif d == 1:
            b = (6 * s**2 - 6 * s, 3 * s**2 - 4 * s + 1, -6 * s**2 + 6 * s, 3 * s**2 - 2 * s)
            k = 1.0 / hi
        else:
            b = (12 * s - 6, 6 * s - 4, -12 * s + 6, 6 * s - 2)
            k = 1.0 / hi**2
        out = (b[0] * Q[i] + b[1] * hi * m[i] + b[2] * Q[i + 1] + b[3] * hi * m[i + 1]) * k
        return out if len(out) > 1 else out[0]
    return f


def point(chain: Chain, q, d_mm: float) -> np.ndarray:
    """A point d_mm out along the flange's z axis (0: the flange)."""
    M, _ = chain.frames(q)
    return M[:3, 3] + d_mm * M[:3, 2]


def plan(chain: Chain, q_start, joint_speed: float = 0.9 * MAX_SPEED, joint_accel: float = MAX_ACCEL,
         tcp_limit: float = 1000.0, release_shift_s: float = -0.2, dt: float = 0.01, poses=POSES) -> dict:
    """Waj's plan_throw: the spline start -> second -> last, uniformly slowed until every limit holds; the
    release at peak ball speed, shifted by release_shift_s."""
    Q = np.array([q_start, poses["second"], poses["last"]], float)
    seg = np.max(np.abs(np.diff(Q, axis=0)), axis=1)
    knots = np.concatenate([[0.0], np.cumsum(seg)]) / 100.0
    for _ in range(6):
        cs = clamped_spline(knots, Q)
        tt = np.linspace(0, knots[-1], 400)
        P = np.array([[point(chain, q, d) for d in (0.0, 175.0)] for q in cs(tt)])
        tool = np.linalg.norm(np.diff(P, axis=0), axis=2).max(1).max() / (tt[1] - tt[0])
        k = max(np.abs(cs(tt, 1)).max() / joint_speed, math.sqrt(np.abs(cs(tt, 2)).max() / joint_accel), tool / tcp_limit)
        knots = knots * k
    cs = clamped_spline(knots, Q)
    times = np.append(np.arange(0.0, knots[-1] - dt / 2, dt), knots[-1])
    qs = cs(times)
    ball = np.array([point(chain, q, BALL_FROM_FLANGE_MM) for q in qs])
    vel = np.gradient(ball, times, axis=0)
    i = int(np.argmax(np.linalg.norm(vel, axis=1)))
    i = int(np.clip(i + round(release_shift_s / dt), 1, len(times) - 2))
    return {"times": times, "q": qs, "release_index": i, "release_t": float(times[i]),
            "ball_speed": float(np.linalg.norm(vel[i]) / 1000), "ball_vel": vel[i] / 1000.0,
            "ball_pos": ball[i], "duration": float(times[-1]), "peak_joint_speed": float(np.abs(cs(times, 1)).max())}


class Releaser(threading.Thread):
    """Opens the gripper on its own thread, event loop and Viam connection, `delay` s after the stream starts
    (fire_from), on a wall-clock timer. cancel() before then keeps the gripper closed."""

    def __init__(self, connect, gripper_name: str, delay: float):
        super().__init__(daemon=True)
        self.connect, self.gripper_name, self.delay = connect, gripper_name, delay
        self.ready, self.armed, self.cancelled = threading.Event(), threading.Event(), threading.Event()
        self.t_start = None
        self.result: dict = {}

    def fire_from(self, t_start: float) -> None:
        self.t_start = t_start
        self.armed.set()

    def cancel(self) -> None:
        self.cancelled.set()
        self.armed.set()

    def run(self) -> None:
        asyncio.run(self._main())

    async def _main(self) -> None:
        from viam.components.gripper import Gripper
        machine = None
        try:
            machine = await self.connect()
            gripper = Gripper.from_robot(machine, self.gripper_name)
            await gripper.do_command({"get": True})                 # warm the connection
            self.ready.set()
            await asyncio.to_thread(self.armed.wait)
            if self.cancelled.is_set():
                return
            await asyncio.sleep(max(0.0, self.t_start + self.delay - time.monotonic()))
            if self.cancelled.is_set():
                return
            self.result["sent_s"] = round(time.monotonic() - self.t_start, 3)
            await gripper.open()
            self.result["open_returned_s"] = round(time.monotonic() - self.t_start, 3)
        except Exception as exc:                                     # reported, never raised into the throw
            self.result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            self.ready.set()
            if machine is not None:
                await machine.close()


async def stream(arm, p: dict, releaser: Releaser) -> dict:
    """Stream the planned trajectory while the releaser opens the gripper on time; a fault before the release
    keeps the ball held."""
    from viam.components.arm import Arm

    async def batches():
        for t, q in zip(p["times"], p["q"]):
            yield [Arm.TrajectoryPoint(time=timedelta(seconds=float(t)), positions=[float(v) for v in q])]

    t0 = time.monotonic()
    releaser.fire_from(t0)
    try:
        async for _ in arm.move_through_joint_positions_streamed(batches()):
            pass
    except BaseException:
        releaser.cancel()
        raise
    t_end = time.monotonic() - t0
    await asyncio.to_thread(releaser.join, 5.0)
    rel = dict(releaser.result)
    if "sent_s" in rel:
        rel["mid_motion"] = rel["sent_s"] < p["duration"] and rel.get("open_returned_s", 1e9) < t_end
        rel["planned_release_s"] = round(p["release_t"], 3)
    return {"stream_s": round(t_end, 3), "planned_s": round(p["duration"], 3), "points": len(p["times"]),
            "release": rel or None}
