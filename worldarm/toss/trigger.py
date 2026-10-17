"""During a swing: read the joints as fast as the arm answers, and tell the gripper to let go on time.

predict   Each reading says how far the swing has got; with the profile, when it will reach the trigger
          angle. When that's due before the next reading could come back, wait for it and send the open
          command then. Steady even if the swing started late; it needs readings that come back quickly.
timed     Count from when the swing command went out: the start latency `dry` measured, plus the profile's
          time to the trigger angle. Needs no readings; as steady as the start latency is.

The trigger angle is the release point less release_delay_s of swing (the plan), so the ball is free right at
the release point if the open command takes release_delay_s to free it. Everything is kept for the log: every
reading, when the command went out, and how far the swing had got by then.
"""

from __future__ import annotations

import asyncio
import statistics
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import numpy as np

from worldarm.toss.profile import Trapezoid
from worldarm.toss.swing import SwingSpec


def progress_of(q, spec: SwingSpec) -> float:
    """How far the swing has got: the swinging joints' mean turn from the wind-up pose (deg)."""
    return float(np.mean([s * (q[j] - spec.start_deg[j]) for j, s in zip(spec.joints, spec.signs)]))


@dataclass
class Recorder:
    """Joint readings, one request at a time, as fast as the arm answers: (time it was most likely read,
    joints, round trip). The time is the middle of the request: the arm reads its joints somewhere in there."""
    read: Callable[[], Awaitable[list[float]]]
    clock: Callable[[], float]
    readings: list = field(default_factory=list)
    errors: int = 0

    def __post_init__(self):
        self._new = asyncio.Event()

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            t_req = self.clock()
            try:
                q = list(await self.read())
            except asyncio.CancelledError:
                raise
            except Exception:
                self.errors += 1
                await asyncio.sleep(0.005)
                continue
            t_got = self.clock()
            self.readings.append((0.5 * (t_req + t_got), q, t_got - t_req))
            self._new.set()

    async def next(self) -> tuple:
        self._new.clear()
        await self._new.wait()
        return self.readings[-1]

    def progress(self, spec: SwingSpec) -> tuple[np.ndarray, np.ndarray]:
        """(times, progress) of every reading so far."""
        if not self.readings:
            return np.zeros(0), np.zeros(0)
        return (np.array([r[0] for r in self.readings]), np.array([progress_of(r[1], spec) for r in self.readings]))


@dataclass
class Sent:
    mode: str
    at: float | None = None          # when the open command went out (the loop's clock)
    due: float | None = None         # when it was meant to go out
    late_deg: float = 0.0            # > 0: the trigger angle had already passed when it was decided
    reason: str = ""

    def to_json(self) -> dict:
        return {"mode": self.mode, "at": self.at, "due": self.due, "late_deg": round(self.late_deg, 3),
                "reason": self.reason}


async def release(open_gripper: Callable[[], Awaitable], spec: SwingSpec, prof: Trapezoid, trigger_deg: float,
                  mode: str, t_cmd: float, start_latency_s: float | None, recorder: Recorder | None,
                  clock: Callable[[], float], timeout_s: float) -> Sent:
    """Send the open command at the trigger angle (see the module doc). prof: the swing's profile with t0 = 0
    (the planned or the measured speed and acceleration)."""
    sent = Sent(mode)
    to_trigger = prof.time_at(trigger_deg) - prof.t0       # s from the start of motion to the trigger angle
    if mode == "timed":
        if start_latency_s is None:
            raise ValueError("timed release needs the start latency: run toss.py dry first")
        sent.due = t_cmd + start_latency_s + to_trigger
        await asyncio.sleep(max(0.0, sent.due - clock()))
    elif mode == "predict":
        if recorder is None:
            raise ValueError("predicted release needs joint readings")
        deadline = t_cmd + timeout_s
        while True:
            if clock() > deadline:
                sent.reason = "the swing never reached the trigger angle"
                return sent
            try:
                t_read, q, _ = await asyncio.wait_for(recorder.next(), max(0.05, deadline - clock()))
            except asyncio.TimeoutError:
                sent.reason = "no joint readings came back"
                return sent
            p = progress_of(q, spec)
            if p >= trigger_deg:
                sent.due, sent.late_deg = t_read, p - trigger_deg   # the window passed between readings: now
                break
            if p <= 0.5:
                continue                                   # not moving yet: no phase to go by
            sent.due = t_read + (prof.time_at(trigger_deg) - prof.time_at(p))
            rtts = [r[2] for r in recorder.readings[-5:]]
            if sent.due - clock() <= 1.5 * statistics.median(rtts):   # the next reading would come too late
                await asyncio.sleep(max(0.0, sent.due - clock()))
                break
    else:
        raise ValueError(f"unknown release mode {mode!r}")
    sent.at = clock()
    await open_gripper()
    return sent
