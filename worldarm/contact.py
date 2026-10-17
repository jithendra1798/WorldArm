"""Contact check for a stepwise descent: a joint load that breaks its trend means the fingers touched.

Same idea and limits as the drawer service's watched descent (a ~7 N push reaches them): gravity makes
each joint load drift smoothly as the arm goes down, so every reading is compared with the straight-line
trend of the previous readings. J2, J3 and J5 carry a vertical push; J1, J4 and J6 are ignored.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

import numpy as np

LIMITS = (None, 3.0, 1.5, None, 1.5, None)
MIN_TREND = 3  # readings needed before judging (a line through 2 noisy readings extrapolates badly)


class ContactGuard:
    """Call with the gripper height after each descent step; returns True on contact."""

    def __init__(self, read_loads: Callable[[], Awaitable[list[float] | None]],
                 settle_s: float = 0.3, window: int = 4, active_below: float | None = None, confirm: int = 2):
        """active_below: only judge contact once the gripper is this low (where the fingers can reach the
        block or the table at all); higher up, readings only build the trend. confirm: reads at the same
        height that must all be off the trend. The arm still ringing from a step settles by the next
        read; a push doesn't. The off-trend reads never join the trend, so they can't bend it."""
        self.read_loads, self.settle_s, self.window = read_loads, settle_s, window
        self.active_below, self.confirm = active_below, max(1, confirm)
        self.history: list[tuple[float, list[float]]] = []
        self.contact_z: float | None = None
        self.worst = 0.0  # largest off-trend reading within reach, as a fraction of its limit (for the log)

    def _off(self, z: float, loads: list[float]) -> float:
        """Largest off-trend load at height z, as a fraction of its limit (0 without enough trend)."""
        recent = self.history[-self.window:]
        if len(recent) < MIN_TREND:
            return 0.0
        zs = np.array([h[0] for h in recent])
        if np.ptp(zs) <= 1e-6:
            return 0.0
        off = 0.0
        for j, limit in enumerate(LIMITS):
            if limit is not None:
                fit = np.polyfit(zs, [h[1][j] for h in recent], 1)
                off = max(off, abs(loads[j] - float(np.polyval(fit, z))) / limit)
        return off

    async def __call__(self, z: float) -> bool:
        reach = self.active_below is None or z <= self.active_below
        for _ in range(self.confirm):
            await asyncio.sleep(self.settle_s)  # let the step (or the last read) finish ringing
            loads = await self.read_loads()
            if not loads:
                return False
            off = self._off(z, loads)
            if reach:
                self.worst = max(self.worst, off)
            if off <= 1.0:
                break
        else:
            if reach:  # off the trend on every read: pressing on something
                self.contact_z = z
                return True
            # Up here nothing can be touched: a lasting shift is the arm's own, so the trend follows it.
        self.history.append((z, list(loads)))
        return False
