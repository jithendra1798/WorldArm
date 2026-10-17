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


class ContactGuard:
    """Call with the gripper height after each descent step; returns True on contact."""

    def __init__(self, read_loads: Callable[[], Awaitable[list[float] | None]],
                 settle_s: float = 0.15, window: int = 4):
        self.read_loads, self.settle_s, self.window = read_loads, settle_s, window
        self.history: list[tuple[float, list[float]]] = []
        self.contact_z: float | None = None
        self.worst = 0.0  # largest off-trend reading as a fraction of its limit (for the log)

    async def __call__(self, z: float) -> bool:
        await asyncio.sleep(self.settle_s)  # let the step finish ringing before reading
        loads = await self.read_loads()
        if not loads:
            return False
        contact = False
        recent = self.history[-self.window:]
        if len(recent) >= 2:
            zs = np.array([h[0] for h in recent])
            if np.ptp(zs) > 1e-6:
                for j, limit in enumerate(LIMITS):
                    if limit is None:
                        continue
                    fit = np.polyfit(zs, [h[1][j] for h in recent], 1)
                    off = abs(loads[j] - float(np.polyval(fit, z))) / limit
                    self.worst = max(self.worst, off)
                    contact |= off > 1.0
        self.history.append((z, list(loads)))
        if contact:
            self.contact_z = z
        return contact
