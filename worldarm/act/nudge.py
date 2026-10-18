"""Wait, standing still, until someone nudges the arm.

While the arm holds still its joint loads are steady; a push on the arm moves them. First the resting
loads are measured for a moment (the baseline), then a nudge is a load off its baseline by more than
the contact limits (config `contact`: the joints a push actually loads, J2/J3/J5), for `confirm`
readings in a row: the same joints and limits the picks use to feel contact, here against a still arm.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np


async def wait_for_nudge(rig, *, scale: float = 1.0, timeout_s: float | None = None, settle_s: float = 1.0,
                         say=print) -> dict:
    """Returns {"nudged": True, ...} once nudged, or {"nudged": False, ...} on timeout / no load readings."""
    c = rig.cfg.contact
    joints, limits = list(c.joints), np.array(c.limits, float) * scale
    await asyncio.sleep(settle_s)                     # let the arm stop swaying after its last move
    base = []
    t_end = time.time() + 1.0
    while time.time() < t_end:
        loads = await rig.loads()
        if loads is not None:
            base.append([loads[j] for j in joints])
        await asyncio.sleep(c.poll_s)
    if len(base) < 3:
        return {"nudged": False, "why": "no joint load readings from the arm"}
    baseline = np.median(np.array(base), axis=0)
    say("    waiting: nudge the arm to go on (Ctrl-C to stop)")
    t0, over = time.time(), 0
    while timeout_s is None or time.time() - t0 < timeout_s:
        loads = await rig.loads()
        if loads is not None:
            dev = np.abs(np.array([loads[j] for j in joints]) - baseline) / limits
            over = over + 1 if dev.max() > 1.0 else 0
            if over >= c.confirm:
                return {"nudged": True, "after_s": round(time.time() - t0, 1),
                        "joint": int(joints[int(dev.argmax())]) + 1, "ratio": round(float(dev.max()), 2)}
        await asyncio.sleep(c.poll_s)
    return {"nudged": False, "why": f"no nudge in {timeout_s:.0f} s"}
