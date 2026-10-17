"""Where the thrown ball came to rest, seen by the overhead camera, and the two arms' frames.

The overhead camera's table map (task B: `ball.py calibrate` on armfarm03) turns the ball's pixel into
armfarm03's world (mm), for a ball lying on the table. Thrown onto a cloth the ball stops within a few cm of
where it lands, so its resting point is the landing to within that roll; a ball resting higher than the table
(in the bowl) is off by parallax, ~6% of its distance from the point under the camera per 8 cm of height.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Awaitable, Callable

import numpy as np

from worldarm.ball.detect import find_ball
from worldarm.ball.table import TableMap


@dataclass(frozen=True)
class ArmsFrame:
    """The thrower's base in the catcher's frame: position (mm) and heading (deg about z)."""
    x_mm: float
    y_mm: float
    yaw_deg: float

    @classmethod
    def of(cls, d: dict) -> ArmsFrame:
        return cls(float(d["x_mm"]), float(d["y_mm"]), float(d["yaw_deg"]))

    def _R(self) -> np.ndarray:
        c, s = math.cos(math.radians(self.yaw_deg)), math.sin(math.radians(self.yaw_deg))
        return np.array([[c, -s], [s, c]])

    def to_catcher(self, xy) -> np.ndarray:
        return self._R() @ np.asarray(xy, float) + np.array([self.x_mm, self.y_mm])

    def to_thrower(self, xy) -> np.ndarray:
        return self._R().T @ (np.asarray(xy, float) - np.array([self.x_mm, self.y_mm]))

    def to_json(self) -> dict:
        return {"x_mm": round(self.x_mm, 1), "y_mm": round(self.y_mm, 1), "yaw_deg": round(self.yaw_deg, 2)}


async def resting_ball(frame: Callable[[], Awaitable[tuple[float, np.ndarray | None]]], table: TableMap,
                       radius_px=(8.0, 30.0), table_px=None, still_mm: float = 6.0, need: int = 4,
                       timeout_s: float = 8.0, clock=None) -> tuple[np.ndarray | None, str]:
    """Wait until the ball lies still (found within still_mm in `need` frames in a row), then where it is
    (median of those frames, catcher's frame, mm). frame: async () -> (time, BGR image or None)."""
    clock = clock or (lambda: asyncio.get_running_loop().time())
    t_end = clock() + timeout_s
    run: list[np.ndarray] = []
    seen = 0
    while clock() < t_end:
        _, img = await frame()
        s = find_ball(img, radius_px, table_px) if img is not None else None
        if s is None:
            run = []
            continue
        seen += 1
        xy = table.to_world((s.u, s.v))
        if run and np.linalg.norm(xy - np.median(run, axis=0)) > still_mm:
            run = []
        run.append(xy)
        if len(run) >= need:
            return np.median(run, axis=0), "at rest"
    return None, ("the ball wasn't seen on the table" if seen == 0 else "the ball never came to rest in view")
