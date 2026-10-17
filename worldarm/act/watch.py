"""Contact sensing during one smooth move: a joint load that breaks its own trend means the fingers hit
something. The drawer service's proven logic (trend over the last 0.6 s, two readings in a row over the
limit, nothing judged in the first 0.6 s or above the reach height), judging only the joints a push on
the fingers actually loads: J2, J3, J5. J6 (the wrist roll) rang after wrist turns and stopped descents."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from worldarm.cell.config import ContactCfg


@dataclass
class WatchResult:
    contact: bool
    z: float | None = None            # gripper frame height when contact was confirmed
    ratio: float = 0.0                # off-trend / limit when it was confirmed
    worst: float = 0.0                # largest off-trend / limit seen while judging
    per_joint: list | None = None
    seconds: float = 0.0

    def to_json(self) -> dict:
        return {"contact": self.contact, "z": None if self.z is None else round(self.z, 1),
                "ratio": round(self.ratio, 2), "worst": round(self.worst, 2), "per_joint": self.per_joint,
                "seconds": round(self.seconds, 2)}


@dataclass
class ContactWatch:
    cfg: ContactCfg
    active_below_z: float | None = None  # judge only once the gripper frame is this low
    t0: float | None = None
    worst: float = 0.0
    _hist: list = field(default_factory=list)
    _over: int = 0

    def feed(self, t: float, z: float | None, loads: list[float] | None) -> WatchResult | None:
        """Add one reading; returns a contact result when contact is confirmed, else None."""
        if self.t0 is None:
            self.t0 = t
        if not loads:
            return None
        sample = (t, [float(loads[j]) for j in self.cfg.joints])
        if t - self.t0 < self.cfg.warmup_s:
            self._hist.append(sample)
            return None
        past = [h for h in self._hist if t - h[0] <= self.cfg.window_s]
        judging = z is None or self.active_below_z is None or z <= self.active_below_z
        if not judging or len(past) < 4 or t - past[0][0] < 0.6 * self.cfg.window_s:
            self._hist.append(sample)
            self._over = 0
            return None
        ts = np.array([h[0] for h in past])
        dev = []
        for j in range(len(self.cfg.joints)):
            vs = np.array([h[1][j] for h in past])
            slope, icept = np.polyfit(ts - ts.mean(), vs, 1)
            if float(np.max(np.abs(vs - (icept + slope * (ts - ts.mean()))))) > 0.5 * self.cfg.limits[j]:
                self._hist.append(sample)  # the window itself isn't a clean trend (a step went by): wait
                self._over = 0
                return None
            dev.append(sample[1][j] - (icept + slope * (t - ts.mean())))
        ratios = [abs(d) / lim for d, lim in zip(dev, self.cfg.limits)]
        r = max(ratios)
        self.worst = max(self.worst, r)
        if r <= 1.0:
            self._hist.append(sample)  # off-trend readings never join the trend: a blip can't bend it,
        self._over = self._over + 1 if r > 1.0 else 0  # and a lasting push can't be absorbed into it
        if self._over >= self.cfg.confirm:
            return WatchResult(True, z, r, self.worst, [round(d, 2) for d in dev], t - self.t0)
        return None
