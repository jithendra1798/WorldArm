"""The calibrated constants, from logged throws.

Each logged throw is predicted from what was measured during it: the swing (joint speed asked for, its fitted
profile) and how far it had got when the open command went out. Unknowns:

  speed_factor     the ball leaves this much faster than the joint speed says (speed tracking, the ball
                   rolling in the fingers)
  release_delay_s  open command sent -> ball free (the network, the gripper's reaction)
  x, y, yaw        where the thrower's base is in the catcher's frame (the overhead camera's)

fit to the measured landings by least squares, with weak priors (the tape measure, 1.0, 50 ms) so a handful of
throws already give an answer, and standard deviations to say how well each is pinned down. Speed and delay
both move the landing along the throw, differently with speed (the delay's part grows with speed, the
speed's part with its square): throws at one speed can't tell them apart, so use two speeds at least, and
two base angles to pin the yaw.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from worldarm.toss.kinematics import Chain
from worldarm.toss.landing import ArmsFrame
from worldarm.toss.lsq import least_squares
from worldarm.toss.swing import SwingSpec, landing_for

PARAMS = ("speed_factor", "release_delay_s", "x_mm", "y_mm", "yaw_deg")
PRIOR_SD = {"speed_factor": 0.1, "release_delay_s": 0.05, "x_mm": 50.0, "y_mm": 50.0, "yaw_deg": 3.0}


def spec_of(throw: dict) -> SwingSpec:
    s = throw["spec"]
    return SwingSpec(list(s["start_deg"]), list(s["joints"]), list(s["signs"]), float(s["amplitude_deg"]),
                     float(s["accel_deg_s2"]), tuple(s["tool_mm"]), float(s.get("table_z_mm", 0.0)),
                     float(s.get("clearance_mm", 80.0)))


def predicted(chain: Chain, throw: dict, speed_factor: float, delay_s: float) -> np.ndarray | None:
    """Where this throw should have come down (thrower's frame, mm), had the constants been these."""
    w_fit = throw.get("swing_w_deg_s") or throw["speed_deg_s"]
    progress = throw["sent_progress_deg"] + w_fit * delay_s          # where the ball came free
    return landing_for(chain, spec_of(throw), throw["base_deg"], progress, throw["speed_deg_s"],
                       throw["catch_z_mm"], speed_factor)


@dataclass
class Calibration:
    values: dict
    sd: dict
    residuals_mm: list            # per throw, measured - predicted (catcher's frame)
    rms_mm: float
    n: int
    note: str = ""


def fit_constants(chain: Chain, throws: list[dict], prior: dict, fit: tuple = PARAMS) -> Calibration:
    """throws: logged throws with a measured landing ("landing_catcher_mm"). prior: current values of PARAMS
    (the config). fit: which to fit; the rest stay at the prior."""
    used = [t for t in throws if t.get("landing_catcher_mm") is not None and t.get("sent_progress_deg") is not None]
    names = [p for p in PARAMS if p in fit]
    x0 = np.array([prior[p] for p in names], float)

    def full(x):
        v = dict(prior)
        v.update(dict(zip(names, x)))
        return v

    def resid(x):
        v = full(x)
        arms = ArmsFrame(v["x_mm"], v["y_mm"], v["yaw_deg"])
        r = []
        for t in used:
            p = predicted(chain, t, v["speed_factor"], v["release_delay_s"])
            if p is None:
                r += [1e4, 1e4]
            else:
                r += list(arms.to_catcher(p) - np.asarray(t["landing_catcher_mm"], float))
        r += [(xi - x0i) / PRIOR_SD[n] * 5.0 for xi, x0i, n in zip(x, x0, names)]   # 5 mm per prior sd
        return np.array(r, float)

    res = least_squares(resid, x0)
    v = full(res.x)
    k = 2 * len(used)
    per = res.residuals[:k].reshape(-1, 2) if k else np.zeros((0, 2))
    speeds = {round(t["speed_deg_s"], 0) for t in used}
    bases = {round(t["base_deg"], 0) for t in used}
    note = []
    if len(speeds) < 2 and {"speed_factor", "release_delay_s"} <= set(names):
        note.append("all throws at one speed: speed factor and release delay can't be told apart (the priors "
                    "decide the split); throw at a second speed")
    if len(bases) < 2 and "yaw_deg" in names:
        note.append("all throws at one base angle: the yaw is mostly the prior; throw at a second base angle")
    return Calibration({n: float(v[n]) for n in PARAMS}, {n: float(s) for n, s in zip(names, res.sd)},
                       [list(np.round(r, 1)) for r in per],
                       float(np.sqrt(np.mean(np.sum(per * per, axis=1)))) if len(per) else 0.0, len(used), "; ".join(note))
