"""Is a joint motion safe? Every pose along it against the joint limits, and every point of the arm (each
joint, points along the links between them, the flange, the ball) against the table, the ceiling and the
walls. Direct joint moves ignore Viam's obstacles, so this check is the only guard.

A joint move turns all joints together and they finish together, so its path is the straight line between
the two poses in joint space; it's checked every degree. Not checked: the arm hitting itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from worldarm.toss.kinematics import Chain


@dataclass
class KeepOut:
    table_z_mm: float = 0.0
    clearance_mm: float = 80.0
    base_radius_mm: float = 130.0     # the base column is always near the table: not checked against it
    ceiling_z_mm: float = 1300.0
    walls: list = field(default_factory=list)   # [[x0, y0, z0], [x1, y1, z1]] boxes
    margin_mm: float = 120.0          # the arm is checked along its centre line: links, gripper and camera stick out
    self_mm: float = 130.0            # the wrist, gripper and ball keep this far from the base column and upper arm
    reach_mm: float = 450.0           # the wrist stays this far from the shoulder: the arm open, never folded on itself

    @classmethod
    def of(cls, cfg) -> KeepOut:
        return cls(cfg.table_z_mm, cfg.clearance_mm, cfg.base_radius_mm, cfg.ceiling_z_mm, list(cfg.walls),
                   cfg.wall_margin_mm)


def arm_points(chain: Chain, q_deg, tool_mm, per_link: int = 4) -> np.ndarray:
    """(N, 3) points along the arm, base to ball: the joints, points along the links between them, the
    flange and the ball."""
    M, joints = chain.frames(q_deg)
    key = [o for o, _ in joints] + [M[:3, 3], (M @ np.r_[np.asarray(tool_mm, float), 1.0])[:3]]
    pts = []
    for a, b in zip(key[:-1], key[1:]):
        pts += [a + f * (b - a) for f in np.linspace(0.0, 1.0, per_link, endpoint=False)]
    pts.append(key[-1])
    return np.array(pts)


def _seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from each point (N, 3) to the segment a-b."""
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-9), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def self_problem(chain: Chain, q_deg, tool_mm, keep: KeepOut) -> str | None:
    """The arm folding into itself: the wrist, gripper and ball too close to the base column or the upper arm.
    (The xArm's controller stops on self-collision too; this keeps the planner from choosing such poses.)"""
    M, joints = chain.frames(q_deg)
    o = [j[0] for j in joints]
    ball = (M @ np.r_[np.asarray(tool_mm, float), 1.0])[:3]
    wrist = np.array([o[-2] + f * (ball - o[-2]) for f in np.linspace(0.0, 1.0, 6)])   # wrist pitch -> ball
    column = (np.array([0.0, 0.0, 0.0]), o[1].copy())                                     # base to shoulder
    if float(np.linalg.norm(o[-2] - o[1])) < keep.reach_mm:
        return f"the arm folds on itself (wrist {np.linalg.norm(o[-2] - o[1]):.0f} mm from the shoulder)"
    if float(_seg_dist(wrist, *column).min()) < keep.self_mm:
        return "the wrist or gripper comes too close to the arm's own base"
    if float(_seg_dist(wrist, o[1], o[2]).min()) < keep.self_mm:
        return "the wrist or gripper folds back onto the upper arm"
    return None


def pose_problem(chain: Chain, q_deg, tool_mm, keep: KeepOut) -> str | None:
    """What's wrong with one pose (None: nothing)."""
    if not chain.within_limits(q_deg):
        bad = [i + 1 for i, (q, (lo, hi)) in enumerate(zip(q_deg, chain.limits_deg)) if not lo <= q <= hi]
        return f"joint {bad[0]} past its limit"
    pts = arm_points(chain, q_deg, tool_mm)
    off_base = np.hypot(pts[:, 0], pts[:, 1]) > keep.base_radius_mm
    low = pts[off_base, 2] < keep.table_z_mm + keep.clearance_mm
    if low.any():
        z = float(pts[off_base, 2][low].min())
        return f"the arm comes within {z - keep.table_z_mm:.0f} mm of the table (needs {keep.clearance_mm:.0f})"
    if (pts[:, 2] > keep.ceiling_z_mm - keep.margin_mm).any():
        return f"the arm comes within {keep.margin_mm:.0f} mm of the ceiling limit {keep.ceiling_z_mm:.0f} mm"
    for lo, hi in keep.walls:
        lo_m, hi_m = np.asarray(lo, float) - keep.margin_mm, np.asarray(hi, float) + keep.margin_mm
        if np.all((pts >= lo_m) & (pts <= hi_m), axis=1).any():
            return f"the arm comes within {keep.margin_mm:.0f} mm of the keep-out box {list(lo)}-{list(hi)}"
    return self_problem(chain, q_deg, tool_mm, keep)


def motion_problem(chain: Chain, q_from, q_to, tool_mm, keep: KeepOut, step_deg: float = 1.0) -> str | None:
    """The first problem along a joint move from q_from to q_to (None: safe all the way)."""
    a, b = np.asarray(q_from, float), np.asarray(q_to, float)
    n = max(1, int(np.ceil(float(np.max(np.abs(b - a))) / step_deg)))
    for f in np.linspace(0.0, 1.0, n + 1):
        why = pose_problem(chain, a + f * (b - a), tool_mm, keep)
        if why is not None:
            return f"{why} ({100 * f:.0f}% of the way)"
    return None
