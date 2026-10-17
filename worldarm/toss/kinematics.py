"""Forward kinematics from the arm's own model (what `arm.get_kinematics()` returns): Viam's SVA JSON or URDF.

Gives, for joint angles, where each joint is and which way it turns; from that, where the ball held in
the gripper is and - for joint speeds - how fast it's moving: v = sum_i w_i * (axis_i x (ball - joint_i)).
Joints turning about parallel axes (the xArm's shoulder, elbow and wrist pitch) add their speeds at the
ball. Units: mm and degrees at the interface (like Viam), metres and radians inside the toss physics.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from worldarm.cell.geometry import rot_from_ov, rotvec_matrix


@dataclass
class Step:
    kind: str                   # "fixed" | "revolute"
    T: np.ndarray | None        # 4x4 for fixed steps
    axis: np.ndarray | None     # unit axis in the step's parent frame, for revolute steps
    name: str


def _T(R=None, t=(0.0, 0.0, 0.0)) -> np.ndarray:
    M = np.eye(4)
    if R is not None:
        M[:3, :3] = R
    M[:3, 3] = t
    return M


def _rpy(r, p, y) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _sva_orientation(o: dict | None) -> np.ndarray | None:
    if not o:
        return None
    kind, val = o.get("type", "ov_degrees"), o.get("value", {})
    if kind == "ov_degrees":
        return rot_from_ov(val.get("x", 0.0), val.get("y", 0.0), val.get("z", 1.0), val.get("th", 0.0))
    if kind == "ov_radians":
        return rot_from_ov(val.get("x", 0.0), val.get("y", 0.0), val.get("z", 1.0), math.degrees(val.get("th", 0.0)))
    if kind == "euler_angles":
        return _rpy(val.get("roll", 0.0), val.get("pitch", 0.0), val.get("yaw", 0.0))
    if kind == "axis_angles":
        return rotvec_matrix(np.array([val.get("x", 0.0), val.get("y", 0.0), val.get("z", 0.0)]) * val.get("th", 0.0))
    raise ValueError(f"unsupported orientation type {kind!r} in the kinematics file")


class Chain:
    """A serial chain of fixed transforms and revolute joints, base to flange."""

    def __init__(self, steps: list[Step], limits_deg: list[tuple[float, float]]):
        self.steps, self.limits_deg = steps, limits_deg
        self.n = sum(s.kind == "revolute" for s in steps)

    @classmethod
    def from_sva(cls, data: dict | str | bytes) -> Chain:
        d = json.loads(data) if isinstance(data, (str, bytes)) else data
        frames: dict[str, tuple[str, dict, str]] = {}
        for link in d.get("links", []):
            frames[link["id"]] = (link.get("parent", "world"), link, "link")
        for joint in d.get("joints", []):
            frames[joint["id"]] = (joint.get("parent", "world"), joint, "joint")
        children = {parent: fid for fid, (parent, _, _) in frames.items()}
        chain, steps, limits, cur = [], [], [], "world"
        while cur in children:
            cur = children[cur]
            chain.append(cur)
        for fid in chain:
            _, f, kind = frames[fid]
            if kind == "link":
                t = f.get("translation", {})
                steps.append(Step("fixed", _T(_sva_orientation(f.get("orientation")),
                                              (t.get("x", 0.0), t.get("y", 0.0), t.get("z", 0.0))), None, fid))
            elif f.get("type", "revolute") == "revolute":
                a = f["axis"]
                ax = np.array([a.get("x", 0.0), a.get("y", 0.0), a.get("z", 0.0)], float)
                steps.append(Step("revolute", None, ax / np.linalg.norm(ax), fid))
                limits.append((float(f.get("min", -360.0)), float(f.get("max", 360.0))))
            else:
                raise ValueError(f"joint type {f.get('type')!r} isn't supported")
        return cls(steps, limits)

    @classmethod
    def from_urdf(cls, text: str | bytes) -> Chain:
        root = ET.fromstring(text)
        joints = {j.get("name"): j for j in root.findall("joint")}
        by_parent = {j.find("parent").get("link"): j for j in joints.values()}
        child_links = {j.find("child").get("link") for j in joints.values()}
        roots = [l.get("name") for l in root.findall("link") if l.get("name") not in child_links]
        base = next((r for r in roots if r in by_parent), roots[0])   # not a dangling "world" link listed first
        steps, limits, cur = [], [], base
        while cur in by_parent:
            j = by_parent[cur]
            o = j.find("origin")
            xyz = [float(v) * 1000.0 for v in (o.get("xyz", "0 0 0").split() if o is not None else [0, 0, 0])]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0").split() if o is not None else [0, 0, 0])]
            steps.append(Step("fixed", _T(_rpy(*rpy), xyz), None, j.get("name") + ":origin"))
            if j.get("type") in ("revolute", "continuous"):
                ax_el = j.find("axis")
                ax = np.array([float(v) for v in (ax_el.get("xyz", "1 0 0").split() if ax_el is not None else [1, 0, 0])])
                steps.append(Step("revolute", None, ax / np.linalg.norm(ax), j.get("name")))
                lim = j.find("limit")
                limits.append((math.degrees(float(lim.get("lower", -math.pi))), math.degrees(float(lim.get("upper", math.pi))))
                              if lim is not None else (-360.0, 360.0))
            cur = j.find("child").get("link")
        return cls(steps, limits)

    def frames(self, q_deg) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        """Flange transform (4x4, mm) and, per joint, (its position, its axis) in the base frame."""
        q = np.radians(np.asarray(q_deg, float))
        if len(q) != self.n:
            raise ValueError(f"{len(q)} joint angles for a {self.n}-joint arm")
        M, joints, i = np.eye(4), [], 0
        for s in self.steps:
            if s.kind == "fixed":
                M = M @ s.T
            else:
                axis_w = M[:3, :3] @ s.axis
                joints.append((M[:3, 3].copy(), axis_w))
                M = M @ _T(rotvec_matrix(s.axis * q[i]))
                i += 1
        return M, joints

    def point(self, q_deg, tool_mm=(0.0, 0.0, 0.0)) -> np.ndarray:
        """Where a point fixed to the flange (tool_mm in the flange frame) is, in the base frame (mm)."""
        M, _ = self.frames(q_deg)
        return (M @ np.r_[np.asarray(tool_mm, float), 1.0])[:3]

    def velocity(self, q_deg, qdot_deg_s, tool_mm=(0.0, 0.0, 0.0)) -> np.ndarray:
        """Velocity (mm/s) of that point for joint speeds qdot (deg/s)."""
        M, joints = self.frames(q_deg)
        p = (M @ np.r_[np.asarray(tool_mm, float), 1.0])[:3]
        v = np.zeros(3)
        for (o, a), w in zip(joints, np.radians(np.asarray(qdot_deg_s, float))):
            v += w * np.cross(a, p - o)
        return v

    def within_limits(self, q_deg) -> bool:
        return all(lo <= q <= hi for q, (lo, hi) in zip(q_deg, self.limits_deg))


def load_chain(path_or_text: str, fmt: str | None = None) -> Chain:
    """From a saved kinematics file (`.json` SVA or `.urdf`), or its text."""
    text = path_or_text
    if fmt is None:
        if path_or_text.lstrip().startswith(("{", "<")):
            fmt = "sva" if path_or_text.lstrip().startswith("{") else "urdf"
        else:
            with open(path_or_text) as f:
                text = f.read()
            fmt = "urdf" if path_or_text.endswith(".urdf") or text.lstrip().startswith("<") else "sva"
    return Chain.from_urdf(text) if fmt == "urdf" else Chain.from_sva(text)


def solve_down(chain: Chain, xyz_mm, q0, tool_mm=(0.0, 0.0, 0.0), iters: int = 200, pointing=(0.0, 0.0, -1.0)) -> np.ndarray | None:
    """Joint angles (deg) that put the tool point at xyz_mm with the gripper pointing straight down (or along
    `pointing`), starting from q0 (the wrist roll stays free). Damped least squares on our own model; None if it
    doesn't get within 1 mm."""
    target = np.asarray(xyz_mm, float)
    down = np.asarray(pointing, float) / np.linalg.norm(pointing)
    lo, hi = np.array([l for l, _ in chain.limits_deg]), np.array([h for _, h in chain.limits_deg])
    q = np.clip(np.asarray(q0, float), lo, hi)

    def err(q):
        M, _ = chain.frames(q)
        p = (M @ np.r_[np.asarray(tool_mm, float), 1.0])[:3]
        return np.r_[target - p, 300.0 * (down - M[:3, 2])]     # 300 mm per unit of pointing error

    for _ in range(iters):
        e = err(q)
        if np.linalg.norm(e[:3]) < 1.0 and np.linalg.norm(e[3:]) < 0.3:
            return q
        J = np.empty((6, chain.n))
        for i in range(chain.n):
            dq = np.zeros(chain.n)
            dq[i] = 0.01
            J[:, i] = (e - err(q + dq)) / 0.01                    # d(position)/d(q): e falls as the tool moves on
        step = J.T @ np.linalg.solve(J @ J.T + 25.0 * np.eye(6), e)
        q = np.clip(q + np.clip(step, -5.0, 5.0), lo, hi)
    return None
