"""Pose math shared by every skill. Orientations go through Viam's own spatialmath, not hand-derived
conventions, so a pose we compute is the pose the arm takes."""

from __future__ import annotations

import math

import numpy as np
from viam.proto.common import Pose
from viam.spatialmath import OrientationVector

from worldarm.grasp import finger_axis_deg, theta_for_finger_axis  # noqa: F401  (re-exported)

DOWN = (0.0, 0.0, -1.0)  # gripper pointing straight down


def rot_from_ov(ox: float, oy: float, oz: float, theta_deg: float) -> np.ndarray:
    """Rotation matrix (frame axes as columns, in world) of a Viam orientation vector."""
    q = OrientationVector(ox, oy, oz, math.radians(theta_deg)).to_quaternion()
    return np.array(q.to_rotation_matrix().elements, float).reshape(3, 3)


def pose_matrix(pose: Pose | dict) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) of a Viam pose: p_world = R @ p_frame + t."""
    p = pose if isinstance(pose, dict) else {k: getattr(pose, k) for k in ("x", "y", "z", "o_x", "o_y", "o_z", "theta")}
    return rot_from_ov(p["o_x"], p["o_y"], p["o_z"], p["theta"]), np.array([p["x"], p["y"], p["z"]], float)


def down_pose(x: float, y: float, z: float, theta_deg: float) -> Pose:
    """Gripper pointing straight down at (x, y, z), wrist turned theta."""
    return Pose(x=float(x), y=float(y), z=float(z), o_x=0.0, o_y=0.0, o_z=-1.0, theta=float(theta_deg))


def pose_dict(pose: Pose) -> dict:
    return {k: round(float(getattr(pose, k)), 3) for k in ("x", "y", "z", "o_x", "o_y", "o_z", "theta")}


def to_pose(d: dict) -> Pose:
    return Pose(**{k: float(d[k]) for k in ("x", "y", "z", "o_x", "o_y", "o_z", "theta")})


def tilt_deg(pose: Pose | dict) -> float:
    """How far the gripper points away from straight down."""
    oz = pose["o_z"] if isinstance(pose, dict) else pose.o_z
    ox = pose["o_x"] if isinstance(pose, dict) else pose.o_x
    oy = pose["o_y"] if isinstance(pose, dict) else pose.o_y
    n = math.sqrt(ox * ox + oy * oy + oz * oz) or 1.0
    return math.degrees(math.acos(max(-1.0, min(1.0, -oz / n))))


def wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def axis_diff(a_deg: float, b_deg: float) -> float:
    """Angle between two undirected axes (0-90)."""
    d = (a_deg - b_deg) % 180.0
    return min(d, 180.0 - d)


def other_wrist(theta_deg: float) -> float:
    """The same finger axis with the wrist turned half a turn (a parallel gripper is symmetric)."""
    return theta_deg - 180.0 if theta_deg > 0 else theta_deg + 180.0


def rotvec_matrix(w: np.ndarray) -> np.ndarray:
    """Rodrigues: rotation of angle |w| (rad) about w."""
    w = np.asarray(w, float)
    a = float(np.linalg.norm(w))
    if a < 1e-12:
        return np.eye(3)
    k = w / a
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * kx + (1 - math.cos(a)) * kx @ kx


def unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def perp(v) -> np.ndarray:
    """2-D vector turned +90 degrees."""
    return np.array([-v[1], v[0]], float)


def deg_of(v) -> float:
    return math.degrees(math.atan2(v[1], v[0]))


def vec_of(deg: float) -> np.ndarray:
    r = math.radians(deg)
    return np.array([math.cos(r), math.sin(r)])


def point_in_polygon(xy: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorised even-odd test: which of the (N, 2) points lie inside the (M, 2) polygon."""
    xy = np.atleast_2d(np.asarray(xy, float))
    poly = np.asarray(poly, float)
    x, y = xy[:, 0], xy[:, 1]
    inside = np.zeros(len(xy), bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        cross = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi)
        inside ^= cross
        j = i
    return inside
