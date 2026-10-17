"""Gripper yaw for a block seen from above: the fingers close across the block's short side.

The finger axis is the gripper frame's y axis (the claws' 120 mm span; the wrist camera sees the
fingertips side by side along its x axis). Which world direction that is for a given wrist angle
comes from Viam's own orientation math, not a hand-derived formula: for a gripper pointing straight
down, finger-axis angle = 90 deg - theta.
"""

from __future__ import annotations

import math
from typing import Awaitable, Callable

import cv2
import numpy as np
from viam.spatialmath import OrientationVector
from viam.spatialmath.vector3 import Vector3

ROUND_ASPECT = 1.25  # an outline this close to square ...
ROUND_FILL = 0.88    # ... whose convex hull fills this little of its rectangle looks round (a disc: 0.785).
                     # Only a label: the rectangle's short side is used either way, since a round
                     # object can be gripped at any angle but a box mistaken for round can't.


def finger_axis_deg(theta_deg: float, ov: tuple[float, float, float] = (0.0, 0.0, -1.0)) -> float:
    """World angle (deg, from +x toward +y) of the finger axis for a gripper pose with this theta."""
    q = OrientationVector(ov[0], ov[1], ov[2], math.radians(theta_deg)).to_quaternion()
    y = q.rotate_vector(Vector3(0.0, 1.0, 0.0))
    return math.degrees(math.atan2(y.y, y.x))


def theta_for_finger_axis(axis_deg: float, base_theta_deg: float,
                          ov: tuple[float, float, float] = (0.0, 0.0, -1.0)) -> float:
    """The wrist theta within +-90 deg of `base_theta_deg` whose finger axis lies along `axis_deg`.

    A parallel gripper is symmetric under 180 deg, so a match always exists within +-90 deg;
    staying near the base keeps the wrist from winding up.
    """
    best, best_err = base_theta_deg, 180.0
    for step in range(-180, 181):
        theta = base_theta_deg + step * 0.5
        diff = (finger_axis_deg(theta, ov) - axis_deg) % 180.0
        err = min(diff, 180.0 - diff)
        if err < best_err - 1e-9:
            best, best_err = theta, err
    return best


def short_axis_px(mask: np.ndarray):
    """Short-side direction of a block's mask, from its minimum-area rectangle.

    Returns (center_uv, unit_uv, short_px, long_px, fill) or None for an empty mask. `fill` is the
    convex hull's area over the rectangle's: depth holes (black rubber, dark wood) don't lower it,
    so a patchy box still reads as a box (~1.0) and only a disc reads round (~0.785).
    """
    ys, xs = np.nonzero(mask)
    if xs.size < 20:
        return None
    pts = np.column_stack((xs, ys)).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    e1, e2 = box[1] - box[0], box[2] - box[1]
    n1, n2 = float(np.hypot(*e1)), float(np.hypot(*e2))
    short_vec, short_len, long_len = (e1, n1, n2) if n1 <= n2 else (e2, n2, n1)
    fill = float(cv2.contourArea(cv2.convexHull(pts))) / max(1.0, n1 * n2)
    return rect[0], short_vec / max(short_len, 1e-6), short_len, long_len, fill


async def block_grasp_theta(mask: np.ndarray, depth: np.ndarray,
                            to_world: Callable[[float, float, float], Awaitable[tuple[float, float]]],
                            base_theta_deg: float,
                            ov: tuple[float, float, float] = (0.0, 0.0, -1.0)) -> tuple[float, dict]:
    """Wrist theta that closes the fingers across the block's short side, plus what was measured.

    `to_world(u, v, depth_mm)` maps a pixel at a depth to world (x, y) in mm.
    Anything unmeasurable keeps `base_theta_deg`.
    """
    found = short_axis_px(mask)
    if found is None:
        return base_theta_deg, {"shape": "unmeasured"}
    (cu, cv), (du, dv), short_px, long_px, fill = found
    aspect = long_px / max(short_px, 1e-6)
    shape = "round" if aspect < ROUND_ASPECT and fill < ROUND_FILL else "box"
    z = float(np.median(depth[(mask > 0) & (depth > 0)])) if np.any((mask > 0) & (depth > 0)) else 0.0
    if z <= 0:
        return base_theta_deg, {"shape": "no depth"}
    half = max(8.0, short_px / 2.0)  # pixels along the short side, either way from the center
    ax, ay = await to_world(cu - du * half, cv - dv * half, z)
    bx, by = await to_world(cu + du * half, cv + dv * half, z)
    short_deg = math.degrees(math.atan2(by - ay, bx - ax))
    theta = theta_for_finger_axis(short_deg, base_theta_deg, ov)
    return theta, {"shape": shape, "short_axis_deg": round(short_deg, 1), "short_mm": round(math.hypot(bx - ax, by - ay), 1),
                   "aspect": round(aspect, 2), "fill": round(fill, 2), "theta": round(theta, 1)}


def place_theta_for_drawer(open_dir: tuple[float, float], pick_theta_deg: float,
                           ov: tuple[float, float, float] = (0.0, 0.0, -1.0)) -> float:
    """Wrist theta at the drawer: the block's long side along the pull direction, so it fits the
    open part (which is longer along the pull than across). Fingers then lie across the pull."""
    pull_deg = math.degrees(math.atan2(open_dir[1], open_dir[0]))
    return theta_for_finger_axis(pull_deg + 90.0, pick_theta_deg, ov)
