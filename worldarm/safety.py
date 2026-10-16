"""Our own motion guard, checked before every computed motion.move.

Viam's table obstacle on armfarm03 sits about 25 mm below the real surface, and direct joint moves
ignore obstacles entirely, so we don't rely on the planner to keep the fingertips out of the table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class SafetyError(RuntimeError):
    pass


@dataclass(frozen=True)
class Limits:
    table_z_mm: float = 2.0       # measured table surface in world z (verify_camera.py, per arm)
    floor_margin_mm: float = 5.0  # fingertips stay at least this far above the table
    fingertip_mm: float = 60.0    # fingertips are this far along the gripper frame's z axis
    x_mm: tuple[float, float] = (100.0, 640.0)   # wall-front starts at x = 690
    y_mm: tuple[float, float] = (-600.0, 400.0)  # wall-side starts at y = 450
    z_max_mm: float = 900.0                      # ceiling at z = 1000


def fingertip_xyz(pose, fingertip_mm: float) -> tuple[float, float, float]:
    """Fingertip position of a world-frame gripper pose; (o_x, o_y, o_z) is the frame's z axis."""
    n = math.sqrt(pose.o_x ** 2 + pose.o_y ** 2 + pose.o_z ** 2) or 1.0
    return (pose.x + fingertip_mm * pose.o_x / n,
            pose.y + fingertip_mm * pose.o_y / n,
            pose.z + fingertip_mm * pose.o_z / n)


def check_gripper_target(pose, limits: Limits) -> None:
    """Raise SafetyError unless a world-frame gripper pose keeps the gripper and fingertips in bounds."""
    floor = limits.table_z_mm + limits.floor_margin_mm
    problems = []
    for name, (x, y, z) in (("gripper", (pose.x, pose.y, pose.z)),
                            ("fingertips", fingertip_xyz(pose, limits.fingertip_mm))):
        if z < floor:
            problems.append(f"{name} z={z:.0f} mm is below the floor ({floor:.0f} mm)")
        if z > limits.z_max_mm:
            problems.append(f"{name} z={z:.0f} mm is above {limits.z_max_mm:.0f} mm")
        if not limits.x_mm[0] <= x <= limits.x_mm[1]:
            problems.append(f"{name} x={x:.0f} mm is outside {limits.x_mm}")
        if not limits.y_mm[0] <= y <= limits.y_mm[1]:
            problems.append(f"{name} y={y:.0f} mm is outside {limits.y_mm}")
    if problems:
        raise SafetyError("; ".join(problems))
