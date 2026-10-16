"""Machine and task configuration.

Resource names come from the env file (see .env.example). Numbers default to what we measured on
armfarm03; override them per arm in that arm's env file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from . import connection  # noqa: F401  (loads the env file)
from .safety import Limits
from .schema import COLORS


def parse_assignment(text: str) -> dict[str, int]:
    """'blue:1,yellow:2,green:3' -> {'blue': 1, 'yellow': 2, 'green': 3}"""
    pairs = (item.split(":") for item in text.split(",") if item.strip())
    return {color.strip(): int(k) for color, k in pairs}


@dataclass(frozen=True)
class Config:
    machine: str          # short machine name, used for state and log files
    arm: str
    gripper: str
    camera: str
    vision: str           # detections-to-segments service returning 3D objects
    recorder: str         # arm-recorder sensor holding the drawer recordings
    pose_home: str        # arm-position-saver switch names
    pose_table_view: str
    colors: tuple[str, ...]
    n_drawers: int
    assignment: dict[str, int]  # color -> drawer
    containers: str       # "drawers" (organizer) or "bins" (open-top bins: open/close do nothing)
    limits: Limits
    approach_mm: float    # pick approach height above the block center
    settle_s: float       # wait after gripper open/grab


def load_config() -> Config:
    g = os.getenv
    return Config(
        machine=g("VIAM_MACHINE_ADDRESS", "sim").split(".")[0],
        arm=g("VIAM_ARM", "arm"),
        gripper=g("VIAM_GRIPPER", "gripper"),
        camera=g("VIAM_CAMERA", "cam"),
        vision=g("WORLDARM_VISION", "vision-segment"),
        recorder=g("WORLDARM_RECORDER", "drawer-recorder"),
        pose_home=g("WORLDARM_POSE_HOME", "home-pose"),
        pose_table_view=g("WORLDARM_POSE_TABLE_VIEW", "table-view"),
        colors=COLORS,
        n_drawers=3,
        assignment=parse_assignment(g("WORLDARM_ASSIGNMENT", "blue:1,yellow:2,green:3")),
        containers=g("WORLDARM_CONTAINERS", "drawers"),
        limits=Limits(table_z_mm=float(g("WORLDARM_TABLE_Z_MM", "2.0")),
                      fingertip_mm=float(g("WORLDARM_FINGERTIP_MM", "60"))),
        approach_mm=100.0,
        settle_s=0.3,
    )
