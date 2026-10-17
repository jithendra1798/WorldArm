"""Chapter 2 settings for one arm: the table map, the defended lane, the arm's lane speed, the learned
slowdown. Measured by `ball.py` drills and saved to config/ball_<machine>.json."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

from worldarm.ball.intercept import ArmModel, Lane
from worldarm.ball.table import TableMap
from worldarm.cell.config import CONFIG_DIR


@dataclass
class BallCfg:
    machine: str = "armfarm03"
    camera: str = "overview-cam"
    ball_radius_mm: float = 25.0            # measure it
    ball_radius_px: tuple = (8.0, 30.0)     # Riyaj's detector limits at 1.31 m
    camera_latency_s: float = 0.06          # exposure to arrival; refined from the lane-test log
    paddle_half_width_mm: float = 60.0      # half the width of what the gripper holds across the lane
    zone_front_mm: float = 20.0             # the ball isn't looked for this close in front of the lane (the paddle),
    zone_back_mm: float = 150.0             # this far behind it (gripper, wrist and arm: the higher, the further out
    zone_side_mm: float = 30.0              # the camera sees them), within the paddle's half width + this along it
    field_length_mm: float = 1400.0         # how far out from the lane the ball is looked for
    field_margin_mm: float = 150.0          # beyond the lane ends, and behind the lane (to see goals)
    lane_speed_deg_s: float = 30.0          # arm speed along the lane (raise only with everyone clear)
    max_lane_speed_deg_s: float = 90.0
    decel_mm_s2: float = 0.0                # learned from recorded rolls (ball.py fit); 0 until then
    decel_spread: float | None = None
    table_map: dict | None = None           # TableMap.to_json() (ball.py calibrate)
    table_px: list | None = None            # polygon of the table in the overhead image (search area)
    lane: dict | None = None                # Lane.to_json() (ball.py lane --teach a / b)
    arm: dict | None = None                 # {"latency_s", "speed_mm_s"} (ball.py lane-test)
    arm_speed_deg_s: float | None = None    # the lane speed `arm` was measured at
    arm_retarget_s: float | None = None     # a new target mid-move -> the paddle turns back (stop + restart; lane-test)

    @property
    def path(self) -> Path:
        return CONFIG_DIR / f"ball_{self.machine}.json"

    @classmethod
    def load(cls, machine: str = "armfarm03") -> BallCfg:
        cfg = cls(machine=machine)
        if cfg.path.exists():
            for k, v in json.loads(cfg.path.read_text()).items():
                if hasattr(cfg, k):
                    setattr(cfg, k, tuple(v) if isinstance(getattr(cfg, k), tuple) and isinstance(v, list) else v)
        return cfg

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(dataclasses.asdict(self), indent=2) + "\n")
        return self.path

    def table(self) -> TableMap | None:
        return TableMap.from_json(self.table_map) if self.table_map else None

    def get_lane(self) -> Lane | None:
        return Lane.from_json(self.lane) if self.lane and "a" in self.lane and "b" in self.lane else None

    def arm_model(self) -> ArmModel:
        known = {f.name for f in dataclasses.fields(ArmModel)}      # an extra key in the file must not crash it
        return ArmModel(**{k: v for k, v in self.arm.items() if k in known}) if self.arm else ArmModel()
