"""Every number the sorting task uses for one arm, in one place, with where it came from.

Defaults below are armfarm03's. config/<machine>.json overrides them, and the drills that measure
something (touch, calibrate, view --teach) write their result back there. Credentials and component
names stay in the env file (.env).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
LEGACY_VIEW_POSE = ROOT / ".pick_place_view_pose.armfarm03.json"  # taught with sort_bench.py --teach-view


@dataclass
class GripperCfg:
    frame_below_flange_mm: float = 105.0   # Viam frame config: gripper frame origin below the flange
    tip_below_frame_mm: float = 60.0       # fingertip ends below the gripper frame. Waj's GRIPPER_LENGTH;
                                           # matches the drawer grip (94.5 frame z, knob centre ~44). `drill touch`
    pad_length_mm: float = 20.0            # finger pad height above the tip ends (pad centre = 50 mm below frame)
    max_open_mm: float = 84.0              # fully open
    full_open_counts: float = 850.0        # gripper position units at max_open_mm
    finger_width_mm: float = 18.0          # finger size along the jaw (across the closing axis), for clearance
    finger_depth_mm: float = 12.0          # finger thickness along the closing axis
    block_torque_pct: int = 30             # grip force for blocks
    knob_torque_pct: int = 25              # drawer service's proven knob grip
    grip_speed: int = 1500
    settle_s: float = 0.4

    @property
    def pad_center_below_frame_mm(self) -> float:
        return self.tip_below_frame_mm - self.pad_length_mm / 2.0

    def counts(self, mm: float) -> float:
        return max(0.0, min(self.full_open_counts, mm / self.max_open_mm * self.full_open_counts))

    def mm(self, counts: float) -> float:
        return counts / self.full_open_counts * self.max_open_mm


@dataclass
class WorkcellCfg:
    table_z_mm: float = 2.0                # real table top in world z (verify_camera.py); re-fit every look
    obstacle_table_top_z_mm: float = -23.0  # the table obstacle the planner already knows (frame config)
    min_tip_above_table_mm: float = 3.0    # fingertips never go lower than this above the table
    wall_front_x_mm: float = 740.0         # armfarm03 front wall, x 690-790
    wall_side_y_mm: float = 500.0          # armfarm03 side wall (armfarm05's is at -500)
    ceiling_z_mm: float = 1050.0
    x_min_mm: float = -400.0
    x_max_mm: float = 680.0
    y_min_mm: float = -720.0
    y_max_mm: float = 440.0                # keep clear of the side wall
    z_max_mm: float = 800.0
    reach_mm: float = 690.0                # drawer service: 662 mm worked, 738 mm was unreachable


@dataclass
class SpeedCfg:
    travel_deg_s: float = 30.0             # free moves
    slow_deg_s: float = 10.0               # near anything: descents, drawer pulls
    view_deg_s: float = 40.0               # back to the overview pose


@dataclass
class CameraCfg:
    correction: dict | None = None         # cell.calibration.CameraCorrection.to_json(), from `drill calibrate`
    min_depth_mm: float = 250.0            # the depth camera can't see closer (drawer service: needs > ~280)


@dataclass
class OrganizerCfg:
    n_drawers: int = 3
    knob_diameter_mm: float = 23.0
    knob_spacing_mm: float = 75.0          # drawer service find_knobs: 73-76 mm
    knob_z_mm: float = 44.0                # knob centre height (drawer service: 42.6-45.6)
    knob_grip_frame_z_mm: float = 94.5     # gripper frame height that grips a knob at knob_z_mm (proven)
    knob_protrusion_mm: float = 20.0       # knob centre in front of the drawer face (not measured yet)
    grasp_outward_mm: float = 8.0          # grip this far in front of the knob centre (5 mm brushed the box)
    approach_open_mm: float = 35.0         # fingers open on the way down (knob is 23 mm)
    hold_min_mm: float = 14.8              # holding the knob reads 21 mm; outside this range = lost it
    hold_max_mm: float = 29.6
    release_open_mm: float = 30.0
    hover_mm: float = 120.0                # start of the watched descent, above the grip height
    lift_mm: float = 60.0                  # straight up after letting go
    open_target_mm: float = 120.0          # 75% of the travel: the drawers come off past that
    closed_below_mm: float = 12.0          # a knob less than this out of line: closed
    usable_above_mm: float = 90.0          # open at least this far to drop into
    drawer_wall_mm: float = 6.0            # side wall thickness, for the inside of a drawer
    face_mm: float = 10.0                  # drawer front thickness
    floor_above_table_mm: float = 10.0     # used when the floor can't be measured
    rim_above_table_mm: float = 70.0       # drawer walls' top, used when it can't be measured
    clear_mm: float = 160.0                # nothing within this of a knob is a table block


@dataclass
class ContactCfg:
    joints: tuple[int, ...] = (1, 2, 4)    # J2, J3, J5 carry a push on the fingers; J1, J4, J6 don't (J6 rings)
    limits: tuple[float, ...] = (3.0, 1.5, 1.5)
    window_s: float = 0.6                  # trend of each load over this window (drawer service)
    poll_s: float = 0.1
    warmup_s: float = 0.6                  # the start of a move shows acceleration, not contact
    confirm: int = 2                       # readings in a row over the limit
    reach_margin_mm: float = 25.0          # judge contact only once the fingertips are this close to anything
    blocked_margin_mm: float = 12.0        # contact this far above the planned grip height = landed on something


@dataclass
class GraspCfg:
    approach_clearance_mm: float = 12.0    # each side, between an open finger and the block on the way down
    neighbor_clearance_mm: float = 6.0     # between an open finger and a neighbouring block
    hover_above_top_mm: float = 60.0       # fingertips this far above the tallest nearby top before descending
    tip_clearance_mm: float = 4.0          # fingertips above the surface the block stands on
    hold_tolerance_mm: float = 9.0         # finger width after grabbing: block width +- this, else not a hold
    release_gap_mm: float = 8.0            # block bottom above the drop surface when letting go
    carry_clearance_mm: float = 35.0       # block bottom above walls and rims while carrying


@dataclass
class TaskCfg:
    colors: tuple[str, ...] = ("green", "yellow", "black", "blue", "orange", "purple")  # red means a knob
    max_steps: int = 60
    max_tries_per_block: int = 2
    max_tries_per_drawer_move: int = 2


@dataclass
class CellConfig:
    machine: str = "armfarm03"
    gripper: GripperCfg = field(default_factory=GripperCfg)
    workcell: WorkcellCfg = field(default_factory=WorkcellCfg)
    speeds: SpeedCfg = field(default_factory=SpeedCfg)
    camera: CameraCfg = field(default_factory=CameraCfg)
    organizer: OrganizerCfg = field(default_factory=OrganizerCfg)
    contact: ContactCfg = field(default_factory=ContactCfg)
    grasp: GraspCfg = field(default_factory=GraspCfg)
    task: TaskCfg = field(default_factory=TaskCfg)
    view_pose: dict | None = None          # gripper pose for the overview look: x, y, z, o_x, o_y, o_z, theta

    @property
    def path(self) -> Path:
        return CONFIG_DIR / f"{self.machine}.json"

    @classmethod
    def load(cls, machine: str = "armfarm03", path: Path | None = None) -> CellConfig:
        cfg = cls(machine=machine)
        p = path or cfg.path
        if p.exists():
            _merge(cfg, json.loads(p.read_text()))
        if cfg.view_pose is None and LEGACY_VIEW_POSE.exists() and machine == "armfarm03":
            cfg.view_pose = json.loads(LEGACY_VIEW_POSE.read_text())
        return cfg

    def save(self, path: Path | None = None) -> Path:
        p = path or self.path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(dataclasses.asdict(self), indent=2) + "\n")
        return p


def _merge(obj: Any, data: dict) -> None:
    """Overwrite dataclass fields from a (possibly partial) dict, recursing into nested sections."""
    for name, value in data.items():
        if not hasattr(obj, name):
            continue  # an old or misspelled key: ignore rather than fail
        current = getattr(obj, name)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        elif isinstance(current, tuple) and isinstance(value, list):
            setattr(obj, name, tuple(value))
        else:
            setattr(obj, name, value)
