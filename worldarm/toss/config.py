"""Task C settings: the thrower (armfarm05), the catcher and the overhead camera (armfarm03), the swing, the
limits and keep-outs, and the calibrated constants. Measured by toss.py drills, saved to config/toss.json.

Frames: the thrower's base frame for the swing and the target (mm, like Viam); the catcher's frame (armfarm03
world, the overhead camera's table map) for landings, related by `thrower_in_catcher`.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

from worldarm.cell.config import CONFIG_DIR
from worldarm.toss.swing import SwingSpec


@dataclass
class TossCfg:
    thrower_env: str = ".env.arm2"           # armfarm05: Viam address, key, component names
    catcher_env: str = ".env"                # armfarm03: the overhead camera, and the arm that holds the bowl
    thrower: str = "armfarm05"
    catcher: str = "armfarm03"
    camera: str = "overview-cam"
    kinematics: str = "config/kinematics_armfarm05.json"   # the thrower's own model (toss.py probe)
    # the ball and the grip
    ball_diameter_mm: float = 50.0          # measure it
    hold_mm: float | None = None            # finger opening that just holds the ball (toss.py grip --teach); unset:
    grip_torque_pct: int = 67               # close until the fingers stop on the ball, at this torque (holds, not crushes)
    grip_speed: int = 1500
    release_speed: int = 5000               # the gripper's top speed: open as fast as it can, at the trigger
    release_mm: float = 84.0                # opening on release: fully open, the fingers clear the ball fastest
    max_open_mm: float = 84.0               # xArm gripper: 850 counts at 84 mm (chapter 1's GripperCfg)
    full_open_counts: float = 850.0
    # picking the ball up (toss.py pick, throw --pick), Waj's armfarm05 method (throw/real_throw_armfarm05.py):
    # wrist-camera scan from four poses over the table on the arm's left, dark foam + depth, close look from above
    pick_scan: list = field(default_factory=lambda: [[487.7, 495.1, 499.7], [330.0, 330.0, 500.0], [480.0, 220.0, 500.0],
                                                     [300.0, 560.0, 500.0]])   # gripper frame, looking down
    pick_look_z_mm: float = 330.0           # gripper frame height for the close look from straight above the ball
    pick_table_z_mm: float = -23.0          # armfarm05's table top (Waj's measured geometry)
    pick_ball_height_mm: float = 45.0       # the foam ball, when depth reads it flat
    gripper_frame_mm: float = 105.0         # gripper frame below the flange
    tcp_to_tip_mm: float = 55.0
    grasp_lower_mm: float = 6.0
    grasp_raise_mm: float = 10.0            # Waj: grip 10 mm higher than WorldArm's height
    min_tcp_above_table_mm: float = 55.0
    pick_hover_mm: float = 100.0
    close_pct: float = 67.0                 # close the fingers to this share of their travel on the ball (foam squeezes)
    pick_clearance_mm: float = 5.0          # table clearance for the last few cm down to the ball
    pick_fingers_px: int = 110              # the bottom strip of the wrist image where the fingertips show
    # the swing: start pose and directions from `plan --design` (or edited by hand)
    start_deg: list | None = None           # wind-up pose, all joints
    joints: list = field(default_factory=lambda: [1, 2, 4])   # xArm: shoulder, elbow and wrist pitch (0-based)
    signs: list | None = None               # each swinging joint's direction (+1 / -1)
    amplitude_deg: float = 90.0             # how far each swinging joint turns
    accel_deg_s2: float = 600.0             # joint speed-up and slow-down
    tool_mm: tuple = (0.0, 0.0, 155.0)      # ball centre in the flange frame: gripper frame 105 + pad centre 50
    elevation_deg: float | None = None      # release elevation; None: the one where a late release lands the same
    # keep-outs, thrower's base frame (mm). Direct joint moves ignore Viam's obstacles: these are the only guard
    table_z_mm: float = 0.0
    clearance_mm: float = 80.0              # the ball, flange, joints and links stay this far above the table
    base_radius_mm: float = 130.0           # the base column (always near the table) is not checked against it
    ceiling_z_mm: float = 1300.0
    walls: list = field(default_factory=list)   # boxes [[x0, y0, z0], [x1, y1, z1]] the arm must stay out of
    wall_margin_mm: float = 120.0           # kept from walls and ceiling (the check follows the arm's centre line)
    # limits
    max_speed_deg_s: float = 180.0          # the plan uses the least that reaches; this is the most it may ask
    max_accel_deg_s2: float = 1100.0        # xArm850 joints: about 1145 deg/s^2 at most
    windup_speed_deg_s: float = 20.0
    # the target, in the thrower's base frame
    target_mm: tuple = (800.0, 0.0)         # where the ball should come down (x, y)
    catch_z_mm: float = 25.0                # height it comes down to: ball centre on the table, or the bowl's rim
    min_throw_mm: float = 400.0             # the ball must fly at least this far from the release (else it's a drop)
    throw_mm: float | None = 600.0          # how far the ball flies from the release, along heading_deg (None: aim at target_mm)
    heading_deg: float | None = None        # the ball's direction at release (None: toward armfarm03, along the arms' line)
    clear_base_mm: float = 250.0            # the ball's path passes at least this far from the thrower's own base
    # calibrated constants (toss.py dry, grip, calibrate)
    speed_factor: float = 1.0               # ball speed vs. what the joint speed says
    release_delay_s: float = 0.05           # open command sent -> ball free (a guess until calibrated)
    trigger: str = "predict"                # "predict" (from joint readings) or "timed" (from the measured start)
    start_latency_s: float | None = None    # swing command -> the arm starts moving (toss.py dry)
    # where the thrower stands in the catcher's frame (tape measure first; `calibrate` refines it)
    thrower_in_catcher: dict = field(default_factory=lambda: {"x_mm": 1780.0, "y_mm": 0.0, "yaw_deg": 180.0})
    # the catch: armfarm03 holds the bowl
    bowl_diameter_mm: float = 300.0
    bowl_hold: dict | None = None           # flange pose relative to the bowl centre, as held (toss.py catch --teach)
    catch_speed_deg_s: float = 20.0

    @property
    def path(self) -> Path:
        return CONFIG_DIR / "toss.json"

    @classmethod
    def load(cls) -> TossCfg:
        cfg = cls()
        if cfg.path.exists():
            for k, v in json.loads(cfg.path.read_text()).items():
                if hasattr(cfg, k):                    # unknown keys (an older or newer file) are ignored
                    setattr(cfg, k, tuple(v) if isinstance(getattr(cfg, k), tuple) and isinstance(v, list) else v)
        return cfg

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(dataclasses.asdict(self), indent=2) + "\n")
        return self.path

    def spec(self) -> SwingSpec | None:
        """The swing as the planner sees it (None until a start pose and directions exist)."""
        if self.start_deg is None or self.signs is None:
            return None
        return SwingSpec(list(self.start_deg), list(self.joints), list(self.signs), self.amplitude_deg,
                         self.accel_deg_s2, tuple(self.tool_mm), self.table_z_mm, self.clearance_mm)

    def counts(self, mm: float) -> float:
        return max(0.0, min(self.full_open_counts, mm / self.max_open_mm * self.full_open_counts))

    def mm(self, counts: float) -> float:
        return counts / self.full_open_counts * self.max_open_mm
