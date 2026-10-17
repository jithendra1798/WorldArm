"""A simulated arm, gripper and camera with simple physics, standing in for cell.rig.Rig in tests.

Looks are rendered (tests/synth) from the world as it is, so the real perception runs on them. Closing
the fingers across a block holds it; a held knob drags its drawer; opening the fingers drops the held
block onto whatever is under it. Driving the fingertips into anything outside a contact-watched move
raises Crash: a skill that would push into something fails its test.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

import numpy as np
from viam.proto.common import Pose

from tests.synth import Box, Scene3D, block as draw_block
from worldarm.act.watch import WatchResult
from worldarm.cell.calibration import CameraCorrection
from worldarm.cell.config import CellConfig
from worldarm.cell.geometry import finger_axis_deg, pose_matrix, vec_of
from worldarm.cell.rig import Rig


class Crash(AssertionError):
    pass


@dataclass
class FBlock:
    color: str
    x: float
    y: float
    base: float
    h: float
    short: float = 33.0
    long: float = 41.0
    axis: float = 20.0            # world direction of its short side
    slot: int | None = None       # drawer slot it's in, None = on the table

    @property
    def top(self) -> float:
        return self.base + self.h

    def extent(self, deg: float) -> float:
        d = math.radians(deg - self.axis)
        return abs(self.short * math.cos(d)) + abs(self.long * math.sin(d))

    def covers(self, px: float, py: float, pad: float = 0.0) -> bool:
        s, l = vec_of(self.axis), vec_of(self.axis + 90.0)
        d = np.array([px - self.x, py - self.y])
        return abs(d @ s) <= self.short / 2 + pad and abs(d @ l) <= self.long / 2 + pad


@dataclass
class FakeWorld:
    box: Box
    blocks: list[FBlock] = field(default_factory=list)

    def __post_init__(self):
        self.openings = list(self.box.openings)

    # ---- geometry of the organizer ---------------------------------------------------------------
    def _st(self, px, py, slot):
        """(along-row offset from the slot centre, pull coordinate from the box front) of a point."""
        b = self.box
        rel = np.array([px, py]) - (np.asarray(b.front) + b.slot_along(slot) * b.row)
        return float(rel @ b.row), float(rel @ b.pull)

    def knob(self, slot) -> np.ndarray:
        return dataclasses.replace(self.box, openings=tuple(self.openings)).knob(slot)

    def in_drawer(self, px, py) -> int | None:
        b = self.box
        for i in range(3):
            o = self.openings[i]
            t, s = self._st(px, py, i)
            if o > 1 and abs(t) <= b.spacing / 2 - b.wall and 0 <= s <= o - b.face:
                return i
        return None

    def height_at(self, px: float, py: float, skip: FBlock | None = None) -> float:
        """Top of whatever is at (px, py): table, box, drawer parts, knobs, blocks."""
        b = self.box
        top = 0.0
        for i in range(3):
            o = self.openings[i]
            t, s = self._st(px, py, i)
            if abs(t) <= b.spacing / 2 + (5 if i in (0, 2) else 0):
                if -b.depth <= s <= 0:
                    top = max(top, b.height)                                    # box body
                if o > 1 and 0 < s <= o - b.face:
                    if abs(t) <= b.spacing / 2 - b.wall:
                        top = max(top, b.floor)                                 # drawer floor
                    elif abs(t) <= b.spacing / 2:
                        top = max(top, b.rim)                                   # side walls
                if max(0.0, o - b.face) < s <= o and abs(t) <= b.spacing / 2:
                    top = max(top, b.height - 5)                                # drawer front
            k = self.knob(i)
            d = math.hypot(px - k[0], py - k[1])
            if d < b.knob_r:
                top = max(top, k[2] + math.sqrt(b.knob_r ** 2 - d * d))
        for blk in self.blocks:
            if blk is not skip and blk.covers(px, py):
                top = max(top, blk.top)
        return top

    def move_drawer(self, slot: int, delta: float) -> None:
        self.openings[slot] += delta
        for blk in self.blocks:
            if blk.slot == slot:
                blk.x += delta * self.box.pull[0]
                blk.y += delta * self.box.pull[1]

    # ---- the camera --------------------------------------------------------------------------------
    def render(self, cam=None, rot=None):
        s = Scene3D() if cam is None else Scene3D(cam_xy=(cam[0], cam[1]), cam_z=cam[2], rot=rot)
        for blk in self.blocks:
            if blk.slot is None:
                draw_block(s, blk.color, (blk.x, blk.y), (blk.short, blk.long), blk.axis + 90.0, blk.top)
        dataclasses.replace(self.box, openings=tuple(self.openings), contents={}).draw(s)
        for blk in self.blocks:
            if blk.slot is not None and self.openings[blk.slot] > 1:
                draw_block(s, blk.color, (blk.x, blk.y), (blk.short, blk.long), blk.axis + 90.0, blk.top)
        return s.frame()

    # ---- what a judge would check ------------------------------------------------------------------
    def drawer_colors(self) -> dict[int, list[str]]:
        return {i: sorted(b.color for b in self.blocks if b.slot == i) for i in range(3)}

    def table_colors(self) -> list[str]:
        return sorted(b.color for b in self.blocks if b.slot is None)


# The wrist camera on the gripper, like armfarm03's: 72 mm off the gripper axis, 87 mm above the gripper
# frame, looking along it (turned half a turn about its axis, so the overview renders as it always has).
CAM_R = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
CAM_T = np.array([68.4, -23.9, -87.0])
# where the gripper is for the overview: the camera then sits at synth's default (260, -420, 620)
VIEW = {"x": 328.4, "y": -396.1, "z": 533.0, "o_x": 0.0, "o_y": 0.0, "o_z": -1.0, "theta": 0.0}


def fake_cfg() -> CellConfig:
    cfg = CellConfig()
    cfg.workcell.table_z_mm = 0.0           # the synthetic table is at z = 0
    cfg.view_pose = dict(VIEW)
    cfg.gripper.settle_s = 0.0
    return cfg


class FakeRig:
    def __init__(self, world: FakeWorld, cfg: CellConfig | None = None):
        self.world, self.cfg = world, cfg or fake_cfg()
        self.correction = CameraCorrection()
        self.x, self.y, self.z, self.theta = VIEW["x"], VIEW["y"], VIEW["z"], VIEW["theta"]
        self.opening = self.cfg.gripper.max_open_mm
        self.held: FBlock | None = None
        self.held_offset = 0.0
        self.knob: int | None = None
        self.events: list[str] = []
        self.grabs: list[tuple] = []        # (torque, speed) of every grab
        self.looks = 0
        self.look_dir = None

    check = Rig.check

    # ---- reading ---------------------------------------------------------------------------------
    async def look(self):
        """What the wrist camera sees from where the gripper is now."""
        Rg, tg = pose_matrix({"x": self.x, "y": self.y, "z": self.z, "o_x": 0.0, "o_y": 0.0, "o_z": -1.0,
                              "theta": self.theta})
        self.looks += 1
        return self.world.render(tg + Rg @ CAM_T, Rg @ CAM_R)

    async def camera_in_gripper(self):
        return CAM_R, CAM_T

    async def gripper_pose(self) -> Pose:
        return Pose(x=self.x, y=self.y, z=self.z, o_x=0.0, o_y=0.0, o_z=-1.0, theta=self.theta)

    async def frame_z(self):
        return self.z

    async def loads(self):
        p = getattr(self, "pressing", 0.0)
        return [0.0, 1.5 * p, 1.0 * p, 0.0, 0.3 * p, 0.0]   # pressing shows on J2 and J3

    async def joints(self):
        return [0.0, -30.0, -40.0, 0.0, 70.0, self.theta]

    async def move_joints(self, values):
        self.theta = float(values[5])
        self.events.append(f"joints: wrist {self.theta:.0f}")

    async def set_speed(self, deg_s):
        pass

    async def stop(self):
        pass

    async def settle(self, seconds):
        pass  # simulated: nothing rings or sways

    # ---- fingers -----------------------------------------------------------------------------------
    def _probes(self, x, y, theta, opening):
        g = self.cfg.gripper
        u = vec_of(finger_axis_deg(theta))
        a = np.array([-u[1], u[0]])
        pts = []
        for side in (-1.0, 1.0):
            c = np.array([x, y]) + side * (opening / 2 + g.finger_depth_mm / 2) * u
            for da in (-g.finger_depth_mm / 2, 0.0, g.finger_depth_mm / 2):
                for db in (-g.finger_width_mm / 2, 0.0, g.finger_width_mm / 2):
                    pts.append(c + da * u + db * a)
        return pts

    def _contact_z(self, x, y, theta) -> float:
        """Highest gripper frame height at which something under the fingers (or under the held block) is touched."""
        tip = self.cfg.gripper.tip_below_frame_mm
        z = max(self.world.height_at(p[0], p[1], skip=self.held) + tip for p in self._probes(x, y, theta, self.opening))
        if self.held is not None:
            hb = self.held
            for da in (-0.45, 0.0, 0.45):
                for db in (-0.45, 0.0, 0.45):
                    s, l = vec_of(hb.axis), vec_of(hb.axis + 90)
                    p = np.array([x, y]) + da * hb.short * s + db * hb.long * l
                    z = max(z, self.world.height_at(p[0], p[1], skip=hb) + self.held_offset)
        return z

    # ---- moving ------------------------------------------------------------------------------------
    async def move(self, pose, *, linear=False, extra=(), label="move", floor_z=None):
        self.check(pose, label, floor_z)
        self._go(pose, label, None)

    async def move_watched(self, pose, watch, *, linear=True, extra=(), label="watched move", floor_z=None):
        self.check(pose, label, floor_z)
        return self._go(pose, label, watch)

    def _go(self, pose, label, watch) -> WatchResult:
        tx, ty, tz, th = pose.x, pose.y, pose.z, pose.theta
        if self.knob is not None:
            b = self.world.box
            d = np.array([tx - self.x, ty - self.y])
            if abs(float(d @ b.row)) > 3 or abs(tz - self.z) > 3:
                raise Crash(f"{label}: pulled a held knob sideways or up")
            along = float(d @ b.pull)
            new = self.world.openings[self.knob] + along
            if new > 165:
                raise Crash(f"{label}: drawer pulled off ({new:.0f} mm)")
            if new < -0.5 and watch is not None:                   # a watched push stops on pressing
                along = -self.world.openings[self.knob]
                self.world.move_drawer(self.knob, along)
                self.x, self.y = self.x + along * b.pull[0], self.y + along * b.pull[1]
                self.pressing = 1.0
                self.events.append(f"{label}: contact at shut")
                return WatchResult(True, self.z, 2.0, 2.0, None, 1.0)
            if new < -0.5:
                self.pressing = -new                               # the arm presses on a shut drawer
                if self.pressing > 6.0:
                    raise Crash(f"{label}: pushed {self.pressing:.0f} mm past shut (that shoves the box)")
                along = -self.world.openings[self.knob]          # the drawer stops at shut; so does the knob
                self.world.move_drawer(self.knob, along)
                self.x, self.y = self.x + along * b.pull[0], self.y + along * b.pull[1]
                self.events.append(f"{label}: pressing {self.pressing:.0f} mm past shut")
                return WatchResult(False, self.z)
            self.pressing = 0.0
            self.world.move_drawer(self.knob, along)
            self.x, self.y = tx, ty
            self.events.append(f"{label}")
            return WatchResult(False, self.z)
        if math.hypot(tx - self.x, ty - self.y) < 1.0 and tz < self.z:
            hit = self._contact_z(tx, ty, th)
            if hit > tz + 0.5:
                if watch is None or (watch.active_below_z is not None and hit > watch.active_below_z):
                    raise Crash(f"{label}: drove into something at gripper z {hit:.0f} (target {tz:.0f})")
                self._set(tx, ty, hit, th)
                self.events.append(f"{label}: contact at {hit:.0f}")
                return WatchResult(True, hit, 2.0, 2.0, None, 1.0)
        elif self._contact_z(tx, ty, th) > tz + 0.5:
            raise Crash(f"{label}: target ({tx:.0f}, {ty:.0f}, {tz:.0f}) is inside something")
        self._set(tx, ty, tz, th)
        self.events.append(label)
        return WatchResult(False, tz)

    def _set(self, x, y, z, th):
        if self.held is not None:
            self.held.x += x - self.x
            self.held.y += y - self.y
            self.held.base = z - self.held_offset
            self.held.axis += finger_axis_deg(th) - finger_axis_deg(self.theta)
        self.x, self.y, self.z, self.theta = x, y, z, th

    # ---- gripper -----------------------------------------------------------------------------------
    async def open_mm(self, mm):
        if self.held is not None and mm > self.opening + 0.5:
            hb, self.held = self.held, None
            hb.base = max(self.world.height_at(p[0], p[1], skip=hb)
                          for p in [(hb.x, hb.y)] + [(hb.x + dx, hb.y + dy) for dx in (-8, 8) for dy in (-8, 8)])
            hb.slot = self.world.in_drawer(hb.x, hb.y)
            self.events.append(f"let go of {hb.color} at ({hb.x:.0f}, {hb.y:.0f}) on z {hb.base:.0f}"
                               + (f" in slot {hb.slot}" if hb.slot is not None else ""))
        if self.knob is not None and mm > 22.0:
            self.knob = None
            self.pressing = 0.0
        self.opening = mm

    async def width_mm(self):
        return self.opening

    async def grab(self, torque_pct, speed=None, position_mm=None):
        self.grabs.append((torque_pct, speed))
        g = self.cfg.gripper
        u_deg = finger_axis_deg(self.theta)
        u = vec_of(u_deg)
        a = np.array([-u[1], u[0]])
        tip = self.z - g.tip_below_frame_mm
        for blk in self.world.blocks:
            rel = np.array([blk.x - self.x, blk.y - self.y])
            ext_u, ext_a = blk.extent(u_deg), blk.extent(u_deg + 90.0)
            if abs(float(rel @ a)) > ext_a / 2 + g.finger_width_mm / 2 - 3:
                continue
            if abs(float(rel @ u)) + ext_u / 2 > self.opening / 2 + 0.5:
                continue
            if min(tip + g.pad_length_mm, blk.top) - max(tip, blk.base) < 3:
                continue
            blk.x, blk.y = (np.array([self.x, self.y]) + float(rel @ a) * a)   # the fingers centre it
            self.held, self.held_offset, self.opening = blk, self.z - blk.base, ext_u
            blk.slot = None
            self.events.append(f"holding {blk.color}, fingers at {ext_u:.0f} mm")
            return
        for i in range(3):
            k = self.world.knob(i)
            rel = k[:2] - np.array([self.x, self.y])
            # the pads (finger_width across) must close on a chord of the ball at least 16 mm wide
            r = self.world.box.knob_r
            off = max(0.0, abs(float(rel @ a)) - g.finger_width_mm / 2)   # pad edge to ball centre, along pull
            if (off < r and 2 * math.sqrt(r * r - off * off) >= 16.0 and abs(float(rel @ u)) <= self.opening / 2
                    and tip <= k[2] <= tip + g.pad_length_mm):
                self.knob, self.opening = i, 21.0
                self.events.append(f"holding knob {i}")
                return
        self.opening = position_mm or 0.0
        self.events.append("closed on nothing")

    async def holding(self):
        return self.held is not None or self.knob is not None
