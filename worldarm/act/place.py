"""Set a held block down: into an open drawer, or onto the table.

The drop is planned from geometry measured in the last look: the drawer's inside and the top of its
walls. The block is carried with its bottom clear of the walls to the centre of the drawer, and let go
with the fingertips just above the walls, the fingers opened only a little: it drops the rest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from worldarm.act.pick import PickResult
from worldarm.act.watch import ContactWatch, WatchResult
from worldarm.blocks import Block
from worldarm.cell.geometry import deg_of, down_pose, other_wrist, theta_for_finger_axis
from worldarm.see.organizer import Organizer

PLACED, DROPPED, STUCK, UNREACHABLE, NO_ROOM = "placed", "dropped", "stuck", "unreachable", "no room"
RELEASE_EXTRA_MM = 12.0   # open this much past the block to let go (wider would hit drawer walls)


@dataclass
class DropTarget:
    x: float
    y: float
    surface_z: float                  # what the block will stand on
    clear_z: float                    # the highest thing to clear on the way in (walls, rims)
    finger_axis_deg: float | None     # fingers along this at the drop; None: keep the pick's
    label: str = ""
    release_z: float | None = None    # gripper frame height to let go at; None: block bottom just above the surface
    lowered: bool = True              # False: let go above the walls and let it drop the rest

    def to_json(self) -> dict:
        return {"x": round(self.x, 1), "y": round(self.y, 1), "surface_z": round(self.surface_z, 1),
                "clear_z": round(self.clear_z, 1),
                "finger_axis_deg": None if self.finger_axis_deg is None else round(self.finger_axis_deg, 1),
                "label": self.label, "release_z": None if self.release_z is None else round(self.release_z, 1),
                "lowered": self.lowered}


@dataclass
class PlaceResult:
    status: str
    target: DropTarget | None
    theta: float | None = None
    contact: WatchResult | None = None
    detail: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == PLACED

    def to_json(self) -> dict:
        return {"status": self.status, "target": self.target.to_json() if self.target else None, "theta": self.theta,
                "contact": self.contact.to_json() if self.contact else None, **self.detail}


def drawer_target(org: Organizer, k: int, held: PickResult, contents: list[Block], cfg) -> tuple[DropTarget | None, str]:
    """Where and how to drop a held block into open drawer k, or (None, why not).

    Every block is dropped the same way: over the centre of the drawer's open part, let go with the
    fingertips just above the walls (and anything already inside), turned so its long side lies along
    the drawer's long side. It is never lowered in among the walls.
    """
    d = org.drawer(k)
    if d.inside is None or d.floor_z is None:
        return None, f"can't see into drawer {k}"
    pull, row = org.pull, org.row
    t_in, s_in = d.inside @ row, d.inside @ pull
    t0, t1, s0, s1 = float(t_in.min()), float(t_in.max()), float(s_in.min()), float(s_in.max())
    plan, g, oc = held.plan, cfg.gripper, cfg.organizer
    across = held.width_mm or plan.expected_width_mm      # block size between the fingers
    jaw = plan.jaw_extent_mm                              # block size along the jaws
    # the walls' top; a measurement far below the real walls (seen once: 26 mm) would carry it into the box
    rim = max(d.rim_z if d.rim_z is not None else -1e9, d.floor_z + oc.rim_above_table_mm - oc.floor_above_table_mm)
    if jaw <= (s1 - s0) - 2 and across <= (t1 - t0) - 2:
        axis = row                                        # fingers across the drawer: long side along it
    elif jaw <= (t1 - t0) - 2 and across <= (s1 - s0) - 2:
        axis = pull
    else:
        return None, f"no room in drawer {k} for a {across:.0f} x {jaw:.0f} mm block"
    p = 0.5 * (t0 + t1) * row + 0.5 * (s0 + s1) * pull
    tops = max([rim] + [b.table_z_mm + b.height_mm for b in contents])
    release = tops + 8.0 + g.tip_below_frame_mm           # fingertips just above the walls: it drops the rest
    return DropTarget(float(p[0]), float(p[1]), d.floor_z, tops, deg_of(axis) % 180.0, f"drawer {k}",
                      release_z=float(release), lowered=False), ""


def table_target(xy, table_z: float, neighbors: list[Block] = (), clear_mm: float = 150.0) -> DropTarget:
    near = [b.table_z_mm + b.height_mm for b in neighbors if float(np.linalg.norm(b.center[:2] - np.asarray(xy))) < clear_mm]
    return DropTarget(float(xy[0]), float(xy[1]), table_z, max([table_z] + near), None, "table")


async def place(rig, held: PickResult, target: DropTarget, *, travel_obstacles=(), say=print) -> PlaceResult:
    cfg = rig.cfg
    bottom = held.bottom_below_frame_mm
    carry_z = target.clear_z + cfg.grasp.carry_clearance_mm + bottom
    base_theta = held.theta if held.theta is not None else held.plan.theta
    theta = base_theta if target.finger_axis_deg is None else theta_for_finger_axis(target.finger_axis_deg, base_theta)

    await rig.set_speed(cfg.speeds.slow_deg_s)
    here = await rig.gripper_pose()
    if here.z < carry_z:  # straight up first: never drag a block sideways at its pick height
        await rig.move(down_pose(here.x, here.y, carry_z, base_theta), linear=True, label="up to carry height")
    await rig.set_speed(cfg.speeds.travel_deg_s)
    used, errors = None, []
    for th in (theta, other_wrist(theta)):
        try:
            await rig.move(down_pose(target.x, target.y, carry_z, th), extra=travel_obstacles, label=f"carry to {target.label}")
            used = th
            break
        except Exception as e:
            errors.append(f"wrist {th:.0f}: {str(e)[:160]}")
    if used is None:
        return PlaceResult(UNREACHABLE, target, detail={"errors": errors})
    width = await rig.width_mm()
    lo = max(3.0, held.plan.expected_width_mm - cfg.grasp.hold_tolerance_mm)
    if width is None or width < lo:
        return PlaceResult(DROPPED, target, used, detail={"width_mm": width})

    await rig.set_speed(cfg.speeds.slow_deg_s)
    release_z = target.release_z if target.release_z is not None else target.surface_z + cfg.grasp.release_gap_mm + bottom
    watch = ContactWatch(cfg.contact, active_below_z=release_z + 45.0)
    res = await rig.move_watched(down_pose(target.x, target.y, release_z, used), watch, label=f"lower into {target.label}")
    await rig.open_mm(min(cfg.gripper.max_open_mm, (held.width_mm or held.plan.expected_width_mm) + RELEASE_EXTRA_MM))
    await rig.move(down_pose(target.x, target.y, carry_z, used), linear=True, label="up from the drop")
    after = await rig.width_mm()
    if after is not None and after < (held.width_mm or 0.0) + 5.0:
        return PlaceResult(STUCK, target, used, res, {"width_mm": after})
    return PlaceResult(PLACED, target, used, res)
