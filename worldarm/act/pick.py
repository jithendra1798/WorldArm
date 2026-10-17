"""Pick one block by its grasp plan, and say exactly how it went.

  open to the planned width -> fly above it (wrist turned for the planned finger axis; the half-turn
  alternative if that can't be planned) -> straight down to the grip height, stopping on contact ->
  contact well above the grip height means a finger landed on something: back off, don't close ->
  grab -> finger width must match the block -> lift -> width checked again (it didn't slip out)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from worldarm.act.grasp_plan import GraspPlan
from worldarm.act.watch import ContactWatch, WatchResult
from worldarm.cell.geometry import down_pose

HELD, MISSED, BLOCKED, SLIPPED, UNREACHABLE, INFEASIBLE = "held", "missed", "blocked", "slipped", "unreachable", "infeasible"


@dataclass
class PickResult:
    status: str
    plan: GraspPlan
    theta: float | None = None             # wrist angle used
    width_mm: float | None = None          # finger width while holding
    grip_frame_z: float | None = None      # where the fingers closed
    contact: WatchResult | None = None
    detail: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == HELD

    @property
    def bottom_below_frame_mm(self) -> float:
        """Gripper frame height minus the block's bottom while held (constant until it's let go)."""
        z = self.grip_frame_z if self.grip_frame_z is not None else self.plan.frame_z_grip
        return z - self.plan.base_z

    def to_json(self) -> dict:
        return {"status": self.status, "theta": self.theta, "width_mm": None if self.width_mm is None else round(self.width_mm, 1),
                "grip_frame_z": None if self.grip_frame_z is None else round(self.grip_frame_z, 1),
                "contact": self.contact.to_json() if self.contact else None, "plan": self.plan.to_json(), **self.detail}


async def pick(rig, plan: GraspPlan, *, travel_obstacles=(), say=print) -> PickResult:
    cfg = rig.cfg
    if not plan.feasible:
        return PickResult(INFEASIBLE, plan, detail={"reason": plan.reason})
    tol = cfg.grasp.hold_tolerance_mm
    lo = max(3.0, plan.expected_width_mm - tol)

    await rig.open_mm(plan.open_mm)
    await rig.set_speed(cfg.speeds.travel_deg_s)
    used, errors = None, []
    for theta in (plan.theta, plan.alt_theta):
        try:
            await rig.move(down_pose(plan.x, plan.y, plan.frame_z_hover, theta), extra=travel_obstacles,
                           label="above the block")
            used = theta
            break
        except Exception as e:  # e.g. this wrist angle hits the camera: try the other one
            errors.append(f"wrist {theta:.0f}: {str(e)[:160]}")
    if used is None:
        return PickResult(UNREACHABLE, plan, detail={"errors": errors})

    await rig.set_speed(cfg.speeds.slow_deg_s)
    await rig.settle(cfg.gripper.settle_s)  # let a wrist turn stop ringing before judging loads
    watch = ContactWatch(cfg.contact, active_below_z=plan.reach_z)
    res = await rig.move_watched(down_pose(plan.x, plan.y, plan.frame_z_grip, used), watch, label="down to grip height")
    grip_z = plan.frame_z_grip
    if res.contact:
        if res.z is not None and res.z > plan.frame_z_grip + cfg.contact.blocked_margin_mm:
            await rig.move(down_pose(plan.x, plan.y, plan.frame_z_hover, used), linear=True, label="back up: blocked")
            say(f"    touched something at z={res.z:.0f}, {res.z - plan.frame_z_grip:.0f} mm above the grip height")
            return PickResult(BLOCKED, plan, used, contact=res)
        grip_z = res.z if res.z is not None else grip_z  # touched down near the bottom: grip right there

    await rig.grab(cfg.gripper.block_torque_pct)
    width = await rig.width_mm()
    if width is None or width < lo:
        await rig.open_mm(plan.open_mm)
        await rig.move(down_pose(plan.x, plan.y, plan.frame_z_hover, used), linear=True, label="back up: missed")
        return PickResult(MISSED, plan, used, width, grip_z, res, {"expected_width_mm": round(plan.expected_width_mm, 1)})
    detail = {}
    if width > plan.expected_width_mm + tol:
        detail["warning"] = f"fingers stopped at {width:.0f} mm, expected {plan.expected_width_mm:.0f}: gripped off-axis?"

    await rig.move(down_pose(plan.x, plan.y, plan.frame_z_hover, used), linear=True, label="lift")
    after = await rig.width_mm()
    if after is None or after < lo or abs(after - width) > 6.0:
        await rig.open_mm(plan.open_mm)
        return PickResult(SLIPPED, plan, used, after, grip_z, res, {"width_at_grab_mm": round(width, 1), **detail})
    return PickResult(HELD, plan, used, after, grip_z, res, detail)
