"""Open or close one drawer.

ClientDrawers (default) works from the knob positions in our own latest look, so it handles any start:
a drawer left open by hand, a partly open one, the box moved. Mechanics are the drawer service's proven
ones: fingers 35 mm open, grip 8 mm in front of the knob centre at the measured knob height, 25% torque,
pull or push in a straight line, let go, straight up. Contact on the way down is judged on J2/J3/J5 only.

ServiceDrawers drives the deployed drawer service (PickBottle) instead. It finds the knobs itself and can
only close a drawer it opened; kept as a fallback (--drawers service).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from worldarm.act.watch import ContactWatch, WatchResult
from worldarm.cell.geometry import deg_of, down_pose, other_wrist, theta_for_finger_axis
from worldarm.see.organizer import Organizer

OPENED, CLOSED, NOTHING, BLOCKED, MISSED_KNOB, LOST_KNOB, UNREACHABLE, FAILED = (
    "opened", "closed", "nothing to do", "blocked", "missed the knob", "lost the knob", "unreachable", "failed")


@dataclass
class DrawerResult:
    status: str
    drawer: int
    moved_mm: float = 0.0
    contact: WatchResult | None = None
    detail: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in (OPENED, CLOSED, NOTHING)

    def to_json(self) -> dict:
        return {"status": self.status, "drawer": self.drawer, "moved_mm": round(self.moved_mm, 1),
                "contact": self.contact.to_json() if self.contact else None, **self.detail}


class ClientDrawers:
    name = "client"

    def __init__(self, rig, say=print):
        self.rig, self.say = rig, say

    async def open(self, org: Organizer, k: int, target_mm: float | None = None) -> DrawerResult:
        d = org.drawer(k)
        target = self.rig.cfg.organizer.open_target_mm if target_mm is None else target_mm
        return await self._slide(org, k, target - d.opening_mm, OPENED)

    async def close(self, org: Organizer, k: int) -> DrawerResult:
        """Push it to 10 mm short of shut, then on in 2 mm steps until it bottoms out (felt in the loads), at
        most 18 mm more: never a fixed distance past shut, which shoves the whole box when the opening is
        overestimated. Stops within a few mm of pressing whether the opening was over- or underestimated."""
        d = org.drawer(k)
        return await self._slide(org, k, -(d.opening_mm - 10.0), CLOSED, finish_mm=18.0)

    async def _slide(self, org: Organizer, k: int, delta_mm: float, done: str, finish_mm: float = 0.0) -> DrawerResult:
        rig, oc = self.rig, self.rig.cfg.organizer
        if abs(delta_mm) < 5.0 and finish_mm <= 0.0:
            return DrawerResult(NOTHING, k)
        d = org.drawer(k)
        grip_xy = d.knob[:2] + org.pull * oc.grasp_outward_mm
        dz = float(np.clip(d.knob[2] - oc.knob_z_mm, -8.0, 8.0))  # measured knob height, sanity-limited
        grip_z = oc.knob_grip_frame_z_mm + dz
        hover_z = grip_z + oc.hover_mm
        theta = theta_for_finger_axis(deg_of(org.row) % 180.0, 0.0)  # fingers close along the row, beside the ball

        await rig.open_mm(oc.approach_open_mm)
        await rig.set_speed(rig.cfg.speeds.travel_deg_s)
        used, errors = None, []
        for th in (theta, other_wrist(theta)):
            try:
                await rig.move(down_pose(grip_xy[0], grip_xy[1], hover_z, th), label=f"above drawer {k}'s knob")
                used = th
                break
            except Exception as e:
                errors.append(f"wrist {th:.0f}: {str(e)[:160]}")
        if used is None:
            return DrawerResult(UNREACHABLE, k, detail={"errors": errors})

        await rig.set_speed(rig.cfg.speeds.slow_deg_s)
        await rig.settle(0.5)  # a wrist turn leaves joint 6 ringing
        watch = ContactWatch(rig.cfg.contact, active_below_z=grip_z + 50.0)
        res = await rig.move_watched(down_pose(grip_xy[0], grip_xy[1], grip_z, used), watch, label=f"down to drawer {k}'s knob")
        if res.contact:
            await rig.move(down_pose(grip_xy[0], grip_xy[1], hover_z, used), linear=True, label="back up: blocked")
            return DrawerResult(BLOCKED, k, contact=res)

        await rig.grab(rig.cfg.gripper.knob_torque_pct)
        w = await rig.width_mm()
        if w is None or not (oc.hold_min_mm <= w <= oc.hold_max_mm):
            await rig.open_mm(oc.approach_open_mm)
            await rig.move(down_pose(grip_xy[0], grip_xy[1], hover_z, used), linear=True, label="back up: missed knob")
            return DrawerResult(MISSED_KNOB, k, detail={"width_mm": w})

        end = grip_xy + org.pull * delta_mm
        detail = {"from_opening_mm": round(org.drawer(k).opening_mm, 1), "width_mm": round(w, 1)}
        try:
            if abs(delta_mm) >= 1.0:
                await rig.move(down_pose(end[0], end[1], grip_z, used), linear=True,
                               label=f"{'pull' if delta_mm > 0 else 'push'} drawer {k} {abs(delta_mm):.0f} mm")
            if finish_mm > 0.0:  # the last bit: push on in small steps until it bottoms out
                pushed, felt, ratio = await self._seat(end, grip_z, used, org.pull, finish_mm, k)
                end = end - org.pull * pushed
                detail["seated"] = {"pushed_mm": pushed, "felt_shut": felt, "load_ratio": round(ratio, 2)}
                delta_mm = float((end - grip_xy) @ org.pull)
        except Exception as e:
            await self._let_go(grip_xy, grip_z, used)
            return DrawerResult(FAILED, k, detail={"error": str(e)[:200], **detail})
        w2 = await rig.width_mm()
        if w2 is None or not (oc.hold_min_mm <= w2 <= oc.hold_max_mm):
            await self._let_go(end, grip_z, used)
            return DrawerResult(LOST_KNOB, k, detail={"width_after_mm": w2, **detail})
        await self._let_go(end, grip_z, used)
        return DrawerResult(done, k, delta_mm, res, detail)

    async def _seat(self, xy, grip_z: float, theta: float, pull: np.ndarray, max_mm: float, k: int,
                    step_mm: float = 2.0) -> tuple[float, bool, float]:
        """Push 2 mm at a time, reading the joint loads at rest after each step against the loads before the
        first: once the drawer is shut the arm presses on it, and the loads show it. Stops right there."""
        rig, c = self.rig, self.rig.cfg.contact
        await rig.settle(0.25)
        base = await rig.loads()
        pushed, worst = 0.0, 0.0
        while pushed + step_mm <= max_mm + 1e-6:
            nxt = np.asarray(xy) - pull * (pushed + step_mm)
            await rig.move(down_pose(nxt[0], nxt[1], grip_z, theta), linear=True, label=f"seat drawer {k}")
            pushed += step_mm
            await rig.settle(0.25)
            now = await rig.loads()
            if base is None or now is None:
                return pushed, False, worst  # can't feel it: stop after one step rather than push blind
            worst = max(abs(now[j] - base[j]) / lim for j, lim in zip(c.joints, c.limits))
            if worst > 1.0:
                return pushed, True, worst
        return pushed, False, worst

    async def _let_go(self, xy, grip_z: float, theta: float) -> None:
        rig, oc = self.rig, self.rig.cfg.organizer
        await rig.open_mm(oc.release_open_mm)
        await rig.set_speed(rig.cfg.speeds.slow_deg_s)
        await rig.move(down_pose(xy[0], xy[1], grip_z + oc.lift_mm, theta), linear=True, label="straight up from the knob")
        await rig.set_speed(rig.cfg.speeds.travel_deg_s)


class ServiceDrawers:
    """The deployed drawer service. Never sends {} (that runs every drawer)."""
    name = "service"

    def __init__(self, machine, service_name: str = "PickBottle", say=print):
        from viam.services.generic import Generic
        self.svc = Generic.from_robot(machine, service_name)
        self.say = say

    async def open(self, org: Organizer, k: int, target_mm: float | None = None) -> DrawerResult:
        r = dict(await self.svc.do_command({"action": "open", "drawer": int(k), "refine": False}))
        return DrawerResult(OPENED if r.get("success") else FAILED, k, detail=_trim(r))

    async def close(self, org: Organizer, k: int) -> DrawerResult:
        st = dict(await self.svc.do_command({"action": "status"})).get("open_drawers") or {}
        if str(k) not in st:
            return DrawerResult(FAILED, k, detail={"error": f"the service didn't open drawer {k}; it can't close it"})
        r = dict(await self.svc.do_command({"action": "close", "drawer": int(k)}))
        return DrawerResult(CLOSED if r.get("success") else FAILED, k, detail=_trim(r))


def _trim(r: dict) -> dict:
    return {k: v for k, v in r.items() if k in ("success", "stage", "error", "drop_point", "open_dir", "opened", "grip_pos")}
