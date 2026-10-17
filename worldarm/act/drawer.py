"""Open or close one drawer.

ClientDrawers (default) works from the knob positions in our own latest look, so it handles any start:
a drawer left open by hand, a partly open one, the box moved. Every knob is measured by a look just before
it's grabbed, taken on the way where possible: after closing a drawer it looks at the row of knobs on the
way up (for the next one it opens), and right after dropping a block into a drawer it looks at that
drawer's knob (for closing it). Otherwise a close look straight down at the knob first (act/look.py).
Mechanics are the drawer service's proven ones: fingers 35 mm open, grip 8 mm in front of the knob centre
at the measured knob height, pull or push in a straight line, let go, straight up - with a gentle, slow
grip (12% torque, not the service's 25%). Closing lifts the drawer 3 mm first so it doesn't catch.
Contact on the way down is judged on J2/J3/J5 only.

ServiceDrawers drives the deployed drawer service (PickBottle) instead. It finds the knobs itself and can
only close a drawer it opened; kept as a fallback (--drawers service).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from worldarm.act.look import MATCH_KNOB_MM, knobs_in_view
from worldarm.act.watch import ContactWatch, WatchResult
from worldarm.cell.geometry import deg_of, down_pose, other_wrist, theta_for_finger_axis, wrap180
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
        # knobs measured by a look just before they're grabbed (on the way up from a close, or right after
        # a drop), with the wrist angle of that look: used once. drawer -> xyz / theta
        self.fresh: dict[int, np.ndarray] = {}
        self.fresh_theta: dict[int, float] = {}

    def forget(self) -> None:
        """Something went wrong: don't trust the remembered knob positions; look again."""
        self.fresh, self.fresh_theta = {}, {}

    def _remember(self, found: dict, theta) -> None:
        for j, p in found.items():
            self.fresh[j], self.fresh_theta[j] = p, theta

    async def look_at_knob(self, org: Organizer, k: int, theta: float) -> dict:
        """Right after dropping a block into drawer k: look at its knob from above, so closing it grabs the
        knob where it is now (never where it was), with no stop of its own."""
        d = org.drawer(k)
        found, info = await knobs_in_view(self.rig, d.knob, (theta, other_wrist(theta)), {k: d.knob},
                                          label=f"look at drawer {k}'s knob after the drop", say=self.say)
        self.fresh.pop(k, None)
        self.fresh_theta.pop(k, None)
        self._remember(found, info.get("theta"))
        return info

    async def open(self, org: Organizer, k: int, target_mm: float | None = None) -> DrawerResult:
        d = org.drawer(k)
        target = self.rig.cfg.organizer.open_target_mm if target_mm is None else target_mm
        return self._kept(await self._slide(org, k, target - d.opening_mm, OPENED))

    def _kept(self, r: DrawerResult) -> DrawerResult:
        if not r.ok:
            self.forget()
        return r

    async def close(self, org: Organizer, k: int) -> DrawerResult:
        """Grip the knob, lift the drawer close_lift_mm so it doesn't catch on the box, push it to 10 mm short
        of shut, then on in one slow watched push until it bottoms out (felt in the loads), at most 18 mm
        more: enough to shut a drawer whose opening was underestimated by 7 mm. When the push doesn't feel
        it shut, the box takes the rest: tape it down."""
        d = org.drawer(k)
        return self._kept(await self._slide(org, k, -(d.opening_mm - 10.0), CLOSED, finish_mm=18.0))

    async def _slide(self, org: Organizer, k: int, delta_mm: float, done: str, finish_mm: float = 0.0) -> DrawerResult:
        rig, oc = self.rig, self.rig.cfg.organizer
        if abs(delta_mm) < 5.0 and finish_mm <= 0.0:
            return DrawerResult(NOTHING, k)
        theta = theta_for_finger_axis(deg_of(org.row) % 180.0, 0.0)  # fingers close along the row, beside the ball
        await rig.open_mm(oc.approach_open_mm)
        knob, seen = await self._knob(org, k, theta)
        grip_xy = knob[:2] + org.pull * oc.grasp_outward_mm
        dz = float(np.clip(knob[2] - oc.knob_z_mm, -8.0, 8.0))  # measured knob height, sanity-limited
        grip_z = oc.knob_grip_frame_z_mm + dz
        hover_z = grip_z + oc.hover_mm

        await rig.set_speed(rig.cfg.speeds.travel_deg_s)
        used, errors = None, []
        first = seen.get("theta")  # the wrist of the look just taken: no turn before going down (if it grips)
        if first is None or min(abs(wrap180(first - t)) for t in (theta, other_wrist(theta))) > 5.0:
            first = theta
        for th in dict.fromkeys((first, theta, other_wrist(theta))):
            try:
                await rig.move(down_pose(grip_xy[0], grip_xy[1], hover_z, th), label=f"above drawer {k}'s knob")
                used = th
                break
            except Exception as e:
                errors.append(f"wrist {th:.0f}: {str(e)[:160]}")
        if used is None:
            return DrawerResult(UNREACHABLE, k, detail={"errors": errors, "knob": seen})

        await rig.set_speed(rig.cfg.speeds.slow_deg_s)
        turned = seen.get("theta") is None or abs(wrap180(used - seen["theta"])) > 5.0
        await rig.settle(0.5 if turned else 0.15)  # a wrist turn leaves joint 6 ringing; no turn, no wait
        watch = ContactWatch(rig.cfg.contact, active_below_z=grip_z + 50.0)
        res = await rig.move_watched(down_pose(grip_xy[0], grip_xy[1], grip_z, used), watch, label=f"down to drawer {k}'s knob")
        if res.contact:
            await rig.move(down_pose(grip_xy[0], grip_xy[1], hover_z, used), linear=True, label="back up: blocked")
            return DrawerResult(BLOCKED, k, contact=res, detail={"knob": seen})

        g = rig.cfg.gripper
        # gently: low torque, a slow close, and aimed just inside the knob (not fully shut), so the fingers
        # settle on the ball instead of squeezing it; a miss still closes below hold_min and is caught
        await rig.grab(g.knob_torque_pct, g.knob_grip_speed, position_mm=oc.hold_min_mm - 3.0)
        w = await rig.width_mm()
        if w is None or not (oc.hold_min_mm <= w <= oc.hold_max_mm):
            await rig.open_mm(oc.approach_open_mm)
            await rig.move(down_pose(grip_xy[0], grip_xy[1], hover_z, used), linear=True, label="back up: missed knob")
            return DrawerResult(MISSED_KNOB, k, detail={"width_mm": w, "knob": seen})

        end = grip_xy + org.pull * delta_mm
        detail = {"from_opening_mm": round(org.drawer(k).opening_mm, 1), "width_mm": round(w, 1), "knob": seen}
        # closing: lift the drawer a hair first so its front slides into the box instead of catching on it
        slide_z = grip_z + (oc.close_lift_mm if done == CLOSED else 0.0)
        try:
            if slide_z > grip_z:
                try:
                    await rig.move(down_pose(grip_xy[0], grip_xy[1], slide_z, used), linear=True,
                                   label=f"lift drawer {k} {oc.close_lift_mm:.1f} mm")
                    detail["lifted_mm"] = oc.close_lift_mm
                except Exception as e:  # too small a move to plan: the push below still ends that much higher
                    self.say(f"    couldn't lift drawer {k} first ({str(e)[:80]}); lifting it along the push")
            if abs(delta_mm) >= 1.0:
                await rig.move(down_pose(end[0], end[1], slide_z, used), linear=True,
                               label=f"{'pull' if delta_mm > 0 else 'push'} drawer {k} {abs(delta_mm):.0f} mm")
            if finish_mm > 0.0:  # the last bit: push on in small steps until it bottoms out
                pushed, felt, ratio = await self._seat(end, slide_z, used, org.pull, finish_mm, k)
                end = end - org.pull * pushed
                detail["seated"] = {"pushed_mm": pushed, "felt_shut": felt, "load_ratio": round(ratio, 2)}
                delta_mm = float((end - grip_xy) @ org.pull)
        except Exception as e:
            await self._let_go(grip_xy, slide_z, used)
            return DrawerResult(FAILED, k, detail={"error": str(e)[:200], **detail})
        w2 = await rig.width_mm()
        if w2 is None or not (oc.hold_min_mm <= w2 <= oc.hold_max_mm):
            await self._let_go(end, slide_z, used)
            return DrawerResult(LOST_KNOB, k, detail={"width_after_mm": w2, **detail})
        await self._let_go(end, slide_z, used)
        knob_xy = end - org.pull * oc.grasp_outward_mm
        if done == CLOSED:  # the next move is usually the drawer beside it: see its knob on the way up
            detail["look_after"] = await self._look_after(org, k, knob_xy, knob[2], used)
        return DrawerResult(done, k, delta_mm, res, detail)

    async def _knob(self, org: Organizer, k: int, theta: float) -> tuple[np.ndarray, dict]:
        """Where knob k is, always from a look just before grabbing it: the one on the way up from the last
        close or right after the last drop (when it agrees with the latest overview), else a close look
        straight down at it now; the overview's only if neither finds it."""
        d = org.drawer(k)
        known, known_theta = self.fresh.get(k), self.fresh_theta.get(k)
        self.fresh, self.fresh_theta = {}, {}  # used once: the next grab looks again
        if known is not None and float(np.linalg.norm(known[:2] - d.knob[:2])) <= MATCH_KNOB_MM:
            return known, {"from": "seen just before (after the last close or drop)", "theta": known_theta,
                           "off_mm": round(float(np.linalg.norm(known[:2] - d.knob[:2])), 1)}
        found, info = await knobs_in_view(self.rig, d.knob, (theta, other_wrist(theta)), {k: d.knob},
                                          label=f"look at drawer {k}'s knob", say=self.say)
        if k in found:
            self.say(f"    close look: drawer {k}'s knob {info['off_mm'][str(k)]:.0f} mm from where the overview had it")
            return found[k], {"from": "close look", **info}
        self.say(f"    close look: didn't find drawer {k}'s knob; going by the overview")
        return d.knob, {"from": "overview", **info}

    async def _look_after(self, org: Organizer, k: int, knob_xy, knob_z: float, theta: float) -> dict:
        """From above the knob just let go, look at the row of knobs and remember where each one is."""
        expected = {x.index: x.knob.copy() for x in org.drawers}
        expected[k] = np.array([knob_xy[0], knob_xy[1], knob_z])
        centre = np.mean(list(expected.values()), axis=0)
        found, info = await knobs_in_view(self.rig, centre, (theta, other_wrist(theta)), expected,
                                          label="look at the knobs on the way up", say=self.say)
        self._remember(found, info.get("theta"))
        return info

    async def _seat(self, xy, grip_z: float, theta: float, pull: np.ndarray, max_mm: float, k: int,
                    ) -> tuple[float, bool, float]:
        """One slow, continuous push of up to max_mm, stopped the moment the joint loads break their trend
        (the drawer has bottomed out and the arm starts pressing): the same contact watch as descents.
        Smooth - the 2-3 mm stop-and-go steps it replaces looked jerky."""
        rig = self.rig
        await rig.set_speed(rig.cfg.speeds.seat_deg_s)
        start = await rig.gripper_pose()
        end = np.asarray(xy) - pull * max_mm
        res = await rig.move_watched(down_pose(end[0], end[1], grip_z, theta), ContactWatch(rig.cfg.contact),
                                     label=f"seat drawer {k}")
        here = await rig.gripper_pose()
        await rig.set_speed(rig.cfg.speeds.slow_deg_s)
        pushed = float(np.array([start.x - here.x, start.y - here.y]) @ pull)
        return max(0.0, pushed), bool(res.contact), float(res.ratio if res.contact else res.worst)

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

    def forget(self) -> None:
        pass  # the service finds the knobs itself every time

    async def look_at_knob(self, org: Organizer, k: int, theta: float) -> dict:
        return {}

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
