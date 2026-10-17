"""MOVES THE ARM. The hurray at the end of a sorted run: up at the overview pose, the wrist waves left and
right while the fingers clap, then everything goes back where it was. Only the wrist roll (J6) and the
gripper move, high above the table, so it can't reach anything."""

from __future__ import annotations

import asyncio
import math

from worldarm.cell.geometry import to_pose

WAVE_DEG = 25.0         # wrist roll each way
WAVES = 3
WAVE_DEG_S = 180.0      # quick: only the wrist roll turns, high up (the run's moves are 10-50 deg/s)
WAVE_ACCEL = 400.0      # deg/s^2 for the wave only: at the run's gentle 80 a swing took ~2 s
CLAP_MM = (84.0, 50.0)  # open, shut: a short clap. 84-10 mm took the gripper 0.85 s (+0.4 s settle), and
                        # every swing waited for it (measured: swings 0.5-0.8 s, claps 1.2-1.4 s)


async def _clap(rig, mm: float) -> None:
    """Fingers to mm without open_mm's settle wait (nothing is held, nothing to wait for)."""
    gripper = getattr(rig, "gripper", None)
    if gripper is None:  # simulated arm
        await rig.open_mm(mm)
        return
    await gripper.do_command({"set": rig.cfg.gripper.counts(mm)})


async def _accel(rig, deg_s2) -> None:
    if deg_s2:
        try:
            await rig.arm.do_command({"set_acceleration": float(deg_s2)})
        except Exception:
            pass  # simulated arm, or a driver without it


async def hurray(rig, say=print) -> None:
    cfg = rig.cfg
    vp = cfg.view_pose
    await rig.set_speed(cfg.speeds.view_deg_s)
    if vp is not None:  # up at the overview first (where a sorted run already ends)
        here = await rig.gripper_pose()
        if math.dist((here.x, here.y, here.z), (vp["x"], vp["y"], vp["z"])) > 5.0:
            await rig.move(to_pose(vp), label="up for the hurray")
    say("HURRAY! Everything is sorted.")
    j0 = await rig.joints()
    await rig.set_speed(WAVE_DEG_S)
    await _accel(rig, WAVE_ACCEL)
    try:
        for i in range(WAVES):
            for side in (1.0, -1.0):
                j = list(j0)
                j[5] = j0[5] + side * WAVE_DEG
                # the clap happens during the swing, not after it
                await asyncio.gather(rig.move_joints(j), _clap(rig, CLAP_MM[0] if side > 0 else CLAP_MM[1]))
        await asyncio.gather(rig.move_joints(j0), _clap(rig, CLAP_MM[0]))
    finally:
        await rig.set_speed(cfg.speeds.view_deg_s)
        await _accel(rig, cfg.speeds.accel_deg_s2)  # back to the run's gentle ramps
