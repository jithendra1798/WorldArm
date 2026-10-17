"""Isolated real tests, one per piece of the sorting task. Each takes a rig (real, or the simulated one in
tests) and returns a result dict with "pass". drill.py is the command line around them.

They run the same code the full task runs (overview, perception, grasp planning, pick, place, drawers,
runner.execute), so a drill that passes is that piece of the task working.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from viam.proto.common import Pose

from worldarm.act.grasp_plan import plan_grasp
from worldarm.act.pick import pick
from worldarm.act.place import place, table_target
from worldarm.act.watch import ContactWatch
from worldarm.blocks import measure_surface
from worldarm.cell.calibration import Sighting, fit_correction
from worldarm.cell.geometry import down_pose, pose_dict, tilt_deg
from worldarm.metrics import wilson
from worldarm.see.draw import draw_scene
from worldarm.see.frame import Frame
from worldarm.see.scene import Scene, block_json, free_spot, observe
from worldarm.task.belief import SortBelief
from worldarm.task.policy import Action, decide
from worldarm.task.runner import execute, overview, travel_obstacles


def _save(out: Path | None, name: str, frame: Frame, scene: Scene | None, cfg, title: str = "") -> None:
    if out is None:
        return
    out.mkdir(parents=True, exist_ok=True)
    img = draw_scene(frame, scene, cfg, title) if scene is not None else frame.color
    cv2.imwrite(str(out / f"{name}.jpg"), img)
    frame.save(out / f"{name}.npz")


def rate(k: int, n: int) -> str:
    if n == 0:
        return "0/0"
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({100 * k / n:.0f}%, 95% CI {100 * lo:.0f}-{100 * hi:.0f}%)"


# ---- no motion -----------------------------------------------------------------------------------------

async def check(rig, out: Path | None = None) -> dict:
    """Every sensor and setting the task needs, read without moving."""
    cfg, res, problems = rig.cfg, {}, []
    pose = await rig.gripper_pose()
    res["gripper_pose"] = pose_dict(pose)
    res["joints_deg"] = [round(j, 1) for j in await rig.joints()]
    res["finger_width_mm"] = await rig.width_mm()
    if res["finger_width_mm"] is None:
        problems.append("gripper position can't be read")
    reads = []
    for _ in range(12):
        reads.append(await rig.loads())
        await rig.settle(0.05)
    if any(r is None for r in reads):
        problems.append("joint loads can't be read: contact sensing won't work")
    else:
        sd = np.std(np.array(reads), axis=0)
        res["load_noise"] = {f"J{j + 1}": round(float(sd[j]), 3) for j in cfg.contact.joints}
        noisy = [f"J{j + 1}" for j, lim in zip(cfg.contact.joints, cfg.contact.limits) if sd[j] > lim / 3]
        if noisy:
            problems.append(f"joint loads noisy at rest on {noisy}: contact sensing may stop moves falsely")
    frame = await rig.look()
    valid = float(np.mean((frame.depth > 200) & (frame.depth < 1500)))
    res["image"] = [int(frame.color.shape[1]), int(frame.color.shape[0])]
    res["depth_valid"] = round(valid, 3)
    if valid < 0.5:
        problems.append(f"only {valid:.0%} of the depth image is usable")
    try:
        surf = measure_surface(frame.color, frame.depth, frame.intr, frame.rot, frame.trans)
        res["table_z_here"] = round(surf.table_z(*frame.trans[:2]), 1)
        res["camera_above_table_mm"] = round(float(frame.trans[2]) - res["table_z_here"], 0)
    except Exception as e:
        problems.append(f"no table in view from here ({e})")
    res["camera_correction"] = cfg.camera.correction or "none yet (drill calibrate)"
    res["view_pose"] = cfg.view_pose or "none yet (drill view --teach)"
    res["tip_below_frame_mm"] = cfg.gripper.tip_below_frame_mm
    if not cfg.view_pose:
        problems.append("no overview pose: put the arm there and run `drill view --teach`")
    _save(out, "check", frame, None, cfg)
    res["problems"] = problems
    res["pass"] = not problems
    return res


async def teach_view(rig, force: bool = False, out: Path | None = None) -> dict:
    """Save where the arm is now as the overview pose, if the view is good. Moves nothing."""
    cfg = rig.cfg
    pose = await rig.gripper_pose()
    tilt = tilt_deg(pose)
    frame = await rig.look()
    problems = []
    if tilt > 25:
        problems.append(f"the gripper is tilted {tilt:.0f} deg from straight down (max 25)")
    try:
        scene = observe(frame, cfg)
    except Exception as e:
        return {"pass": False, "problems": [f"can't find the table from here: {e}"]}
    if scene.organizer is None:
        problems.append("the drawer organizer isn't in view (all 3 knobs must be)")
    problems += [w for w in scene.warnings if "cut off" in w]
    if scene.edge_cut:
        problems.append(f"{len(scene.edge_cut)} block(s) cut off by the image edge")
    _save(out, "teach_view", frame, scene, cfg, "teach view")
    res = {"pose": pose_dict(pose), "tilt_deg": round(tilt, 1), "blocks": [block_json(b) for b in scene.table_blocks],
           "organizer": scene.organizer is not None, "problems": problems}
    if problems and not force:
        res["pass"] = False
        res["saved"] = False
        return res
    cfg.view_pose = pose_dict(pose)
    res["saved"] = str(cfg.save())
    res["pass"] = True
    return res


async def policy_dry(rig=None, frame: Frame | None = None, cfg=None, out: Path | None = None) -> dict:
    """What the robot would do next from one look (or a saved frame), and why. Moves nothing."""
    cfg = cfg or rig.cfg
    frame = frame or await rig.look()
    scene = observe(frame, cfg)
    belief = SortBelief(cfg)
    belief.observe(scene, 1)
    action = decide(belief)
    _save(out, "policy", frame, scene, cfg, str(action))
    return {"pass": True, "next": str(action), "action": action.to_json(), "belief": belief.to_json(),
            "warnings": scene.warnings}


def describe(scene: Scene, cfg) -> list[str]:
    """The scene as readable lines."""
    lines = []
    for b in scene.table_blocks:
        p = plan_grasp(b, cfg, scene.table_blocks)
        grip = (f"wrist {p.theta:.0f} (or {p.alt_theta:.0f}), open {p.open_mm:.0f} mm, fingers at z "
                f"{p.frame_z_grip - cfg.gripper.tip_below_frame_mm:.0f}" if p.feasible else f"CAN'T GRIP: {p.reason}")
        lines.append(f"  {b.color:<7} at ({b.center[0]:.0f}, {b.center[1]:.0f})  {b.short_mm:.0f} x {b.long_mm:.0f} mm, "
                     f"{b.height_mm:.0f} tall, short side {b.short_axis_deg:.0f} deg -> {grip}")
    if not scene.table_blocks:
        lines.append("  no blocks on the table")
    org = scene.organizer
    if org is None:
        lines.append("  organizer: NOT FOUND")
    else:
        lines.append(f"  organizer: {org.how}; drawers come out toward {math.degrees(math.atan2(org.pull[1], org.pull[0])):.0f} deg")
        for d in org.drawers:
            inside = scene.contents.get(d.index)
            what = "" if inside is None else ("  empty" if not inside else "  holds " + ", ".join(sorted(b.color for b in inside)))
            extra = f" {d.opening_mm:.0f} mm, floor z {d.floor_z:.0f}, walls z {d.rim_z:.0f}" if d.state != "closed" and d.floor_z else ""
            lines.append(f"    drawer {d.index}: {d.state}{extra}  knob ({d.knob[0]:.0f}, {d.knob[1]:.0f}, {d.knob[2]:.0f}){what}")
    lines += [f"  ! {w}" for w in scene.warnings]
    lines += [f"  ! a {e['color']} block is cut off by the image edge near ({e['xy'][0]:.0f}, {e['xy'][1]:.0f})"
              for e in scene.edge_cut]
    return lines


async def scan(rig, here: bool = False, times: int = 1, out: Path | None = None) -> dict:
    """Look (from the overview pose unless `here`) and report; several times to measure repeatability."""
    cfg = rig.cfg
    scenes = []
    for i in range(times):
        frame = await (rig.look() if here else overview(rig))
        scene = observe(frame, cfg)
        scenes.append(scene)
        _save(out, f"scan-{i + 1}", frame, scene, cfg, f"scan {i + 1}")
    res = {"lines": describe(scenes[-1], cfg), "scene": scenes[-1].to_json()}
    if times > 1:
        res["repeatability_mm"] = _jitter(scenes)
    sc = scenes[-1]
    res["pass"] = sc.organizer is not None and not any("cut off" in w for w in sc.warnings)
    return res


def _jitter(scenes: list[Scene]) -> dict:
    """Spread of each block's position, and each knob's, over repeated looks (nothing moved)."""
    out = {}
    first = scenes[0]
    for i, b in enumerate(first.table_blocks):
        pts = [b.center[:2]]
        for sc in scenes[1:]:
            same = [x for x in sc.table_blocks if x.color == b.color]
            if same:
                pts.append(min(same, key=lambda x: np.linalg.norm(x.center[:2] - b.center[:2])).center[:2])
        out[f"{b.color}#{i}"] = round(float(np.max(np.linalg.norm(np.array(pts) - np.mean(pts, axis=0), axis=1))), 1)
    if all(sc.organizer for sc in scenes):
        for d in first.organizer.drawers:
            pts = [sc.organizer.drawer(d.index).knob[:2] for sc in scenes]
            out[f"knob{d.index}"] = round(float(np.max(np.linalg.norm(np.array(pts) - np.mean(pts, axis=0), axis=1))), 1)
    return out


# ---- motion ----------------------------------------------------------------------------------------------

async def calibrate(rig, drop_mm: float = 120.0, save: bool = False, out: Path | None = None) -> dict:
    """Fix the camera offset: look at the same blocks and knobs with the wrist turned half a turn, at two
    heights. A camera mounting error flips with the wrist, so the looks disagree by twice the error."""
    cfg = rig.cfg
    looks: list[tuple[dict, np.ndarray, np.ndarray]] = []

    async def take(name: str):
        frame = await rig.look()
        scene = observe(frame, cfg)
        _save(out, f"calib-{name}", frame, scene, cfg, name)
        counts = Counter(b.color for b in scene.table_blocks)
        pts = {f"block:{b.color}": frame.in_camera(b.center) for b in scene.table_blocks if counts[b.color] == 1}
        if scene.organizer is not None:
            pts.update({f"knob{d.index}": frame.in_camera(d.knob) for d in scene.organizer.drawers})
        looks.append((pts, frame.rot_cfg, frame.trans_cfg))
        return len(pts)

    await overview(rig)
    start = await rig.joints()
    if await take("wrist-A-high") < 3:
        return {"pass": False, "problems": ["need at least 3 things to look at: put 2-3 blocks of different colors "
                                             "on the table, with the organizer in view"]}
    flip = list(start)
    flip[5] = start[5] - 180.0 if start[5] > 0 else start[5] + 180.0
    await rig.set_speed(cfg.speeds.slow_deg_s)
    await rig.move_joints(flip)
    await rig.settle(0.5)
    await take("wrist-B-high")
    p = await rig.gripper_pose()
    await rig.move(Pose(x=p.x, y=p.y, z=p.z - drop_mm, o_x=p.o_x, o_y=p.o_y, o_z=p.o_z, theta=p.theta),
                   linear=True, label="calibration: lower (same tilt)")
    await take("wrist-B-low")
    low = await rig.joints()
    low[5] = start[5]
    await rig.move_joints(low)
    await rig.settle(0.5)
    await take("wrist-A-low")
    await overview(rig)

    sightings = [Sighting(np.asarray(v, float), R, t, k) for pts, R, t in looks for k, v in pts.items()]
    seen = Counter(s.key for s in sightings)
    sightings = [s for s in sightings if seen[s.key] >= 2]
    try:
        corr, report = fit_correction(sightings)
    except ValueError as e:
        return {"pass": False, "problems": [str(e)]}
    res = {"report": report, "correction": corr.to_json(), "points": sorted(seen)}
    res["pass"] = report["rms_after_mm"] < 4.0
    if save and res["pass"]:
        cfg.camera.correction = corr.to_json()
        rig.correction = corr
        res["saved"] = str(cfg.save())
    return res


async def touch(rig, save: bool = False, out: Path | None = None) -> dict:
    """Measure how far below the gripper frame the fingertips end: lower slowly onto bare table until the
    joint loads say it touched (twice: fast to find it, then slow to measure it)."""
    cfg, g = rig.cfg, rig.cfg.gripper
    frame = await overview(rig)
    scene = observe(frame, cfg)
    spot = free_spot(scene, frame, cfg)
    if spot is None:
        return {"pass": False, "problems": ["no clear table spot in view"]}
    table_z = scene.surface.table_z(*spot)
    lowest = table_z + 35.0                           # 25 mm past where 60 mm fingertips would touch
    await rig.open_mm(40.0)
    await rig.set_speed(cfg.speeds.travel_deg_s)
    await rig.move(down_pose(spot[0], spot[1], table_z + 130.0, 0.0), label="touch: above a clear spot")
    found = []
    for speed, start in ((8.0, table_z + 130.0), (3.0, None)):
        if start is None:
            start = found[-1] + 15.0
            await rig.move(down_pose(spot[0], spot[1], start, 0.0), linear=True, label="touch: back up", floor_z=lowest)
        await rig.set_speed(speed)
        await rig.settle(0.5)
        res = await rig.move_watched(down_pose(spot[0], spot[1], lowest, 0.0), ContactWatch(cfg.contact, table_z + 100.0),
                                     label="touch: down to the table", floor_z=lowest - 1.0)
        await rig.move(down_pose(spot[0], spot[1], (res.z or lowest) + 30.0, 0.0), linear=True, label="touch: up",
                       floor_z=lowest)
        if not res.contact:
            return {"pass": False, "problems": [f"no touch felt down to gripper z {lowest:.0f}: contact sensing isn't "
                                                "working, or the fingertips are shorter than 35 mm"]}
        found.append(float(res.z))
    tip = found[-1] - table_z
    result = {"tip_below_frame_mm": round(tip, 1), "fast_pass_mm": round(found[0] - table_z, 1), "table_z": round(table_z, 1),
              "was": g.tip_below_frame_mm}
    result["pass"] = 40.0 <= tip <= 80.0
    if save and result["pass"]:
        g.tip_below_frame_mm = round(tip, 1)
        result["saved"] = str(cfg.save())
    await overview(rig)
    return result


async def pick_test(rig, color: str | None = None, times: int = 1, hold_s: float = 2.0, out: Path | None = None) -> dict:
    """Pick a block, hold it (is the grip firm?), put it back where it was. Repeat."""
    cfg = rig.cfg
    trials = []
    for i in range(times):
        frame = await overview(rig)
        scene = observe(frame, cfg)
        _save(out, f"pick-{i + 1}-before", frame, scene, cfg)
        cands = [b for b in scene.table_blocks if color is None or b.color == color]
        if not cands:
            trials.append({"status": "not seen", "color": color})
            break
        b = cands[0]
        plan = plan_grasp(b, cfg, scene.table_blocks)
        held = await pick(rig, plan, travel_obstacles=travel_obstacles(scene))
        rec = {"color": b.color, "block": block_json(b), "pick": held.to_json(), "status": held.status}
        if held.ok:
            await rig.settle(hold_s)
            w = await rig.width_mm()
            rec["firm"] = w is not None and abs(w - (held.width_mm or 0)) < 3.0
            back = await place(rig, held, table_target(b.center[:2], b.table_z_mm), travel_obstacles=travel_obstacles(scene))
            rec["put_back"] = back.to_json()
            if not rec["firm"]:
                rec["status"] = "loose"
        trials.append(rec)
    ok = sum(1 for t in trials if t["status"] == "held")
    return {"pass": ok == len(trials) and bool(trials), "rate": rate(ok, len(trials)), "trials": trials}


async def drawer_test(rig, drawers, k: int, mode: str = "cycle", times: int = 1, out: Path | None = None) -> dict:
    """Open and/or close drawer k, checking each move with a look: how far it moved, and that the box didn't."""
    cfg = rig.cfg
    ops = {"open": ["open"], "close": ["close"], "cycle": ["open", "close"]}[mode]
    trials, prior = [], None
    for i in range(times):
        for op in ops:
            frame = await overview(rig)
            before = observe(frame, cfg, prior)
            if before.organizer is None:
                return {"pass": False, "problems": ["organizer not in view"], "trials": trials}
            prior = before.organizer.model()
            r = await (drawers.open(before.organizer, k) if op == "open" else drawers.close(before.organizer, k))
            frame2 = await overview(rig)
            after = observe(frame2, cfg, prior)
            _save(out, f"drawer-{i + 1}-{op}", frame2, after, cfg, f"after {op} {k}")
            rec = {"op": op, "result": r.to_json(), "opening_before": round(before.organizer.drawer(k).opening_mm, 1)}
            if after.organizer is None:
                rec.update(status="organizer lost")
            else:
                d = after.organizer.drawer(k)
                moved = float(np.linalg.norm(after.organizer.box.mean(axis=0) - before.organizer.box.mean(axis=0)))
                good = (d.opening_mm >= cfg.organizer.usable_above_mm) if op == "open" else (d.state == "closed")
                rec.update(opening_after=round(d.opening_mm, 1), box_moved_mm=round(moved, 1),
                           status="ok" if (r.ok and good and moved < 15.0) else
                           ("box slid" if moved >= 15.0 else f"drawer at {d.opening_mm:.0f} mm"))
                prior = after.organizer.model()
            trials.append(rec)
    ok = sum(1 for t in trials if t["status"] == "ok")
    return {"pass": ok == len(trials), "rate": rate(ok, len(trials)), "trials": trials}


async def drop_test(rig, drawers, k: int, color: str | None = None, out: Path | None = None) -> dict:
    """One table block into drawer k (opened first if needed); then look: is it in there?"""
    cfg = rig.cfg
    frame = await overview(rig)
    scene = observe(frame, cfg)
    if scene.organizer is None:
        return {"pass": False, "problems": ["organizer not in view"]}
    if scene.organizer.drawer(k).state != "open":
        r = await drawers.open(scene.organizer, k)
        if not r.ok:
            return {"pass": False, "problems": [f"couldn't open drawer {k}: {r.status}"], "drawer": r.to_json()}
        frame = await overview(rig)
        scene = observe(frame, cfg, scene.organizer.model())
    cands = [b for b in scene.table_blocks if color is None or b.color == color]
    if not cands:
        return {"pass": False, "problems": [f"no {color or ''} block on the table"]}
    b = cands[0]
    before = Counter(x.color for x in scene.contents.get(k, []))
    ok, outcome = await execute(rig, drawers, scene, frame, Action("transfer", k, b, b.color))
    frame2 = await overview(rig)
    after = observe(frame2, cfg, scene.organizer.model())
    _save(out, "drop-after", frame2, after, cfg, f"after dropping {b.color} into {k}")
    now = Counter(x.color for x in after.contents.get(k, []))
    seen = now[b.color] > before[b.color]
    return {"pass": ok and seen, "outcome": outcome, "seen_in_drawer": seen, "contents_after": dict(now)}


async def unload_test(rig, drawers, k: int, color: str | None = None, out: Path | None = None) -> dict:
    """One block out of open drawer k onto the table; then look: is it on the table?"""
    cfg = rig.cfg
    frame = await overview(rig)
    scene = observe(frame, cfg)
    org = scene.organizer
    if org is None or org.drawer(k).state != "open":
        return {"pass": False, "problems": [f"drawer {k} isn't open (open it: drill drawer --open {k})"]}
    inside = [b for b in scene.contents.get(k, []) if color is None or b.color == color]
    if not inside:
        return {"pass": False, "problems": [f"no {color or ''} block seen inside drawer {k}"]}
    b = max(inside, key=lambda x: float(x.center[:2] @ org.pull))
    n_before = sum(1 for x in scene.table_blocks if x.color == b.color)
    ok, outcome = await execute(rig, drawers, scene, frame, Action("unload", k, b, b.color))
    frame2 = await overview(rig)
    after = observe(frame2, cfg, org.model())
    _save(out, "unload-after", frame2, after, cfg, f"after taking {b.color} out of {k}")
    on_table = sum(1 for x in after.table_blocks if x.color == b.color) > n_before
    return {"pass": ok and on_table, "outcome": outcome, "on_table": on_table}
