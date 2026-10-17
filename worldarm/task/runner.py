"""The sorting loop: look -> update the belief -> decide -> act -> (look again, or remember).

The arm only goes back up for an overview look when it has to: at the start, after opening a drawer
(what's inside?), after anything failed, and before it stops (it only calls the task sorted, or gives up,
on what a real look shows). After a drawer closed or a block went into a drawer it already knows what
changed, so the next step starts from memory and the arm goes straight on to the next thing. Where exactly
to grip is measured just before touching: a close look at a block, and for knobs where the arm left them
or what it saw on the way up (act/look.py, act/drawer.py). Each step is logged to runs/<stamp>-sort/: the
annotated picture (step-NN.jpg) and raw frame (step-NN.npz, replayable offline) of each overview look,
and a JSON record of the scene, belief, decision, outcome and timings.
"""

from __future__ import annotations

import dataclasses
import math
import time
from pathlib import Path

import cv2
import numpy as np

from worldarm.act.grasp_plan import plan_grasp
from worldarm.act.look import refine_block
from worldarm.act.pick import pick
from worldarm.act.place import UNREACHABLE, drawer_target, place, table_target
from worldarm.cell.geometry import to_pose
from worldarm.cell.rig import box_geometry
from worldarm.see.draw import draw_scene
from worldarm.see.frame import Frame
from worldarm.cell.geometry import point_in_polygon
from worldarm.see.scene import Scene, _organizer_keep_out, free_spot, observe, visible_table
from worldarm.task.belief import SortBelief
from worldarm.task.policy import Action, decide


async def overview(rig) -> Frame:
    """Go to the overview pose (if not there already) and take one look."""
    vp = rig.cfg.view_pose
    if vp is None:
        raise RuntimeError("no overview pose yet: put the arm there and run `python drill.py view --teach`")
    here = await rig.gripper_pose()
    if math.dist((here.x, here.y, here.z), (vp["x"], vp["y"], vp["z"])) > 5.0 or abs(here.theta - vp["theta"]) > 2.0:
        await rig.set_speed(rig.cfg.speeds.view_deg_s)
        await rig.move(to_pose(vp), label="overview")
        await rig.settle(0.3)  # let it stop swaying before the picture
    return await rig.look()


async def survey(rig, scene: Scene, logger=None, say=print) -> Scene:
    """Once, at the start: the overview sees only part of the table. From the overview pose, turn the arm
    about its base alone (same height and tilt: every pose is as clear of the table as the overview) and
    look at each turn; every block found joins the overview's. Then back to the overview."""
    cfg = rig.cfg
    if not cfg.task.survey_turns_deg:
        return scene
    j0 = await rig.joints()
    blocks = list(scene.table_blocks)
    keep_out = _organizer_keep_out(scene.organizer, 30.0) if scene.organizer is not None else []
    await rig.set_speed(cfg.speeds.view_deg_s)
    for turn in cfg.task.survey_turns_deg:
        j = list(j0)
        j[0] += turn
        await rig.move_joints(j)
        await rig.settle(0.3)
        frame = await rig.look()
        sc = observe(frame, cfg)
        x0, x1, y0, y1 = cfg.task.sort_area_xy
        new = [b for b in sc.table_blocks if b.color in cfg.task.colors
               and x0 <= b.center[0] <= x1 and y0 <= b.center[1] <= y1     # our side of the table only
               and b.short_mm <= 45.0 and b.long_mm <= 95.0                 # block-sized (plugs, cables aren't)
               and not any(point_in_polygon(b.center[:2], p)[0] for p in keep_out)
               and all(b.color != o.color or float(np.linalg.norm(b.center[:2] - o.center[:2])) > 50.0 for o in blocks)]
        blocks += new
        say(f"    survey {turn:+.0f} deg: {len(sc.table_blocks)} block(s) in view, new: "
            + (", ".join(f"{b.color}@({b.center[0]:.0f},{b.center[1]:.0f})" for b in new) or "none"))
        if logger is not None:
            pic = draw_scene(frame, sc, cfg, f"survey {turn:+.0f} deg: {len(new)} new")
            cv2.imwrite(str(Path(logger.dir) / f"survey{turn:+.0f}.jpg"), pic)
            frame.save(Path(logger.dir) / f"survey{turn:+.0f}.npz")
    await rig.move_joints(j0)
    return dataclasses.replace(scene, table_blocks=blocks)


def travel_obstacles(scene: Scene) -> list:
    """The organizer box, for the planner, on moves between places (not on the final straight moves)."""
    org = scene.organizer
    if org is None:
        return []
    base = float(np.mean([scene.surface.table_z(*c) for c in org.box])) if scene.surface is not None else 0.0
    return [box_geometry("wa-organizer", org.box, base, org.box_top_z + 10.0)]


async def execute(rig, drawers, scene: Scene, frame: Frame, action: Action, say=print) -> tuple[bool, dict]:
    cfg, org = rig.cfg, scene.organizer
    obs = travel_obstacles(scene)
    if action.kind == "open":
        r = await drawers.open(org, action.drawer)
        return r.ok, {"drawer": r.to_json()}
    if action.kind == "close":
        r = await drawers.close(org, action.drawer)
        return r.ok, {"drawer": r.to_json()}

    if action.kind == "transfer":
        block, k = action.block, action.drawer
        first = plan_grasp(block, cfg, scene.table_blocks)  # its wrist angle, for the close look
        block, neighbors, look = await refine_block(rig, block, scene.table_blocks, (first.theta, first.alt_theta),
                                                    extra=obs, say=say)
        plan = plan_grasp(block, cfg, neighbors)
        held = await pick(rig, plan, travel_obstacles=obs, say=say)
        out = {"look": look, "pick": held.to_json()}
        if not held.ok:
            return False, out
        target, why = drawer_target(org, k, held, scene.contents.get(k, []), cfg)
        if target is None:
            say(f"    {why}: putting it back")
            back = await place(rig, held, table_target(block.center[:2], block.table_z_mm), travel_obstacles=obs, say=say)
            out["place"] = {"status": "no room", "why": why, "put_back": back.to_json()}
            return False, out
        res = await place(rig, held, target, travel_obstacles=obs, say=say)
        out["place"] = res.to_json()
        if res.ok:  # while there: measure the knob again, so closing the drawer grabs it where it is now
            out["knob_look"] = await drawers.look_at_knob(org, k, res.theta)
        if res.status == UNREACHABLE:
            back = await place(rig, held, table_target(block.center[:2], block.table_z_mm), travel_obstacles=obs, say=say)
            out["put_back"] = back.to_json()
        return res.ok, out

    if action.kind == "unload":
        block, k = action.block, action.drawer
        d = org.drawer(k)
        others = scene.contents.get(k, [])
        plan = plan_grasp(block, cfg, others, inside=d.inside, clear_z=d.rim_z)
        held = await pick(rig, plan, travel_obstacles=[], say=say)
        out = {"pick": held.to_json()}
        if not held.ok:
            return False, out
        spot = free_spot(scene, frame, cfg, near_xy=d.knob[:2])
        if spot is None:
            target = None
            out["place"] = {"status": "no free table spot"}
        else:
            target = table_target(spot, scene.surface.table_z(*spot) if scene.surface else cfg.workcell.table_z_mm,
                                  scene.table_blocks)
        if target is None:  # nowhere to put it: back into the drawer where it was
            target = table_target(block.center[:2], block.table_z_mm)
        res = await place(rig, held, target, travel_obstacles=obs, say=say)
        out["place"] = res.to_json()
        return res.ok and spot is not None, out
    raise ValueError(f"unknown action {action.kind}")


def remembered(scene: Scene, action: Action, ok: bool, outcome: dict) -> Scene | None:
    """The scene after an action whose effect is sure enough to skip the overview look, or None (look).
    A close: that drawer shut, everything else as it was. A transfer: the block off the table and in the
    middle of the drawer, where it was dropped. An open needs a look (what's inside?), and so do an unload
    and anything that failed."""
    org, k = scene.organizer, action.drawer
    if not ok or org is None:
        return None
    if action.kind == "close":
        d = org.drawer(k)
        knob = d.knob.copy()
        knob[:2] = knob[:2] - org.pull * d.opening_mm
        shut = dataclasses.replace(d, knob=knob, opening_mm=0.0, state="closed", inside=None, floor_z=None, rim_z=None)
        return dataclasses.replace(scene, organizer=dataclasses.replace(org, drawers=[shut if x.index == k else x
                                                                                      for x in org.drawers]),
                                   contents={j: v for j, v in scene.contents.items() if j != k})
    if action.kind == "transfer":
        target, b = (outcome.get("place") or {}).get("target"), action.block
        if target is None or b is None:
            return None
        dropped = dataclasses.replace(b, center=np.array([target["x"], target["y"], target["surface_z"] + b.height_mm / 2.0]),
                                      table_z_mm=target["surface_z"])
        contents = dict(scene.contents)
        contents[k] = list(contents.get(k, [])) + [dropped]
        return dataclasses.replace(scene, table_blocks=[x for x in scene.table_blocks if x is not b], contents=contents)
    return None


async def run_task(rig, drawers, logger=None, *, max_steps: int | None = None, dry_run: bool = False,
                   say=print) -> dict:
    cfg = rig.cfg
    belief = SortBelief(cfg)
    max_steps = max_steps or cfg.task.max_steps
    t_start = time.time()
    result = {"status": "max steps", "success": False, "steps": 0, "surprises": 0}
    if logger is not None:
        rig.look_dir = Path(logger.dir) / "looks"  # close looks (act/look.py) are kept here
    idle = 0  # drawer moves since a block last moved: a loop the rules didn't foresee shouldn't run forever
    scene: Scene | None = None        # None: the next step starts with an overview look
    frame: Frame | None = None
    for step in range(1, max_steps + 1):
        t0 = time.time()
        looked, visible = scene is None, None
        if looked:
            frame = await overview(rig)
            scene = shown = observe(frame, cfg, belief.row_model)
            if step == 1 and not dry_run:
                scene = await survey(rig, scene, logger, say)  # every block on the table, once
            else:
                visible = visible_table(frame, scene.table)    # blocks outside this view are remembered
        t_look = time.time()
        surprises = belief.observe(scene, step, visible)
        action = decide(belief)
        if not looked and action.kind in ("stop", "unload"):  # stop, or reach into a drawer, only on a real look
            frame = await overview(rig)
            scene = shown = observe(frame, cfg, belief.row_model)
            looked, t_look = True, time.time()
            surprises += belief.observe(scene, step, visible_table(frame, scene.table))
            action = decide(belief)
        scene = dataclasses.replace(scene, table_blocks=list(belief.table))  # what it knows: seen or remembered
        t_think = time.time()
        for s in surprises:
            say(f"    SURPRISE {s}")
        say(f"{step:>3}  {action}" + ("" if looked else "   [from memory, no rescan]"))
        record = {"step": step, "looked": looked, "action": action.to_json(), "scene": scene.to_json(),
                  "belief": belief.to_json(), "surprises": [str(s) for s in surprises],
                  "seconds": {"look": round(t_look - t0, 2), "think": round(t_think - t_look, 2)}}
        if logger is not None and looked:
            pic = draw_scene(frame, shown, cfg, f"step {step}: {action}")
            cv2.imwrite(str(Path(logger.dir) / f"step-{step:02d}.jpg"), pic)
            frame.save(Path(logger.dir) / f"step-{step:02d}.npz")
        if action.kind == "stop" or dry_run:
            result = {"status": "done" if action.kind == "stop" else "dry run", "success": bool(action.success),
                      "reason": action.reason, "steps": step}
            if logger is not None:
                logger.step(record)
            break
        try:
            ok, outcome = await execute(rig, drawers, scene, frame, action, say)
        except Exception as e:  # stop safely and keep the log
            await rig.stop()
            record.update(outcome={"error": f"{type(e).__name__}: {e}"}, ok=False)
            if logger is not None:
                logger.step(record)
            result = {"status": "aborted", "success": False, "reason": f"{action}: {type(e).__name__}: {e}", "steps": step}
            break
        belief.after(action.kind, ok, drawer=action.drawer, color=action.color, block=action.block)
        if not ok:
            drawers.forget()  # something went wrong: look again at knobs too
        if (outcome.get("pick") or {}).get("status") == "infeasible":
            belief.give_up(action.block)  # it can't be gripped where it is: trying again won't change that
        idle = 0 if (action.kind in ("transfer", "unload") and ok) else idle + 1
        if idle > 4 * cfg.organizer.n_drawers:
            record.update(outcome=outcome, ok=ok)
            if logger is not None:
                logger.step(record)
            result = {"status": "no progress", "success": False, "steps": step,
                      "reason": f"{idle} steps without moving a block; last: {action}"}
            break
        record.update(outcome=outcome, ok=ok, seconds={**record["seconds"], "act": round(time.time() - t_think, 2)})
        say(f"       -> {'ok' if ok else 'FAILED'}  {_brief(outcome)}")
        if logger is not None:
            logger.step(record)
        scene = remembered(scene, action, ok, outcome)  # None: look again
    result.update(surprises=len(belief.surprises), seconds=round(time.time() - t_start, 1),
                  final_belief=belief.to_json())
    return result


def _brief(outcome: dict) -> str:
    bits = []
    for key in ("drawer", "pick", "place"):
        v = outcome.get(key)
        if isinstance(v, dict) and "status" in v:
            bits.append(f"{key}: {v['status']}")
    return ", ".join(bits)
