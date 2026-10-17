"""The sorting loop: look -> update the belief -> decide -> act -> look again.

Every iteration starts with a fresh overview look, so every decision is made on what is there now (and
that look also checks the previous action). Each step is logged to runs/<stamp>-sort/: the annotated
picture (step-NN.jpg), the raw frame (step-NN.npz, replayable offline), and a JSON record of the scene,
belief, decision, outcome and timings.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np

from worldarm.act.grasp_plan import plan_grasp
from worldarm.act.pick import pick
from worldarm.act.place import UNREACHABLE, drawer_target, place, table_target
from worldarm.cell.geometry import to_pose
from worldarm.cell.rig import box_geometry
from worldarm.see.draw import draw_scene
from worldarm.see.frame import Frame
from worldarm.see.scene import Scene, free_spot, observe
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
        plan = plan_grasp(block, cfg, scene.table_blocks)
        held = await pick(rig, plan, travel_obstacles=obs, say=say)
        out = {"pick": held.to_json()}
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


async def run_task(rig, drawers, logger=None, *, max_steps: int | None = None, dry_run: bool = False,
                   say=print) -> dict:
    cfg = rig.cfg
    belief = SortBelief(cfg)
    max_steps = max_steps or cfg.task.max_steps
    t_start = time.time()
    result = {"status": "max steps", "success": False, "steps": 0, "surprises": 0}
    idle = 0  # drawer moves since a block last moved: a loop the rules didn't foresee shouldn't run forever
    for step in range(1, max_steps + 1):
        t0 = time.time()
        frame = await overview(rig)
        t_look = time.time()
        scene = observe(frame, cfg, belief.row_model)
        surprises = belief.observe(scene, step)
        action = decide(belief)
        t_think = time.time()
        for s in surprises:
            say(f"    SURPRISE {s}")
        say(f"{step:>3}  {action}")
        record = {"step": step, "action": action.to_json(), "scene": scene.to_json(), "belief": belief.to_json(),
                  "surprises": [str(s) for s in surprises],
                  "seconds": {"look": round(t_look - t0, 2), "think": round(t_think - t_look, 2)}}
        if logger is not None:
            pic = draw_scene(frame, scene, cfg, f"step {step}: {action}")
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
