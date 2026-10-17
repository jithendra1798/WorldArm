"""Close looks on the way: just before the arm touches something, point the wrist camera straight down at
it from close above and measure it again; after closing a drawer, look at the knobs on the way up.

The overview look sees everything, but from about half a metre and at a slant, so its positions carry that
distance's error (and a block's side faces widen it). A straight-down look from look_mm away, right before
touching, is sharper and catches anything that moved since the overview. The target is matched to the same
thing in the close look by color and distance. When it can't be matched, or the look pose can't be reached,
the skill goes by the overview as before and says so.
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import cv2
import numpy as np

from worldarm.blocks import Block, blocks_on_surface, knob_centers, measure_surface
from worldarm.cell.geometry import down_pose, rot_from_ov
from worldarm.see.frame import Frame
from worldarm.see.scene import block_json

MATCH_BLOCK_MM = 40.0   # the same block in a close look: same color, centre within this of where it was
MATCH_KNOB_MM = 30.0    # the same knob: within this (knobs are 75 mm apart)
_count = itertools.count(1)


def look_pose(target, theta: float, cam_in_gripper, dist_mm: float):
    """Gripper pose (pointing down, wrist turned theta) that puts `target` on the camera's optical axis,
    dist_mm in front of it: the target lands in the middle of the picture."""
    Rgc, tgc = cam_in_gripper
    Rg = rot_from_ov(0.0, 0.0, -1.0, theta)
    p = np.asarray(target, float) - dist_mm * (Rg @ Rgc[:, 2]) - Rg @ tgc
    return down_pose(p[0], p[1], p[2], theta)


async def look_at(rig, target, thetas, *, extra=(), label: str = "close look") -> tuple[Frame | None, float | None, list[str]]:
    """Move so the camera looks straight down at `target` from cfg.camera.look_mm and take one look, trying
    each wrist angle in turn. (None, None, why) when none can be reached, or close looks are off."""
    dist = rig.cfg.camera.look_mm
    if dist <= 0:
        return None, None, ["close looks are off (camera.look_mm = 0)"]
    cam = await rig.camera_in_gripper()
    await rig.set_speed(rig.cfg.speeds.travel_deg_s)
    errors = []
    for th in dict.fromkeys(thetas):
        try:
            await rig.move(look_pose(target, th, cam, dist), extra=extra, label=label)
        except Exception as e:  # out of reach or the planner can't get there: try the other wrist angle
            errors.append(f"wrist {th:.0f}: {str(e)[:160]}")
            continue
        await rig.settle(0.3)  # stop swaying before the picture
        return await rig.look(), th, errors
    return None, None, errors


async def refine_block(rig, block: Block, neighbors, thetas, *, extra=(), say=print) -> tuple[Block, list[Block], dict]:
    """The block and its neighbours as a close look at the block sees them; the overview's if it can't."""
    others = [n for n in neighbors if n is not block]
    top = (float(block.center[0]), float(block.center[1]), block.table_z_mm + block.height_mm)
    frame, used, errors = await look_at(rig, top, thetas, extra=extra, label=f"look at the {block.color} block")
    if frame is None:
        say(f"    close look: can't get there ({'; '.join(errors)[:120]}); going by the overview")
        return block, others, {"status": "no look", "errors": errors}
    surf = measure_surface(frame.color, frame.depth, frame.intr, frame.rot, frame.trans)
    seen, _ = blocks_on_surface(surf, clear_mm=rig.cfg.organizer.knob_keep_out_mm)

    def gap(a: Block, b: Block) -> float:
        return float(np.linalg.norm(np.asarray(a.center[:2]) - np.asarray(b.center[:2])))

    best = min((b for b in seen if b.color == block.color), key=lambda b: gap(b, block), default=None)
    found = best is not None and gap(best, block) <= MATCH_BLOCK_MM
    _save(rig, frame, f"block-{block.color}", [(block.center, (255, 255, 255))] + ([(best.center, (0, 255, 0))] if found else []))
    if not found:
        say(f"    close look: no {block.color} block within {MATCH_BLOCK_MM:.0f} mm; going by the overview")
        return block, others, {"status": "not found", "theta": used, "seen": [block_json(b) for b in seen]}
    # neighbours: everything the close look saw, plus the overview's blocks it didn't (out of its view)
    rest = [b for b in seen if b is not best] + [n for n in others if all(gap(n, s) > MATCH_BLOCK_MM for s in seen)]
    say(f"    close look: {block.color} block {gap(best, block):.0f} mm from where the overview had it, "
        f"{best.short_mm:.0f} x {best.long_mm:.0f} mm (overview: {block.short_mm:.0f} x {block.long_mm:.0f})")
    return best, rest, {"status": "refined", "theta": used, "moved_mm": round(gap(best, block), 1),
                        "before": block_json(block), "after": block_json(best)}


async def knobs_in_view(rig, centre, thetas, expected: dict, *, label: str = "look at the knobs",
                        say=print) -> tuple[dict, dict]:
    """Look straight down at `centre` and find the knobs expected near the given positions:
    ({drawer: knob xyz}, info). Knobs cut off by the edge of the picture don't count."""
    frame, used, errors = await look_at(rig, centre, thetas, label=label)
    if frame is None:
        say(f"    close look: can't get there ({'; '.join(errors)[:120]})")
        return {}, {"status": "no look", "errors": errors}
    surf = measure_surface(frame.color, frame.depth, frame.intr, frame.rot, frame.trans)
    knobs, at_edge = knob_centers(surf, rig.cfg.organizer.knob_diameter_mm, with_edge=True)
    knobs = knobs[~at_edge] if len(knobs) else knobs
    knobs = one_per_knob(knobs)
    found = {}
    for k, e in expected.items():
        if len(knobs):
            d = np.linalg.norm(knobs[:, :2] - np.asarray(e, float)[:2], axis=1)
            j = int(np.argmin(d))
            if d[j] <= MATCH_KNOB_MM:
                found[k] = knobs[j].copy()
    _save(rig, frame, label.replace(" ", "-"), [(e, (255, 255, 255)) for e in expected.values()]
          + [(p, (0, 255, 0)) for p in found.values()])
    off = {str(k): round(float(np.linalg.norm(found[k][:2] - np.asarray(expected[k], float)[:2])), 1) for k in found}
    return found, {"status": "seen" if found else "not found", "theta": used, "off_mm": off,
                   "knobs": {str(k): [round(float(v), 1) for v in p] for k, p in found.items()}}


def one_per_knob(knobs: np.ndarray, within_mm: float = 20.0) -> np.ndarray:
    """One centre per knob: of red patches closer together than a knob, keep the highest. Up close, an open
    drawer's knob shows a second red patch ~14 mm in, on the drawer's glossy top edge just under the ball
    (its reflection, ~7 mm lower); drawer 3 was gripped there once and missed (run 12:22)."""
    keep: list[np.ndarray] = []
    for p in sorted(np.asarray(knobs, float).reshape(-1, 3), key=lambda q: -q[2]):
        if all(float(np.linalg.norm(p[:2] - q[:2])) > within_mm for q in keep):
            keep.append(p)
    return np.array(keep).reshape(-1, 3)


def _save(rig, frame: Frame, name: str, marks) -> None:
    """Keep the close look in the run's folder (looks/NN-<name>.jpg with where things were expected in
    white and found in green, and the raw .npz, replayable with `drill.py scan --frame`)."""
    folder = getattr(rig, "look_dir", None)
    if folder is None:
        return
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{next(_count):02d}-{time.strftime('%H%M%S')}-{name}"
    pic = frame.color.copy()
    for xyz, bgr in marks:
        cv2.circle(pic, frame.to_pixel(xyz), 16, bgr, 2)
    cv2.putText(pic, name, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imwrite(str(folder / f"{stem}.jpg"), pic)
    frame.save(folder / f"{stem}.npz")
