"""MOVES THE ARM. Single-arm sort on armfarm03: blocks anywhere on the table go into the drawers by color.

For each block, in drawer order:
  open k     drawer service (PickBottle): grip the knob, pull 12 cm, let go. A drawer that's already open
             is reused; another open drawer is closed first; a remembered one that's already shut is forgotten.
  transfer   Waj's pick-and-drop (pick_place_cubes.move_one_object): wrist turned so the fingers close across
             the block's short side, a contact-checked descent that stops on touch instead of pushing into
             the table, and the block turned at the drawer so its long side runs along the drawer
  check      look into the open drawer for the block's color
  close k    drawer service
At the end it looks at the table again. Success, judged by the cameras: every block seen in its drawer,
none left on the table, every drawer closed. (--ask-truth asks you instead.)

Blocks are found by height above the table and named by color (worldarm/blocks.py): no tape or mat.
Default colors: green -> drawer 1, yellow -> 2, black -> 3 (WORLDARM_ASSIGNMENT=blue:1,yellow:2,green:3 etc.).

  python sort_bench.py --teach-view      # save the overview pose: camera straight down at the table
  python sort_bench.py --preview         # look, print the plan, draw where the fingers will close
  python sort_bench.py --episodes 5      # benchmark; python scoreboard.py summarizes it
"""

import argparse
import asyncio
import math
import os
import time
from pathlib import Path

from worldarm import connection  # loads .env first, so its values win over the defaults below

# Waj's code defaults to armfarm05; these are armfarm03's values. Set before importing it.
ARMFARM03_DEFAULTS = {
    "VIAM_VIEW_POSE_FILE": ".pick_place_view_pose.armfarm03.json",
    "VIAM_WALL_SIDE_Y_MM": "500",           # armfarm03's side wall (verify_camera.py)
    "VIAM_WORKSPACE_Y_MIN_MM": "-700",      # armfarm03 works at negative y
    "VIAM_WORKSPACE_Y_MAX_MM": "440",       # stay clear of the side wall at y = 450-550
    "VIAM_TABLE_TOP_Z_MM": "2",             # armfarm03's real table top (its config says -23, like armfarm05)
    "VIAM_MIN_TCP_ABOVE_TABLE_MM": "65",    # gripper frame >= 65 mm above the table, so the fingertips stay clear
    "VIAM_DEPTH_PX_TO_MM": "1.0",           # our detector gives footprint area in mm^2 directly
    # Motion tuning: the same values as armfarm05's (.env.arm2)
    "VIAM_APPROACH_CLEARANCE_MM": "120", "VIAM_LIFT_MM": "120", "VIAM_APPROACH_SPEED_DEGS_PER_SEC": "32",
    "VIAM_DESCENT_STEP_MM": "25", "VIAM_GRASP_TCP_Z_OFFSET_MM": "45",
    "VIAM_GRASP_OX": "0", "VIAM_GRASP_OY": "0", "VIAM_GRASP_OZ": "-1", "VIAM_GRASP_THETA_DEG": "0",
}
for key, value in ARMFARM03_DEFAULTS.items():
    os.environ.setdefault(key, value)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pick_place_cubes as ppc  # noqa: E402  (needs the settings above)
from viam.components.arm import Arm  # noqa: E402
from viam.components.camera import Camera  # noqa: E402
from viam.components.gripper import Gripper  # noqa: E402
from viam.proto.common import Pose, PoseInFrame  # noqa: E402
from viam.services.generic import Generic  # noqa: E402
from viam.services.motion import MotionClient  # noqa: E402

from modules.drawer_service import DROP_FROM_FACE_MM, GRASP_OUTWARD_MM, KNOB_PROTRUSION_MM, TRAVEL_MM  # noqa: E402
from worldarm.blocks import color_fraction, find_blocks  # noqa: E402
from worldarm.config import parse_assignment  # noqa: E402
from worldarm.contact import ContactGuard  # noqa: E402
from worldarm.frames import frame_to_world  # noqa: E402
from worldarm.grasp import finger_axis_deg, place_theta_for_drawer, theta_for_finger_axis  # noqa: E402
from worldarm.logger import EpisodeLogger  # noqa: E402
from worldarm.metrics import rate  # noqa: E402
from worldarm.perception import capture  # noqa: E402


class Frame:
    """One look through the wrist camera: images, camera pose, and the blocks found."""

    def __init__(self, color, depth, intr, rot, trans, blocks, table):
        self.color, self.depth, self.intr, self.rot, self.trans = color, depth, intr, rot, trans
        self.blocks, self.table = blocks, table

    def to_pixel(self, x: float, y: float, z: float) -> tuple[int, int]:
        cam = self.rot.T @ (np.array([x, y, z]) - self.trans)
        return (int(round(self.intr.focal_x_px * cam[0] / cam[2] + self.intr.center_x_px)),
                int(round(self.intr.focal_y_px * cam[1] / cam[2] + self.intr.center_y_px)))


def describe(blocks, assignment: dict[str, int]) -> str:
    if not blocks:
        return "no blocks on the table"
    names = ", ".join(b.color if b.color in assignment else f"{b.color} (not sorted)" for b in blocks)
    return f"{len(blocks)} block(s): {names}"


def drawer_info(st: dict) -> dict:
    """Drop point, pull direction and wrist of an open drawer, from the drawer service's status entry."""
    rx, ry = st["open_grasp"]
    pull = st["pull"]
    knob = (rx - pull[0] * GRASP_OUTWARD_MM, ry - pull[1] * GRASP_OUTWARD_MM)
    back = TRAVEL_MM + KNOB_PROTRUSION_MM - DROP_FROM_FACE_MM
    return {"drop_point": {"x": knob[0] - pull[0] * back, "y": knob[1] - pull[1] * back, "z": st["knob_z"]},
            "open_dir": pull, "wrist": st.get("wrist")}


def log_step(logger, n, kind, ok, t0, detail, color=None, drawer=None, note=""):
    logger.step({
        "step": n, "t_start": t0, "duration_s": round(time.time() - t0, 3),
        "action": {"kind": kind, "color": color, "drawer": drawer},
        "observation": {"pre_facts": {}, "facts": {"success": ok}, "detail": detail},
        "surprises": [] if ok else [{"key": kind, "expected": True, "observed": False, "note": note}],
        "ignored_surprises": [],
    })


def ask_truth(assignment: dict[str, int]) -> dict:
    print("\nGround truth (look at the drawers and the table):")
    blocks = {}
    for color in assignment:
        a = input(f"  where is {color}? [1-3 = drawer, t = table, ? = don't know] ").strip()
        blocks[color] = f"drawer:{a}" if a.isdigit() else "table" if a == "t" else "unknown"
    closed = input("  are all drawers fully closed? [y/n] ").strip().lower() == "y"
    return {"blocks": blocks, "all_closed": closed, "source": "operator"}


class Bench:
    def __init__(self, machine, args):
        self.m, self.args = machine, args
        self.camera_name = connection.env("VIAM_CAMERA")
        self.gripper_name = connection.env("VIAM_GRIPPER")
        self.camera = Camera.from_robot(machine, self.camera_name)
        self.arm = Arm.from_robot(machine, connection.env("VIAM_ARM"))
        self.gripper = Gripper.from_robot(machine, self.gripper_name)
        self.motion = MotionClient.from_robot(machine, os.getenv("VIAM_MOTION", "builtin"))
        self.drawers = Generic.from_robot(machine, args.drawer_service)
        self.timeout = ppc.setting("VIAM_MOVE_TIMEOUT_SEC", 60.0)
        self.cfg = ppc.required_motion_settings()
        self.margin = self.cfg["VIAM_MIN_TCP_ABOVE_TABLE_MM"]
        self.world = ppc.workcell_world_state(include_ceiling=True,
                                              table_keep_out_mm=ppc.setting("VIAM_TABLE_KEEP_OUT_MM", 0.0))
        self.intr = None

    @property
    def ov(self) -> tuple[float, float, float]:
        return (self.cfg["VIAM_GRASP_OX"], self.cfg["VIAM_GRASP_OY"], self.cfg["VIAM_GRASP_OZ"])

    # ---- seeing -----------------------------------------------------------------------------------

    async def look(self, find: bool = True) -> Frame:
        """Read-only: one aligned color + depth frame, the camera pose, and (if `find`) the blocks."""
        color, depth = await capture(self.camera)
        if self.intr is None:
            self.intr = (await self.camera.get_properties()).intrinsic_parameters
        rot, trans = await frame_to_world(self.m, self.camera_name)
        blocks, table = find_blocks(color, depth, self.intr, rot, trans) if find else ([], None)
        return Frame(color, depth, self.intr, rot, trans, blocks, table)

    def wrist_for(self, block) -> tuple[float, float]:
        """Wrist theta that closes the fingers across the block's short side, and the same grip turned 180."""
        theta = theta_for_finger_axis(block.short_axis_deg, self.cfg["VIAM_GRASP_THETA_DEG"], self.ov)
        return theta, (theta - 180.0 if theta > 0 else theta + 180.0)

    def draw(self, frame: Frame, path: Path, assignment: dict[str, int]) -> None:
        """Save the frame with each block outlined, named, and a green line where the fingers will close."""
        vis = frame.color.copy()
        for b in frame.blocks:
            contours, _ = cv2.findContours(b.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (0, 0, 255), 2)
            a = math.radians(finger_axis_deg(self.wrist_for(b)[0], self.ov))
            ends = [frame.to_pixel(b.center[0] + s * math.cos(a), b.center[1] + s * math.sin(a), b.center[2])
                    for s in (-30.0, 30.0)]
            cv2.line(vis, ends[0], ends[1], (0, 255, 0), 4)
            label = b.color if b.color in assignment else f"{b.color}?"
            cv2.putText(vis, label, (b.pixel[0] + 18, b.pixel[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), vis)

    async def scan(self, assignment: dict[str, int], save_to: Path | None = None) -> Frame:
        """Go to the overview pose and look."""
        await ppc.open_gripper_for_pick(self.gripper)
        await ppc.return_to_view_pose(self.m, self.arm, self.motion, self.gripper_name, self.world,
                                      self.margin, self.timeout)
        await ppc.with_timeout(self.arm.do_command({"set_speed": self.cfg["VIAM_APPROACH_SPEED_DEGS_PER_SEC"]}),
                               10.0, "set approach speed")
        frame = await self.look()
        print(f"  saw: {describe(frame.blocks, assignment)}")
        if save_to is not None:
            self.draw(frame, save_to, assignment)
        return frame

    # ---- drawers ----------------------------------------------------------------------------------

    async def open_drawers(self) -> dict:
        return dict(await self.drawers.do_command({"action": "status"})).get("open_drawers") or {}

    async def ensure_open(self, k: int) -> tuple[bool, dict, str]:
        """Drawer k open, with its drop point. Reuses it if it's open already; closes another open
        drawer first; forgets a remembered drawer that turns out to be shut (nothing to grip)."""
        open_now = await self.open_drawers()
        if str(k) in open_now:
            return True, drawer_info(open_now[str(k)]), "already open, reused"
        for other in list(open_now):
            r = dict(await self.drawers.do_command({"action": "close", "drawer": int(other)}))
            if r.get("success"):
                print(f"  closed drawer {other} first (it was left open)")
            elif r.get("stage") in ("grip", "hover"):
                await self.drawers.do_command({"action": "forget"})
                print(f"  drawer {other} was remembered open but is already shut: forgot it")
            else:
                return False, r, f"couldn't close drawer {other}: {r.get('error') or r.get('stage')}"
        r = dict(await self.drawers.do_command({"action": "open", "drawer": k, "refine": self.args.refine_knobs}))
        if not r.get("success"):
            return False, r, str(r.get("error") or r.get("stage") or "open failed")
        st = (await self.open_drawers()).get(str(k))
        return True, ({**r, **drawer_info(st)} if st else r), "opened"

    async def close(self, k: int) -> dict:
        return dict(await self.drawers.do_command({"action": "close", "drawer": k}))

    # ---- picking ----------------------------------------------------------------------------------

    async def read_loads(self):
        try:
            return [float(v) for v in (await self.arm.do_command({"load": ""})).get("load")]
        except Exception:
            return None

    def detected(self, b) -> "ppc.DetectedObject":
        """Waj's object type for one of our blocks (area in mm^2, since VIAM_DEPTH_PX_TO_MM is 1)."""
        return ppc.DetectedObject(pixel=b.pixel, bbox=cv2.boundingRect(b.mask), area_px=int(round(b.short_mm * b.long_mm)),
                                  center_world=Pose(x=float(b.center[0]), y=float(b.center[1]), z=float(b.center[2])),
                                  height_mm=b.height_mm, point_mask=b.mask, color_name=b.color)

    async def transfer(self, frame: Frame, block, info: dict) -> tuple[str, dict]:
        obj = self.detected(block)
        others = [self.detected(b) for b in frame.blocks if b is not block]
        dp = info["drop_point"]
        target = Pose(x=dp["x"], y=dp["y"], z=dp["z"] + self.args.drop_extra_mm, o_z=-1)
        theta, alt = self.wrist_for(block)
        # At the drawer: block's long side along the pull, wrist as close as possible to the drawer's own wrist
        # (so closing doesn't need a big wrist turn).
        near = info.get("wrist")
        guard, result, used = ContactGuard(self.read_loads), "not tried", None
        for pick_t in (theta, alt):
            place_t = place_theta_for_drawer(tuple(info["open_dir"]), near if near is not None else pick_t, self.ov)
            guard = ContactGuard(self.read_loads)
            try:
                result = await ppc.move_one_object(obj, target, others, others, self.arm, self.gripper, self.motion,
                                                   self.gripper_name, self.world, self.cfg, self.margin, self.timeout,
                                                   pick_theta_deg=pick_t, place_theta_deg=place_t, descent_guard=guard)
                used = (pick_t, place_t)
                break
            except Exception as e:  # e.g. no motion plan: a failed transfer, the drawer still gets closed
                result = f"error: {type(e).__name__}: {str(e)[:160]}"
                held = (await self.gripper.is_holding_something()).is_holding_something
                planning = any(w in str(e) for w in ("IK", "constraint", "no solution", "plan"))
                if planning and not held and pick_t == theta:
                    print(f"  wrist {theta:.0f} deg can't be planned; trying {alt:.0f} deg")
                    continue
                try:
                    await ppc.open_gripper_for_pick(self.gripper)
                except Exception:
                    pass
                break
        return result, {
            "result": result, "pick_theta": used[0] if used else theta, "place_theta": used[1] if used else None,
            "block": {"color": block.color, "short_mm": round(block.short_mm, 1), "long_mm": round(block.long_mm, 1),
                      "height_mm": round(block.height_mm, 1), "short_axis_deg": round(block.short_axis_deg, 1),
                      "center": [round(float(v), 1) for v in block.center]},
            "drop_target": dp, "contact_z": guard.contact_z, "contact_worst": round(guard.worst, 2),
        }

    async def check_in_drawer(self, color: str, info: dict) -> tuple[bool, float]:
        """After the drop: back the gripper up 120 mm and look for the block's color where it was dropped."""
        await self.motion.move(component_name=self.gripper_name,
                               destination=PoseInFrame(reference_frame=self.gripper_name, pose=Pose(z=-120.0, o_z=1)))
        frame = await self.look(find=False)
        dp = info["drop_point"]
        frac = color_fraction(frame.color, frame.to_pixel(dp["x"], dp["y"], dp["z"]), 70, color)
        return frac >= 0.08, frac

    # ---- one episode ------------------------------------------------------------------------------

    async def episode(self, assignment: dict[str, int], logger) -> tuple[str, str, int, dict]:
        n, looks, attempts, where, closes_ok = 0, 0, {}, {}, True
        status, reason = "done", ""
        while True:
            looks += 1
            frame = await self.scan(assignment, Path(logger.dir) / f"look-{looks}.jpg")
            todo = [b for b in frame.blocks if b.color in assignment
                    and where.get(b.color) != f"drawer:{assignment[b.color]}"
                    and attempts.get(b.color, 0) <= self.args.retries]
            if not todo:
                break
            block = min(todo, key=lambda b: (assignment[b.color], b.long_mm))
            color, k = block.color, assignment[block.color]
            attempts[color] = attempts.get(color, 0) + 1
            print(f"\n{color} -> drawer {k}  (try {attempts[color]})")

            n, t0 = n + 1, time.time()
            ok, info, note = await self.ensure_open(k)
            print(f"  open {k}: {note}")
            log_step(logger, n, "open", ok, t0, info, drawer=k, note="" if ok else note)
            if not ok:
                status, reason = "aborted", f"open {k}: {note}"
                break

            n, t0 = n + 1, time.time()
            theta, alt = self.wrist_for(block)
            print(f"  grip: {block.short_mm:.0f} x {block.long_mm:.0f} mm, {block.height_mm:.0f} mm tall; "
                  f"short side along {block.short_axis_deg:.0f} deg -> wrist {theta:.0f} deg (or {alt:.0f})")
            result, detail = await self.transfer(frame, block, info)
            touched = f"  (touched at z={detail['contact_z']:.0f} and stopped)" if detail["contact_z"] else ""
            print(f"  transfer {color}: {result}{touched}")
            log_step(logger, n, "transfer", result == "ok", t0, detail, color=color, drawer=k, note=result)

            if result == "ok":
                n, t0 = n + 1, time.time()
                seen, frac = await self.check_in_drawer(color, info)
                where[color] = f"drawer:{k}" if seen else "unknown"
                print(f"  check drawer {k}: {color} {'seen' if seen else 'NOT seen'} ({frac:.0%} of the window)")
                log_step(logger, n, "check", seen, t0, {"fraction": round(frac, 3)}, color=color, drawer=k)

            n, t0 = n + 1, time.time()
            r = await self.close(k)
            ok = bool(r.get("success"))
            closes_ok &= ok
            loads = (r.get("closed") or {}).get("load_diff_vs_before")
            print(f"  close {k}: {'ok' if ok else r.get('error') or r.get('stage')}"
                  + (f"  (load diff {loads})" if loads is not None else ""))
            log_step(logger, n, "close", ok, t0, r, drawer=k, note=str(r.get("error", "")))
            if not ok:
                status, reason = "aborted", f"close {k}: {r.get('error') or r.get('stage')}"
                break

        if status == "done":  # anything of a sorted color still on the table didn't make it
            looks += 1
            final = await self.scan(assignment, Path(logger.dir) / f"look-{looks}-final.jpg")
            for b in final.blocks:
                if b.color in assignment:
                    where[b.color] = "table"
            reason = ", ".join(f"{c}: {where.get(c, 'not seen')}" for c in assignment)
        truth = {"blocks": {c: where.get(c, "unknown") for c in assignment},
                 "all_closed": closes_ok and not await self.open_drawers(), "source": "cameras"}
        return status, reason, n, truth


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--preview", action="store_true", help="look from the overview pose and print the plan")
    p.add_argument("--teach-view", action="store_true", help="save the arm's current pose as the overview pose")
    p.add_argument("--drawer-service", default="PickBottle")
    p.add_argument("--drop-extra-mm", type=float, default=0.0,
                   help="raise the drop target above the knob height (release is 120 mm above the target)")
    p.add_argument("--refine-knobs", action="store_true",
                   help="drawer service's self-correcting two-look knob check (about 25 s slower per drawer)")
    p.add_argument("--retries", type=int, default=1, help="extra tries per block after a missed or dropped pick")
    p.add_argument("--ask-truth", action="store_true", help="ask where the blocks ended up instead of looking")
    p.add_argument("--yes", action="store_true", help="skip the motion confirmation")
    args = p.parse_args()

    machine_name = connection.env("VIAM_MACHINE_ADDRESS").split(".")[0]
    if "armfarm3" not in machine_name:
        raise SystemExit(f"sort_bench.py carries armfarm03's settings; this env file points at {machine_name}.")
    assignment = parse_assignment(os.getenv("WORLDARM_ASSIGNMENT", "green:1,yellow:2,black:3"))

    async with await connection.connect() as machine:
        bench = Bench(machine, args)
        if args.teach_view:  # read-only: checks the view, then saves where the arm is now
            pose = (await machine.get_pose(bench.gripper_name, destination_frame="world")).pose
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, -pose.o_z))))
            if tilt > 25:
                raise SystemExit(f"Not saved: the gripper is tilted {tilt:.0f} deg from straight down. Point the "
                                 "wrist camera straight down at the table, about 40 cm above the blocks.")
            try:
                frame = await bench.look()
            except RuntimeError as e:
                raise SystemExit(f"Not saved: the wrist camera can't find the table here ({e}).")
            bench.draw(frame, Path("captures") / "teach_view.jpg", assignment)
            print(f"Sees: {describe(frame.blocks, assignment)}   (picture: captures/teach_view.jpg)")
            ppc.save_view_pose(pose)
            print(f"Saved the overview pose to {ppc.VIEW_POSE_PATH}: {ppc.fmt_pose(pose)}")
            return

        status = dict(await bench.drawers.do_command({"action": "status"}))  # never moves
        if "error" in status:
            raise SystemExit(f"{args.drawer_service} has no open/close yet: deploy modules/drawer_service.py "
                             "to its inline module first.")
        if ppc.load_view_pose() is None:
            raise SystemExit("No overview pose for armfarm03 yet: put the arm there, then run --teach-view.")
        print(f"Sorting on {machine_name}: " + ", ".join(f"{c} -> drawer {k}" for c, k in assignment.items()))
        if not args.yes and input("This MOVES the arm. Someone at the E-stop? Type 'go': ").strip() != "go":
            raise SystemExit("Cancelled.")

        if args.preview:
            picture = Path("captures") / "preview.jpg"
            frame = await bench.scan(assignment, picture)
            for b in frame.blocks:
                plan = f"drawer {assignment[b.color]}" if b.color in assignment else "not sorted"
                theta, alt = bench.wrist_for(b)
                print(f"  {b.color:<7} {b.short_mm:.0f} x {b.long_mm:.0f} mm, {b.height_mm:.0f} mm tall at "
                      f"({b.center[0]:.0f}, {b.center[1]:.0f}) -> wrist {theta:.0f} deg (or {alt:.0f}) -> {plan}")
            print(f"  picture: {picture}  (green line = where the fingers will close: across the short side)")
            return

        results = []
        for ep in range(1, args.episodes + 1):
            input(f"\nEpisode {ep}/{args.episodes}: blocks on the table, drawers closed, then Enter ")
            logger = EpisodeLogger("runs", "sort-team", {
                "mode": "real", "goal": "sort-team", "blind": False, "machine": machine_name,
                "assignment": assignment, "args": vars(args), "settings": ARMFARM03_DEFAULTS,
            })
            t0 = time.time()
            try:
                status_, reason, steps, truth = await bench.episode(assignment, logger)
            except Exception as e:  # stop safely, keep the log
                status_, reason, steps, truth = "aborted", f"{type(e).__name__}: {e}", 0, None
                try:
                    await bench.arm.stop()
                    await ppc.open_gripper_for_pick(bench.gripper)
                except Exception:
                    pass
            if args.ask_truth or truth is None:
                truth = ask_truth(assignment)
            ok = bool(truth["all_closed"]) and all(truth["blocks"].get(c) == f"drawer:{k}" for c, k in assignment.items())
            logger.finish({"status": status_, "reason": reason, "steps": steps, "surprises": None,
                           "duration_s": round(time.time() - t0, 2), "belief_success": None,
                           "truth_success": ok, "belief_accuracy": None, "truth_final": truth})
            results.append(ok)
            print(f"\nEpisode {ep}: {status_} ({reason}); sorted: {ok} (judged by {truth['source']}); "
                  f"{time.time() - t0:.0f} s")
        print(f"\nSorted: {rate(sum(results), len(results))}   (python scoreboard.py for per-step numbers)")


if __name__ == "__main__":
    asyncio.run(main())
