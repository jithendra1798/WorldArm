"""MOVES THE ARM. Single-arm sort on armfarm03 with the team's skills, benchmarked and logged.

For each block in the source tape square, in drawer order:
  open k      drawer service (PickBottle): grip the knob, pull 12 cm, let go, remember the drawer
  transfer    Waj's pick-and-drop (pick_place_cubes.move_one_object) into the open drawer
  close k     drawer service
At the end of each episode you say where every block really is, so success is judged on the truth.

Setup on armfarm03 (once):
  1. Deploy modules/drawer_service.py to the inline module behind PickBottle (open/close need it).
  2. One blue-tape square in the wrist camera's overview, with the blocks in it.
     Default colors: green -> drawer 1, yellow -> 2, black -> 3 (change with WORLDARM_ASSIGNMENT).
     Waj's detector names green, yellow, orange, red and black. Blue is the tape; red is the knobs.
  3. Put the arm at an overview pose that sees both squares, then:  python sort_bench.py --teach-view

  python sort_bench.py --preview         # moves to the overview pose, scans, prints the plan
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
    "VIAM_TAPE_ZONES": "1",                # one source square; the drawers are the destination
    # Black by brightness alone: on armfarm03 the black block measured V=10 but S=162 (dark pixels
    # have noisy saturation), so the default "S < 70" test called it green. Green measured V=59.
    "VIAM_BLACK_MAX_SATURATION": "256",
    "VIAM_BLACK_MAX_VALUE": "35",
    "VIAM_WALL_SIDE_Y_MM": "500",           # armfarm03's side wall (verify_camera.py)
    "VIAM_WORKSPACE_Y_MIN_MM": "-700",      # armfarm03 works at negative y
    "VIAM_WORKSPACE_Y_MAX_MM": "440",       # stay clear of the side wall at y = 450-550
    # Motion tuning: the same values as armfarm05's (.env.arm2)
    "VIAM_APPROACH_CLEARANCE_MM": "120", "VIAM_LIFT_MM": "120", "VIAM_APPROACH_SPEED_DEGS_PER_SEC": "32",
    "VIAM_DESCENT_STEP_MM": "25", "VIAM_GRASP_TCP_Z_OFFSET_MM": "45", "VIAM_MIN_TCP_ABOVE_TABLE_MM": "50",
    "VIAM_GRASP_OX": "0", "VIAM_GRASP_OY": "0", "VIAM_GRASP_OZ": "-1", "VIAM_GRASP_THETA_DEG": "0",
}
for key, value in ARMFARM03_DEFAULTS.items():
    os.environ.setdefault(key, value)

import cv2  # noqa: E402
import pick_place_cubes as ppc  # noqa: E402  (needs the settings above)
from viam.components.arm import Arm  # noqa: E402
from viam.components.camera import Camera  # noqa: E402
from viam.components.gripper import Gripper  # noqa: E402
from viam.proto.common import Pose, PoseInFrame  # noqa: E402
from viam.services.generic import Generic  # noqa: E402
from viam.services.motion import MotionClient  # noqa: E402

from worldarm.config import parse_assignment  # noqa: E402
from worldarm.grasp import block_grasp_theta, finger_axis_deg, place_theta_for_drawer  # noqa: E402
from worldarm.logger import EpisodeLogger  # noqa: E402
from worldarm.metrics import rate  # noqa: E402


def log_step(logger, n, kind, ok, t0, detail, color=None, drawer=None, note=""):
    logger.step({
        "step": n, "t_start": t0, "duration_s": round(time.time() - t0, 3),
        "action": {"kind": kind, "color": color, "drawer": drawer},
        "observation": {"pre_facts": {}, "facts": {"success": ok}, "detail": detail},
        "surprises": [] if ok else [{"key": kind, "expected": True, "observed": False, "note": note}],
        "ignored_surprises": [],
    })


def describe(scene, assignment: dict[str, int]) -> str:
    """One line about what a scan saw: the square and each block's color."""
    if not scene.zones:
        return "no blue square in view"
    objs = [o for z in scene.zones for o in z.objects]
    if not objs:
        return "the square is empty"
    names = ", ".join(o.color_name if o.color_name in assignment else f"{o.color_name} (not sorted)" for o in objs)
    hint = ""
    if all(o.color_name == "unknown" for o in objs):
        hint = ("  <- every color unknown: the parser is reading color beside the blocks. Is the wrist "
                "camera's depth aligned to color (\"align_color_depth\": true on cam)?")
    return f"{len(objs)} block(s) in the square: {names}{hint}"


def draw(scene, path: Path) -> None:
    """Save the scan with the square outlined and each block labeled."""
    vis = scene.color.copy()
    for z in scene.zones:
        cv2.polylines(vis, [z.corners], True, (0, 0, 255), 3)
        for o in z.objects:
            cv2.circle(vis, o.pixel, 14, (0, 255, 255), 3)
            cv2.putText(vis, o.color_name, (o.pixel[0] + 16, o.pixel[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 0, 255), 2)
    cv2.imwrite(str(path), vis)


def ask_truth(assignment: dict[str, int]) -> dict:
    print("\nGround truth (look at the drawers and the table):")
    blocks = {}
    for color in assignment:
        a = input(f"  where is {color}? [1-3 = drawer, t = table, ? = don't know] ").strip()
        blocks[color] = f"drawer:{a}" if a.isdigit() else "table" if a == "t" else "unknown"
    closed = input("  are all drawers fully closed? [y/n] ").strip().lower() == "y"
    return {"blocks": blocks, "all_closed": closed}


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

    @property
    def ov(self) -> tuple[float, float, float]:
        return (self.cfg["VIAM_GRASP_OX"], self.cfg["VIAM_GRASP_OY"], self.cfg["VIAM_GRASP_OZ"])

    async def grasp_for(self, scene, obj) -> tuple[float, dict]:
        """Wrist theta that closes the fingers across the block's short side (worldarm/grasp.py)."""
        async def to_world(u, v, z):
            p = await ppc.point_to_world(self.m, self.camera_name, scene.intrinsics, u, v, z)
            return p.x, p.y
        return await block_grasp_theta(obj.point_mask, scene.depth, to_world,
                                       self.cfg["VIAM_GRASP_THETA_DEG"], self.ov)

    async def finger_line_px(self, scene, obj, theta: float, half_mm: float = 30.0):
        """The finger axis for `theta` through the block's center, projected back into the image."""
        a, c, k = math.radians(finger_axis_deg(theta, self.ov)), obj.center_world, scene.intrinsics
        ends = []
        for s in (-half_mm, half_mm):
            world = Pose(x=c.x + s * math.cos(a), y=c.y + s * math.sin(a), z=c.z, o_z=1)
            p = (await self.m.transform_pose(PoseInFrame(reference_frame="world", pose=world), self.camera_name)).pose
            ends.append((int(round(k.focal_x_px * p.x / p.z + k.center_x_px)),
                         int(round(k.focal_y_px * p.y / p.z + k.center_y_px))))
        return ends

    async def scan(self, assignment: dict[str, int], save_to: Path | None = None):
        """Overview pose -> scene. Returns (scene, source square) or (scene, None) if it's empty."""
        await ppc.open_gripper_for_pick(self.gripper)
        await ppc.return_to_view_pose(self.m, self.arm, self.motion, self.gripper_name, self.world,
                                      self.margin, self.timeout)
        await ppc.with_timeout(self.arm.do_command({"set_speed": self.cfg["VIAM_APPROACH_SPEED_DEGS_PER_SEC"]}),
                               10.0, "set approach speed")
        scene = await ppc.capture_scene(self.m, self.camera, self.camera_name)
        print(f"  saw: {describe(scene, assignment)}")
        if save_to is not None:
            draw(scene, save_to)
        source = max(scene.zones, key=lambda z: len(z.objects))
        return scene, (source if source.objects else None)

    async def episode(self, assignment: dict[str, int], logger) -> tuple[str, str, int]:
        n, done, looks = 0, set(), 0
        while True:
            looks += 1
            scene, source = await self.scan(assignment, Path(logger.dir) / f"look-{looks}.jpg")
            todo = [o for o in (source.objects if source else [])
                    if o.color_name in assignment and o.color_name not in done]
            if not todo:
                why = "sorted " + ", ".join(sorted(done)) if done else f"nothing to sort ({describe(scene, assignment)})"
                return "done", why, n
            obj = min(todo, key=lambda o: assignment[o.color_name])
            color, k = obj.color_name, assignment[obj.color_name]
            done.add(color)  # one attempt per block per episode: a failure is a failure
            print(f"\n{color} -> drawer {k}")

            n, t0 = n + 1, time.time()
            reply = dict(await self.drawers.do_command({"action": "open", "drawer": k,
                                                         "refine": self.args.refine_knobs}))
            ok = bool(reply.get("success"))
            print(f"  open {k}: {'ok' if ok else reply.get('error') or reply.get('stage')}")
            log_step(logger, n, "open", ok, t0, reply, drawer=k, note=str(reply.get("error", "")))
            if not ok:
                return "aborted", f"open {k} failed: {reply.get('error') or reply.get('stage')}", n

            n, t0 = n + 1, time.time()
            dp = reply["drop_point"]
            target = Pose(x=dp["x"], y=dp["y"], z=dp["z"] + self.args.drop_extra_mm, o_z=-1)
            neighbors = [o for o in source.objects if o is not obj]
            obstacles = list({o.pixel: o for o in neighbors}.values())
            open_dir = tuple(reply["open_dir"])
            pick_theta, grasp = await self.grasp_for(scene, obj)
            # A parallel gripper grips the same turned 180 deg; the planner may reject one of the two
            # (the wrist camera's box can collide with the wrist link), so keep the other as a fallback.
            alt = pick_theta - 180.0 if pick_theta > 0 else pick_theta + 180.0
            wrists = [(pick_theta, place_theta_for_drawer(open_dir, pick_theta, self.ov)),
                      (alt, place_theta_for_drawer(open_dir, alt, self.ov))]
            print(f"  grasp {color}: {grasp.get('shape')}, short side {grasp.get('short_mm', '-')} mm along "
                  f"{grasp.get('short_axis_deg', '-')} deg -> wrist {pick_theta:.0f} deg (or {alt:.0f}); "
                  f"at the drawer {wrists[0][1]:.0f} deg")
            result, used = "not tried", None
            for pick_t, place_t in wrists:
                try:
                    result = await ppc.move_one_object(obj, target, neighbors, obstacles, self.arm, self.gripper,
                                                       self.motion, self.gripper_name, self.world, self.cfg,
                                                       self.margin, self.timeout,
                                                       pick_theta_deg=pick_t, place_theta_deg=place_t)
                    used = (pick_t, place_t)
                    break
                except Exception as e:  # e.g. no motion plan: a failed transfer, but still close the drawer below
                    result = f"error: {type(e).__name__}: {str(e)[:160]}"
                    held = (await self.gripper.is_holding_something()).is_holding_something
                    planning = any(w in str(e) for w in ("IK", "constraint", "no solution", "plan"))
                    if planning and not held and (pick_t, place_t) != wrists[-1]:
                        print(f"  wrist {pick_t:.0f} deg can't be planned; trying {wrists[-1][0]:.0f} deg")
                        continue
                    try:
                        await ppc.open_gripper_for_pick(self.gripper)
                    except Exception:
                        pass
                    break
            pick_theta, place_theta = used or wrists[0]
            print(f"  transfer {color}: {result}")
            log_step(logger, n, "transfer", result == "ok", t0,
                     {"result": result, "drop_target": dp, "grasp": grasp, "pick_theta": pick_theta,
                      "place_theta": place_theta}, color=color, drawer=k, note=result)

            n, t0 = n + 1, time.time()
            reply = dict(await self.drawers.do_command({"action": "close", "drawer": k}))
            # Pushed back to the closed grasp point while holding the knob = closed. The service's load
            # comparison is across two separate grips and too noisy to judge by (logged as info).
            ok = bool(reply.get("success"))
            loads = (reply.get("closed") or {}).get("load_diff_vs_before")
            print(f"  close {k}: {'ok' if ok else reply.get('error') or reply.get('stage')}"
                  + (f"  (load diff {loads})" if loads is not None else ""))
            log_step(logger, n, "close", ok, t0, reply, drawer=k, note=str(reply.get("error", "")))
            if not reply.get("success"):
                return "aborted", f"close {k} failed: {reply.get('error') or reply.get('stage')}", n


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--preview", action="store_true", help="scan from the overview pose and print the plan")
    p.add_argument("--teach-view", action="store_true", help="save the arm's current pose as the overview pose")
    p.add_argument("--drawer-service", default="PickBottle")
    p.add_argument("--drop-extra-mm", type=float, default=0.0,
                   help="raise the drop target above the knob height (release is 120 mm above the target)")
    p.add_argument("--refine-knobs", action="store_true",
                   help="use the drawer service's self-correcting two-look knob check (about 25 s slower per "
                        "drawer). Default: one overhead look + the stored correction, which two refined runs "
                        "on the aligned camera confirmed to within 1.5 mm")
    p.add_argument("--yes", action="store_true", help="skip the motion confirmation")
    args = p.parse_args()

    machine_name = connection.env("VIAM_MACHINE_ADDRESS").split(".")[0]
    if "armfarm3" not in machine_name:
        raise SystemExit(f"sort_bench.py carries armfarm03's settings; this env file points at {machine_name}.")
    assignment = parse_assignment(os.getenv("WORLDARM_ASSIGNMENT", "green:1,yellow:2,black:3"))

    async with await connection.connect() as machine:
        if args.teach_view:  # read-only: checks the view, then saves where the arm is now
            gripper_name = connection.env("VIAM_GRIPPER")
            pose = (await machine.get_pose(gripper_name, destination_frame="world")).pose
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, -pose.o_z))))
            if tilt > 25:
                raise SystemExit(f"Not saved: the gripper is tilted {tilt:.0f} deg from straight down. Point the "
                                 "wrist camera straight down at the blue square, about 40 cm above it.")
            camera_name = connection.env("VIAM_CAMERA")
            try:
                scene = await ppc.capture_scene(machine, Camera.from_robot(machine, camera_name), camera_name)
            except RuntimeError as e:
                raise SystemExit(f"Not saved: the wrist camera doesn't see a usable blue square here ({e}).")
            Path("captures").mkdir(exist_ok=True)
            draw(scene, Path("captures") / "teach_view.jpg")
            print(f"Sees: {describe(scene, assignment)}   (picture: captures/teach_view.jpg)")
            ppc.save_view_pose(pose)
            print(f"Saved the overview pose to {ppc.VIEW_POSE_PATH}: {ppc.fmt_pose(pose)}")
            return

        bench = Bench(machine, args)
        status = dict(await bench.drawers.do_command({"action": "status"}))  # never moves
        if "error" in status:
            raise SystemExit(f"{args.drawer_service} has no open/close yet: deploy modules/drawer_service.py "
                             "to its inline module first.")
        if ppc.load_view_pose() is None:
            raise SystemExit("No overview pose for armfarm03 yet: put the arm there, then run --teach-view.")
        print(f"Sorting on {machine_name}: " + ", ".join(f"{c} -> drawer {k}" for c, k in assignment.items()))
        if not args.yes and input("This MOVES the arm. Someone at the E-stop? Type 'go': ").strip() != "go":
            raise SystemExit("Cancelled.")
        for k in sorted(status.get("open_drawers") or {}, key=int):  # left open by an interrupted run
            answer = ""
            while answer not in ("o", "c", "q"):
                answer = input(f"Drawer {k} was left open by an earlier run. Is it open right now? "
                               "[o = yes, close it with the arm / c = no, it's closed / q = quit] ").strip().lower()
            if answer == "q":
                raise SystemExit("Stopped.")
            if answer == "c":
                await bench.drawers.do_command({"action": "forget"})
                print(f"  forgot drawer {k}")
                break
            reply = dict(await bench.drawers.do_command({"action": "close", "drawer": int(k)}))
            if not reply.get("success"):
                raise SystemExit(f"Couldn't close drawer {k}: {reply.get('error') or reply.get('stage')}. "
                                 "Close it by hand, then answer c next time.")
            print(f"  closed drawer {k}")

        if args.preview:
            picture = Path("captures") / "preview.jpg"
            scene, source = await bench.scan(assignment, picture)
            vis = cv2.imread(str(picture))
            for o in (source.objects if source else []):
                plan = f"drawer {assignment[o.color_name]}" if o.color_name in assignment else "not sorted"
                theta, grasp = await bench.grasp_for(scene, o)
                p0, p1 = await bench.finger_line_px(scene, o, theta)
                cv2.line(vis, p0, p1, (0, 255, 0), 4)
                print(f"  {o.color_name:<7} {grasp.get('shape')}, short side {grasp.get('short_mm', '-')} mm "
                      f"-> wrist {theta:.0f} deg -> {plan}")
            cv2.imwrite(str(picture), vis)
            print(f"  picture: {picture}  (green line = where the fingers will close; it should cross the "
                  "block's short side)")
            return

        results = []
        for ep in range(1, args.episodes + 1):
            input(f"\nEpisode {ep}/{args.episodes}: blocks in the source square, drawers closed, then Enter ")
            logger = EpisodeLogger("runs", "sort-team", {
                "mode": "real", "goal": "sort-team", "blind": False, "machine": machine_name,
                "assignment": assignment, "args": vars(args), "settings": ARMFARM03_DEFAULTS,
            })
            t0 = time.time()
            try:
                status_, reason, steps = await bench.episode(assignment, logger)
            except Exception as e:  # stop safely, keep the log
                status_, reason, steps = "aborted", f"{type(e).__name__}: {e}", 0
                try:
                    await bench.arm.stop()
                    await ppc.open_gripper_for_pick(bench.gripper)
                except Exception:
                    pass
            truth = ask_truth(assignment)
            ok = truth["all_closed"] and all(truth["blocks"][c] == f"drawer:{k}" for c, k in assignment.items())
            logger.finish({"status": status_, "reason": reason, "steps": steps, "surprises": None,
                           "duration_s": round(time.time() - t0, 2), "belief_success": None,
                           "truth_success": ok, "belief_accuracy": None, "truth_final": truth})
            results.append(ok)
            print(f"Episode {ep}: {status_} ({reason}); truly sorted: {ok}; {time.time() - t0:.0f} s")
        print(f"\nTruly sorted: {rate(sum(results), len(results))}   (python scoreboard.py for per-step numbers)")


if __name__ == "__main__":
    asyncio.run(main())
