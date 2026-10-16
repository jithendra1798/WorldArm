"""MOVES THE GRIPPER ONLY, never the arm. Teach helper for manual mode.

Put the arm in manual mode (Viam app → CONTROL → arm card → Enter manual mode), guide it by hand,
and use these commands to work the gripper, record poses and save camera captures:

  o          open the gripper
  g          grab (close until contact)
  h          is the gripper holding something?
  p          print arm pose, gripper pose in world, joints, gripper position
  s NAME     save the current pose as NAME in poses/<machine>.json
  l          list saved poses
  c NAME     save color + depth captures to captures/NAME_color.png and NAME_depth.png
  q          quit
"""

import asyncio
import json
import time
from pathlib import Path

import cv2
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.proto.common import Pose, PoseInFrame

from worldarm.connection import connect, env
from worldarm.perception import capture

POSES_DIR = Path("poses")
CAPTURES_DIR = Path("captures")


def pose_dict(p: Pose) -> dict:
    return {k: round(getattr(p, k), 3) for k in ("x", "y", "z", "o_x", "o_y", "o_z", "theta")}


def poses_file() -> Path:
    machine = env("VIAM_MACHINE_ADDRESS").split(".")[0]  # e.g. armfarm3-main
    return POSES_DIR / f"{machine}.json"


async def snapshot(machine, arm: Arm, gripper: Gripper) -> dict:
    grip_world = await machine.transform_pose(
        PoseInFrame(reference_frame=env("VIAM_GRIPPER"), pose=Pose(o_z=1)), "world")
    status = await gripper.is_holding_something()
    return {
        "joints_deg": [round(v, 3) for v in (await arm.get_joint_positions()).values],
        "arm_end": pose_dict(await arm.get_end_position()),
        "gripper_world": pose_dict(grip_world.pose),
        # xArm gripper position from the holding meta: ~840 = fully open, lower = more closed
        "gripper_position": (status.meta or {}).get("position"),
        "holding": status.is_holding_something,
    }


async def main():
    path = poses_file()
    poses = json.loads(path.read_text()) if path.exists() else {}
    async with await connect() as machine:
        arm = Arm.from_robot(machine, env("VIAM_ARM"))
        gripper = Gripper.from_robot(machine, env("VIAM_GRIPPER"))
        cam = Camera.from_robot(machine, env("VIAM_CAMERA"))
        print(__doc__.split("\n\n", 1)[1])
        print(f"Poses file: {path}")

        while True:
            cmd, _, arg = (await asyncio.to_thread(input, "teach> ")).strip().partition(" ")
            arg = arg.strip()
            try:
                if cmd == "o":
                    await gripper.open()
                    print("opened")
                elif cmd == "g":
                    print(f"grabbed: {await gripper.grab()}")
                elif cmd == "h":
                    status = await gripper.is_holding_something()
                    print(f"holding: {status.is_holding_something}  meta: {status.meta}")
                elif cmd == "p":
                    print(json.dumps(await snapshot(machine, arm, gripper), indent=2))
                elif cmd == "s" and arg:
                    if arg in poses and (await asyncio.to_thread(input, f"overwrite '{arg}'? [y/N] ")).lower() != "y":
                        continue
                    poses[arg] = {**await snapshot(machine, arm, gripper),
                                  "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
                    path.parent.mkdir(exist_ok=True)
                    path.write_text(json.dumps(poses, indent=2) + "\n")
                    print(f"saved '{arg}' -> {path}")
                elif cmd == "l":
                    for name, p in poses.items():
                        g = p["gripper_world"]
                        print(f"  {name:<24} gripper xyz=({g['x']:.0f}, {g['y']:.0f}, {g['z']:.0f})  {p['saved_at']}")
                elif cmd == "c" and arg:
                    color, depth = await capture(cam)
                    CAPTURES_DIR.mkdir(exist_ok=True)
                    cv2.imwrite(str(CAPTURES_DIR / f"{arg}_color.png"), color)
                    cv2.imwrite(str(CAPTURES_DIR / f"{arg}_depth.png"), depth)
                    print(f"saved {CAPTURES_DIR}/{arg}_color.png and _depth.png")
                elif cmd == "q":
                    break
                else:
                    print("commands: o, g, h, p, s NAME, l, c NAME, q")
            except Exception as e:  # keep the session alive; a failed call shouldn't drop the teach loop
                print(f"error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
