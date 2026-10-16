"""Read-only: print camera intrinsics and the frame system, and save one color + depth capture. Moves nothing."""

import asyncio
from pathlib import Path

import cv2
import numpy as np
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.proto.common import Pose, PoseInFrame

from worldarm.connection import connect, env
from worldarm.perception import capture

OUT = Path("captures")


def fmt_pose(p: Pose) -> str:
    return (f"xyz=({p.x:.1f}, {p.y:.1f}, {p.z:.1f}) mm  "
            f"ov=({p.o_x:.3f}, {p.o_y:.3f}, {p.o_z:.3f}) theta={p.theta:.1f}")


async def main():
    OUT.mkdir(exist_ok=True)
    async with await connect() as machine:
        cam = Camera.from_robot(machine, env("VIAM_CAMERA"))
        arm = Arm.from_robot(machine, env("VIAM_ARM"))

        props = await cam.get_properties()
        print("Camera intrinsics:", props.intrinsic_parameters)
        print("Distortion:", props.distortion_parameters)

        print("\nFrame system:")
        for part in await machine.get_frame_system_config():
            f = part.frame
            geo = f.physical_object
            geo_desc = geo.WhichOneof("geometry_type") if geo.ByteSize() else "none"
            print(f"  {f.reference_frame:<12} parent={f.pose_in_observer_frame.reference_frame:<10} "
                  f"{fmt_pose(f.pose_in_observer_frame.pose)}  geometry={geo_desc}")
            if geo.ByteSize():
                print(f"    {geo}".replace("\n", " "))

        cam_in_world = await machine.transform_pose(
            PoseInFrame(reference_frame=env("VIAM_CAMERA"), pose=Pose(o_z=1)), "world")
        print("\nCamera pose in world:  ", fmt_pose(cam_in_world.pose))
        print("Arm end pose in world: ", fmt_pose(await arm.get_end_position()))

        color, depth = await capture(cam)
        cv2.imwrite(str(OUT / "color.jpg"), color)
        cv2.imwrite(str(OUT / "depth.png"), depth)  # 16-bit, mm
        valid = depth[depth > 0]
        vis = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=255.0 / max(int(valid.max()), 1)), cv2.COLORMAP_TURBO)
        vis[depth == 0] = 0
        cv2.imwrite(str(OUT / "depth_vis.png"), vis)

        h, w = depth.shape
        center = depth[h // 2 - 5:h // 2 + 5, w // 2 - 5:w // 2 + 5]
        print(f"\nColor: {color.shape[1]}x{color.shape[0]}   Depth: {w}x{h}")
        print(f"Depth valid: {valid.size / depth.size:.0%}  range {valid.min()}-{valid.max()} mm  "
              f"center ~{int(np.median(center[center > 0])) if (center > 0).any() else 'n/a'} mm")
        print(f"Saved to {OUT.resolve()}/")


if __name__ == "__main__":
    asyncio.run(main())
