"""Read-only: check that camera pixels map to sensible world coordinates. Moves nothing.

- Compares the local camera->world transform against Viam's transform_pose on a few pixels.
- Fits a plane to the table to check the camera calibration's tilt and the table height.
"""

import asyncio

import numpy as np
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.proto.common import Pose, PoseInFrame

from connection import connect, env
from frames import frame_to_world, to_world
from perception import capture, deproject, pixel_to_cam

OBSTACLES = ["table", "wall-front", "wall-side", "ceiling"]


async def main():
    async with await connect() as machine:
        cam_name = env("VIAM_CAMERA")
        cam = Camera.from_robot(machine, cam_name)
        intr = (await cam.get_properties()).intrinsic_parameters
        color, depth = await capture(cam)
        R, t = await frame_to_world(machine, cam_name)

        print(f"det(R)={np.linalg.det(R):.4f} (should be 1.0)   camera at {np.round(t, 1)} mm")

        # 1. Local transform vs Viam's transform_pose on a few pixels
        h, w = depth.shape
        errs = []
        for u, v in [(w // 2, h // 2), (w // 4, h // 2), (3 * w // 4, h // 3), (w // 2, 2 * h // 3)]:
            p = pixel_to_cam(u, v, depth, intr)
            if p is None:
                continue
            local = to_world(R, t, p[None])[0]
            rpc = (await machine.transform_pose(
                PoseInFrame(reference_frame=cam_name, pose=Pose(x=p[0], y=p[1], z=p[2], o_z=1)), "world")).pose
            errs.append(np.linalg.norm(local - [rpc.x, rpc.y, rpc.z]))
            print(f"  pixel ({u:4d},{v:3d}) -> world {np.round(local, 1)} mm")
        print(f"Local vs Viam transform max error: {max(errs):.3f} mm")

        # 2. Table plane fit
        pts_cam, _ = deproject(depth, intr, step=4)
        pts = to_world(R, t, pts_cam)
        z = pts[:, 2]
        hist, edges = np.histogram(z, bins=np.arange(z.min(), z.max() + 5, 5))
        peak = edges[np.argmax(hist)] + 2.5
        table = pts[np.abs(z - peak) < 15]
        A = np.c_[table[:, 0], table[:, 1], np.ones(len(table))]
        (a, b, c), *_ = np.linalg.lstsq(A, table[:, 2], rcond=None)
        tilt = np.degrees(np.arctan(np.hypot(a, b)))
        resid = table[:, 2] - A @ [a, b, c]
        print(f"\nTable plane: z = {a:.4f}x + {b:.4f}y + {c:.1f}   tilt {tilt:.2f} deg   "
              f"residual std {resid.std():.1f} mm   ({len(table)} pts)")
        print(f"Visible table area (world): x {table[:, 0].min():.0f}..{table[:, 0].max():.0f}  "
              f"y {table[:, 1].min():.0f}..{table[:, 1].max():.0f} mm")

        # 3. Obstacle geometries configured on the machine
        print("\nObstacle geometries:")
        for name in OBSTACLES:
            for g in await Gripper.from_robot(machine, name).get_geometries():
                print(f"  {name:<11} {str(g).replace(chr(10), ' ')}")


if __name__ == "__main__":
    asyncio.run(main())
