import asyncio

import numpy as np
from viam.proto.common import Pose, PoseInFrame
from viam.robot.client import RobotClient


async def frame_to_world(machine: RobotClient, frame: str) -> tuple[np.ndarray, np.ndarray]:
    """Rotation R (3x3) and translation t (3,) such that p_world = R @ p_frame + t, in mm.

    Built by asking Viam to transform the frame's origin and unit axes, so it uses Viam's own
    frame-system math and calibration rather than re-implementing orientation-vector conventions.
    """
    probes = [(0, 0, 0), (100, 0, 0), (0, 100, 0), (0, 0, 100)]
    results = await asyncio.gather(*[
        machine.transform_pose(PoseInFrame(reference_frame=frame, pose=Pose(x=x, y=y, z=z, o_z=1)), "world")
        for x, y, z in probes
    ])
    pts = np.array([[r.pose.x, r.pose.y, r.pose.z] for r in results])
    t = pts[0]
    R = ((pts[1:] - t) / 100.0).T
    return R, t


def to_world(R: np.ndarray, t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ R.T + t
