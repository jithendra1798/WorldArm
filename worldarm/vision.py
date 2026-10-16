"""Block perception on the real arm. Owner: perception track.

v0 reads 3D objects from the vision-segment service (shape-finder → detections-to-segments), so
it only sees squares and rectangles. Improve or replace `detect_blocks`, but keep its return
format so the rest of the system doesn't change.
"""

from __future__ import annotations

from viam.proto.common import PoseInFrame
from viam.robot.client import RobotClient
from viam.services.vision import VisionClient

from .config import Config


async def detect_blocks(machine: RobotClient, cfg: Config, vision: VisionClient) -> list[dict]:
    """Blocks of known colors seen from the arm's current pose.

    Each block is a dict with: color; label; frame (the frame `center` is in); center (Pose);
    world (Pose in the world frame); points_bytes (point-cloud size, a proxy for point count).
    """
    blocks = []
    for obj in await vision.get_object_point_clouds(cfg.camera):
        if not obj.geometries.geometries:
            continue
        geo = obj.geometries.geometries[0]
        color = geo.label.rsplit("-", 1)[-1]  # shape-finder labels: "<shape>-<color>"
        if color not in cfg.colors:
            continue
        frame = obj.geometries.reference_frame or cfg.camera
        world = (await machine.transform_pose(PoseInFrame(reference_frame=frame, pose=geo.center),
                                              "world")).pose
        blocks.append({"color": color, "label": geo.label, "frame": frame, "center": geo.center,
                       "world": world, "points_bytes": len(obj.point_cloud)})
    return blocks
