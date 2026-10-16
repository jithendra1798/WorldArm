"""Pick up a detected block. Owner: Waj (find and pick up any block).

v0 is the pick pattern from Viam's pick-and-place workshop: approach above the block center,
descend, grab, lift. It ignores the block's shape and angle; making it work for every shape is
this track's goal.
"""

from __future__ import annotations

from viam.proto.common import Pose, PoseInFrame

from .manipulation import Manipulator, along_z
from .safety import check_gripper_target
from .schema import Observation


async def pick_block(m: Manipulator, block: dict) -> Observation:
    """Pick a block from `vision.detect_blocks` while the arm is still at the pose it was seen from."""
    cfg = m.cfg
    # Approach above the block center in the camera frame (camera z points at the table), then
    # descend until the fingertips reach the block center.
    c = block["center"]
    approach_cam = Pose(x=c.x, y=c.y, z=c.z - cfg.approach_mm,
                        o_x=c.o_x, o_y=c.o_y, o_z=c.o_z, theta=c.theta)
    approach = (await m.machine.transform_pose(
        PoseInFrame(reference_frame=block["frame"], pose=approach_cam), "world")).pose
    grasp = along_z(approach, cfg.approach_mm - cfg.limits.fingertip_mm)
    check_gripper_target(grasp, cfg.limits)  # check both before moving at all

    await m.move_gripper(approach)
    await m.gripper.open()
    await m.settle()
    await m.move_gripper(grasp)
    await m.gripper.grab()
    await m.settle()
    await m.move_gripper(approach)  # lift
    m.held = block["color"] if await m.holding() else None
    return Observation(facts={"holding": m.held},
                       detail={"block": block, "approach": approach, "grasp": grasp})
