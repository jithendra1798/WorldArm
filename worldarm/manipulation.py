"""Arm skills on the real arm.

v0, built on reused Viam pieces (docs/viam-config.md):
  pick_block    the workshop pattern: approach above the block, descend, grab, lift
  place         go to `drop-k`, open
  open_drawer   `knob-k-pre` → recording grasp-k → grab → recording pull-k → open → back off
  close_drawer  close the fingers → recording push-k (starts at drop-k)

Each skill returns an Observation with the facts it measured. Every computed move goes through
the safety check. Not yet validated on hardware: test with someone on the E-stop.
"""

from __future__ import annotations

import asyncio
import time

from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.components.sensor import Sensor
from viam.components.switch import Switch
from viam.proto.common import Pose, PoseInFrame
from viam.robot.client import RobotClient
from viam.services.motion import MotionClient

from .config import Config
from .drawer_check import colors_in
from .perception import capture
from .safety import check_gripper_target
from .schema import Observation


class SkillError(RuntimeError):
    pass


def along_z(pose: Pose, distance_mm: float) -> Pose:
    """The same pose moved `distance_mm` along its own z axis (o_x, o_y, o_z)."""
    n = (pose.o_x ** 2 + pose.o_y ** 2 + pose.o_z ** 2) ** 0.5 or 1.0
    return Pose(x=pose.x + distance_mm * pose.o_x / n, y=pose.y + distance_mm * pose.o_y / n,
                z=pose.z + distance_mm * pose.o_z / n,
                o_x=pose.o_x, o_y=pose.o_y, o_z=pose.o_z, theta=pose.theta)


class Manipulator:
    def __init__(self, machine: RobotClient, cfg: Config):
        self.machine, self.cfg = machine, cfg
        self.gripper = Gripper.from_robot(machine, cfg.gripper)
        self.motion = MotionClient.from_robot(machine, "builtin")
        self.held: str | None = None  # color we believe is in the gripper (the sensor can't tell colors)

    # --- building blocks ---------------------------------------------------------------------

    async def go(self, pose_name: str) -> None:
        """Replay a saved arm-position-saver pose ("go to" = position 2)."""
        await Switch.from_robot(self.machine, pose_name).set_position(2)

    async def move_gripper(self, world_pose: Pose) -> None:
        check_gripper_target(world_pose, self.cfg.limits)
        await self.motion.move(component_name=self.cfg.gripper,
                               destination=PoseInFrame(reference_frame="world", pose=world_pose))

    async def back_off(self, mm: float = 60.0) -> None:
        """Move the gripper straight back along its own z axis."""
        await self.motion.move(component_name=self.cfg.gripper,
                               destination=PoseInFrame(reference_frame=self.cfg.gripper,
                                                       pose=Pose(z=-mm, o_z=1)))

    async def holding(self) -> bool:
        return (await self.gripper.is_holding_something()).is_holding_something

    async def settle(self) -> None:
        await asyncio.sleep(self.cfg.settle_s)

    async def play(self, session: str, timeout_s: float = 60.0) -> None:
        """Replay an arm-recorder session and wait for it to finish."""
        recorder = Sensor.from_robot(self.machine, self.cfg.recorder)
        await recorder.do_command({"command": "play", "session": session})
        t0, started = time.monotonic(), False
        while True:
            await asyncio.sleep(0.25)
            readings = await recorder.get_readings()
            playing = readings.get("state") == "playing"
            started |= playing
            elapsed = time.monotonic() - t0
            if not playing and (started or elapsed > 2.0):  # finished (or finished before our first poll)
                break
            if elapsed > timeout_s:
                raise SkillError(f"playback {session!r} timed out after {timeout_s:.0f} s")
        if readings.get("last_error"):
            raise SkillError(f"playback {session!r}: {readings['last_error']}")

    # --- skills ------------------------------------------------------------------------------

    async def pick_block(self, block: dict) -> Observation:
        """Pick a block from `vision.detect_blocks` while the arm is still at the pose it was seen from."""
        # Approach above the block center in the camera frame (camera z points at the table), then
        # descend until the fingertips reach the block center.
        c = block["center"]
        approach_cam = Pose(x=c.x, y=c.y, z=c.z - self.cfg.approach_mm,
                            o_x=c.o_x, o_y=c.o_y, o_z=c.o_z, theta=c.theta)
        approach = (await self.machine.transform_pose(
            PoseInFrame(reference_frame=block["frame"], pose=approach_cam), "world")).pose
        grasp = along_z(approach, self.cfg.approach_mm - self.cfg.limits.fingertip_mm)
        check_gripper_target(grasp, self.cfg.limits)  # check both before moving at all

        await self.move_gripper(approach)
        await self.gripper.open()
        await self.settle()
        await self.move_gripper(grasp)
        await self.gripper.grab()
        await self.settle()
        await self.move_gripper(approach)  # lift
        self.held = block["color"] if await self.holding() else None
        return Observation(facts={"holding": self.held},
                           detail={"block": block, "approach": approach, "grasp": grasp})

    async def place(self, k: int) -> Observation:
        await self.go(f"drop-{k}")
        await self.gripper.open()
        await self.settle()
        if not await self.holding():
            self.held = None
        return Observation(facts={"holding": self.held})

    async def look_into(self, k: int) -> Observation:
        """MOVES THE ARM. Look into container k from above and report the block colors inside."""
        await self.go(f"drop-{k}")        # go above the open drawer
        await self.back_off(150)          # back up ~15 cm so the camera sees the whole drawer
        cam = Camera.from_robot(self.machine, self.cfg.camera)
        color, _ = await capture(cam)     # take a color photo (depth not needed)
        seen = colors_in(color)           # your detector: e.g. ["blue"]
        return Observation(facts={f"contents:{k}": seen})

    async def open_drawer(self, k: int) -> Observation:
        await self.go(f"knob-{k}-pre")
        await self.gripper.open()
        await self.settle()
        await self.play(f"grasp-{k}")
        await self.gripper.grab()
        await self.settle()
        if not await self.holding():
            await self.gripper.open()
            await self.back_off()
            return Observation(facts={f"drawer_open:{k}": False},
                               detail={"reason": "fingers did not close on the knob"})
        await self.play(f"pull-{k}")
        await self.gripper.open()
        await self.settle()
        await self.back_off()
        return Observation(detail={"knob_grabbed": True})

    async def close_drawer(self, k: int) -> Observation:
        await self.gripper.grab()  # close the fingers to push with
        await self.settle()
        await self.play(f"push-{k}")
        return Observation(detail={"pushed": True})