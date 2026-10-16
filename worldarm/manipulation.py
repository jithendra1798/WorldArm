"""Shared arm building blocks for the skills in pick.py, drawers.py and placing.py. Owner: Jithendra.

Saved poses, safety-checked moves, gripper helpers and recording playback. Every computed move
goes through the safety check. Not yet validated on hardware: test with someone on the E-stop.
"""

from __future__ import annotations

import asyncio
import time

from viam.components.gripper import Gripper
from viam.components.sensor import Sensor
from viam.components.switch import Switch
from viam.proto.common import Pose, PoseInFrame
from viam.robot.client import RobotClient
from viam.services.motion import MotionClient

from .config import Config
from .safety import check_gripper_target


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
