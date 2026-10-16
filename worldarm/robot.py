"""The real arm, built from reused Viam pieces (docs/viam-config.md).

v0 skills, deliberately simple:
  observe_table  go to `table-view`, read 3D objects from the vision-segment service
  pick           the workshop pattern: approach above the block, descend, grab, lift
  place          go to `drop-k`, open
  open_drawer    `knob-k-pre` → recording grasp-k → grab → recording pull-k → open → back off
  close_drawer   close the fingers → recording push-k (starts at drop-k)

v0 measures only `on_table` (detections), `holding` (gripper sensor) and a failed knob grab.
Not yet validated on hardware; test each skill with someone on the E-stop.
"""

from __future__ import annotations

import asyncio
import time

from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.components.sensor import Sensor
from viam.components.switch import Switch
from viam.proto.common import Pose, PoseInFrame
from viam.robot.client import RobotClient
from viam.services.motion import MotionClient
from viam.services.vision import VisionClient

from .config import Config
from .safety import check_gripper_target
from .schema import Act, Action, Observation


class SkillError(RuntimeError):
    pass


def along_z(pose: Pose, distance_mm: float) -> Pose:
    """The same pose moved `distance_mm` along its own z axis (o_x, o_y, o_z)."""
    n = (pose.o_x ** 2 + pose.o_y ** 2 + pose.o_z ** 2) ** 0.5 or 1.0
    return Pose(x=pose.x + distance_mm * pose.o_x / n, y=pose.y + distance_mm * pose.o_y / n,
                z=pose.z + distance_mm * pose.o_z / n,
                o_x=pose.o_x, o_y=pose.o_y, o_z=pose.o_z, theta=pose.theta)


class ViamRobot:
    def __init__(self, machine: RobotClient, cfg: Config):
        self.machine, self.cfg = machine, cfg
        self.arm = Arm.from_robot(machine, cfg.arm)
        self.gripper = Gripper.from_robot(machine, cfg.gripper)
        self.motion = MotionClient.from_robot(machine, "builtin")
        self.vision = VisionClient.from_robot(machine, cfg.vision)
        self._held: str | None = None

    def truth(self) -> None:
        return None  # the real world doesn't tell us; ask the operator (run_episode --ask-truth)

    async def execute(self, action: Action) -> Observation:
        if action.kind is Act.OBSERVE_TABLE:
            return await self.observe_table()
        if action.kind is Act.PICK:
            return await self.pick(action.color)
        if action.kind is Act.PLACE:
            return await self.place(action.drawer)
        if action.kind is Act.OPEN:
            return await self.open_drawer(action.drawer)
        if action.kind is Act.CLOSE:
            return await self.close_drawer(action.drawer)
        raise ValueError(f"unknown action {action}")

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

    async def detect_blocks(self, move: bool = True) -> list[dict]:
        """Blocks of known colors from the vision-segment service, with world positions."""
        if move:
            await self.go(self.cfg.pose_table_view)
        blocks = []
        for obj in await self.vision.get_object_point_clouds(self.cfg.camera):
            if not obj.geometries.geometries:
                continue
            geo = obj.geometries.geometries[0]
            color = geo.label.rsplit("-", 1)[-1]  # shape-finder labels: "<shape>-<color>"
            if color not in self.cfg.colors:
                continue
            frame = obj.geometries.reference_frame or self.cfg.camera
            world = (await self.machine.transform_pose(PoseInFrame(reference_frame=frame, pose=geo.center),
                                                       "world")).pose
            blocks.append({"color": color, "label": geo.label, "frame": frame, "center": geo.center,
                           "world": world, "points_bytes": len(obj.point_cloud)})
        return blocks

    # --- skills ------------------------------------------------------------------------------

    async def observe_table(self) -> Observation:
        blocks = await self.detect_blocks()
        return Observation(facts={"on_table": sorted({b["color"] for b in blocks})},
                           detail={"blocks": blocks})

    async def pick(self, color: str) -> Observation:
        blocks = await self.detect_blocks()
        seen = sorted({b["color"] for b in blocks})
        candidates = [b for b in blocks if b["color"] == color]
        if not candidates:
            return Observation(pre_facts={"on_table": seen}, facts={"holding": None},
                               detail={"reason": f"{color} not detected"})
        block = max(candidates, key=lambda b: b["points_bytes"])
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
        await asyncio.sleep(self.cfg.settle_s)
        await self.move_gripper(grasp)
        await self.gripper.grab()
        await asyncio.sleep(self.cfg.settle_s)
        await self.move_gripper(approach)  # lift
        held = await self.holding()
        self._held = color if held else None
        return Observation(pre_facts={"on_table": seen}, facts={"holding": self._held},
                           detail={"block": block, "approach": approach, "grasp": grasp})

    async def place(self, k: int) -> Observation:
        await self.go(f"drop-{k}")
        await self.gripper.open()
        await asyncio.sleep(self.cfg.settle_s)
        still_held = await self.holding()
        self._held = self._held if still_held else None
        return Observation(facts={"holding": self._held})

    async def open_drawer(self, k: int) -> Observation:
        await self.go(f"knob-{k}-pre")
        await self.gripper.open()
        await asyncio.sleep(self.cfg.settle_s)
        await self.play(f"grasp-{k}")
        await self.gripper.grab()
        await asyncio.sleep(self.cfg.settle_s)
        if not await self.holding():
            await self.gripper.open()
            await self.back_off()
            return Observation(facts={f"drawer_open:{k}": False},
                               detail={"reason": "fingers did not close on the knob"})
        await self.play(f"pull-{k}")
        await self.gripper.open()
        await asyncio.sleep(self.cfg.settle_s)
        await self.back_off()
        return Observation(detail={"knob_grabbed": True})

    async def close_drawer(self, k: int) -> Observation:
        await self.gripper.grab()  # close the fingers to push with
        await asyncio.sleep(self.cfg.settle_s)
        await self.play(f"push-{k}")
        return Observation(detail={"pushed": True})
