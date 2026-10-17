"""The arm, gripper, wrist camera and motion planner of one machine, and the primitives every skill uses.

Everything that touches hardware goes through Rig, so skills run against a fake in tests (tests/fake_rig.py)
and every motion passes the same safety checks: inside the work area, fingertips above the table.
Gripper commands are the drawer service's proven ones ({"set"}, {"get"}, {"grab_with_torque"}).
"""

from __future__ import annotations

import asyncio
import math
import os
import time

import numpy as np
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.proto.common import Geometry, GeometriesInFrame, Pose, PoseInFrame, RectangularPrism, Vector3, WorldState
from viam.proto.component.arm import JointPositions
from viam.proto.service.motion import Constraints, LinearConstraint
from viam.services.motion import MotionClient

from worldarm import connection
from worldarm.act.watch import ContactWatch, WatchResult
from worldarm.cell.calibration import CameraCorrection
from worldarm.cell.config import CellConfig
from worldarm.cell.geometry import pose_dict, pose_matrix
from worldarm.frames import frame_to_world
from worldarm.perception import capture
from worldarm.see.frame import Frame, Intrinsics


class MotionRefused(RuntimeError):
    """A target outside the safe work area: never sent to the arm."""


def box_geometry(label: str, corners_xy: np.ndarray, z_bottom: float, z_top: float) -> Geometry:
    """A rectangular obstacle from its four footprint corners (any rotation about z)."""
    c = np.asarray(corners_xy, float)
    e1, e2 = c[1] - c[0], c[2] - c[1]
    centre = c.mean(axis=0)
    return Geometry(label=label,
                    center=Pose(x=float(centre[0]), y=float(centre[1]), z=(z_bottom + z_top) / 2.0, o_z=1.0,
                                theta=math.degrees(math.atan2(e1[1], e1[0]))),
                    box=RectangularPrism(dims_mm=Vector3(x=float(np.linalg.norm(e1)), y=float(np.linalg.norm(e2)),
                                                         z=float(z_top - z_bottom))))


class Rig:
    def __init__(self, machine, cfg: CellConfig, arm: str, gripper: str, camera: str, motion: str = "builtin"):
        self.machine, self.cfg = machine, cfg
        self.arm_name, self.gripper_name, self.camera_name = arm, gripper, camera
        self.arm = Arm.from_robot(machine, arm)
        self.gripper = Gripper.from_robot(machine, gripper)
        self.camera = Camera.from_robot(machine, camera)
        self.motion = MotionClient.from_robot(machine, motion)
        self.correction = CameraCorrection.from_json(cfg.camera.correction)
        self._intr: Intrinsics | None = None
        self._cam_in_gripper: tuple[np.ndarray, np.ndarray] | None = None
        self.look_dir = None  # where close looks are saved (sort.py and drill.py set it to the run's folder)

    @classmethod
    async def connect(cls, cfg: CellConfig) -> Rig:
        machine = await connection.connect()
        rig = cls(machine, cfg, connection.env("VIAM_ARM"), connection.env("VIAM_GRIPPER"),
                  connection.env("VIAM_CAMERA"), os.getenv("VIAM_MOTION", "builtin"))
        if cfg.speeds.accel_deg_s2:
            try:  # gentle ramps at the start and end of every move; a driver without it just ignores this
                await rig.arm.do_command({"set_acceleration": float(cfg.speeds.accel_deg_s2)})
            except Exception:
                pass
        return rig

    async def close(self) -> None:
        await self.machine.close()

    # ---- seeing ----------------------------------------------------------------------------------
    async def look(self) -> Frame:
        """Read-only: one aligned color + depth frame and where the camera was (corrected)."""
        color, depth = await capture(self.camera)
        if self._intr is None:
            self._intr = Intrinsics.from_viam((await self.camera.get_properties()).intrinsic_parameters)
        R, t = await frame_to_world(self.machine, self.camera_name)
        Rc, tc = self.correction.apply(R, t)
        pose = await self.gripper_pose()
        return Frame(color, depth, self._intr, Rc, tc, R, t, pose_dict(pose), time.time())

    async def camera_in_gripper(self) -> tuple[np.ndarray, np.ndarray]:
        """Where the wrist camera sits on the gripper: (R, t) with p_gripper = R @ p_camera + t, corrected.
        The mount is rigid, so it's read once from the frame system. Measured: 72 mm to the side of the
        gripper axis and 87 mm above the gripper frame, looking along it within 2 degrees."""
        if self._cam_in_gripper is None:
            R, t = await frame_to_world(self.machine, self.camera_name)
            Rc, tc = self.correction.apply(R, t)
            Rg, tg = pose_matrix(await self.gripper_pose())
            self._cam_in_gripper = (Rg.T @ Rc, Rg.T @ (tc - tg))
        return self._cam_in_gripper

    # ---- where things are ------------------------------------------------------------------------
    async def gripper_pose(self) -> Pose:
        return (await self.machine.get_pose(self.gripper_name, destination_frame="world")).pose

    async def frame_z(self) -> float | None:
        """Gripper frame height, fast (flange - offset; valid while pointing down, as all our moves do)."""
        try:
            return float((await self.arm.get_end_position()).z) - self.cfg.gripper.frame_below_flange_mm
        except Exception:
            return None

    async def joints(self) -> list[float]:
        return list((await self.arm.get_joint_positions()).values)

    async def loads(self) -> list[float] | None:
        try:
            return [float(v) for v in (await self.arm.do_command({"load": ""})).get("load")]
        except Exception:
            return None

    # ---- moving ----------------------------------------------------------------------------------
    def check(self, pose: Pose, label: str, floor_z: float | None = None) -> None:
        """Refuse targets outside the work area or with the fingertips into the table. floor_z overrides the
        lowest allowed gripper frame height (only the fingertip-length drill, which touches the table on purpose)."""
        w = self.cfg.workcell
        if not (w.x_min_mm <= pose.x <= w.x_max_mm and w.y_min_mm <= pose.y <= w.y_max_mm and pose.z <= w.z_max_mm):
            raise MotionRefused(f"{label}: ({pose.x:.0f}, {pose.y:.0f}, {pose.z:.0f}) is outside the work area")
        floor = floor_z if floor_z is not None else w.table_z_mm + self.cfg.gripper.tip_below_frame_mm + w.min_tip_above_table_mm
        if pose.o_z < -0.5 and pose.z < floor:
            raise MotionRefused(f"{label}: gripper frame z {pose.z:.0f} would put the fingertips below "
                                f"{w.min_tip_above_table_mm:.0f} mm above the table (floor {floor:.0f})")

    def world_state(self, extra: list[Geometry] | tuple = ()) -> WorldState:
        w = self.cfg.workcell
        top = w.obstacle_table_top_z_mm
        geoms = [
            Geometry(label="wa-table", center=Pose(x=0, y=0, z=top - 100.0, o_z=1),
                     box=RectangularPrism(dims_mm=Vector3(x=3000, y=3000, z=200))),
            Geometry(label="wa-wall-front", center=Pose(x=w.wall_front_x_mm, y=0, z=300, o_z=1),
                     box=RectangularPrism(dims_mm=Vector3(x=100, y=3000, z=1600))),
            Geometry(label="wa-wall-side", center=Pose(x=0, y=w.wall_side_y_mm, z=300, o_z=1),
                     box=RectangularPrism(dims_mm=Vector3(x=3000, y=100, z=1600))),
            Geometry(label="wa-ceiling", center=Pose(x=0, y=0, z=w.ceiling_z_mm, o_z=1),
                     box=RectangularPrism(dims_mm=Vector3(x=3000, y=3000, z=100))),
            *extra,
        ]
        return WorldState(obstacles=[GeometriesInFrame(reference_frame="world", geometries=geoms)])

    async def set_speed(self, deg_s: float) -> None:
        try:
            await self.arm.do_command({"set_speed": float(deg_s)})
        except Exception:
            pass

    async def move(self, pose: Pose, *, linear: bool = False, extra: list[Geometry] | tuple = (),
                   label: str = "move", timeout: float = 60.0, floor_z: float | None = None) -> None:
        """Gripper to `pose` (world) with the motion planner; linear=True keeps it on a straight line."""
        self.check(pose, label, floor_z)
        cons = (Constraints(linear_constraint=[LinearConstraint(line_tolerance_mm=3.0, orientation_tolerance_degs=2.0)])
                if linear else None)
        try:
            ok = await asyncio.wait_for(
                self.motion.move(component_name=self.gripper_name,
                                 destination=PoseInFrame(reference_frame="world", pose=pose),
                                 world_state=self.world_state(extra), constraints=cons, timeout=timeout),
                timeout + 5.0)
        except asyncio.TimeoutError:
            await self.stop()  # a cancelled call doesn't stop the arm; this does
            raise RuntimeError(f"{label}: no answer from the motion service in {timeout + 5:.0f} s; arm stopped")
        if ok is False:
            raise RuntimeError(f"{label}: the motion service reported failure")

    async def move_watched(self, pose: Pose, watch: ContactWatch, *, linear: bool = True,
                           extra: list[Geometry] | tuple = (), label: str = "watched move",
                           floor_z: float | None = None) -> WatchResult:
        """Move while reading joint loads; stop the arm the moment `watch` confirms contact."""
        self.check(pose, label, floor_z)
        loop = asyncio.get_running_loop()
        task = asyncio.ensure_future(self.move(pose, linear=linear, extra=extra, label=label, floor_z=floor_z))
        t0 = loop.time()
        while not task.done():
            await asyncio.sleep(self.cfg.contact.poll_s)
            loads, z = await self.loads(), await self.frame_z()
            hit = watch.feed(loop.time(), z, loads)
            if hit is not None and not task.done():
                await self.stop()
                task.cancel()
                try:
                    await task
                except BaseException:
                    pass
                try:
                    await self.arm.do_command({"clear_error": True})
                except Exception:
                    pass
                return hit
        await task  # re-raise a planning error
        return WatchResult(False, await self.frame_z(), 0.0, watch.worst, None, loop.time() - t0)

    async def move_joints(self, values: list[float]) -> None:
        await self.arm.move_to_joint_positions(JointPositions(values=list(values)))

    async def stop(self) -> None:
        try:
            await self.arm.stop()
        except Exception:
            pass

    async def settle(self, seconds: float) -> None:
        """Wait for the arm to stop ringing or swaying. The simulated rig in tests doesn't wait."""
        await asyncio.sleep(seconds)

    # ---- gripper ---------------------------------------------------------------------------------
    async def open_mm(self, mm: float) -> None:
        await self.gripper.do_command({"set": self.cfg.gripper.counts(mm)})
        await asyncio.sleep(self.cfg.gripper.settle_s)

    async def width_mm(self) -> float | None:
        try:
            return self.cfg.gripper.mm(float((await self.gripper.do_command({"get": True})).get("pos")))
        except Exception:
            return None

    async def grab(self, torque_pct: int, speed: int | None = None, position_mm: float | None = None) -> None:
        """Close with limited torque toward position_mm (default: fully shut)."""
        g = self.cfg.gripper
        speed = speed or g.grip_speed
        target = g.counts(position_mm) if position_mm is not None else 0
        start = await self.width_mm()
        t0 = time.time()
        await self.gripper.do_command({"grab_with_torque": {"position": target, "speed": speed, "torque": torque_pct}})
        # wait until the fingers have closed and stopped (on the object), not a fixed time; at most as long
        # as the slowest close takes. A fixed 1.5 s left the arm standing still holding the knob.
        deadline = t0 + g.settle_s * max(1.0, g.grip_speed / speed)
        last = None
        while time.time() < deadline:
            await asyncio.sleep(0.1)
            w = await self.width_mm()
            moved = start is None or w is None or w < start - 2.0 or time.time() - t0 > 0.6
            if w is not None and last is not None and abs(w - last) < 0.3 and moved:
                return
            last = w

    async def holding(self) -> bool | None:
        try:
            return bool((await self.gripper.is_holding_something()).is_holding_something)
        except Exception:
            return None
