"""Viam Generic service to move objects between two blue-tape zones.

DoCommand actions:
  {"action": "preview"}                         Read-only RGB-D scene analysis.
  {"action": "move", "confirm": true}           Motion; also requires enable_motion=true.

Service creation and preview never move hardware. Motion begins only after an
explicit move command and a second opt-in in the machine configuration.
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any, ClassVar, Mapping, Optional, Sequence, Tuple

from typing_extensions import Self
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Pose, ResourceName
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic
from viam.services.motion import Motion
from viam.utils import ValueTypes, struct_to_dict

from models.transfer import (
    TABLE_TOP_Z_MM,
    assert_in_workspace,
    capture_scene,
    drop_slot_pixels,
    fmt_pose,
    move_gripper,
    move_one_object,
    number,
    surface_down_orientation,
    workspace_bounds,
    with_timeout,
    workcell_world_state,
    zone_surface_pose,
)

logger = logging.getLogger("moving_blocks")


def _required_name(attrs: Mapping[str, Any], key: str) -> str:
    name = attrs.get(key)
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"attribute {key!r} (non-empty resource name) is required")
    return name


class MovingBlocks(Generic, EasyResource):
    MODEL: ClassVar[Model] = Model(ModelFamily("hackathons", "moving-blocks"), "transfer")

    arm: Arm
    camera: Camera
    gripper: Gripper
    motion: Motion
    arm_name: str
    camera_name: str
    gripper_name: str
    enable_motion: bool
    options: dict[str, Any]
    run_lock: asyncio.Lock

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Tuple[Sequence[str], Sequence[str]]:
        attrs = struct_to_dict(config.attributes)
        deps = [
            _required_name(attrs, "arm"),
            _required_name(attrs, "camera"),
            _required_name(attrs, "gripper"),
        ]
        motion_name = attrs.get("motion") or "builtin"
        if not isinstance(motion_name, str) or not motion_name:
            raise ValueError("attribute 'motion' must be a non-empty service name")
        # Built-in services must be declared by their full API resource name in
        # registry modules; a bare "builtin" is ambiguous to the resolver.
        deps.append(f"rdk:service:motion/{motion_name}")
        return deps, []

    @classmethod
    def new(cls, config: ComponentConfig,
            dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        self = super().new(config, dependencies)
        attrs = struct_to_dict(config.attributes)
        self.arm_name = _required_name(attrs, "arm")
        self.camera_name = _required_name(attrs, "camera")
        self.gripper_name = _required_name(attrs, "gripper")
        motion_name = str(attrs.get("motion") or "builtin")
        self.enable_motion = attrs.get("enable_motion", False) is True
        self.arm = dependencies[Arm.get_resource_name(self.arm_name)]  # type: ignore[assignment]
        self.camera = dependencies[Camera.get_resource_name(self.camera_name)]  # type: ignore[assignment]
        self.gripper = dependencies[Gripper.get_resource_name(self.gripper_name)]  # type: ignore[assignment]
        self.motion = dependencies[Motion.get_resource_name(motion_name)]  # type: ignore[assignment]
        self.options = dict(attrs)
        self.run_lock = asyncio.Lock()
        self.logger.info(
            "moving-blocks ready: arm=%s camera=%s gripper=%s motion=%s enable_motion=%s",
            self.arm_name, self.camera_name, self.gripper_name, motion_name, self.enable_motion,
        )
        return self

    async def _return_to_view_pose(self, world_state, margin: float, timeout: float) -> None:
        target_data = self.options.get("view_pose")
        if not isinstance(target_data, Mapping):
            raise RuntimeError("Configure view_pose with the calibrated overview pose before enabling motion.")
        keys = ("x", "y", "z", "o_x", "o_y", "o_z", "theta")
        try:
            target = Pose(**{key: float(target_data[key]) for key in keys})
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Configured view_pose is incomplete or invalid.") from exc
        bounds = workspace_bounds(self.options)
        bounds["z_max"] = number(self.options, "view_z_max_mm", 800.0)
        assert_in_workspace(target.x, target.y, target.z, bounds)
        current = (await self.motion.get_pose(self.gripper_name, destination_frame="world")).pose
        pos_error = math.sqrt((current.x-target.x)**2 + (current.y-target.y)**2 + (current.z-target.z)**2)
        ori_error = math.sqrt((current.o_x-target.o_x)**2 + (current.o_y-target.o_y)**2 + (current.o_z-target.o_z)**2)
        if pos_error <= 5.0 and ori_error <= 0.02 and abs(current.theta-target.theta) <= 2.0:
            self.logger.info("Arm is already at the saved observation pose.")
            return
        view_speed = number(self.options, "view_speed_degs_per_sec", 25.0)
        if not 3.0 <= view_speed <= 180.0:
            raise RuntimeError("view_speed_degs_per_sec must be between 3 and 180.")
        self.logger.info("Moving to the saved overview pose before scene capture: %s", fmt_pose(target))
        await with_timeout(self.arm.do_command({"set_speed": view_speed}), 10.0, "setting overview speed")
        try:
            await move_gripper(self.motion, self.gripper_name, target, world_state,
                               linear=False, timeout=timeout,
                               label="move to saved observation pose", margin_mm=margin)
        except Exception:
            try:
                await self.arm.stop()
            except Exception:
                self.logger.exception("Could not stop arm after overview-pose motion failure")
            raise

    async def do_command(self, command: Mapping[str, ValueTypes], *,
                         timeout: Optional[float] = None, **kwargs) -> Mapping[str, ValueTypes]:
        action = command.get("action", "preview")
        if action not in ("preview", "move"):
            raise ValueError("action must be 'preview' or 'move'")
        moving = action == "move"
        async with self.run_lock:
            timeout_s = float(timeout or number(self.options, "move_timeout_sec", 60.0))
            world_state = None
            margin = number(self.options, "min_tcp_above_table_mm", 55.0)
            if moving:
                if not self.enable_motion:
                    raise RuntimeError("Motion is disabled in this service configuration (enable_motion=false).")
                if command.get("confirm") is not True:
                    raise RuntimeError("Motion requires DoCommand {'action':'move','confirm':true}.")
                if margin < 50:
                    raise RuntimeError("min_tcp_above_table_mm must be at least 50 mm.")
                if number(self.options, "grasp_tcp_z_offset_mm", 65.0) < 50:
                    raise RuntimeError("grasp_tcp_z_offset_mm must be at least 50 mm for this setup.")
                for key in ("approach_clearance_mm", "lift_mm", "descent_step_mm"):
                    if number(self.options, key, 25.0) <= 0:
                        raise RuntimeError(f"{key} must be greater than zero.")
                if not 3 <= number(self.options, "approach_speed_degs_per_sec", 8.0) <= 180:
                    raise RuntimeError("approach_speed_degs_per_sec must be between 3 and 180.")
                holding = await with_timeout(self.gripper.is_holding_something(), 10.0, "checking gripper before transfer")
                if holding.is_holding_something:
                    raise RuntimeError("Gripper reports an object; refusing to start another transfer.")
                world_state = workcell_world_state(number(self.options, "table_keep_out_mm", 0.0))
                # This move occurs only inside the explicit, confirmed DoCommand.
                await self._return_to_view_pose(world_state, margin, timeout_s)
                await with_timeout(
                    self.arm.do_command({"set_speed": number(self.options, "approach_speed_degs_per_sec", 8.0)}),
                    10.0, "setting transfer speed",
                )

            scene = await capture_scene(self.camera, self.motion, self.camera_name, self.options)
            counts = [len(zone.objects) for zone in scene.zones]
            areas = [[obj.area_px for obj in zone.objects] for zone in scene.zones]
            if counts.count(0) != 1 or sum(counts) == 0:
                raise RuntimeError(
                    f"Expected one occupied source zone and one empty destination zone; "
                    f"detected counts {counts}, component areas (px) {areas}. Refusing to guess."
                )
            source_index = 0 if counts[0] else 1
            source, destination_zone = scene.zones[source_index], scene.zones[1-source_index]
            max_objects = int(number(self.options, "max_objects_per_run", 10))
            if max_objects < 1 or len(source.objects) > max_objects:
                raise RuntimeError(f"Object count {len(source.objects)} exceeds max_objects_per_run={max_objects}.")
            slots = drop_slot_pixels(destination_zone, source.objects, self.options)
            destinations = []
            for slot in slots:
                pose = await zone_surface_pose(self.motion, scene, self.camera_name,
                                               destination_zone, slot, self.options)
                # Validate every drop orientation before the first object is picked.
                surface_down_orientation(pose)
                assert_in_workspace(pose.x, pose.y, pose.z, workspace_bounds(self.options))
                destinations.append(pose)
            for idx, obj in enumerate(source.objects, 1):
                self.logger.info("Object %d: pixel=%s bbox=%s height≈%.1fmm center=%s",
                                 idx, obj.pixel, obj.bbox, obj.height_mm, fmt_pose(obj.center_world))
                self.logger.info("Drop slot %d: %s", idx, fmt_pose(destinations[idx-1]))
            if not moving:
                return {
                    "status": "preview_only",
                    "source_zone": source_index + 1,
                    "destination_zone": 2 - source_index,
                    "object_count": len(source.objects),
                    "object_centers_mm": [
                        {"x": obj.center_world.x, "y": obj.center_world.y, "z": obj.center_world.z}
                        for obj in source.objects
                    ],
                    "message": "No motion commanded. Use action=move with confirm=true only after reviewing this preview.",
                }

            assert world_state is not None
            try:
                for idx, (obj, destination) in enumerate(zip(source.objects, destinations), 1):
                    await move_one_object(
                        obj, destination, self.arm, self.gripper, self.motion,
                        self.gripper_name, world_state, self.options, margin, timeout_s,
                    )
                    self.logger.info("Transferred %d/%d object(s).", idx, len(source.objects))
            except Exception:
                try:
                    await self.arm.stop()
                except Exception:
                    self.logger.exception("Could not stop arm after transfer error")
                raise
            return {"status": "complete", "transferred": len(source.objects)}

    async def get_status(self, *, timeout: Optional[float] = None, **kwargs) -> Mapping[str, ValueTypes]:
        return {"enable_motion": self.enable_motion, "arm": self.arm_name,
                "camera": self.camera_name, "gripper": self.gripper_name}
