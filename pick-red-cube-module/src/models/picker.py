"""Viam generic service: detect a red cube and optionally pick + hold it.

Reuses machine resources (no invented extrinsics):
  - vision color_detector + detections-to-segments
  - builtin motion (uses configured workcell obstacles)
  - arm / gripper

DoCommand:
  {"action": "preview"}  — detect + log planned poses; never moves
  {"action": "pick"}     — requires attributes.enable_pick == true
"""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar, Mapping, Optional, Sequence, Tuple

from typing_extensions import Self

from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Pose, PoseInFrame, ResourceName, WorldState
from viam.proto.service.motion import Constraints, LinearConstraint
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic
from viam.services.motion import Motion
from viam.services.vision import Vision
from viam.utils import ValueTypes, struct_to_dict

from models.workcell_obstacles import TABLE_TOP_Z_MM, workcell_world_state


def _req_str(attrs: Mapping[str, Any], key: str) -> str:
    val = attrs.get(key)
    if not isinstance(val, str) or not val:
        raise ValueError(f"attribute '{key}' (non-empty string) is required")
    return val


def _num(attrs: Mapping[str, Any], key: str, default: float | None = None) -> float:
    if key not in attrs or attrs[key] is None or attrs[key] == "":
        if default is None:
            raise ValueError(f"attribute '{key}' is required and must be a number")
        return float(default)
    return float(attrs[key])


def _pose_dict(p: Pose) -> dict[str, float]:
    return {
        "x": float(p.x),
        "y": float(p.y),
        "z": float(p.z),
        "o_x": float(p.o_x),
        "o_y": float(p.o_y),
        "o_z": float(p.o_z),
        "theta": float(p.theta),
    }


class PickRedCube(Generic, EasyResource):
    # Do not change MODEL when pasting into an *inline* module — the app generates it.
    # For this repo module / registry package, this is the registered model id.
    MODEL: ClassVar[Model] = Model(ModelFamily("hackathons", "pick-red-cube"), "picker")

    arm: Arm
    gripper: Gripper
    detector: Vision
    segmenter: Vision
    motion: Motion

    arm_name: str
    gripper_name: str
    camera_name: str
    detector_name: str
    segmenter_name: str
    motion_name: str
    detect_label: str
    enable_pick: bool

    min_detection_confidence: float
    vision_timeout_sec: float
    move_timeout_sec: float
    min_camera_depth_mm: float
    max_camera_depth_mm: float
    min_point_cloud_bytes: int
    table_top_z_mm: float
    table_z_tolerance_mm: float

    workspace_x_min_mm: float
    workspace_x_max_mm: float
    workspace_y_min_mm: float
    workspace_y_max_mm: float
    workspace_z_min_mm: float
    workspace_z_max_mm: float

    approach_clearance_mm: float
    lift_mm: float
    approach_speed_degs_per_sec: float
    descent_step_mm: float
    grasp_tcp_z_offset_mm: float
    grasp_ox: float
    grasp_oy: float
    grasp_oz: float
    grasp_theta_deg: float
    gripper_speed: int
    gripper_torque_pct: int
    min_tcp_above_table_mm: float
    table_keep_out_mm: float
    # Max extra tool-axis descent after the camera-frame approach (mm). Kept small
    # so we do not double-apply approach_clearance and dive through the table.
    max_fine_descent_mm: float

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Tuple[Sequence[str], Sequence[str]]:
        attrs = struct_to_dict(config.attributes)
        deps = [
            _req_str(attrs, "arm"),
            _req_str(attrs, "gripper"),
            _req_str(attrs, "detector"),
            _req_str(attrs, "segmenter"),
        ]
        # camera is a string dependency of the vision services; also required for PCD calls
        _req_str(attrs, "camera")
        motion_name = attrs.get("motion") or "builtin"
        if not isinstance(motion_name, str) or not motion_name:
            raise ValueError("attribute 'motion' must be a non-empty string when set")
        deps.append(motion_name)
        return deps, []

    @classmethod
    def new(
        cls,
        config: ComponentConfig,
        dependencies: Mapping[ResourceName, ResourceBase],
    ) -> Self:
        self = super().new(config, dependencies)
        attrs = struct_to_dict(config.attributes)

        self.arm_name = _req_str(attrs, "arm")
        self.gripper_name = _req_str(attrs, "gripper")
        self.camera_name = _req_str(attrs, "camera")
        self.detector_name = _req_str(attrs, "detector")
        self.segmenter_name = _req_str(attrs, "segmenter")
        self.motion_name = str(attrs.get("motion") or "builtin")
        self.detect_label = str(attrs.get("detect_label") or "red-cube")
        self.enable_pick = bool(attrs.get("enable_pick", False))

        self.min_detection_confidence = _num(attrs, "min_detection_confidence", 0.5)
        self.vision_timeout_sec = _num(attrs, "vision_timeout_sec", 30.0)
        self.move_timeout_sec = _num(attrs, "move_timeout_sec", 60.0)
        self.min_camera_depth_mm = _num(attrs, "min_camera_depth_mm", 100.0)
        self.max_camera_depth_mm = _num(attrs, "max_camera_depth_mm", 1000.0)
        self.min_point_cloud_bytes = int(_num(attrs, "min_point_cloud_bytes", 50))
        # Table top from armfarm obstacles fragment: center z=-123, half-height 100 → -23
        self.table_top_z_mm = _num(attrs, "table_top_z_mm", -23.0)
        self.table_z_tolerance_mm = _num(attrs, "table_z_tolerance_mm", 100.0)

        self.workspace_x_min_mm = _num(attrs, "workspace_x_min_mm", -1400.0)
        self.workspace_x_max_mm = _num(attrs, "workspace_x_max_mm", 690.0)
        self.workspace_y_min_mm = _num(attrs, "workspace_y_min_mm", -450.0)
        self.workspace_y_max_mm = _num(attrs, "workspace_y_max_mm", 1400.0)
        self.workspace_z_min_mm = _num(attrs, "workspace_z_min_mm", -23.0)
        self.workspace_z_max_mm = _num(attrs, "workspace_z_max_mm", 150.0)

        # Motion geometry — required only when enable_pick is true (validated at pick time)
        self.approach_clearance_mm = _num(attrs, "approach_clearance_mm", 100.0)
        self.lift_mm = _num(attrs, "lift_mm", 100.0)
        self.approach_speed_degs_per_sec = _num(attrs, "approach_speed_degs_per_sec", 8.0)
        self.descent_step_mm = _num(attrs, "descent_step_mm", 25.0)
        self.grasp_tcp_z_offset_mm = _num(attrs, "grasp_tcp_z_offset_mm", -6.0)
        self.grasp_ox = _num(attrs, "grasp_ox", 0.0)
        self.grasp_oy = _num(attrs, "grasp_oy", 0.0)
        self.grasp_oz = _num(attrs, "grasp_oz", -1.0)
        self.grasp_theta_deg = _num(attrs, "grasp_theta_deg", 0.0)
        self.gripper_speed = int(_num(attrs, "gripper_speed", 800))
        self.gripper_torque_pct = int(_num(attrs, "gripper_torque_pct", 40))
        self.min_tcp_above_table_mm = _num(attrs, "min_tcp_above_table_mm", 40.0)
        self.table_keep_out_mm = _num(attrs, "table_keep_out_mm", 40.0)
        self.max_fine_descent_mm = _num(attrs, "max_fine_descent_mm", 15.0)

        self.arm = dependencies[Arm.get_resource_name(self.arm_name)]  # type: ignore[assignment]
        self.gripper = dependencies[Gripper.get_resource_name(self.gripper_name)]  # type: ignore[assignment]
        self.detector = dependencies[Vision.get_resource_name(self.detector_name)]  # type: ignore[assignment]
        self.segmenter = dependencies[Vision.get_resource_name(self.segmenter_name)]  # type: ignore[assignment]
        self.motion = dependencies[Motion.get_resource_name(self.motion_name)]  # type: ignore[assignment]

        self.logger.info(
            "pick-red-cube ready: camera=%s detector=%s segmenter=%s enable_pick=%s",
            self.camera_name,
            self.detector_name,
            self.segmenter_name,
            self.enable_pick,
        )
        return self

    async def _timeout(self, awaitable, seconds: float, what: str):
        try:
            return await asyncio.wait_for(awaitable, timeout=seconds)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(f"Timed out after {seconds:.1f}s waiting for {what}") from exc

    async def _localize(self) -> dict[str, Any]:
        detections = await self._timeout(
            self.detector.get_detections_from_camera(self.camera_name),
            self.vision_timeout_sec,
            "detections",
        )
        matched = [
            d
            for d in detections
            if (not self.detect_label or d.class_name == self.detect_label)
            and d.confidence >= self.min_detection_confidence
        ]
        if not matched:
            raise RuntimeError(
                f"No unambiguous detection for label={self.detect_label!r} "
                f"(min_conf={self.min_detection_confidence})"
            )

        def area(d) -> float:
            return max(0.0, (d.x_max - d.x_min) * (d.y_max - d.y_min))

        matched = sorted(matched, key=area, reverse=True)
        if len(matched) > 1 and area(matched[1]) > area(matched[0]) * 0.60:
            raise RuntimeError(
                f"More than one similarly sized detection ({len(matched)}); refusing to guess."
            )
        det = matched[0]

        objects = await self._timeout(
            self.segmenter.get_object_point_clouds(self.camera_name),
            self.vision_timeout_sec,
            "object point clouds",
        )
        if not objects:
            raise RuntimeError("Segmenter returned no 3D objects; refusing to move.")

        labeled = [
            o
            for o in objects
            if o.geometries.geometries and o.geometries.geometries[0].label == self.detect_label
        ]
        candidates = labeled or list(objects)
        if len(candidates) > 1:
            sizes = sorted((len(o.point_cloud) for o in candidates), reverse=True)
            if len(sizes) > 1 and sizes[1] > sizes[0] * 0.60:
                raise RuntimeError(
                    f"Segmenter found {len(candidates)} similar 3D objects; refusing to guess."
                )
        obj = max(candidates, key=lambda o: len(o.point_cloud))
        if len(obj.point_cloud) < self.min_point_cloud_bytes:
            raise RuntimeError(
                f"3D segment too small ({len(obj.point_cloud)} bytes); invalid depth."
            )
        if not obj.geometries.geometries:
            raise RuntimeError("3D object has no geometry/center.")

        center_cam = obj.geometries.geometries[0].center
        if not self.min_camera_depth_mm <= center_cam.z <= self.max_camera_depth_mm:
            raise RuntimeError(
                f"Camera-frame depth {center_cam.z:.1f} mm outside "
                f"[{self.min_camera_depth_mm}, {self.max_camera_depth_mm}]"
            )

        # Resolve camera-frame point into world using Viam's frame system (motion.get_pose
        # of a supplemental frame is not available; use parent TransformPose via dialing
        # is awkward in-module). Instead: ask motion for the camera pose in world and
        # apply the calibrated frame — but OV composition invents math. Prefer handing
        # camera-frame destinations to motion.move (workshop Phase 5), and also try
        # TransformPose through the motion service's parent by moving a zero-cost query:
        # get_pose(camera) gives the calibrated extrinsics origin.
        cam_world = await self._timeout(
            self.motion.get_pose(self.camera_name, "world"),
            self.vision_timeout_sec,
            "camera pose in world",
        )

        return {
            "detection": {
                "label": det.class_name,
                "confidence": float(det.confidence),
                "bbox": {
                    "x_min": float(det.x_min),
                    "y_min": float(det.y_min),
                    "x_max": float(det.x_max),
                    "y_max": float(det.y_max),
                },
            },
            "center_camera": _pose_dict(center_cam),
            "camera_in_world": _pose_dict(cam_world.pose),
            "point_cloud_bytes": len(obj.point_cloud),
            "center_cam_pose": center_cam,
        }

    def _orientation(self) -> dict[str, float]:
        return {
            "o_x": self.grasp_ox,
            "o_y": self.grasp_oy,
            "o_z": self.grasp_oz,
            "theta": self.grasp_theta_deg,
        }

    def _validate_motion_attrs(self) -> None:
        if not 3.0 <= self.approach_speed_degs_per_sec <= 180.0:
            raise RuntimeError("approach_speed_degs_per_sec must be between 3 and 180")
        if self.descent_step_mm <= 0 or self.approach_clearance_mm <= 0 or self.lift_mm <= 0:
            raise RuntimeError("approach_clearance_mm, lift_mm, descent_step_mm must be > 0")

    def _table_floor_z(self) -> float:
        return max(
            TABLE_TOP_Z_MM + self.min_tcp_above_table_mm,
            TABLE_TOP_Z_MM + self.table_keep_out_mm,
        )

    async def _gripper_world_z(self) -> float:
        pose = await self._timeout(
            self.motion.get_pose(self.gripper_name, "world"),
            self.move_timeout_sec,
            "gripper pose in world",
        )
        return float(pose.pose.z)

    async def _move_gripper(
        self,
        destination: PoseInFrame,
        world_state: WorldState,
        *,
        linear: bool,
        label: str,
    ) -> None:
        constraints = None
        if linear:
            constraints = Constraints(
                linear_constraint=[
                    LinearConstraint(line_tolerance_mm=2.0, orientation_tolerance_degs=5.0)
                ]
            )
        # WorldState always includes safety-table / safety-wall-front / safety-wall-side
        # (unique labels). Component obstacles on the machine still apply as well.
        ok = await self._timeout(
            self.motion.move(
                component_name=self.gripper_name,
                destination=destination,
                world_state=world_state,
                constraints=constraints,
                timeout=self.move_timeout_sec,
            ),
            self.move_timeout_sec + 5.0,
            label,
        )
        if not ok:
            raise RuntimeError(f"Motion service reported failure for {label}")

    async def preview(self) -> Mapping[str, ValueTypes]:
        loc = await self._localize()
        center_cam: Pose = loc["center_cam_pose"]
        clearance = self.approach_clearance_mm
        approach_cam = Pose(
            x=center_cam.x,
            y=center_cam.y,
            z=center_cam.z - clearance,
            **self._orientation(),
        )
        grasp_cam = Pose(
            x=center_cam.x,
            y=center_cam.y,
            z=center_cam.z + self.grasp_tcp_z_offset_mm,
            **self._orientation(),
        )
        self.logger.info(
            "preview detection=%s cam_xyz=(%.1f,%.1f,%.1f) approach_z=%.1f "
            "obstacles=safety-table,safety-wall-front,safety-wall-side table_floor=%.1f",
            loc["detection"]["label"],
            center_cam.x,
            center_cam.y,
            center_cam.z,
            approach_cam.z,
            self._table_floor_z(),
        )
        return {
            "action": "preview",
            "moved": False,
            "enable_pick": self.enable_pick,
            "detection": loc["detection"],
            "center_camera_mm": loc["center_camera"],
            "camera_in_world_mm": loc["camera_in_world"],
            "point_cloud_bytes": loc["point_cloud_bytes"],
            "planned_approach_camera_mm": _pose_dict(approach_cam),
            "planned_grasp_camera_mm": _pose_dict(grasp_cam),
            "table_top_z_mm": TABLE_TOP_Z_MM,
            "table_safety_floor_z_mm": self._table_floor_z(),
            "motion_obstacles": [
                "safety-table",
                "safety-wall-front",
                "safety-wall-side",
                "safety-ceiling",
            ],
            "note": (
                "Preview only. Pick uses WorldState obstacles mirroring machine "
                "table/wall-front/wall-side/ceiling, plus a live world-Z floor check "
                "so descent cannot drive the TCP through the table."
            ),
        }

    async def pick(self) -> Mapping[str, ValueTypes]:
        if not self.enable_pick:
            raise RuntimeError(
                "Pick refused: set attributes.enable_pick=true on this service in the "
                "Viam machine config (double gate with DoCommand action=pick)."
            )
        self._validate_motion_attrs()
        if self.min_tcp_above_table_mm < 0:
            raise RuntimeError("min_tcp_above_table_mm must be >= 0")

        loc = await self._localize()
        center_cam: Pose = loc["center_cam_pose"]
        orientation = self._orientation()
        clearance = self.approach_clearance_mm
        step = self.descent_step_mm
        z_floor = self._table_floor_z()
        world_state = workcell_world_state(
            include_ceiling=True, table_keep_out_mm=self.table_keep_out_mm
        )

        # Camera-frame approach only: standoff toward the camera (away from the table).
        # Do NOT apply a second full-clearance tool-axis dive afterward (that was
        # driving through the table).
        approach = PoseInFrame(
            reference_frame=self.camera_name,
            pose=Pose(
                x=center_cam.x,
                y=center_cam.y,
                z=center_cam.z - clearance,
                **orientation,
            ),
        )
        grasp = approach  # grab at the approach standoff unless a tiny fine descent runs

        self.logger.info(
            "pick start: cam_target=(%.1f,%.1f,%.1f) speed=%.1f deg/s "
            "obstacles=safety-table(+keepout %.0f),walls floor_z=%.1f max_fine_descent=%.1f",
            center_cam.x,
            center_cam.y,
            center_cam.z,
            self.approach_speed_degs_per_sec,
            self.table_keep_out_mm,
            z_floor,
            self.max_fine_descent_mm,
        )

        await self._timeout(
            self.arm.do_command({"set_speed": self.approach_speed_degs_per_sec}),
            10.0,
            "arm set_speed",
        )
        await self._timeout(self.gripper.open(), 30.0, "gripper.open")

        try:
            # Free (non-linear) plan so the planner can route around table/walls.
            await self._move_gripper(
                approach, world_state, linear=False, label="pre-grasp"
            )

            gz = await self._gripper_world_z()
            if gz < z_floor:
                raise RuntimeError(
                    f"Pre-grasp left gripper at world Z={gz:.1f} mm below table floor "
                    f"{z_floor:.1f} mm; refusing further descent."
                )

            # Optional tiny fine approach only — capped, with live world-Z abort.
            remaining = max(0.0, min(self.max_fine_descent_mm, step))
            while remaining > 1e-3:
                delta = min(step, remaining)
                gz = await self._gripper_world_z()
                if gz - delta < z_floor:
                    self.logger.warning(
                        "Stopping fine descent: Z=%.1f mm, step=%.1f would cross floor %.1f",
                        gz,
                        delta,
                        z_floor,
                    )
                    break
                remaining -= delta
                waypoint = PoseInFrame(
                    reference_frame=self.gripper_name,
                    pose=Pose(x=0, y=0, z=delta, o_x=0, o_y=0, o_z=1, theta=0),
                )
                self.logger.info(
                    "fine descent %.1f mm (gripper_world_z=%.1f, floor=%.1f)",
                    delta,
                    gz,
                    z_floor,
                )
                await self._move_gripper(
                    waypoint, world_state, linear=True, label="fine descent"
                )

            if self.gripper_speed > 0:
                await self._timeout(
                    self.gripper.do_command({"set_gripper_speed": float(self.gripper_speed)}),
                    10.0,
                    "set_gripper_speed",
                )
            if self.gripper_torque_pct > 0:
                try:
                    await self._timeout(
                        self.gripper.do_command({"set_gripper_torque": int(self.gripper_torque_pct)}),
                        10.0,
                        "set_gripper_torque",
                    )
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning("set_gripper_torque skipped: %s", exc)

            grabbed = await self._timeout(self.gripper.grab(), 30.0, "gripper.grab")
            if not grabbed:
                holding = await self._timeout(
                    self.gripper.is_holding_something(), 10.0, "is_holding_something"
                )
                grabbed = bool(holding.is_holding_something)
            if not grabbed:
                await self._move_gripper(
                    approach, world_state, linear=True, label="retreat after failed grasp"
                )
                raise RuntimeError("Gripper did not confirm a grasp; did not lift.")

            # Lift opposite tool +Z (away from table) while still obstacle-aware.
            lift = PoseInFrame(
                reference_frame=self.gripper_name,
                pose=Pose(x=0, y=0, z=-self.lift_mm, o_x=0, o_y=0, o_z=1, theta=0),
            )
            await self._move_gripper(lift, world_state, linear=True, label="lift")

            still = await self._timeout(
                self.gripper.is_holding_something(), 10.0, "post-lift hold check"
            )
            if not still.is_holding_something:
                raise RuntimeError("Object lost during/after lift.")

            self.logger.info("Pick complete; holding cube (not releasing).")
            return {
                "action": "pick",
                "moved": True,
                "success": True,
                "holding": True,
                "detection": loc["detection"],
                "center_camera_mm": loc["center_camera"],
                "planned_approach_camera_mm": _pose_dict(approach.pose),
                "planned_grasp_camera_mm": _pose_dict(grasp.pose),
                "table_safety_floor_z_mm": z_floor,
                "motion_obstacles": [
                    "safety-table",
                    "safety-wall-front",
                    "safety-wall-side",
                    "safety-ceiling",
                ],
            }
        except Exception:
            try:
                await self.arm.stop()
            except Exception:  # noqa: BLE001
                self.logger.exception("arm.stop failed during error handling")
            raise

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        action = str(command.get("action") or "preview").lower()
        if action in ("preview", "detect"):
            return await self.preview()
        if action == "pick":
            return await self.pick()
        raise ValueError(
            f"Unknown action {action!r}. Use {{\"action\": \"preview\"}} or {{\"action\": \"pick\"}}."
        )
