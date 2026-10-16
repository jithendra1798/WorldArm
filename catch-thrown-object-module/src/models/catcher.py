"""Viam generic service: soft intercept / catch of a colored object.

Safety posture (non-negotiable):
  - Motion off unless attributes.enable_catch is true AND DoCommand includes
    safety_ack == "I_UNDERSTAND_CATCH_RISKS".
  - Hard caps on arm speed, object speed, session time, and TCP height that
    config cannot raise above.
  - Every motion target must lie inside a tight catch volume AABB.
  - WorldState always includes table / wall-front / wall-side / ceiling.
  - Lost track, out-of-volume prediction, or timeout → stop arm and open gripper.
  - Soft throws only: refuses trajectories faster than HARD_MAX_OBJECT_SPEED_MM_S.

DoCommand:
  {"action": "preview"}     — track + report; never moves
  {"action": "arm_ready"}   — slow move to ready pose (requires enable_catch)
  {"action": "catch", "safety_ack": "I_UNDERSTAND_CATCH_RISKS"}
  {"action": "abort"}       — stop arm immediately
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from typing import Any, ClassVar, Deque, Mapping, Optional, Sequence, Tuple

from typing_extensions import Self

from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Pose, PoseInFrame, ResourceName, Transform, WorldState
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic
from viam.services.motion import Motion
from viam.services.vision import Vision
from viam.utils import ValueTypes, struct_to_dict

from models.workcell_obstacles import TABLE_TOP_Z_MM, workcell_world_state

# --- Hard caps (config may only LOWER these, never raise) ---
SAFETY_ACK_PHRASE = "I_UNDERSTAND_CATCH_RISKS"
HARD_MAX_ARM_SPEED_DEGS = 15.0
HARD_MAX_OBJECT_SPEED_MM_S = 900.0  # ~2 mph; soft lob only
HARD_MAX_SESSION_SEC = 8.0
HARD_MIN_TCP_ABOVE_TABLE_MM = 70.0
HARD_MAX_CATCH_VOLUME_DIAG_MM = 900.0  # refuse huge catch boxes
HARD_MAX_LEAD_SEC = 0.45
HARD_MIN_TRACK_SAMPLES = 3


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


def _clamp_down(requested: float, hard_max: float) -> float:
    return min(requested, hard_max)


class CatchThrownObject(Generic, EasyResource):
    # Keep MODEL exactly as generated when pasting into a Viam Control Code inline module.
    MODEL: ClassVar[Model] = Model(ModelFamily("hackathons", "catch-thrown-object"), "catcher")

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
    enable_catch: bool

    min_detection_confidence: float
    vision_timeout_sec: float
    move_timeout_sec: float
    poll_interval_sec: float

    arm_speed_degs_per_sec: float
    max_object_speed_mm_s: float
    max_session_sec: float
    min_tcp_above_table_mm: float
    close_radius_mm: float
    lead_time_sec: float
    gripper_speed: int
    gripper_torque_pct: int

    # Catch volume AABB in world mm (TCP must stay inside).
    catch_x_min_mm: float
    catch_x_max_mm: float
    catch_y_min_mm: float
    catch_y_max_mm: float
    catch_z_min_mm: float
    catch_z_max_mm: float

    ready_x_mm: float
    ready_y_mm: float
    ready_z_mm: float
    grasp_ox: float
    grasp_oy: float
    grasp_oz: float
    grasp_theta_deg: float

    _abort: bool

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Tuple[Sequence[str], Sequence[str]]:
        attrs = struct_to_dict(config.attributes)
        deps = [
            _req_str(attrs, "arm"),
            _req_str(attrs, "gripper"),
            _req_str(attrs, "detector"),
            _req_str(attrs, "segmenter"),
        ]
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
        self.detect_label = str(attrs.get("detect_label") or "red")
        self.enable_catch = bool(attrs.get("enable_catch", False))

        self.min_detection_confidence = _num(attrs, "min_detection_confidence", 0.55)
        self.vision_timeout_sec = _num(attrs, "vision_timeout_sec", 2.0)
        self.move_timeout_sec = _num(attrs, "move_timeout_sec", 8.0)
        self.poll_interval_sec = max(0.05, _num(attrs, "poll_interval_sec", 0.08))

        self.arm_speed_degs_per_sec = _clamp_down(
            _num(attrs, "arm_speed_degs_per_sec", 10.0), HARD_MAX_ARM_SPEED_DEGS
        )
        self.max_object_speed_mm_s = _clamp_down(
            _num(attrs, "max_object_speed_mm_s", 700.0), HARD_MAX_OBJECT_SPEED_MM_S
        )
        self.max_session_sec = _clamp_down(
            _num(attrs, "max_session_sec", 6.0), HARD_MAX_SESSION_SEC
        )
        self.min_tcp_above_table_mm = max(
            HARD_MIN_TCP_ABOVE_TABLE_MM,
            _num(attrs, "min_tcp_above_table_mm", HARD_MIN_TCP_ABOVE_TABLE_MM),
        )
        self.close_radius_mm = _num(attrs, "close_radius_mm", 80.0)
        self.lead_time_sec = _clamp_down(_num(attrs, "lead_time_sec", 0.25), HARD_MAX_LEAD_SEC)
        self.gripper_speed = int(_num(attrs, "gripper_speed", 800))
        self.gripper_torque_pct = int(_num(attrs, "gripper_torque_pct", 40))

        # Default catch volume: above table, inside walls, modest reach.
        self.catch_x_min_mm = _num(attrs, "catch_x_min_mm", 150.0)
        self.catch_x_max_mm = _num(attrs, "catch_x_max_mm", 450.0)
        self.catch_y_min_mm = _num(attrs, "catch_y_min_mm", 250.0)
        self.catch_y_max_mm = _num(attrs, "catch_y_max_mm", 550.0)
        floor = TABLE_TOP_Z_MM + self.min_tcp_above_table_mm
        self.catch_z_min_mm = max(floor, _num(attrs, "catch_z_min_mm", floor))
        self.catch_z_max_mm = _num(attrs, "catch_z_max_mm", 280.0)

        dx = self.catch_x_max_mm - self.catch_x_min_mm
        dy = self.catch_y_max_mm - self.catch_y_min_mm
        dz = self.catch_z_max_mm - self.catch_z_min_mm
        if dx <= 0 or dy <= 0 or dz <= 0:
            raise ValueError("catch volume AABB must have positive extents")
        diag = math.sqrt(dx * dx + dy * dy + dz * dz)
        if diag > HARD_MAX_CATCH_VOLUME_DIAG_MM:
            raise ValueError(
                f"catch volume diagonal {diag:.0f} mm exceeds hard max "
                f"{HARD_MAX_CATCH_VOLUME_DIAG_MM:.0f} mm; shrink the box"
            )
        if self.catch_z_min_mm < TABLE_TOP_Z_MM + HARD_MIN_TCP_ABOVE_TABLE_MM:
            raise ValueError("catch_z_min_mm too low relative to table safety floor")

        self.ready_x_mm = _num(attrs, "ready_x_mm", (self.catch_x_min_mm + self.catch_x_max_mm) / 2)
        self.ready_y_mm = _num(attrs, "ready_y_mm", (self.catch_y_min_mm + self.catch_y_max_mm) / 2)
        self.ready_z_mm = _num(attrs, "ready_z_mm", (self.catch_z_min_mm + self.catch_z_max_mm) / 2)
        self.grasp_ox = _num(attrs, "grasp_ox", 0.0)
        self.grasp_oy = _num(attrs, "grasp_oy", 0.0)
        self.grasp_oz = _num(attrs, "grasp_oz", -1.0)
        self.grasp_theta_deg = _num(attrs, "grasp_theta_deg", 0.0)

        self.arm = dependencies[Arm.get_resource_name(self.arm_name)]  # type: ignore[assignment]
        self.gripper = dependencies[Gripper.get_resource_name(self.gripper_name)]  # type: ignore[assignment]
        self.detector = dependencies[Vision.get_resource_name(self.detector_name)]  # type: ignore[assignment]
        self.segmenter = dependencies[Vision.get_resource_name(self.segmenter_name)]  # type: ignore[assignment]
        self.motion = dependencies[Motion.get_resource_name(self.motion_name)]  # type: ignore[assignment]

        self._abort = False
        self._assert_in_catch_volume(self.ready_x_mm, self.ready_y_mm, self.ready_z_mm, "ready pose")

        self.logger.info(
            "catch-thrown-object ready: label=%s enable_catch=%s speed<=%.1f deg/s "
            "obj_speed<=%.0f mm/s session<=%.1fs volume=[(%.0f,%.0f,%.0f)-(%.0f,%.0f,%.0f)]",
            self.detect_label,
            self.enable_catch,
            self.arm_speed_degs_per_sec,
            self.max_object_speed_mm_s,
            self.max_session_sec,
            self.catch_x_min_mm,
            self.catch_y_min_mm,
            self.catch_z_min_mm,
            self.catch_x_max_mm,
            self.catch_y_max_mm,
            self.catch_z_max_mm,
        )
        return self

    def _assert_in_catch_volume(self, x: float, y: float, z: float, context: str) -> None:
        if not (
            self.catch_x_min_mm <= x <= self.catch_x_max_mm
            and self.catch_y_min_mm <= y <= self.catch_y_max_mm
            and self.catch_z_min_mm <= z <= self.catch_z_max_mm
        ):
            raise RuntimeError(
                f"{context}: ({x:.1f},{y:.1f},{z:.1f}) mm outside catch volume "
                f"[({self.catch_x_min_mm:.0f},{self.catch_y_min_mm:.0f},{self.catch_z_min_mm:.0f})-"
                f"({self.catch_x_max_mm:.0f},{self.catch_y_max_mm:.0f},{self.catch_z_max_mm:.0f})]"
            )
        floor = TABLE_TOP_Z_MM + self.min_tcp_above_table_mm
        if z < floor:
            raise RuntimeError(f"{context}: Z={z:.1f} below table safety floor {floor:.1f}")

    def _clamp_to_catch_volume(self, x: float, y: float, z: float) -> Tuple[float, float, float]:
        return (
            min(max(x, self.catch_x_min_mm), self.catch_x_max_mm),
            min(max(y, self.catch_y_min_mm), self.catch_y_max_mm),
            min(max(z, self.catch_z_min_mm), self.catch_z_max_mm),
        )

    def _make_pose(self, x: float, y: float, z: float) -> Pose:
        return Pose(
            x=x,
            y=y,
            z=z,
            o_x=self.grasp_ox,
            o_y=self.grasp_oy,
            o_z=self.grasp_oz,
            theta=self.grasp_theta_deg,
        )

    async def _timeout(self, awaitable, seconds: float, what: str):
        try:
            return await asyncio.wait_for(awaitable, timeout=seconds)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(f"Timed out after {seconds:.1f}s waiting for {what}") from exc

    async def _stop_arm(self) -> None:
        try:
            await self.arm.stop()
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("arm.stop failed: %s", exc)

    async def _safe_open_gripper(self) -> None:
        try:
            await self._timeout(self.gripper.open(), 15.0, "gripper.open")
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("gripper.open failed: %s", exc)

    async def _world_state(self) -> WorldState:
        return workcell_world_state(include_ceiling=True)

    async def _move_tcp(self, pose: Pose, *, label: str) -> None:
        self._assert_in_catch_volume(pose.x, pose.y, pose.z, label)
        ws = await self._world_state()
        ok = await self._timeout(
            self.motion.move(
                component_name=self.gripper_name,
                destination=PoseInFrame(reference_frame="world", pose=pose),
                world_state=ws,
                timeout=self.move_timeout_sec,
            ),
            self.move_timeout_sec + 2.0,
            label,
        )
        if not ok:
            raise RuntimeError(f"Motion reported failure for {label}")

    async def _localize_object(self) -> Optional[dict[str, Any]]:
        """Return world XYZ of the labeled object, or None if not confidently seen."""
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
            return None

        def area(d) -> float:
            return max(0.0, (d.x_max - d.x_min) * (d.y_max - d.y_min))

        det = max(matched, key=area)
        objects = await self._timeout(
            self.segmenter.get_object_point_clouds(self.camera_name),
            self.vision_timeout_sec,
            "segments",
        )
        labeled = [
            o
            for o in objects
            if o.geometries.geometries and o.geometries.geometries[0].label == self.detect_label
        ]
        if not labeled:
            return None
        obj = max(labeled, key=lambda o: len(o.point_cloud))
        if not obj.geometries.geometries or len(obj.point_cloud) < 40:
            return None
        center_cam = obj.geometries.geometries[0].center
        if center_cam.z <= 0:
            return None
        # Supplemental frame → world via motion.get_pose (no robot client required).
        detected = Transform(
            reference_frame="catch-detected-object",
            pose_in_observer_frame=PoseInFrame(
                reference_frame=self.camera_name,
                pose=center_cam,
            ),
        )
        center_world = await self._timeout(
            self.motion.get_pose(
                component_name="catch-detected-object",
                destination_frame="world",
                supplemental_transforms=[detected],
            ),
            self.vision_timeout_sec,
            "cam→world via supplemental transform",
        )
        p = center_world.pose
        return {
            "label": det.class_name,
            "confidence": float(det.confidence),
            "x": float(p.x),
            "y": float(p.y),
            "z": float(p.z),
            "t": time.monotonic(),
        }

    @staticmethod
    def _estimate_velocity(
        track: Deque[dict[str, Any]],
    ) -> Optional[Tuple[float, float, float, float]]:
        if len(track) < HARD_MIN_TRACK_SAMPLES:
            return None
        a, b = track[0], track[-1]
        dt = b["t"] - a["t"]
        if dt < 0.04:
            return None
        vx = (b["x"] - a["x"]) / dt
        vy = (b["y"] - a["y"]) / dt
        vz = (b["z"] - a["z"]) / dt
        speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        return vx, vy, vz, speed

    def _predict(self, sample: dict[str, Any], vel: Tuple[float, float, float, float], lead: float):
        vx, vy, vz, _ = vel
        return (
            sample["x"] + vx * lead,
            sample["y"] + vy * lead,
            sample["z"] + vz * lead,
        )

    async def _arm_ready(self) -> dict[str, Any]:
        if not self.enable_catch:
            raise RuntimeError("arm_ready refused: set attributes.enable_catch=true first")
        await self.arm.do_command({"set_speed": self.arm_speed_degs_per_sec})
        await self._safe_open_gripper()
        pose = self._make_pose(self.ready_x_mm, self.ready_y_mm, self.ready_z_mm)
        await self._move_tcp(pose, label="arm_ready")
        return {
            "ok": True,
            "ready_pose_mm": {"x": pose.x, "y": pose.y, "z": pose.z},
            "arm_speed_degs_per_sec": self.arm_speed_degs_per_sec,
        }

    async def _preview(self) -> dict[str, Any]:
        sample = await self._localize_object()
        return {
            "ok": True,
            "motion": False,
            "enable_catch": self.enable_catch,
            "detect_label": self.detect_label,
            "sample": sample,
            "catch_volume_mm": {
                "min": [self.catch_x_min_mm, self.catch_y_min_mm, self.catch_z_min_mm],
                "max": [self.catch_x_max_mm, self.catch_y_max_mm, self.catch_z_max_mm],
            },
            "hard_caps": {
                "max_arm_speed_degs": HARD_MAX_ARM_SPEED_DEGS,
                "max_object_speed_mm_s": HARD_MAX_OBJECT_SPEED_MM_S,
                "max_session_sec": HARD_MAX_SESSION_SEC,
                "min_tcp_above_table_mm": HARD_MIN_TCP_ABOVE_TABLE_MM,
                "safety_ack_phrase": SAFETY_ACK_PHRASE,
            },
            "note": (
                "preview never moves. Soft-lob catch only; clear the workcell; "
                "keep a hand on e-stop. This is not a sports catcher."
            ),
        }

    async def _catch(self, command: Mapping[str, ValueTypes]) -> dict[str, Any]:
        if not self.enable_catch:
            raise RuntimeError("catch refused: attributes.enable_catch is false")
        ack = str(command.get("safety_ack") or "")
        if ack != SAFETY_ACK_PHRASE:
            raise RuntimeError(
                f"catch refused: pass safety_ack={SAFETY_ACK_PHRASE!r} "
                "(people clear of workcell, soft underhand lob only, e-stop ready)"
            )

        self._abort = False
        await self.arm.do_command({"set_speed": self.arm_speed_degs_per_sec})
        await self._safe_open_gripper()
        ready = self._make_pose(self.ready_x_mm, self.ready_y_mm, self.ready_z_mm)
        await self._move_tcp(ready, label="pre-catch ready")

        track: Deque[dict[str, Any]] = deque(maxlen=8)
        started = time.monotonic()
        moved = False
        grabbed = False
        reason = "session_timeout"

        while time.monotonic() - started < self.max_session_sec:
            if self._abort:
                reason = "aborted"
                break
            try:
                sample = await self._localize_object()
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("localize failed: %s", exc)
                sample = None

            if sample is None:
                await asyncio.sleep(self.poll_interval_sec)
                continue

            track.append(sample)
            vel = self._estimate_velocity(track)
            if vel is None:
                await asyncio.sleep(self.poll_interval_sec)
                continue

            _vx, _vy, _vz, speed = vel
            if speed > self.max_object_speed_mm_s:
                reason = f"object_too_fast:{speed:.0f}mm/s"
                await self._stop_arm()
                break

            px, py, pz = self._predict(sample, vel, self.lead_time_sec)
            if not (
                self.catch_x_min_mm <= px <= self.catch_x_max_mm
                and self.catch_y_min_mm <= py <= self.catch_y_max_mm
                and self.catch_z_min_mm <= pz <= self.catch_z_max_mm
            ):
                if not (
                    self.catch_x_min_mm <= sample["x"] <= self.catch_x_max_mm
                    and self.catch_y_min_mm <= sample["y"] <= self.catch_y_max_mm
                    and self.catch_z_min_mm <= sample["z"] <= self.catch_z_max_mm
                ):
                    await asyncio.sleep(self.poll_interval_sec)
                    continue
                px, py, pz = sample["x"], sample["y"], sample["z"]

            cx, cy, cz = self._clamp_to_catch_volume(px, py, pz)
            target = self._make_pose(cx, cy, cz)
            dist = math.sqrt(
                (sample["x"] - cx) ** 2 + (sample["y"] - cy) ** 2 + (sample["z"] - cz) ** 2
            )
            try:
                await self._move_tcp(target, label="intercept step")
                moved = True
            except Exception as exc:  # noqa: BLE001
                self.logger.error("intercept move failed: %s", exc)
                reason = f"motion_failed:{exc}"
                await self._stop_arm()
                break

            if dist <= self.close_radius_mm:
                if self.gripper_speed > 0:
                    try:
                        await self.gripper.do_command(
                            {"set_gripper_speed": float(self.gripper_speed)}
                        )
                    except Exception:  # noqa: BLE001
                        pass
                if self.gripper_torque_pct > 0:
                    try:
                        await self.gripper.do_command(
                            {"set_gripper_torque": int(self.gripper_torque_pct)}
                        )
                    except Exception:  # noqa: BLE001
                        pass
                grabbed = bool(await self._timeout(self.gripper.grab(), 10.0, "gripper.grab"))
                reason = "grabbed" if grabbed else "grab_missed"
                break

            await asyncio.sleep(self.poll_interval_sec)

        if not grabbed:
            await self._stop_arm()
            await self._safe_open_gripper()
            if moved and not self._abort:
                try:
                    await self._move_tcp(ready, label="retreat after miss")
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning("retreat failed: %s", exc)
        else:
            hold = self._make_pose(self.ready_x_mm, self.ready_y_mm, self.catch_z_max_mm)
            try:
                await self._move_tcp(hold, label="hold after catch")
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("hold move failed: %s", exc)

        return {
            "ok": grabbed,
            "reason": reason,
            "moved": moved,
            "grabbed": grabbed,
            "track_samples": len(track),
            "session_sec": round(time.monotonic() - started, 3),
            "last_sample": track[-1] if track else None,
        }

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Mapping[str, ValueTypes]:
        del timeout, kwargs
        action = str(command.get("action") or "preview").lower()

        if action == "abort":
            self._abort = True
            await self._stop_arm()
            await self._safe_open_gripper()
            return {"ok": True, "action": "abort"}

        if action == "preview":
            return await self._preview()

        if action == "arm_ready":
            return await self._arm_ready()

        if action == "catch":
            try:
                return await self._catch(command)
            except Exception:
                await self._stop_arm()
                await self._safe_open_gripper()
                raise

        raise RuntimeError(
            f"Unknown action {action!r}. Use preview | arm_ready | catch | abort."
        )
