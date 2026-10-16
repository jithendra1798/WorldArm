"""Detect one red cube via Viam vision blocks, then optionally pick and hold it.

Uses the machine's configured resources:
  - camera ``cam`` (RealSense RGB-D, PCD-capable)
  - vision ``{color}-object-detector`` (builtin color_detector; any shape/size)
  - vision ``{color}-object-segmenter`` (viam:detections-to-segments)
  - arm ``arm``, gripper ``gripper``, obstacles, and builtin motion

Default is preview-only. Physical motion requires ``VIAM_PICK_ENABLE=1`` plus
explicit motion settings (see ``required_motion_settings``). Arm speed is set
through the xArm module's ``set_speed`` DoCommand (deg/s); pauses do not slow
motion. A persistent default can also be set as ``speed_degs_per_sec`` on the
arm component in the Viam machine / fragment config (currently 60 deg/s).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Sequence

from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.proto.common import Pose, PoseInFrame, WorldState
from viam.proto.service.motion import Constraints, LinearConstraint
from viam.services.motion import MotionClient
from viam.services.vision import Detection, VisionClient

from connection import connect, env
from workcell_obstacles import (
    TABLE_TOP_Z_MM,
    assert_tcp_above_table,
    compute_grasp_tcp_z,
    workcell_world_state,
)

logger = logging.getLogger("pick_red_cube")

# Workspace bounds derived from configured obstacles (front wall x=740, side wall y=-500)
# with margins matching the existing armfarm workcell envelope.
DEFAULT_WORKSPACE = {
    "VIAM_WORKSPACE_X_MIN_MM": -1400.0,
    "VIAM_WORKSPACE_X_MAX_MM": 690.0,
    "VIAM_WORKSPACE_Y_MIN_MM": -450.0,
    "VIAM_WORKSPACE_Y_MAX_MM": 1400.0,
    "VIAM_WORKSPACE_Z_MIN_MM": TABLE_TOP_Z_MM,
    "VIAM_WORKSPACE_Z_MAX_MM": 150.0,
}

# Settings that must be supplied before any physical motion (no invented extrinsics).
REQUIRED_MOTION_SETTINGS = (
    "VIAM_APPROACH_CLEARANCE_MM",
    "VIAM_LIFT_MM",
    "VIAM_APPROACH_SPEED_DEGS_PER_SEC",
    "VIAM_DESCENT_STEP_MM",
    # Millimeters *above* the detected world Z (must be >= 0). Never lowers toward the table.
    "VIAM_GRASP_TCP_Z_OFFSET_MM",
    # Minimum TCP height above the configured table top (mm).
    "VIAM_MIN_TCP_ABOVE_TABLE_MM",
    "VIAM_GRASP_OX",
    "VIAM_GRASP_OY",
    "VIAM_GRASP_OZ",
    "VIAM_GRASP_THETA_DEG",
)

def setting(name: str, default: float | None = None) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        if default is None:
            raise RuntimeError(f"{name} is not set; refuse to continue without an explicit value.")
        return float(default)
    return float(raw)


def optional_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def fmt_pose(p: Pose) -> str:
    return (
        f"xyz=({p.x:.1f}, {p.y:.1f}, {p.z:.1f}) mm  "
        f"ov=({p.o_x:.3f}, {p.o_y:.3f}, {p.o_z:.3f}) theta={p.theta:.1f}"
    )


def required_motion_settings() -> dict[str, float]:
    missing = [n for n in REQUIRED_MOTION_SETTINGS if not os.getenv(n)]
    if missing:
        raise RuntimeError(
            "Motion refused: set these environment variables explicitly before "
            f"VIAM_PICK_ENABLE=1: {', '.join(missing)}"
        )
    values = {n: float(os.environ[n]) for n in REQUIRED_MOTION_SETTINGS}
    speed = values["VIAM_APPROACH_SPEED_DEGS_PER_SEC"]
    if not 3.0 <= speed <= 180.0:
        raise RuntimeError(
            "VIAM_APPROACH_SPEED_DEGS_PER_SEC must be between 3 and 180 "
            "(xArm module set_speed / speed_degs_per_sec range)."
        )
    if values["VIAM_DESCENT_STEP_MM"] <= 0:
        raise RuntimeError("VIAM_DESCENT_STEP_MM must be > 0.")
    if values["VIAM_APPROACH_CLEARANCE_MM"] <= 0 or values["VIAM_LIFT_MM"] <= 0:
        raise RuntimeError("VIAM_APPROACH_CLEARANCE_MM and VIAM_LIFT_MM must be > 0.")
    return values


def workspace_bounds() -> dict[str, float]:
    return {k: setting(k, v) for k, v in DEFAULT_WORKSPACE.items()}


def assert_in_workspace(x: float, y: float, z: float, bounds: dict[str, float]) -> None:
    if not (
        bounds["VIAM_WORKSPACE_X_MIN_MM"] <= x <= bounds["VIAM_WORKSPACE_X_MAX_MM"]
        and bounds["VIAM_WORKSPACE_Y_MIN_MM"] <= y <= bounds["VIAM_WORKSPACE_Y_MAX_MM"]
        and bounds["VIAM_WORKSPACE_Z_MIN_MM"] <= z <= bounds["VIAM_WORKSPACE_Z_MAX_MM"]
    ):
        raise RuntimeError(
            f"Detected target outside configured work area: ({x:.1f}, {y:.1f}, {z:.1f}) mm"
        )


def filter_color_detections(detections: Sequence[Detection], label: str, min_conf: float) -> list[Detection]:
    """Pick the largest in-label detection. Shape/size are not assumed (any blob of that color)."""
    matched = [
        d for d in detections
        if (not label or d.class_name == label) and d.confidence >= min_conf
    ]
    if not matched:
        raise RuntimeError(
            f"No unambiguous detection (label={label!r}, min_conf={min_conf}). "
            "Check lighting, color_detector settings, and camera view."
        )

    def area(d: Detection) -> float:
        return max(0.0, (d.x_max - d.x_min) * (d.y_max - d.y_min))

    matched = sorted(matched, key=area, reverse=True)
    if len(matched) > 1 and area(matched[1]) > area(matched[0]) * 0.60:
        raise RuntimeError(
            f"More than one similarly sized detection for label={label!r} "
            f"({len(matched)}); refusing to guess."
        )
    return matched


def grasp_orientation(cfg: dict[str, float]) -> dict[str, float]:
    return {
        "o_x": cfg["VIAM_GRASP_OX"],
        "o_y": cfg["VIAM_GRASP_OY"],
        "o_z": cfg["VIAM_GRASP_OZ"],
        "theta": cfg["VIAM_GRASP_THETA_DEG"],
    }


def make_pose(x: float, y: float, z: float, orientation: dict[str, float]) -> Pose:
    return Pose(x=x, y=y, z=z, **orientation)


async def with_timeout(awaitable, seconds: float, what: str):
    try:
        return await asyncio.wait_for(awaitable, timeout=seconds)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"Timed out after {seconds:.1f}s waiting for {what}") from exc


async def localize_red_cube(
    machine,
    cam_name: str,
    detector: VisionClient,
    segmenter: VisionClient,
    *,
    label: str,
    min_conf: float,
    vision_timeout: float,
) -> tuple[Detection, Pose, Pose, int]:
    """Return (detection, center_in_camera, center_in_world, point_cloud_bytes)."""
    detections = await with_timeout(
        detector.get_detections_from_camera(cam_name),
        vision_timeout,
        "color detections",
    )
    chosen = filter_color_detections(detections, label=label, min_conf=min_conf)[0]
    logger.info(
        "2D detection: label=%s conf=%.3f bbox=(%.0f,%.0f)-(%.0f,%.0f)",
        chosen.class_name,
        chosen.confidence,
        chosen.x_min,
        chosen.y_min,
        chosen.x_max,
        chosen.y_max,
    )

    objects = await with_timeout(
        segmenter.get_object_point_clouds(cam_name),
        vision_timeout,
        "3D object point clouds",
    )
    if not objects:
        raise RuntimeError(
            "Segmenter returned no 3D objects; depth/PCD projection failed. Refusing to move."
        )

    # Associate the 3D segment with the configured detector label. Never substitute
    # an unrelated/largest cloud: that could command a pick at the wrong object.
    labeled = [
        o for o in objects
        if o.geometries.geometries and o.geometries.geometries[0].label == label
    ]
    if not labeled:
        labels = [
            geometry.label
            for obj in objects
            for geometry in obj.geometries.geometries
        ]
        raise RuntimeError(
            f"No 3D segment matched detector label {label!r}; segment labels={labels!r}. "
            "Refusing to guess which object to pick."
        )
    candidates = labeled
    if len(candidates) > 1:
        sizes = sorted((len(o.point_cloud) for o in candidates), reverse=True)
        if len(sizes) > 1 and sizes[1] > sizes[0] * 0.60:
            raise RuntimeError(
                f"Segmenter found {len(candidates)} similar 3D objects; refusing to guess."
            )
    obj = max(candidates, key=lambda o: len(o.point_cloud))
    if len(obj.point_cloud) < optional_int("VIAM_MIN_POINT_CLOUD_BYTES", 50):
        raise RuntimeError(
            f"3D segment too small ({len(obj.point_cloud)} bytes); invalid depth. Refusing to move."
        )
    if not obj.geometries.geometries:
        raise RuntimeError("3D object has no geometry/center; refusing to move.")

    center_cam = obj.geometries.geometries[0].center
    if center_cam.z <= 0:
        raise RuntimeError(f"Invalid camera-frame depth z={center_cam.z:.1f} mm; refusing to move.")

    # Segmenter reports centers in the camera frame; transform with Viam's frame system.
    center_world = await with_timeout(
        machine.transform_pose(
            PoseInFrame(reference_frame=cam_name, pose=center_cam),
            "world",
        ),
        vision_timeout,
        "camera→world transform",
    )
    return chosen, center_cam, center_world.pose, len(obj.point_cloud)


async def move_gripper(
    motion: MotionClient,
    gripper_name: str,
    destination: Pose,
    world_state: WorldState,
    *,
    linear: bool,
    timeout: float,
    label: str,
    table_margin_mm: float,
) -> None:
    """Move the gripper TCP in world frame with workcell obstacles in WorldState.

    WorldState carries safety-* labeled copies of table / wall-front / wall-side
    (and ceiling). No gripper/table collision is whitelisted; the commanded TCP
    height must preserve the configured table clearance.
    """
    assert_tcp_above_table(destination.z, margin_mm=table_margin_mm, context=label)
    if linear:
        constraints = Constraints(
            linear_constraint=[
                LinearConstraint(line_tolerance_mm=2.0, orientation_tolerance_degs=5.0)
            ]
        )
    else:
        constraints = None
    ok = await with_timeout(
        motion.move(
            component_name=gripper_name,
            destination=PoseInFrame(reference_frame="world", pose=destination),
            world_state=world_state,
            constraints=constraints,
            timeout=timeout,
        ),
        timeout + 5.0,
        label,
    )
    if not ok:
        raise RuntimeError(f"Motion service reported failure for {label}.")


async def gentle_grab(gripper: Gripper, *, speed: int | None, torque: int | None) -> bool:
    """Close the gripper; prefer module gentle-close commands when configured."""
    if speed is not None:
        await with_timeout(
            gripper.do_command({"set_gripper_speed": float(speed)}),
            10.0,
            "set_gripper_speed",
        )
    if torque is not None:
        # G2 only; fall back to grab() if the hardware rejects torque commands.
        try:
            await with_timeout(
                gripper.do_command({"set_gripper_torque": int(torque)}),
                10.0,
                "set_gripper_torque",
            )
        except Exception as exc:  # noqa: BLE001 — hardware variance
            logger.warning("Gentle torque not applied (%s); continuing with gripper.grab().", exc)

    grabbed = await with_timeout(gripper.grab(), 30.0, "gripper.grab")
    if grabbed:
        return True
    holding = await with_timeout(gripper.is_holding_something(), 10.0, "is_holding_something")
    return bool(holding.is_holding_something)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cam_name = env("VIAM_CAMERA")
    arm_name = env("VIAM_ARM")
    gripper_name = env("VIAM_GRIPPER")
    detector_name = os.getenv("VIAM_DETECTOR", "red-object-detector")
    segmenter_name = os.getenv("VIAM_SEGMENTER", "red-object-segmenter")
    motion_name = os.getenv("VIAM_MOTION", "builtin")
    label = os.getenv("VIAM_DETECT_LABEL", "red")
    min_conf = setting("VIAM_MIN_DETECTION_CONFIDENCE", 0.5)
    vision_timeout = setting("VIAM_VISION_TIMEOUT_SEC", 30.0)
    move_timeout = setting("VIAM_MOVE_TIMEOUT_SEC", 60.0)
    maximal_depth_mm = setting("VIAM_MAX_CAMERA_DEPTH_MM", 2000.0)
    minimal_depth_mm = setting("VIAM_MIN_CAMERA_DEPTH_MM", 100.0)
    z_table_tolerance = setting("VIAM_TABLE_Z_TOLERANCE_MM", 100.0)

    bounds = workspace_bounds()
    pick_enabled = os.getenv("VIAM_PICK_ENABLE", "0") == "1"

    async with await connect() as machine:
        arm = Arm.from_robot(machine, arm_name)
        gripper = Gripper.from_robot(machine, gripper_name)
        detector = VisionClient.from_robot(machine, detector_name)
        segmenter = VisionClient.from_robot(machine, segmenter_name)
        motion = MotionClient.from_robot(machine, motion_name)

        detection, center_cam, center_world, cloud_bytes = await localize_red_cube(
            machine,
            cam_name,
            detector,
            segmenter,
            label=label,
            min_conf=min_conf,
            vision_timeout=vision_timeout,
        )

        if not minimal_depth_mm <= center_cam.z <= maximal_depth_mm:
            raise RuntimeError(
                f"Camera-frame depth {center_cam.z:.1f} mm outside "
                f"[{minimal_depth_mm}, {maximal_depth_mm}]; refusing to move."
            )

        x, y, z = center_world.x, center_world.y, center_world.z
        assert_in_workspace(x, y, z, bounds)

        # Object center must sit on or above the configured table top, within tolerance.
        if z < TABLE_TOP_Z_MM:
            raise RuntimeError(
                f"World Z {z:.1f} mm is below configured table top {TABLE_TOP_Z_MM:.1f} mm."
            )
        if z > TABLE_TOP_Z_MM + z_table_tolerance:
            raise RuntimeError(
                f"World Z {z:.1f} mm is more than {z_table_tolerance:.1f} mm above "
                f"table top {TABLE_TOP_Z_MM:.1f} mm; depth/calibration suspect."
            )

        logger.info("Target in camera frame: %s", fmt_pose(center_cam))
        logger.info(
            "Target in world frame: (%0.1f, %0.1f, %0.1f) mm  point_cloud_bytes=%d  detection=%s",
            x,
            y,
            z,
            cloud_bytes,
            detection.class_name,
        )

        if not pick_enabled:
            logger.info(
                "Preview only (VIAM_PICK_ENABLE!=1). Planned pick would use world "
                "target (%.1f, %.1f, %.1f) mm via motion service %r moving gripper %r.",
                x,
                y,
                z,
                motion_name,
                gripper_name,
            )
            logger.info(
                "Safety: motion will pass WorldState obstacles safety-table, "
                "safety-wall-front, safety-wall-side, safety-ceiling (mirrors machine "
                "table / wall-front / wall-side / ceiling) and refuse TCP Z below "
                "table top + VIAM_MIN_TCP_ABOVE_TABLE_MM."
            )
            logger.info(
                "Arm speed note: xArm joint speed is controlled by DoCommand "
                "{'set_speed': deg_per_sec} (3–180) or by speed_degs_per_sec in the "
                "Viam arm config. Sleep/pauses do not slow motion."
            )
            return

        cfg = required_motion_settings()
        orientation = grasp_orientation(cfg)
        table_margin = cfg["VIAM_MIN_TCP_ABOVE_TABLE_MM"]
        # Keep-out is optional; prefer 0. Table collision is never whitelisted.
        keep_out = setting("VIAM_TABLE_KEEP_OUT_MM", 0.0)
        if table_margin < 0:
            raise RuntimeError("VIAM_MIN_TCP_ABOVE_TABLE_MM must be >= 0")
        if keep_out < 0:
            raise RuntimeError("VIAM_TABLE_KEEP_OUT_MM must be >= 0")
        # Claws hang well below the TCP; a low margin lets the jaws hit the slab
        # even when the TCP itself is above the table top.
        if table_margin < 50.0:
            raise RuntimeError(
                "VIAM_MIN_TCP_ABOVE_TABLE_MM must be >= 50 (≈2 in above table top) "
                f"so the claws clear the table at close; got {table_margin:.1f}."
            )
        if cfg["VIAM_GRASP_TCP_Z_OFFSET_MM"] < 50.0:
            raise RuntimeError(
                "VIAM_GRASP_TCP_Z_OFFSET_MM must be >= 50 (≈2 in above detection) "
                f"so the close happens above the table; got "
                f"{cfg['VIAM_GRASP_TCP_Z_OFFSET_MM']:.1f}."
            )
        if keep_out > 0:
            logger.warning(
                "VIAM_TABLE_KEEP_OUT_MM=%.0f is set; prefer 0 for grasping.",
                keep_out,
            )

        # Grasp TCP stays at or above the detected surface (never toward the table).
        grasp_tcp_z = compute_grasp_tcp_z(
            z,
            offset_above_detection_mm=cfg["VIAM_GRASP_TCP_Z_OFFSET_MM"],
            table_margin_mm=table_margin,
        )
        z_floor = TABLE_TOP_Z_MM + table_margin

        clearance = cfg["VIAM_APPROACH_CLEARANCE_MM"]
        lift = cfg["VIAM_LIFT_MM"]
        step = cfg["VIAM_DESCENT_STEP_MM"]
        speed = cfg["VIAM_APPROACH_SPEED_DEGS_PER_SEC"]

        # World-vertical approach only (never dive along camera/tool axes into the table).
        pregrasp = make_pose(x, y, grasp_tcp_z + clearance, orientation)
        grasp_pose = make_pose(x, y, grasp_tcp_z, orientation)
        lift_pose = make_pose(x, y, grasp_tcp_z + lift, orientation)
        for name, pose in (
            ("pre-grasp", pregrasp),
            ("grasp", grasp_pose),
            ("lift", lift_pose),
        ):
            assert_tcp_above_table(pose.z, margin_mm=table_margin, context=f"planned {name}")

        # Table + walls + ceiling. No keep-out by default so claws can enter the cube.
        world_state = workcell_world_state(
            include_ceiling=True, table_keep_out_mm=keep_out
        )
        logger.info(
            "Motion WorldState obstacles: safety-table, safety-wall-front, "
            "safety-wall-side, safety-ceiling%s",
            f", safety-table-keepout({keep_out:.0f} mm)" if keep_out else "",
        )
        logger.info(
            "Detection Z=%.1f mm → grasp TCP Z=%.1f mm (offset +%.1f above detection); "
            "TCP floor=%.1f mm (table top %.1f + margin %.1f); no claws↔table allow",
            z,
            grasp_tcp_z,
            cfg["VIAM_GRASP_TCP_Z_OFFSET_MM"],
            z_floor,
            TABLE_TOP_Z_MM,
            table_margin,
        )
        logger.info("Planned pre-grasp: %s", fmt_pose(pregrasp))
        logger.info("Planned grasp:     %s", fmt_pose(grasp_pose))
        logger.info("Planned lift:      %s", fmt_pose(lift_pose))
        logger.info(
            "Setting xArm joint speed to %.1f deg/s via set_speed "
            "(also configurable as speed_degs_per_sec on the arm resource).",
            speed,
        )

        await with_timeout(arm.do_command({"set_speed": speed}), 10.0, "arm set_speed")

        gripper_speed = optional_int("VIAM_GRIPPER_SPEED", 0) or None
        gripper_torque = optional_int("VIAM_GRIPPER_TORQUE_PCT", 0) or None

        await with_timeout(gripper.open(), 30.0, "gripper.open")
        try:
            # Free plan to pre-grasp (obstacle-aware). No linear constraint so the
            # planner can route around table/walls instead of cutting through them.
            await move_gripper(
                motion,
                gripper_name,
                pregrasp,
                world_state,
                linear=False,
                timeout=move_timeout,
                label="pre-grasp",
                table_margin_mm=table_margin,
            )

            # Descend in small *world-Z* steps. Do NOT whitelist claws↔table: that
            # let the planner drive the jaws into the table. Grasp TCP is held high
            # enough (offset + min-above-table) that claws meet the cube, not the slab.
            remaining = clearance
            current_z = grasp_tcp_z + clearance
            while remaining > 1e-3:
                delta = min(step, remaining)
                current_z -= delta
                remaining -= delta
                if current_z < z_floor:
                    raise RuntimeError(
                        f"Descent would reach Z={current_z:.1f} mm below table floor "
                        f"{z_floor:.1f} mm; aborting."
                    )
                waypoint = make_pose(x, y, current_z, orientation)
                logger.info("Descent waypoint: %s", fmt_pose(waypoint))
                await move_gripper(
                    motion,
                    gripper_name,
                    waypoint,
                    world_state,
                    linear=True,
                    timeout=move_timeout,
                    label="descent step",
                    table_margin_mm=table_margin,
                )

            logger.info("At grasp pose; closing gripper, then lifting (hold, no release).")
            grabbed = await gentle_grab(
                gripper,
                speed=gripper_speed,
                torque=gripper_torque,
            )
            if not grabbed:
                logger.error("Grasp failed; retreating to pre-grasp without lifting.")
                await move_gripper(
                    motion,
                    gripper_name,
                    pregrasp,
                    world_state,
                    linear=True,
                    timeout=move_timeout,
                    label="retreat after failed grasp",
                    table_margin_mm=table_margin,
                )
                raise RuntimeError("Gripper did not confirm a grasp; cube was not lifted.")

            await move_gripper(
                motion,
                gripper_name,
                lift_pose,
                world_state,
                linear=True,
                timeout=move_timeout,
                label="lift",
                table_margin_mm=table_margin,
            )
            still_holding = await with_timeout(
                gripper.is_holding_something(),
                10.0,
                "post-lift is_holding_something",
            )
            if not still_holding.is_holding_something:
                raise RuntimeError("Object lost during/after lift; not treating pick as success.")

            logger.info("Pick complete; holding the cube at the lift pose (not releasing).")
        except Exception:
            try:
                await arm.stop()
            except Exception:  # noqa: BLE001
                logger.exception("Failed to stop arm during error handling")
            raise


if __name__ == "__main__":
    asyncio.run(main())
