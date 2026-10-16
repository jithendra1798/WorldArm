"""RGB-D localization and conservative pick/place helpers for MovingBlocks."""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.proto.common import (
    Geometry,
    GeometriesInFrame,
    Pose,
    PoseInFrame,
    RectangularPrism,
    Transform,
    Vector3,
    WorldState,
)
from viam.proto.service.motion import Constraints, LinearConstraint
from viam.services.motion import Motion

logger = logging.getLogger("moving_blocks")
TABLE_TOP_Z_MM = -23.0
DEPTH_HEADER_BYTES = 24


@dataclass
class DetectedObject:
    pixel: tuple[int, int]
    bbox: tuple[int, int, int, int]
    area_px: int
    center_world: Pose
    height_mm: float


@dataclass
class Zone:
    corners: np.ndarray
    interior: np.ndarray
    center_px: tuple[int, int]
    table_plane: tuple[float, float, float]
    depth_noise_mm: float
    objects: list[DetectedObject]


@dataclass
class Scene:
    depth: np.ndarray
    zones: list[Zone]
    intrinsics: Any


def value(options: Mapping[str, Any], key: str, default: Any) -> Any:
    raw = options.get(key, default)
    return default if raw is None or raw == "" else raw


def number(options: Mapping[str, Any], key: str, default: float) -> float:
    result = float(value(options, key, default))
    if not math.isfinite(result):
        raise RuntimeError(f"Configuration value {key!r} must be finite.")
    return result


def fmt_pose(pose: Pose) -> str:
    return f"xyz=({pose.x:.1f}, {pose.y:.1f}, {pose.z:.1f}) mm"


def workspace_bounds(options: Mapping[str, Any]) -> dict[str, float]:
    return {
        "x_min": number(options, "workspace_x_min_mm", -1400.0),
        "x_max": number(options, "workspace_x_max_mm", 690.0),
        "y_min": number(options, "workspace_y_min_mm", -450.0),
        "y_max": number(options, "workspace_y_max_mm", 1400.0),
        "z_min": number(options, "workspace_z_min_mm", TABLE_TOP_Z_MM),
        "z_max": number(options, "workspace_z_max_mm", 150.0),
    }


def assert_in_workspace(x: float, y: float, z: float, bounds: Mapping[str, float]) -> None:
    if not (bounds["x_min"] <= x <= bounds["x_max"]
            and bounds["y_min"] <= y <= bounds["y_max"]
            and bounds["z_min"] <= z <= bounds["z_max"]):
        raise RuntimeError(f"Target ({x:.1f}, {y:.1f}, {z:.1f}) mm is outside the configured workspace.")


def decode_color(data: bytes) -> np.ndarray:
    color = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if color is None:
        raise RuntimeError("Could not decode the camera color image.")
    return color


def decode_depth(data: bytes) -> np.ndarray:
    if len(data) < DEPTH_HEADER_BYTES:
        raise RuntimeError("Depth image payload is shorter than its Viam header.")
    width = int.from_bytes(data[8:16], "big")
    height = int.from_bytes(data[16:24], "big")
    expected = DEPTH_HEADER_BYTES + width * height * 2
    if width < 1 or height < 1 or len(data) < expected:
        raise RuntimeError("Depth image payload dimensions do not match its data.")
    return np.frombuffer(data[DEPTH_HEADER_BYTES:expected], dtype=">u2").reshape(height, width).astype(np.uint16)


def blue_tape_mask(color: np.ndarray, options: Mapping[str, Any]) -> np.ndarray:
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    low = np.array((int(value(options, "tape_h_min", 90)),
                    int(value(options, "tape_s_min", 45)),
                    int(value(options, "tape_v_min", 25))))
    high = np.array((int(value(options, "tape_h_max", 135)),
                     int(value(options, "tape_s_max", 255)),
                     int(value(options, "tape_v_max", 255))))
    mask = cv2.inRange(hsv, low, high)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def fit_table_depth_plane(
    depth: np.ndarray, color: np.ndarray, interior: np.ndarray, options: Mapping[str, Any]
) -> tuple[tuple[float, float, float], float]:
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 1] < int(value(options, "table_max_saturation", 100)))
             & (hsv[:, :, 2] > int(value(options, "table_min_value", 80))))
    ys, xs = np.nonzero((interior > 0) & white & (depth > 0))
    if xs.size > 12000:
        stride = int(math.ceil(xs.size / 12000))
        xs, ys = xs[::stride], ys[::stride]
    if xs.size < 100:
        raise RuntimeError("Too few tabletop depth samples inside one blue-tape zone.")
    z = depth[ys, xs].astype(np.float64)
    design = np.column_stack((xs.astype(np.float64), ys.astype(np.float64), np.ones(xs.size)))
    inv_z = 1000.0 / z
    inliers = np.ones(xs.size, dtype=bool)
    coeff = np.zeros(3, dtype=np.float64)
    for _ in range(5):
        if np.count_nonzero(inliers) < 100:
            raise RuntimeError("Could not robustly estimate the tabletop depth plane.")
        coeff, *_ = np.linalg.lstsq(design[inliers], inv_z[inliers], rcond=None)
        predicted = 1000.0 / np.maximum(design @ coeff, 1e-9)
        residual = z - predicted
        median = float(np.median(residual[inliers]))
        mad = float(np.median(np.abs(residual[inliers] - median)))
        inliers = np.abs(residual - median) <= max(3.0, 2.5 * 1.4826 * mad)
    residual = z - 1000.0 / np.maximum(design @ coeff, 1e-9)
    residual = residual[inliers]
    med = float(np.median(residual))
    noise = 1.4826 * float(np.median(np.abs(residual - med)))
    return tuple(float(v) for v in coeff), noise


def contour_center(contour: np.ndarray) -> tuple[int, int]:
    moments = cv2.moments(contour)
    if abs(moments["m00"]) < 1e-6:
        raise RuntimeError("Could not estimate the center of a blue-tape zone.")
    return int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])


def build_zones(color: np.ndarray, depth: np.ndarray, options: Mapping[str, Any]) -> list[Zone]:
    mask = blue_tape_mask(color, options)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    zones: list[Zone] = []
    min_area = number(options, "min_tape_zone_area_px", 5000.0)
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        (_, _), (rw, rh), _ = rect
        if min(rw, rh) < 40 or rw * rh < min_area:
            continue
        if max(rw, rh) / min(rw, rh) > number(options, "max_zone_aspect_ratio", 2.5):
            continue
        polygon = np.rint(cv2.boxPoints(rect)).astype(np.int32)
        box_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.fillConvexPoly(box_mask, polygon, 255)
        coverage = cv2.countNonZero(cv2.bitwise_and(mask, box_mask)) / max(1.0, rw * rh)
        if coverage < number(options, "min_tape_coverage", 0.025):
            continue
        rectangularity = abs(cv2.contourArea(contour)) / max(1.0, rw * rh)
        if rectangularity < number(options, "min_zone_rectangularity", 0.7):
            logger.info(
                "Ignoring non-rectangular blue region at (%.0f, %.0f), rectangularity %.2f.",
                rect[0][0], rect[0][1], rectangularity,
            )
            continue
        interior = np.zeros(mask.shape, dtype=np.uint8)
        cv2.fillConvexPoly(interior, polygon, 255)
        inset = int(number(options, "zone_inset_px", 12.0))
        if inset > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * inset + 1, 2 * inset + 1))
            interior = cv2.erode(interior, kernel)
        plane, noise = fit_table_depth_plane(depth, color, interior, options)
        zones.append(Zone(polygon, interior, contour_center(cv2.convexHull(polygon)), plane, noise, []))
    if len(zones) != 2:
        raise RuntimeError(f"Expected exactly two blue-tape work zones; detected {len(zones)}. Refusing to move.")
    return zones


def plane_depth_at(zone: Zone, u: int, v: int) -> float:
    a, b, c = zone.table_plane
    inv = a * u + b * v + c
    if inv <= 0:
        raise RuntimeError("Fitted tabletop depth plane is invalid at the target pixel.")
    return 1000.0 / inv


def component_center(mask: np.ndarray) -> tuple[int, int]:
    moments = cv2.moments(mask.astype(np.uint8))
    if abs(moments["m00"]) < 1e-6:
        raise RuntimeError("Depth component has no measurable center.")
    return int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])


def find_foreground_components(depth: np.ndarray, zone: Zone, options: Mapping[str, Any]):
    rows, cols = np.indices(depth.shape, dtype=np.float32)
    a, b, c = zone.table_plane
    predicted = 1000.0 / np.maximum(a * cols + b * rows + c, 1e-9)
    delta = max(number(options, "object_depth_delta_mm", 15.0), zone.depth_noise_mm * 4.0)
    fg = ((predicted - depth.astype(np.float32) >= delta) & (zone.interior > 0) & (depth > 0))
    mask = cv2.morphologyEx(fg.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    zone_area = max(1, cv2.countNonZero(zone.interior))
    minimum = int(number(options, "min_object_area_px", max(300, zone_area * 0.0015)))
    maximum_fraction = number(options, "max_object_zone_fraction", 0.35)
    objects = []
    for idx in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[idx])
        if area < minimum:
            continue
        if area > zone_area * maximum_fraction:
            raise RuntimeError("A merged foreground region is too large to safely treat as one object.")
        objects.append((x, y, w, h, area, labels[y:y+h, x:x+w] == idx))
    return sorted(objects, key=lambda obj: obj[4], reverse=True)


async def point_to_world(motion: Motion, camera_name: str, intrinsics: Any,
                         u: int, v: int, depth_mm: float) -> Pose:
    if not math.isfinite(depth_mm) or depth_mm <= 0:
        raise RuntimeError("Depth sample is invalid; refusing to target motion.")
    point = Pose(
        x=(u - intrinsics.center_x_px) * depth_mm / intrinsics.focal_x_px,
        y=(v - intrinsics.center_y_px) * depth_mm / intrinsics.focal_y_px,
        z=depth_mm, o_z=1,
    )
    detected = Transform(
        reference_frame="moving-blocks-detected-point",
        pose_in_observer_frame=PoseInFrame(reference_frame=camera_name, pose=point),
    )
    return (await motion.get_pose(
        component_name="moving-blocks-detected-point",
        destination_frame="world",
        supplemental_transforms=[detected],
    )).pose


async def localize_objects(motion: Motion, scene: Scene, camera_name: str,
                           zone: Zone, options: Mapping[str, Any]) -> list[DetectedObject]:
    found: list[DetectedObject] = []
    max_height = number(options, "max_object_height_mm", 300.0)
    max_table_error = number(options, "table_plane_tolerance_mm", 35.0)
    min_depth = number(options, "min_camera_depth_mm", 100.0)
    max_depth = number(options, "max_camera_depth_mm", 2000.0)
    for x, y, w, h, area, local_mask in find_foreground_components(scene.depth, zone, options):
        u0, v0 = component_center(local_mask)
        u, v = x + u0, y + v0
        roi = scene.depth[y:y+h, x:x+w]
        samples = roi[local_mask & (roi > 0)]
        if samples.size < 5:
            continue
        z_top = float(np.median(samples))
        if not min_depth <= z_top <= max_depth:
            raise RuntimeError(f"Object depth {z_top:.1f} mm is outside [{min_depth:.0f}, {max_depth:.0f}] mm.")
        z_table = plane_depth_at(zone, u, v)
        top = await point_to_world(motion, camera_name, scene.intrinsics, u, v, z_top)
        surface = await point_to_world(motion, camera_name, scene.intrinsics, u, v, z_table)
        if abs(surface.z - TABLE_TOP_Z_MM) > max_table_error:
            raise RuntimeError(f"Estimated tabletop Z={surface.z:.1f} mm conflicts with configured {TABLE_TOP_Z_MM:.1f} mm.")
        height = top.z - surface.z
        if not 3.0 <= height <= max_height:
            raise RuntimeError(f"Object height {height:.1f} mm is outside 3–{max_height:.0f} mm.")
        center = Pose(x=top.x, y=top.y, z=surface.z + height / 2, o_z=1)
        assert_in_workspace(center.x, center.y, center.z, workspace_bounds(options))
        found.append(DetectedObject((u, v), (x, y, w, h), area, center, height))
    return found


async def capture_scene(camera: Camera, motion: Motion, camera_name: str,
                        options: Mapping[str, Any]) -> Scene:
    images, _ = await camera.get_images(filter_source_names=["color", "depth"])
    by_name = {image.name: image for image in images}
    if "color" not in by_name or "depth" not in by_name:
        raise RuntimeError("Camera must return aligned 'color' and 'depth' images.")
    color, depth = decode_color(by_name["color"].data), decode_depth(by_name["depth"].data)
    if color.shape[:2] != depth.shape:
        raise RuntimeError("RGB and depth dimensions differ; refusing to use unaligned pixels.")
    intrinsics = (await camera.get_properties()).intrinsic_parameters
    if intrinsics is None or intrinsics.focal_x_px <= 0 or intrinsics.focal_y_px <= 0:
        raise RuntimeError("Camera intrinsics are invalid; refusing to calculate robot targets.")
    zones = build_zones(color, depth, options)
    scene = Scene(depth, zones, intrinsics)
    for zone in zones:
        zone.objects = await localize_objects(motion, scene, camera_name, zone, options)
    return scene


def drop_slot_pixels(zone: Zone, objects: list[DetectedObject], options: Mapping[str, Any]) -> list[tuple[int, int]]:
    (cx, cy), (rw, rh), angle = cv2.minAreaRect(zone.corners)
    if rw < rh:
        rw, rh, angle = rh, rw, angle + 90.0
    inset = number(options, "drop_slot_inset_px", 12.0)
    gap = number(options, "drop_slot_gap_px", 8.0)
    sizes = []
    for obj in objects:
        _, _, w, h = obj.bbox
        major, minor = max(w, h), min(w, h)
        footprint = math.sqrt(obj.area_px) if minor / max(major, 1) < 0.55 else major
        sizes.append(max(20.0, footprint))
    span = sum(sizes) + gap * max(0, len(sizes) - 1)
    if span > rw - 2 * inset:
        raise RuntimeError("Destination zone is too narrow for safe object spacing; refusing to move.")
    theta = math.radians(angle)
    points = []
    cursor = -span / 2
    contour = zone.corners.reshape(-1, 1, 2)
    for size in sizes:
        dx = cursor + size / 2
        u = int(round(cx + dx * math.cos(theta)))
        v = int(round(cy + dx * math.sin(theta)))
        if cv2.pointPolygonTest(contour, (float(u), float(v)), True) < size / 2 + inset:
            raise RuntimeError("Planned drop point is too close to the tape edge.")
        points.append((u, v))
        cursor += size + gap
    return points


async def zone_surface_pose(motion: Motion, scene: Scene, camera_name: str,
                            zone: Zone, pixel: tuple[int, int], options: Mapping[str, Any]) -> Pose:
    pose = await point_to_world(motion, camera_name, scene.intrinsics,
                                pixel[0], pixel[1], plane_depth_at(zone, *pixel))
    tolerance = number(options, "table_plane_tolerance_mm", 35.0)
    if abs(pose.z - TABLE_TOP_Z_MM) > tolerance:
        raise RuntimeError(f"Drop surface Z={pose.z:.1f} mm conflicts with configured table top.")
    return pose


def workcell_world_state(table_keep_out_mm: float) -> WorldState:
    if table_keep_out_mm < 0:
        raise ValueError("table_keep_out_mm must be >= 0")
    geoms = [
        Geometry(label="safety-table", center=Pose(x=0, y=0, z=-123, o_z=1),
                 box=RectangularPrism(dims_mm=Vector3(x=3000, y=3000, z=200))),
        Geometry(label="safety-wall-front", center=Pose(x=740, y=0, z=300, o_z=1),
                 box=RectangularPrism(dims_mm=Vector3(x=100, y=3000, z=1600))),
        Geometry(label="safety-wall-side", center=Pose(x=0, y=-500, z=300, o_z=1),
                 box=RectangularPrism(dims_mm=Vector3(x=3000, y=100, z=1600))),
        Geometry(label="safety-ceiling", center=Pose(x=0, y=0, z=1050, o_z=1),
                 box=RectangularPrism(dims_mm=Vector3(x=3000, y=3000, z=100))),
    ]
    if table_keep_out_mm:
        geoms.append(Geometry(
            label="safety-table-keepout",
            center=Pose(x=0, y=0, z=TABLE_TOP_Z_MM + table_keep_out_mm / 2, o_z=1),
            box=RectangularPrism(dims_mm=Vector3(x=3000, y=3000, z=table_keep_out_mm)),
        ))
    return WorldState(obstacles=[GeometriesInFrame(reference_frame="world", geometries=geoms)])


def assert_tcp_above_table(z_mm: float, margin_mm: float, context: str) -> None:
    floor = TABLE_TOP_Z_MM + margin_mm
    if z_mm < floor:
        raise RuntimeError(f"{context}: TCP Z={z_mm:.1f} mm is below safety floor {floor:.1f} mm.")


def grasp_z(detected_z: float, offset_mm: float, margin_mm: float) -> float:
    if offset_mm < 0:
        raise RuntimeError("grasp_tcp_z_offset_mm must be non-negative.")
    z = max(detected_z + offset_mm, detected_z)
    floor = TABLE_TOP_Z_MM + margin_mm
    if z < floor:
        raise RuntimeError(f"Grasp TCP Z={z:.1f} mm is below table safety floor {floor:.1f} mm.")
    return z


def surface_down_orientation(surface: Pose) -> dict[str, float]:
    """Use the camera-derived downward surface frame for destination IK.

    The measured machine's destination surface frame carries a useful yaw
    (theta≈82°). Replacing it with the grasp's fixed theta=0 made the xArm
    report zero IK solutions at the far blue zone even though the XYZ was
    detected correctly.
    """
    norm = math.sqrt(surface.o_x**2 + surface.o_y**2 + surface.o_z**2)
    if not 0.95 <= norm <= 1.05:
        raise RuntimeError("Destination surface orientation is invalid; refusing to pick before placement is checked.")
    ox, oy, oz = surface.o_x / norm, surface.o_y / norm, surface.o_z / norm
    if oz > -0.90:
        raise RuntimeError(
            f"Destination frame is not facing down (o_z={oz:.3f}); check camera/arm frame calibration."
        )
    return {"o_x": ox, "o_y": oy, "o_z": oz, "theta": float(surface.theta)}


async def with_timeout(awaitable, seconds: float, label: str):
    try:
        return await asyncio.wait_for(awaitable, timeout=seconds)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"Timed out after {seconds:.1f}s while {label}.") from exc


async def move_gripper(motion: Motion, gripper_name: str, pose: Pose, world_state: WorldState,
                       *, linear: bool, timeout: float, label: str, margin_mm: float) -> None:
    assert_tcp_above_table(pose.z, margin_mm, label)
    constraints = (Constraints(linear_constraint=[LinearConstraint(
        line_tolerance_mm=2.0, orientation_tolerance_degs=5.0)]) if linear else None)
    ok = await with_timeout(motion.move(
        component_name=gripper_name,
        destination=PoseInFrame(reference_frame="world", pose=pose),
        world_state=world_state,
        constraints=constraints,
        timeout=timeout,
    ), timeout + 5.0, label)
    if not ok:
        raise RuntimeError(f"Viam motion planning failed for {label}.")


async def gentle_grab(gripper: Gripper, speed: int, torque: int) -> bool:
    if speed > 0:
        await with_timeout(gripper.do_command({"set_gripper_speed": float(speed)}), 10.0, "setting gripper speed")
    if torque > 0:
        try:
            await with_timeout(gripper.do_command({"set_gripper_torque": int(torque)}), 10.0, "setting gripper torque")
        except Exception as exc:  # Hardware versions differ; retain the gripper's normal close behavior.
            logger.warning("Could not set gripper torque (%s); using normal grab().", exc)
    grabbed = await with_timeout(gripper.grab(), 30.0, "closing gripper")
    if grabbed:
        return True
    holding = await with_timeout(gripper.is_holding_something(), 10.0, "checking grasp")
    return bool(holding.is_holding_something)


async def move_one_object(obj: DetectedObject, destination: Pose, arm: Arm, gripper: Gripper,
                          motion: Motion, gripper_name: str, world_state: WorldState,
                          options: Mapping[str, Any], margin_mm: float, timeout: float) -> None:
    orientation = {
        "o_x": number(options, "grasp_ox", 0.0),
        "o_y": number(options, "grasp_oy", 0.0),
        "o_z": number(options, "grasp_oz", -1.0),
        "theta": number(options, "grasp_theta_deg", 0.0),
    }
    place_orientation = surface_down_orientation(destination)
    def pose(x: float, y: float, z: float) -> Pose:
        return Pose(x=x, y=y, z=z, **orientation)
    def place_pose(x: float, y: float, z: float) -> Pose:
        return Pose(x=x, y=y, z=z, **place_orientation)
    grasp = pose(obj.center_world.x, obj.center_world.y,
                 grasp_z(obj.center_world.z, number(options, "grasp_tcp_z_offset_mm", 65), margin_mm))
    clearance = number(options, "approach_clearance_mm", 100.0)
    lift = pose(grasp.x, grasp.y, grasp.z + number(options, "lift_mm", 100.0))
    pregrasp = pose(grasp.x, grasp.y, grasp.z + clearance)
    release_h = max(margin_mm, number(options, "place_release_height_mm", 100.0))
    place_z = max(destination.z + release_h, TABLE_TOP_Z_MM + margin_mm)
    # Preserve the camera-derived surface yaw at the destination. A fixed grasp
    # yaw (theta=0) caused the logged "zero IK solutions" on the far drop zone.
    place = place_pose(destination.x, destination.y, place_z)
    place_approach = place_pose(destination.x, destination.y,
                                max(place_z, destination.z + margin_mm + clearance))
    logger.info("Grasping object at %s (height %.1f mm); dropping at %s",
                fmt_pose(obj.center_world), obj.height_mm, fmt_pose(place))

    await with_timeout(gripper.open(), 30.0, "opening gripper")
    await move_gripper(motion, gripper_name, pregrasp, world_state, linear=False,
                       timeout=timeout, label="pre-grasp", margin_mm=margin_mm)
    step = number(options, "descent_step_mm", 25.0)
    if step <= 0:
        raise RuntimeError("descent_step_mm must be greater than zero.")
    z = pregrasp.z
    while z > grasp.z + 1e-3:
        z = max(grasp.z, z - step)
        await move_gripper(motion, gripper_name, pose(grasp.x, grasp.y, z), world_state,
                           linear=True, timeout=timeout, label="grasp descent", margin_mm=margin_mm)
    if not await gentle_grab(gripper, int(number(options, "gripper_speed", 800)),
                             int(number(options, "gripper_torque_pct", 40))):
        await move_gripper(motion, gripper_name, pregrasp, world_state, linear=True,
                           timeout=timeout, label="retreat after missed grasp", margin_mm=margin_mm)
        raise RuntimeError("Gripper did not confirm a grasp; stopping before lift.")
    await move_gripper(motion, gripper_name, lift, world_state, linear=True,
                       timeout=timeout, label="lift object", margin_mm=margin_mm)
    await move_gripper(motion, gripper_name, place_approach, world_state, linear=False,
                       timeout=timeout, label="travel above destination", margin_mm=margin_mm)
    await move_gripper(motion, gripper_name, place, world_state, linear=True,
                       timeout=timeout, label="lower into destination", margin_mm=margin_mm)
    await with_timeout(gripper.open(), 30.0, "releasing object")
    holding = await with_timeout(gripper.is_holding_something(), 10.0, "verifying release")
    if holding.is_holding_something:
        raise RuntimeError("Gripper still reports holding an object after release.")
    await move_gripper(motion, gripper_name, place_approach, world_state, linear=True,
                       timeout=timeout, label="retreat after release", margin_mm=margin_mm)
