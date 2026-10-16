"""Transfer arbitrary separated objects between two blue-tape work zones.

The program identifies two blue-tape rectangles in the RGB image, chooses the
one with depth foregrounds as the source, labels each object green, red,
orange, yellow, or black, and uses Viam's frame transforms and obstacle-aware
motion service. It does not assume objects are cubes or square. Preview-only
unless VIAM_PICK_ENABLE=1.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.proto.common import Pose, PoseInFrame
from viam.services.motion import MotionClient

from connection import connect, env
from perception import decode_color, decode_depth
from pick_red_cube import (
    assert_in_workspace,
    fmt_pose,
    make_pose,
    move_gripper,
    optional_int,
    required_motion_settings,
    setting,
    with_timeout,
    workspace_bounds,
)
from workcell_obstacles import TABLE_TOP_Z_MM, compute_grasp_tcp_z, workcell_world_state

logger = logging.getLogger("pick_place_cubes")
VIEW_POSE_PATH = Path(__file__).with_name(".pick_place_view_pose.json")


@dataclass
class Zone:
    corners: np.ndarray
    interior: np.ndarray
    center_px: tuple[int, int]
    table_plane: tuple[float, float, float]
    depth_noise_mm: float
    objects: list["DetectedObject"]


@dataclass
class Scene:
    depth: np.ndarray
    blue_tape: np.ndarray
    zones: list[Zone]
    intrinsics: object
    color: np.ndarray


@dataclass
class DetectedObject:
    pixel: tuple[int, int]
    bbox: tuple[int, int, int, int]
    area_px: int
    center_world: Pose
    height_mm: float
    point_mask: np.ndarray
    color_name: str = "unknown"


def blue_tape_mask(color: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    lower = np.array((
        int(setting("VIAM_TAPE_H_MIN", 90)),
        int(setting("VIAM_TAPE_S_MIN", 45)),
        int(setting("VIAM_TAPE_V_MIN", 25)),
    ))
    upper = np.array((
        int(setting("VIAM_TAPE_H_MAX", 135)),
        int(setting("VIAM_TAPE_S_MAX", 255)),
        int(setting("VIAM_TAPE_V_MAX", 255)),
    ))
    mask = cv2.inRange(hsv, lower, upper)
    # Bridge small tape seams, but preserve the gap between the two boxes.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def object_color_name(color: np.ndarray, mask: np.ndarray) -> str:
    """Label a depth object as green, red, orange, yellow, or black.

    Black is the gray cube. Its hue is unstable, so it is recognized by low
    saturation and low brightness instead of the Viam color detector.
    """
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    pixels = hsv[mask > 0]
    if pixels.size == 0:
        return "unknown"
    hue, sat, val = (float(v) for v in np.median(pixels, axis=0))
    if sat < setting("VIAM_BLACK_MAX_SATURATION", 70.0) and val < setting("VIAM_BLACK_MAX_VALUE", 90.0):
        return "black"
    if hue <= 4 or hue >= 170:
        return "red"
    if hue < 15:
        return "orange"
    if hue < 42:
        return "yellow"
    if hue < 95:
        return "green"
    return "unknown"


def contour_center(contour: np.ndarray) -> tuple[int, int]:
    moments = cv2.moments(contour)
    if abs(moments["m00"]) < 1e-6:
        raise RuntimeError("Could not estimate center of a blue-tape zone.")
    return int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])


def fit_table_depth_plane(
    depth: np.ndarray, color: np.ndarray, interior: np.ndarray
) -> tuple[tuple[float, float, float], float]:
    """Robustly fit camera depth over the mostly-white tabletop in a zone."""
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    white_table = (hsv[:, :, 1] < int(setting("VIAM_TABLE_MAX_SATURATION", 100))) & (
        hsv[:, :, 2] > int(setting("VIAM_TABLE_MIN_VALUE", 80))
    )
    valid = (interior > 0) & white_table & (depth > 0)
    ys, xs = np.nonzero(valid)
    if xs.size > 12000:
        stride = int(math.ceil(xs.size / 12000))
        xs, ys = xs[::stride], ys[::stride]
    if xs.size < 100:
        raise RuntimeError("Too few visible tabletop depth samples inside a tape box.")

    z = depth[ys, xs].astype(np.float64)
    inverse_z = 1000.0 / z
    design = np.column_stack((xs.astype(np.float64), ys.astype(np.float64), np.ones(xs.size)))
    inliers = np.ones(xs.size, dtype=bool)
    coefficients = np.zeros(3, dtype=np.float64)
    for _ in range(5):
        if np.count_nonzero(inliers) < 100:
            raise RuntimeError("Could not robustly estimate the tabletop depth plane.")
        coefficients, *_ = np.linalg.lstsq(design[inliers], inverse_z[inliers], rcond=None)
        predicted_inverse = design @ coefficients
        predicted_depth = 1000.0 / np.maximum(predicted_inverse, 1e-9)
        residual = z - predicted_depth
        median = float(np.median(residual[inliers]))
        mad = float(np.median(np.abs(residual[inliers] - median)))
        tolerance = max(3.0, 2.5 * 1.4826 * mad)
        inliers = np.abs(residual - median) <= tolerance
    final_residual = z - 1000.0 / np.maximum(design @ coefficients, 1e-9)
    final_residual = final_residual[inliers]
    residual_median = float(np.median(final_residual))
    noise_sigma = 1.4826 * float(np.median(np.abs(final_residual - residual_median)))
    return tuple(float(v) for v in coefficients), noise_sigma


def build_zones(color: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, list[Zone]]:
    mask = blue_tape_mask(color)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = setting("VIAM_MIN_TAPE_ZONE_AREA_PX", 5000.0)
    zones: list[Zone] = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        (_, _), (rw, rh), _ = rect
        if min(rw, rh) < 40:
            continue
        box_area = rw * rh
        if box_area < min_area:
            continue
        ratio = max(rw, rh) / min(rw, rh)
        if ratio > setting("VIAM_MAX_ZONE_ASPECT_RATIO", 2.5):
            continue
        # Measure blue pixels within the candidate box, rather than requiring a
        # filled-blue square: these are tape outlines around ordinary white table.
        polygon = np.rint(cv2.boxPoints(rect)).astype(np.int32)
        box_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.fillConvexPoly(box_mask, polygon, 255)
        tape_coverage = cv2.countNonZero(cv2.bitwise_and(mask, box_mask)) / max(1.0, box_area)
        if tape_coverage < setting("VIAM_MIN_TAPE_COVERAGE", 0.025):
            continue
        # A closed tape square's outer contour fills its box. The extra blue
        # region along the bottom of the frame is a smear, not a rectangle.
        rectangularity = abs(cv2.contourArea(contour)) / max(1.0, box_area)
        if rectangularity < setting("VIAM_MIN_ZONE_RECTANGULARITY", 0.7):
            logger.info(
                "Ignoring non-rectangular blue region at (%.0f, %.0f), rectangularity %.2f.",
                rect[0][0], rect[0][1], rectangularity,
            )
            continue

        inset = int(setting("VIAM_ZONE_INSET_PX", 12.0))
        interior = np.zeros(mask.shape, dtype=np.uint8)
        cv2.fillConvexPoly(interior, polygon, 255)
        if inset > 0:
            interior = cv2.erode(
                interior,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * inset + 1, 2 * inset + 1)),
            )
        center_px = contour_center(cv2.convexHull(polygon))
        plane, depth_noise_mm = fit_table_depth_plane(depth, color, interior)
        zones.append(Zone(polygon, interior, center_px, plane, depth_noise_mm, []))

    # Blue tape from other objects/background must not be guessed as a work zone.
    if len(zones) != 2:
        raise RuntimeError(
            f"Expected exactly two blue-tape zones, detected {len(zones)}. "
            "Check the camera view/tape thresholds; refusing to move."
        )
    return mask, zones


def plane_depth_at(zone: Zone, u: int, v: int) -> float:
    a, b, c = zone.table_plane
    inverse_depth = a * u + b * v + c
    if inverse_depth <= 0:
        raise RuntimeError("Fitted tabletop depth plane is invalid at this pixel.")
    return 1000.0 / inverse_depth


def component_center(component_mask: np.ndarray) -> tuple[int, int]:
    moments = cv2.moments(component_mask.astype(np.uint8))
    if abs(moments["m00"]) < 1e-6:
        raise RuntimeError("Object depth segment has no measurable center.")
    return int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])


def find_foreground_components(depth: np.ndarray, zone: Zone) -> list[tuple[int, int, int, int, int, np.ndarray]]:
    rows, cols = np.indices(depth.shape, dtype=np.float32)
    a, b, c = zone.table_plane
    predicted_inverse_depth = a * cols + b * rows + c
    predicted_table = 1000.0 / np.maximum(predicted_inverse_depth, 1e-9)
    # Require a change well above table noise so depth speckle does not look like
    # a pile of tiny objects. Still allow explicit tuning for the user's camera.
    delta = max(setting("VIAM_OBJECT_DEPTH_DELTA_MM", 15.0), zone.depth_noise_mm * 4.0)
    foreground = ((predicted_table - depth.astype(np.float32)) >= delta) & (zone.interior > 0) & (depth > 0)
    mask = foreground.astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    zone_area = max(1, cv2.countNonZero(zone.interior))
    # 242-pixel depth islands were detected in the otherwise-empty destination
    # zone in the latest run. Keep the floor above that speckle size; lower this
    # explicitly only if the workcell contains real objects that small.
    minimum = int(setting("VIAM_MIN_OBJECT_AREA_PX", max(300, zone_area * 0.0015)))
    maximum_fraction = setting("VIAM_MAX_OBJECT_ZONE_FRACTION", 0.35)
    objects = []
    for idx in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[idx])
        if area < minimum:
            continue
        if area > zone_area * maximum_fraction:
            raise RuntimeError(
                "A connected foreground region occupies too much of a tape box to "
                "safely treat as one object; separate objects and retry."
            )
        component = labels[y:y + h, x:x + w] == idx
        objects.append((x, y, w, h, area, component))
    objects.sort(key=lambda item: item[4], reverse=True)
    return objects


async def point_to_world(machine, camera_name: str, intrinsics, u: int, v: int, z_depth: float) -> Pose:
    if z_depth <= 0 or not math.isfinite(z_depth):
        raise RuntimeError("Depth sample is invalid; refusing to create a robot target.")
    p_cam = Pose(
        x=(u - intrinsics.center_x_px) * z_depth / intrinsics.focal_x_px,
        y=(v - intrinsics.center_y_px) * z_depth / intrinsics.focal_y_px,
        z=z_depth,
        o_z=1,
    )
    return (await machine.transform_pose(
        PoseInFrame(reference_frame=camera_name, pose=p_cam), "world"
    )).pose


async def localize_objects(machine, scene: Scene, camera_name: str, zone: Zone) -> list[DetectedObject]:
    found: list[DetectedObject] = []
    max_height = setting("VIAM_MAX_OBJECT_HEIGHT_MM", 300.0)
    # New overview poses shift the depth→world table estimate; keep this loose.
    max_table_error = setting("VIAM_TABLE_PLANE_TOLERANCE_MM", 55.0)
    for x, y, w, h, area, local_mask in find_foreground_components(scene.depth, zone):
        u0, v0 = component_center(local_mask)
        u, v = x + u0, y + v0
        mask = np.zeros(scene.depth.shape, dtype=np.uint8)
        mask[y:y + h, x:x + w] = local_mask.astype(np.uint8) * 255
        roi = scene.depth[y:y + h, x:x + w]
        samples = roi[local_mask & (roi > 0)]
        if samples.size < 5:
            continue
        object_depth = float(np.median(samples))
        table_depth = plane_depth_at(zone, u, v)
        min_depth = setting("VIAM_MIN_CAMERA_DEPTH_MM", 100.0)
        max_depth = setting("VIAM_MAX_CAMERA_DEPTH_MM", 2000.0)
        if not min_depth <= object_depth <= max_depth:
            raise RuntimeError(
                f"Object depth {object_depth:.1f} mm is outside the configured camera "
                f"range [{min_depth:.0f}, {max_depth:.0f}] mm; refusing to move."
            )
        top_world = await point_to_world(machine, camera_name, scene.intrinsics, u, v, object_depth)
        surface_world = await point_to_world(machine, camera_name, scene.intrinsics, u, v, table_depth)
        if abs(surface_world.z - TABLE_TOP_Z_MM) > max_table_error:
            raise RuntimeError(
                f"Table depth/frame estimate Z={surface_world.z:.1f} mm conflicts with "
                f"configured table top {TABLE_TOP_Z_MM:.1f} mm; refusing to move."
            )
        height_mm = top_world.z - surface_world.z
        if not 3.0 <= height_mm <= max_height:
            raise RuntimeError(
                f"Foreground object height {height_mm:.1f} mm is outside the supported "
                f"range 3–{max_height:.0f} mm. Refusing to guess a grasp height."
            )
        center = Pose(
            x=top_world.x,
            y=top_world.y,
            z=surface_world.z + height_mm / 2,
            o_x=0,
            o_y=0,
            o_z=1,
            theta=0,
        )
        assert_in_workspace(center.x, center.y, center.z, workspace_bounds())
        found.append(DetectedObject(
            (u, v), (x, y, w, h), area, center, height_mm, mask,
            object_color_name(scene.color, mask),
        ))
    return found


async def capture_scene(machine, camera: Camera, camera_name: str) -> Scene:
    images, _ = await camera.get_images(filter_source_names=["color", "depth"])
    by_name = {image.name: image for image in images}
    if "color" not in by_name or "depth" not in by_name:
        raise RuntimeError("Camera must provide aligned 'color' and 'depth' images.")
    color = decode_color(by_name["color"].data)
    depth = decode_depth(by_name["depth"].data)
    if color is None or color.shape[:2] != depth.shape:
        raise RuntimeError("RGB/depth dimensions do not match; refusing to use unaligned pixels.")
    tape_mask, zones = build_zones(color, depth)
    intrinsics = (await camera.get_properties()).intrinsic_parameters
    if intrinsics.focal_x_px <= 0 or intrinsics.focal_y_px <= 0:
        raise RuntimeError("Camera intrinsics are missing or invalid; refusing to move.")
    scene = Scene(depth, tape_mask, zones, intrinsics, color)
    for zone in zones:
        zone.objects = await localize_objects(machine, scene, camera_name, zone)
    return scene


def drop_slot_pixels(
    zone: Zone,
    objects: list[DetectedObject],
    prefer_toward: tuple[int, int] | None = None,
) -> list[tuple[int, int]]:
    """Compact drop grid, biased toward the source so later slots stay reachable."""
    rect = cv2.minAreaRect(zone.corners)
    (cx, cy), (rw, rh), angle = rect
    if rw < rh:
        rw, rh = rh, rw
        angle += 90.0
    inset = setting("VIAM_DROP_SLOT_INSET_PX", 28.0)
    gap = setting("VIAM_DROP_SLOT_GAP_PX", 36.0)
    n = len(objects)
    cols = 2 if n >= 3 else max(1, n)
    rows = int(math.ceil(n / cols))
    cell = max(28.0, gap + max(math.sqrt(obj.area_px) for obj in objects))
    grid_w = (cols - 1) * cell
    grid_h = (rows - 1) * cell
    if grid_w > rw - 2 * inset or grid_h > rh - 2 * inset:
        raise RuntimeError(
            f"Destination box is too small for {n} separated drop positions. "
            "Clear or enlarge the destination square and preview again."
        )

    # Bias the whole grid toward the source zone so the far edge is unused.
    ox, oy = 0.0, 0.0
    if prefer_toward is not None:
        px, py = prefer_toward
        theta = math.radians(angle)
        cos_a, sin_a = math.cos(theta), math.sin(theta)
        # Source offset in the destination box's local frame.
        dx = (px - cx) * cos_a + (py - cy) * sin_a
        dy = -(px - cx) * sin_a + (py - cy) * cos_a
        max_ox = max(0.0, (rw - 2 * inset - grid_w) / 2)
        max_oy = max(0.0, (rh - 2 * inset - grid_h) / 2)
        ox = max(-max_ox, min(max_ox, 0.55 * dx))
        oy = max(-max_oy, min(max_oy, 0.55 * dy))

    theta = math.radians(angle)
    cos_a, sin_a = math.cos(theta), math.sin(theta)
    contour = zone.corners.reshape(-1, 1, 2)
    points: list[tuple[int, int]] = []
    for idx in range(n):
        row, col = divmod(idx, cols) if cols > 1 else (0, idx)
        # Fill row-major starting at the near corner of the compact grid.
        local_x = ox - grid_w / 2 + col * cell
        local_y = oy - grid_h / 2 + row * cell
        u = int(round(cx + local_x * cos_a - local_y * sin_a))
        v = int(round(cy + local_x * sin_a + local_y * cos_a))
        if cv2.pointPolygonTest(contour, (float(u), float(v)), True) < inset:
            raise RuntimeError("A planned drop position is too close to the tape boundary.")
        points.append((u, v))
    return points


async def zone_surface_pose(machine, scene: Scene, camera_name: str, zone: Zone, pixel: tuple[int, int]) -> Pose:
    u, v = pixel
    pose = await point_to_world(
        machine, camera_name, scene.intrinsics, u, v, plane_depth_at(zone, u, v)
    )
    max_table_error = setting("VIAM_TABLE_PLANE_TOLERANCE_MM", 55.0)
    if abs(pose.z - TABLE_TOP_Z_MM) > max_table_error:
        raise RuntimeError(
            f"Destination table estimate Z={pose.z:.1f} mm conflicts with configured "
            f"table top {TABLE_TOP_Z_MM:.1f} mm; refusing to use this drop position."
        )
    return pose


def surface_down_orientation(surface: Pose) -> dict[str, float]:
    """Keep the camera-derived surface yaw so the far drop pose has solvable IK."""
    norm = math.sqrt(surface.o_x**2 + surface.o_y**2 + surface.o_z**2)
    if not 0.95 <= norm <= 1.05:
        raise RuntimeError("Destination surface orientation is invalid; refusing to move.")
    ox, oy, oz = surface.o_x / norm, surface.o_y / norm, surface.o_z / norm
    if oz > -0.90:
        raise RuntimeError(
            f"Destination frame is not facing down (o_z={oz:.3f}); check camera/arm frame calibration."
        )
    return {"o_x": ox, "o_y": oy, "o_z": oz, "theta": float(surface.theta)}


GRIPPER_FULL_OPEN = 850.0


async def gripper_position(gripper: Gripper) -> float:
    response = await with_timeout(gripper.do_command({"get": True}), 10.0, "read gripper position")
    raw = response.get("pos", response.get("position"))
    if raw is None:
        raise RuntimeError("Gripper did not report a position.")
    return float(raw)


async def set_gripper_position(gripper: Gripper, position: float, *, label: str) -> None:
    position = max(0.0, min(GRIPPER_FULL_OPEN, float(position)))
    await with_timeout(gripper.do_command({"set": position}), 20.0, label)


async def open_gripper_full(gripper: Gripper) -> None:
    await set_gripper_position(gripper, GRIPPER_FULL_OPEN, label="fully open gripper")


async def release_gripper_partial(gripper: Gripper, closed_position: float) -> None:
    """Open just enough to free the object; keep jaws narrow near neighbors."""
    extra = setting("VIAM_RELEASE_OPEN_EXTRA", 220.0)
    target = min(GRIPPER_FULL_OPEN, closed_position + extra)
    # Never release with a target at/below the closed pose.
    if target <= closed_position + 40.0:
        target = min(GRIPPER_FULL_OPEN, closed_position + 120.0)
    await set_gripper_position(gripper, target, label="partial release open")


async def move_to_place_approach(
    motion: MotionClient,
    gripper_name: str,
    destination: Pose,
    orientation: dict[str, float],
    place_approach_z: float,
    world_state,
    table_margin: float,
    timeout: float,
) -> Pose:
    """Travel above the drop; raise or nudge inward if a slot has no IK."""
    candidates = [
        place_approach_z,
        place_approach_z + 40.0,
        place_approach_z + 80.0,
        place_approach_z + 120.0,
    ]
    last_error: Exception | None = None
    for height in candidates:
        for dx, dy in ((0.0, 0.0), (-30.0, 0.0), (0.0, -30.0), (30.0, 0.0), (0.0, 30.0)):
            target = make_pose(destination.x + dx, destination.y + dy, height, orientation)
            try:
                await move_gripper(
                    motion, gripper_name, target, world_state, linear=False,
                    timeout=timeout, label="travel above destination zone",
                    table_margin_mm=table_margin,
                )
                return target
            except Exception as exc:  # noqa: BLE001 — try the next reachable hover pose
                last_error = exc
                continue
    assert last_error is not None
    raise last_error


async def move_one_object(
    obj: DetectedObject,
    destination: Pose,
    arm: Arm,
    gripper: Gripper,
    motion: MotionClient,
    gripper_name: str,
    world_state,
    cfg: dict[str, float],
    table_margin: float,
    timeout: float,
) -> None:
    orientation = {
        "o_x": cfg["VIAM_GRASP_OX"],
        "o_y": cfg["VIAM_GRASP_OY"],
        "o_z": cfg["VIAM_GRASP_OZ"],
        "theta": cfg["VIAM_GRASP_THETA_DEG"],
    }
    # Same TCP orientation for pick and place — camera surface yaw breaks IK.
    grasp_z = compute_grasp_tcp_z(
        obj.center_world.z,
        offset_above_detection_mm=cfg["VIAM_GRASP_TCP_Z_OFFSET_MM"],
        table_margin_mm=table_margin,
    )
    clearance = cfg["VIAM_APPROACH_CLEARANCE_MM"]
    floor = TABLE_TOP_Z_MM + table_margin
    pregrasp = make_pose(obj.center_world.x, obj.center_world.y, grasp_z + clearance, orientation)
    grasp = make_pose(obj.center_world.x, obj.center_world.y, grasp_z, orientation)
    lift = make_pose(obj.center_world.x, obj.center_world.y, grasp_z + cfg["VIAM_LIFT_MM"], orientation)
    # Stay high enough that opening jaws clear neighboring placed objects.
    release_height = max(
        table_margin + 40.0,
        setting("VIAM_PLACE_RELEASE_HEIGHT_MM", 120.0),
    )
    place_z = max(destination.z + release_height, floor)
    place = make_pose(destination.x, destination.y, place_z, orientation)
    place_approach_z = max(place_z + 30.0, destination.z + table_margin + clearance)

    logger.info("Object at %s, estimated height %.1f mm", fmt_pose(obj.center_world), obj.height_mm)
    logger.info("Destination %s", fmt_pose(destination))
    logger.info("Grasp TCP %s; release TCP %s", fmt_pose(grasp), fmt_pose(place))

    # Full open for picking so fingers clear neighboring source objects.
    await open_gripper_full(gripper)
    await move_gripper(motion, gripper_name, pregrasp, world_state, linear=False,
                       timeout=timeout, label="pre-grasp", table_margin_mm=table_margin)
    # One linear descent — fewer motion stops than stepped waypoints.
    await move_gripper(motion, gripper_name, grasp, world_state, linear=True,
                       timeout=timeout, label="grasp descent", table_margin_mm=table_margin)

    from pick_red_cube import gentle_grab
    grip_speed = optional_int("VIAM_GRIPPER_SPEED", 0) or None
    grip_torque = optional_int("VIAM_GRIPPER_TORQUE_PCT", 0) or None
    if not await gentle_grab(gripper, speed=grip_speed, torque=grip_torque):
        await move_gripper(motion, gripper_name, pregrasp, world_state, linear=True,
                           timeout=timeout, label="retreat after missed grasp",
                           table_margin_mm=table_margin)
        raise RuntimeError("Gripper did not confirm a grasp; stopping without lifting.")
    closed_position = await gripper_position(gripper)

    await move_gripper(motion, gripper_name, lift, world_state, linear=True,
                       timeout=timeout, label="lift object", table_margin_mm=table_margin)
    hover = await move_to_place_approach(
        motion, gripper_name, destination, orientation, place_approach_z,
        world_state, table_margin, timeout,
    )
    place_above = make_pose(hover.x, hover.y, place_z, orientation)
    await move_gripper(motion, gripper_name, place_above, world_state, linear=True,
                       timeout=timeout, label="lower into destination zone",
                       table_margin_mm=table_margin)
    # Partial open frees the object without swinging jaws into neighbors; lift
    # straight up before any sideways move; fully reopen once clear.
    await release_gripper_partial(gripper, closed_position)
    await move_gripper(motion, gripper_name, hover, world_state, linear=True,
                       timeout=timeout, label="retreat after release",
                       table_margin_mm=table_margin)
    holding = await with_timeout(gripper.is_holding_something(), 5.0, "verify release")
    if holding.is_holding_something:
        raise RuntimeError("Gripper still reports an object after opening; stopping the sequence.")
    await open_gripper_full(gripper)


def save_view_pose(pose: Pose) -> None:
    values = {
        key: float(getattr(pose, key))
        for key in ("x", "y", "z", "o_x", "o_y", "o_z", "theta")
    }
    VIEW_POSE_PATH.write_text(json.dumps(values, indent=2) + "\n")


def load_view_pose() -> Pose | None:
    if not VIEW_POSE_PATH.exists():
        return None
    try:
        values = json.loads(VIEW_POSE_PATH.read_text())
        return Pose(**{key: float(values[key]) for key in (
            "x", "y", "z", "o_x", "o_y", "o_z", "theta"
        )})
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise RuntimeError(
            f"Saved observation pose {VIEW_POSE_PATH} is invalid. "
            "Re-teach it with VIAM_SAVE_VIEW_POSE=1."
        ) from exc


def poses_match(a: Pose, b: Pose) -> bool:
    position_error = math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
    orientation_error = math.sqrt(
        (a.o_x - b.o_x) ** 2 + (a.o_y - b.o_y) ** 2 + (a.o_z - b.o_z) ** 2
    )
    return position_error <= 5.0 and orientation_error <= 0.02 and abs(a.theta - b.theta) <= 2.0


async def return_to_view_pose(
    machine,
    arm: Arm,
    motion: MotionClient,
    gripper_name: str,
    world_state,
    table_margin: float,
    timeout: float,
) -> None:
    target = load_view_pose()
    if target is None:
        raise RuntimeError(
            f"No saved observation pose. Put the arm in the desired overview position, "
            "then run once with VIAM_SAVE_VIEW_POSE=1 to teach it; that mode only reads "
            "and saves the pose and does not move the arm."
        )
    view_bounds = workspace_bounds()
    view_bounds["VIAM_WORKSPACE_Z_MAX_MM"] = setting("VIAM_VIEW_Z_MAX_MM", 800.0)
    assert_in_workspace(target.x, target.y, target.z, view_bounds)
    current = (await machine.get_pose(gripper_name, destination_frame="world")).pose
    if poses_match(current, target):
        logger.info("Arm is already at the saved observation pose.")
        return

    view_speed = setting("VIAM_VIEW_SPEED_DEGS_PER_SEC", 25.0)
    if not 3.0 <= view_speed <= 180.0:
        raise RuntimeError("VIAM_VIEW_SPEED_DEGS_PER_SEC must be between 3 and 180.")
    logger.info("Returning to the saved overview pose before viewing the scene: %s", fmt_pose(target))
    await with_timeout(arm.do_command({"set_speed": view_speed}), 10.0, "set observation speed")
    try:
        await move_gripper(
            motion,
            gripper_name,
            target,
            world_state,
            linear=False,
            timeout=timeout,
            label="return to observation pose",
            table_margin_mm=table_margin,
        )
    except Exception:
        try:
            await arm.stop()
        except Exception:  # noqa: BLE001
            logger.exception("Failed to stop arm after observation-pose motion error")
        raise


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    camera_name = env("VIAM_CAMERA")
    arm_name = env("VIAM_ARM")
    gripper_name = env("VIAM_GRIPPER")
    enabled = os.getenv("VIAM_PICK_ENABLE", "0") == "1"
    teach_view_pose = os.getenv("VIAM_SAVE_VIEW_POSE", "0") == "1"
    max_objects = optional_int("VIAM_MAX_OBJECTS_PER_RUN", 10)
    if max_objects < 1:
        raise RuntimeError("VIAM_MAX_OBJECTS_PER_RUN must be >= 1.")

    async with await connect() as machine:
        camera = Camera.from_robot(machine, camera_name)
        arm = Arm.from_robot(machine, arm_name)
        gripper = Gripper.from_robot(machine, gripper_name)
        motion = MotionClient.from_robot(machine, os.getenv("VIAM_MOTION", "builtin"))

        # Teaching is an explicit, read-only operation: the operator first places
        # the arm at the desired wide-view pose, then this saves that pose without
        # commanding any hardware.
        if teach_view_pose:
            holding = await with_timeout(gripper.is_holding_something(), 10.0, "check gripper before teaching")
            if holding.is_holding_something:
                raise RuntimeError("Gripper reports an object. Clear/recover it before teaching the view pose.")
            current = (await machine.get_pose(gripper_name, destination_frame="world")).pose
            save_view_pose(current)
            logger.info("Saved observation pose to %s (no motion commanded).", VIEW_POSE_PATH)
            logger.info("Saved pose: %s", fmt_pose(current))
            return

        cfg = None
        world_state = None
        timeout = setting("VIAM_MOVE_TIMEOUT_SEC", 60.0)
        table_margin = 0.0
        if enabled:
            cfg = required_motion_settings()
            table_margin = cfg["VIAM_MIN_TCP_ABOVE_TABLE_MM"]
            if table_margin < 50:
                raise RuntimeError("VIAM_MIN_TCP_ABOVE_TABLE_MM must be >= 50 for table clearance.")
            if cfg["VIAM_GRASP_TCP_Z_OFFSET_MM"] < 50:
                raise RuntimeError("VIAM_GRASP_TCP_Z_OFFSET_MM must be >= 50 for the current grasp setup.")
            world_state = workcell_world_state(
                include_ceiling=True,
                table_keep_out_mm=setting("VIAM_TABLE_KEEP_OUT_MM", 0.0),
            )
            holding = await with_timeout(gripper.is_holding_something(), 10.0, "check gripper before startup move")
            if holding.is_holding_something:
                raise RuntimeError(
                    "Gripper reports an object; refusing the startup move to the saved view pose. "
                    "Recover/release it safely before starting another transfer."
                )
            await return_to_view_pose(
                machine, arm, motion, gripper_name, world_state,
                table_margin, timeout,
            )
            await with_timeout(
                arm.do_command({"set_speed": cfg["VIAM_APPROACH_SPEED_DEGS_PER_SEC"]}),
                10.0,
                "set approach speed",
            )

        scene = await capture_scene(machine, camera, camera_name)
        counts = [len(zone.objects) for zone in scene.zones]
        if counts.count(0) != 1 or sum(counts) == 0:
            areas = [[obj.area_px for obj in zone.objects] for zone in scene.zones]
            raise RuntimeError(
                f"Expected one occupied source zone and one empty destination zone; "
                f"detected object counts {counts} and component areas (px) {areas}. "
                "Refusing to guess roles. If these are noise, increase "
                "VIAM_OBJECT_DEPTH_DELTA_MM or VIAM_MIN_OBJECT_AREA_PX."
            )
        source_index = 0 if counts[0] else 1
        destination_index = 1 - source_index
        source, destination_zone = scene.zones[source_index], scene.zones[destination_index]
        if len(source.objects) > max_objects:
            raise RuntimeError(
                f"Found {len(source.objects)} objects, above VIAM_MAX_OBJECTS_PER_RUN="
                f"{max_objects}; increase explicitly after reviewing preview."
            )

        logger.info("Found two blue-tape zones; source zone=%d (%d objects), destination zone=%d (empty).",
                    source_index + 1, len(source.objects), destination_index + 1)
        for idx, obj in enumerate(source.objects, 1):
            logger.info(
                "Source object %d: color=%s pixel=%s bbox=%s height≈%.1f mm center=%s",
                idx, obj.color_name, obj.pixel, obj.bbox, obj.height_mm, fmt_pose(obj.center_world),
            )

        slot_pixels = drop_slot_pixels(
            destination_zone, source.objects, prefer_toward=source.center_px,
        )
        destinations: list[Pose] = []
        for idx, slot in enumerate(slot_pixels, 1):
            pose = await zone_surface_pose(machine, scene, camera_name, destination_zone, slot)
            assert_in_workspace(pose.x, pose.y, pose.z, workspace_bounds())
            destinations.append(pose)
            logger.info("Destination slot %d: pixel=%s world=%s", idx, slot, fmt_pose(pose))

        if not enabled:
            logger.info("Preview only. Set VIAM_PICK_ENABLE=1 to execute the transfers.")
            return
        assert cfg is not None and world_state is not None
        try:
            for index, (obj, target) in enumerate(zip(source.objects, destinations), 1):
                await move_one_object(
                    obj, target, arm, gripper, motion, gripper_name,
                    world_state, cfg, table_margin, timeout,
                )
                logger.info("Transferred %d/%d object(s).", index, len(source.objects))
            logger.info("Transfer complete: all detected objects are in the other taped zone.")
        except Exception:
            try:
                await arm.stop()
            except Exception:  # noqa: BLE001
                logger.exception("Failed to stop arm after a transfer error")
            raise


if __name__ == "__main__":
    asyncio.run(main())
