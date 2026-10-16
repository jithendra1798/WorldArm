"""Transfer arbitrary separated objects between two blue-tape work zones.

The program returns to the saved overview pose, identifies two blue-tape
rectangles, chooses the occupied square as the source, labels each object
green, red, orange, yellow, or black, and transfers with recovery on misses
and drops. Set VIAM_PICK_ENABLE=0 for a view-only scan (still moves to overview).
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
from viam.proto.common import (
    Geometry,
    GeometriesInFrame,
    Pose,
    PoseInFrame,
    RectangularPrism,
    Vector3,
    WorldState,
)
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
    # Broken/partial tape (objects on the border) drops rectangularity a lot —
    # use a soft floor and rank by size; hard-reject only junk outlines.
    soft_rectangularity = setting("VIAM_MIN_ZONE_RECTANGULARITY", 0.18)
    bottom_reject_frac = setting("VIAM_TAPE_BOTTOM_REJECT_FRAC", 0.82)
    candidates: list[tuple[float, np.ndarray, tuple[int, int], tuple[float, float, float], float]] = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        (cx, cy), (rw, rh), _ = rect
        if min(rw, rh) < 40:
            continue
        box_area = rw * rh
        if box_area < min_area:
            continue
        ratio = max(rw, rh) / min(rw, rh)
        if ratio > setting("VIAM_MAX_ZONE_ASPECT_RATIO", 2.5):
            continue
        # Ignore the blue smear along the bottom of the camera frame.
        if cy > mask.shape[0] * bottom_reject_frac:
            logger.info("Ignoring bottom-frame blue smear at (%.0f, %.0f).", cx, cy)
            continue
        polygon = np.rint(cv2.boxPoints(rect)).astype(np.int32)
        box_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.fillConvexPoly(box_mask, polygon, 255)
        tape_coverage = cv2.countNonZero(cv2.bitwise_and(mask, box_mask)) / max(1.0, box_area)
        if tape_coverage < setting("VIAM_MIN_TAPE_COVERAGE", 0.025):
            continue
        rectangularity = abs(cv2.contourArea(contour)) / max(1.0, box_area)
        if rectangularity < soft_rectangularity:
            logger.info(
                "Ignoring junk blue region at (%.0f, %.0f), rectangularity %.2f.",
                cx, cy, rectangularity,
            )
            continue
        if rectangularity < 0.30:
            logger.info(
                "Keeping partial tape box at (%.0f, %.0f), rectangularity %.2f (soft accept).",
                cx, cy, rectangularity,
            )

        inset = int(setting("VIAM_ZONE_INSET_PX", 12.0))
        interior = np.zeros(mask.shape, dtype=np.uint8)
        cv2.fillConvexPoly(interior, polygon, 255)
        if inset > 0:
            interior = cv2.erode(
                interior,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * inset + 1, 2 * inset + 1)),
            )
        try:
            plane, depth_noise_mm = fit_table_depth_plane(depth, color, interior)
        except RuntimeError as exc:
            logger.info("Skipping blue region at (%.0f, %.0f): %s", cx, cy, exc)
            continue
        center_px = contour_center(cv2.convexHull(polygon))
        # Prefer large tape boxes; rectangularity is only a soft preference.
        score = box_area * (0.35 + rectangularity) * (0.5 + tape_coverage)
        candidates.append((score, polygon, center_px, plane, depth_noise_mm))

    candidates.sort(key=lambda item: item[0], reverse=True)
    chosen: list[Zone] = []
    for score, polygon, center_px, plane, depth_noise_mm in candidates:
        if any(math.hypot(center_px[0] - z.center_px[0], center_px[1] - z.center_px[1]) < 120 for z in chosen):
            continue
        interior = np.zeros(mask.shape, dtype=np.uint8)
        cv2.fillConvexPoly(interior, polygon, 255)
        inset = int(setting("VIAM_ZONE_INSET_PX", 12.0))
        if inset > 0:
            interior = cv2.erode(
                interior,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * inset + 1, 2 * inset + 1)),
            )
        chosen.append(Zone(polygon, interior, center_px, plane, depth_noise_mm, []))
        if len(chosen) == 2:
            break

    if len(chosen) != 2:
        raise RuntimeError(
            f"Expected exactly two blue-tape zones, detected {len(chosen)}. "
            "Check the camera view/tape thresholds; refusing to move."
        )
    return mask, chosen


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


def depth_patch_clear(
    depth: np.ndarray,
    zone: Zone,
    u: int,
    v: int,
    *,
    radius_px: int | None = None,
) -> bool:
    """True when a depth patch at (u,v) is table-flat (no object sticking up)."""
    if radius_px is None:
        radius_px = int(setting("VIAM_DROP_DEPTH_CLEAR_RADIUS_PX", 12.0))
    delta = setting("VIAM_OBJECT_DEPTH_DELTA_MM", 15.0)
    h, w = depth.shape
    x0, x1 = max(0, u - radius_px), min(w, u + radius_px + 1)
    y0, y1 = max(0, v - radius_px), min(h, v + radius_px + 1)
    patch = depth[y0:y1, x0:x1]
    interior = zone.interior[y0:y1, x0:x1]
    valid = (patch > 0) & (interior > 0)
    if not np.any(valid):
        return False
    rows, cols = np.indices(patch.shape, dtype=np.float32)
    a, b, c = zone.table_plane
    pred = 1000.0 / np.maximum(a * (cols + x0) + b * (rows + y0) + c, 1e-9)
    raised = (pred - patch.astype(np.float32)) >= delta
    # Allow a little noise; reject if a meaningful fraction is elevated.
    frac = float(np.count_nonzero(raised & valid)) / float(np.count_nonzero(valid))
    return frac < setting("VIAM_DROP_DEPTH_MAX_RAISED_FRAC", 0.12)


def object_obstacle_geometry(obj: DetectedObject, *, index: int, pad_mm: float) -> Geometry:
    """Axis-aligned keep-out box from a depth-localized object."""
    # Footprint from bbox area scaled roughly at table distance; pad for open jaws.
    side = max(35.0, math.sqrt(max(obj.area_px, 1)) * setting("VIAM_DEPTH_PX_TO_MM", 0.65))
    side = min(side + pad_mm, setting("VIAM_DEPTH_OBSTACLE_MAX_XY_MM", 120.0))
    height = max(obj.height_mm + pad_mm, 40.0)
    # Center sits at object mid-height so the box covers table→top.
    return Geometry(
        label=f"depth-object-{index}-{obj.color_name}",
        center=Pose(
            x=obj.center_world.x,
            y=obj.center_world.y,
            z=TABLE_TOP_Z_MM + height / 2.0,
            o_z=1,
        ),
        box=RectangularPrism(dims_mm=Vector3(x=side, y=side, z=height)),
    )


def _obstacle_excluded(
    obj: DetectedObject,
    *,
    exclude: DetectedObject | None,
    exclude_pixels: set[tuple[int, int]] | None,
) -> bool:
    if exclude is not None and (obj is exclude or obj.pixel == exclude.pixel):
        return True
    if exclude_pixels is not None and obj.pixel in exclude_pixels:
        return True
    return False


def world_state_with_depth_obstacles(
    base: WorldState,
    objects: list[DetectedObject],
    *,
    exclude: DetectedObject | None = None,
    exclude_pixels: set[tuple[int, int]] | None = None,
    pad_mm: float | None = None,
) -> WorldState:
    """Merge static workcell obstacles with depth-derived object keep-outs."""
    pad = setting("VIAM_DEPTH_OBSTACLE_PAD_MM", 30.0) if pad_mm is None else pad_mm
    extras: list[Geometry] = []
    for idx, obj in enumerate(objects):
        if _obstacle_excluded(obj, exclude=exclude, exclude_pixels=exclude_pixels):
            continue
        extras.append(object_obstacle_geometry(obj, index=idx, pad_mm=pad))
    if not extras:
        return base
    frames = list(base.obstacles)
    frames.append(GeometriesInFrame(reference_frame="world", geometries=extras))
    return WorldState(obstacles=frames)


def drop_slot_pixels(
    zone: Zone,
    objects: list[DetectedObject],
    prefer_toward: tuple[int, int] | None = None,
    occupied: list[DetectedObject] | None = None,
    tape_mask: np.ndarray | None = None,
    depth: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Spread drop slots inside the square interior — never on the blue tape border."""
    if not objects:
        return []
    # Plan inside the eroded interior (past the tape), not on the outer box edge.
    ys, xs = np.where(zone.interior > 0)
    if len(xs) < 50:
        raise RuntimeError("Destination square interior is too small for drop planning.")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    # Extra margin so object footprints stay off the tape lines.
    margin = int(setting("VIAM_DROP_SLOT_INSET_PX", 55.0))
    gap = setting("VIAM_DROP_SLOT_GAP_PX", 52.0)
    keepout = setting("VIAM_DROP_OCCUPIED_KEEPOUT_PX", 55.0)
    n = len(objects)
    occupied_count = len(occupied or [])
    inner_w = max(1, x1 - x0 - 2 * margin)
    inner_h = max(1, y1 - y0 - 2 * margin)
    cols = max(2, int(math.ceil(math.sqrt(n + occupied_count))) + 1)
    rows = max(2, int(math.ceil((n + occupied_count) / cols)) + 1)
    cell_x = max(gap * 0.6, inner_w / max(cols, 1))
    cell_y = max(gap * 0.6, inner_h / max(rows, 1))

    # Dilate tape so candidates near the blue border are rejected.
    tape_keepout = None
    if tape_mask is not None:
        radius = int(setting("VIAM_DROP_TAPE_KEEPOUT_PX", 14.0))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1),
        )
        tape_keepout = cv2.dilate(tape_mask, kernel)

    occupied_px = [obj.pixel for obj in (occupied or [])]
    zcx, zcy = zone.center_px
    candidates: list[tuple[float, int, int]] = []
    for row in range(rows):
        for col in range(cols):
            u = int(round(x0 + margin + (col + 0.5) * cell_x))
            v = int(round(y0 + margin + (row + 0.5) * cell_y))
            if not (0 <= v < zone.interior.shape[0] and 0 <= u < zone.interior.shape[1]):
                continue
            if zone.interior[v, u] == 0:
                continue
            if tape_keepout is not None and tape_keepout[v, u] > 0:
                continue
            if any(math.hypot(u - ox, v - oy) < keepout for ox, oy in occupied_px):
                continue
            # Depth must show an empty table patch — not another cube or the lip.
            if depth is not None and not depth_patch_clear(depth, zone, u, v):
                continue
            # Prefer the open center of the square; lightly bias toward the source.
            center_dist = math.hypot(u - zcx, v - zcy)
            toward = 0.0
            if prefer_toward is not None:
                toward = 0.15 * math.hypot(u - prefer_toward[0], v - prefer_toward[1])
            candidates.append((center_dist + toward, u, v))

    candidates.sort(key=lambda item: item[0])
    points: list[tuple[int, int]] = []
    blocked = list(occupied_px)
    pool = [(u, v) for _, u, v in candidates]
    while len(points) < n and pool:
        best = None
        best_score = -1.0
        for cand in pool:
            if any(math.hypot(cand[0] - bx, cand[1] - by) < keepout for bx, by in blocked):
                continue
            score = (
                1e6 if not blocked
                else min(math.hypot(cand[0] - bx, cand[1] - by) for bx, by in blocked)
            )
            # Prefer spread; among equals prefer center (pool already center-sorted).
            if score > best_score:
                best_score = score
                best = cand
        if best is None:
            break
        points.append(best)
        blocked.append(best)
        pool = [c for c in pool if c != best]

    if len(points) < n:
        raise RuntimeError(
            f"Could only plan {len(points)}/{n} interior drop slots clear of tape/depth. "
            "Spread or clear objects and retry."
        )
    return points


def choose_source_and_destination(
    zones: list[Zone],
    locked_destination_center: tuple[int, int] | None,
) -> tuple[Zone, Zone]:
    """Pick source/destination; lock destination across recovery rescans."""
    if locked_destination_center is not None:
        destination = min(
            zones,
            key=lambda zone: math.hypot(
                zone.center_px[0] - locked_destination_center[0],
                zone.center_px[1] - locked_destination_center[1],
            ),
        )
        source = next(zone for zone in zones if zone is not destination)
        return source, destination

    counts = [len(zone.objects) for zone in zones]
    if sum(counts) == 0:
        raise RuntimeError("No objects detected in either blue-tape square.")
    if counts.count(0) == 1:
        source_index = 0 if counts[0] else 1
        return zones[source_index], zones[1 - source_index]
    # Both squares already have objects (mid-recovery). Move from the fuller one.
    source_index = 0 if counts[0] >= counts[1] else 1
    logger.info(
        "Both squares contain objects (counts %s); clearing zone %d into zone %d.",
        counts, source_index + 1, 2 - source_index,
    )
    return zones[source_index], zones[1 - source_index]


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


@dataclass(frozen=True)
class GripperGeom:
    """armfarm05 xArm850 + UFactory gripper — matched to Viam's pick-and-place workshop.

    Workshop reference (viam-devrel/pick-and-place):
      GRIPPER_LENGTH_MM = 60   # TCP stops this far above the contact point
      APPROACH_MM = 100        # standoff above the block before descending
    Machine config: gripper frame z=105 mm from the arm flange.
    """

    flange_to_tcp_mm: float = 105.0
    # Workshop GRIPPER_LENGTH: TCP above fingertip contact at cube center.
    tcp_to_fingertip_mm: float = 60.0
    # Workshop APPROACH standoff above the object center before open+descend.
    approach_above_center_mm: float = 100.0
    max_opening_mm: float = 84.0
    finger_pad_mm: float = 10.0
    settle_s: float = 0.3


def gripper_geom() -> GripperGeom:
    return GripperGeom(
        flange_to_tcp_mm=setting("VIAM_GRIPPER_FLANGE_TO_TCP_MM", 105.0),
        tcp_to_fingertip_mm=setting("VIAM_GRIPPER_TCP_TO_TIP_MM", 60.0),
        approach_above_center_mm=setting("VIAM_APPROACH_ABOVE_CENTER_MM", 100.0),
        max_opening_mm=setting("VIAM_GRIPPER_MAX_OPENING_MM", 84.0),
        finger_pad_mm=setting("VIAM_GRIPPER_FINGER_PAD_MM", 10.0),
        settle_s=setting("VIAM_GRIPPER_SETTLE_S", 0.3),
    )


def jaw_opening_mm(position: float, geom: GripperGeom | None = None) -> float:
    g = geom or gripper_geom()
    return max(0.0, min(GRIPPER_FULL_OPEN, position)) / GRIPPER_FULL_OPEN * g.max_opening_mm


def jaw_half_span_mm(position: float, geom: GripperGeom | None = None) -> float:
    g = geom or gripper_geom()
    return jaw_opening_mm(position, g) / 2.0 + g.finger_pad_mm


def object_half_width_mm(obj: DetectedObject) -> float:
    """Approximate planar half-width from depth footprint."""
    return max(15.0, 0.5 * math.sqrt(max(obj.area_px, 1)) * setting("VIAM_DEPTH_PX_TO_MM", 0.65))


def grasp_tcp_z_for_object(obj: DetectedObject, table_margin_mm: float) -> tuple[float, float, float]:
    """Workshop rule: TCP stops GRIPPER_LENGTH above the cube center (contact point).

    Returns (grasp_tcp_z, contact_z, top_z).
    """
    g = gripper_geom()
    # center_world is already mid-height — that is the fingertip contact target.
    contact_z = obj.center_world.z
    top_z = contact_z + obj.height_mm / 2.0
    # Slightly lower than classic mid for a firmer seat without table hits.
    contact_z = contact_z - setting("VIAM_GRASP_LOWER_MM", 4.0)
    grasp_z = compute_grasp_tcp_z(
        contact_z,
        offset_above_detection_mm=g.tcp_to_fingertip_mm,
        table_margin_mm=table_margin_mm,
    )
    return grasp_z, contact_z, top_z


def pick_open_position_for_scene(
    obj: DetectedObject,
    neighbors: list[DetectedObject],
) -> float:
    """Open generously for a clean pick; clamp only when a neighbor would be hit."""
    g = gripper_geom()
    need_opening = (
        2.0 * object_half_width_mm(obj)
        + setting("VIAM_PICK_OPEN_MARGIN_MM", 20.0)
    )
    preferred = need_opening + setting("VIAM_PICK_OPEN_EXTRA_MM", 14.0)
    max_opening = g.max_opening_mm
    if neighbors:
        nearest = min(neighbors, key=lambda other: planar_distance_mm(obj, other))
        gap = planar_distance_mm(obj, nearest)
        # Yawed side approach: jaws open across the clear axis, not into the neighbor.
        clear = gap - object_half_width_mm(nearest) - g.finger_pad_mm - setting(
            "VIAM_NEIGHBOR_JAW_CLEAR_MM", 6.0,
        )
        max_safe = max(0.0, 2.0 * clear)
        if max_safe < preferred:
            logger.warning(
                "Neighbor gap %.1f mm limits jaw open (prefer %.1f mm, safe≤%.1f mm).",
                gap, preferred, max_safe,
            )
            preferred = max(setting("VIAM_PICK_OPEN_MIN_MM", 45.0), min(preferred, max_safe))
        max_opening = min(max_opening, max(preferred, max_safe))
    opening = min(max(preferred, setting("VIAM_PICK_OPEN_MIN_MM", 45.0)), max_opening)
    position = opening / g.max_opening_mm * GRIPPER_FULL_OPEN
    cap = setting("VIAM_PICK_OPEN_POSITION", 640.0)
    return max(320.0, min(cap, position))


def grasp_orientation_for_neighbors(
    obj: DetectedObject,
    neighbors: list[DetectedObject],
    base: dict[str, float],
) -> dict[str, float]:
    """Yaw jaws so they open across the clear axis (not into the nearest neighbor)."""
    orientation = dict(base)
    if not neighbors:
        return orientation
    nearest = min(neighbors, key=lambda other: planar_distance_mm(obj, other))
    dx = obj.center_world.x - nearest.center_world.x
    dy = obj.center_world.y - nearest.center_world.y
    if math.hypot(dx, dy) < 1.0:
        return orientation
    # Fingers approach along the clear vector; jaws open perpendicular to it.
    orientation["theta"] = math.degrees(math.atan2(dy, dx)) + 90.0
    return orientation


async def gripper_position(gripper: Gripper) -> float:
    response = await with_timeout(gripper.do_command({"get": True}), 10.0, "read gripper position")
    raw = response.get("pos", response.get("position"))
    if raw is None:
        raise RuntimeError("Gripper did not report a position.")
    return float(raw)


async def set_gripper_position(gripper: Gripper, position: float, *, label: str) -> None:
    position = max(0.0, min(GRIPPER_FULL_OPEN, float(position)))
    await with_timeout(gripper.do_command({"set": position}), 20.0, label)


async def open_gripper_for_pick(
    gripper: Gripper,
    position: float | None = None,
) -> None:
    """Open just wide enough for this pick — never full open near neighbors."""
    target = position if position is not None else setting("VIAM_PICK_OPEN_POSITION", 520.0)
    target = max(280.0, min(GRIPPER_FULL_OPEN, target))
    await set_gripper_position(gripper, target, label="narrow open for pick")


async def release_gripper_partial(gripper: Gripper, closed_position: float) -> None:
    """Open just enough to free the object; keep jaws narrow near neighbors."""
    extra = setting("VIAM_RELEASE_OPEN_EXTRA", 180.0)
    pick_open = setting("VIAM_PICK_OPEN_POSITION", 520.0)
    target = min(pick_open, closed_position + extra)
    # Never release with a target at/below the closed pose.
    if target <= closed_position + 40.0:
        target = min(pick_open, closed_position + 120.0)
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
    """Travel above the drop with few waypoints; raise only if IK fails."""
    # Prefer a single free-space flight to the primary hover; fall back sparsely.
    candidates = [
        (0.0, 0.0, place_approach_z),
        (0.0, 0.0, place_approach_z + 50.0),
        (0.0, 0.0, place_approach_z + 100.0),
        (-25.0, 0.0, place_approach_z + 50.0),
        (0.0, -25.0, place_approach_z + 50.0),
    ]
    last_error: Exception | None = None
    for dx, dy, height in candidates:
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


def planar_distance_mm(a: DetectedObject, b: DetectedObject) -> float:
    return math.hypot(a.center_world.x - b.center_world.x, a.center_world.y - b.center_world.y)


def order_picks_outer_first(objects: list[DetectedObject]) -> list[DetectedObject]:
    """Pick the most isolated objects first so the open gripper clears neighbors."""
    if len(objects) <= 1:
        return list(objects)

    def isolation(obj: DetectedObject) -> float:
        others = [planar_distance_mm(obj, other) for other in objects if other is not obj]
        return min(others) if others else 1e9

    return sorted(objects, key=isolation, reverse=True)


def neighbor_clearance_mm(
    obj: DetectedObject,
    neighbors: list[DetectedObject],
    pick_open_position: float,
) -> tuple[float, float, float]:
    """Return (approach_xy_offset, hover_clearance, nearest_gap) using gripper jaw span."""
    g = gripper_geom()
    half_span = jaw_half_span_mm(pick_open_position, g)
    if not neighbors:
        return 0.0, setting("VIAM_NEIGHBOR_HEIGHT_CLEARANCE_MM", 70.0), 1e9
    nearest = min(neighbors, key=lambda other: planar_distance_mm(obj, other))
    gap = planar_distance_mm(obj, nearest)
    neighbor_half = object_half_width_mm(nearest)
    # Open jaws occupy ±half_span about the approach path — stay clear of neighbor body.
    needed_lateral = half_span + neighbor_half + setting("VIAM_NEIGHBOR_JAW_CLEAR_MM", 8.0)
    # Hover: fingertips hang tcp_to_tip below TCP; clear neighbor tops by that + pad.
    tallest = max(other.height_mm for other in neighbors)
    hover_extra = max(
        setting("VIAM_NEIGHBOR_HEIGHT_CLEARANCE_MM", 70.0),
        tallest + g.tcp_to_fingertip_mm + setting("VIAM_DEPTH_HOVER_PAD_MM", 45.0),
    )
    lateral = 0.0
    trigger = max(
        setting("VIAM_NEIGHBOR_LATERAL_TRIGGER_MM", 120.0),
        2.0 * half_span + 2.0 * neighbor_half,
    )
    if gap < trigger:
        lateral = max(
            setting("VIAM_NEIGHBOR_APPROACH_OFFSET_MM", 70.0),
            needed_lateral,
            half_span + max(0.0, trigger - gap) * 0.5,
        )
        logger.info(
            "Neighbor gap %.1f mm, jaw half-span %.1f mm → lateral approach %.1f mm, hover %.1f mm",
            gap, half_span, lateral, hover_extra,
        )
    return lateral, hover_extra, gap


def approach_offset_xy(
    obj: DetectedObject,
    neighbors: list[DetectedObject],
    offset_mm: float,
) -> tuple[float, float]:
    if offset_mm <= 0 or not neighbors:
        return obj.center_world.x, obj.center_world.y
    nearest = min(neighbors, key=lambda other: planar_distance_mm(obj, other))
    dx = obj.center_world.x - nearest.center_world.x
    dy = obj.center_world.y - nearest.center_world.y
    length = math.hypot(dx, dy)
    if length < 1.0:
        return obj.center_world.x + offset_mm, obj.center_world.y
    return (
        obj.center_world.x + offset_mm * dx / length,
        obj.center_world.y + offset_mm * dy / length,
    )


async def move_one_object(
    obj: DetectedObject,
    destination: Pose,
    remaining_neighbors: list[DetectedObject],
    scene_obstacles: list[DetectedObject],
    arm: Arm,
    gripper: Gripper,
    motion: MotionClient,
    gripper_name: str,
    world_state,
    cfg: dict[str, float],
    table_margin: float,
    timeout: float,
) -> str:
    """Pick and place one object.

    Returns ``"ok"``, ``"missed_grasp"``, or ``"dropped"``. Recovery is left to
    the caller (re-scan the source square and retry leftovers).
    Depth-localized neighbors/dest objects are injected as motion obstacles.
    """
    # Fixed wrist orientation for the whole pick/place — never yaw mid-move
    # (dynamic theta was twisting joints and missing grasps).
    orientation = {
        "o_x": cfg["VIAM_GRASP_OX"],
        "o_y": cfg["VIAM_GRASP_OY"],
        "o_z": cfg["VIAM_GRASP_OZ"],
        "theta": cfg["VIAM_GRASP_THETA_DEG"],
    }
    geom = gripper_geom()
    pick_open = pick_open_position_for_scene(obj, remaining_neighbors)
    # Proven pattern (pick_red_cube / Viam workshop): TCP above cube center by
    # GRIPPER_LENGTH, then short world-Z descent only — no wrist twists.
    grasp_z, contact_z, top_z = grasp_tcp_z_for_object(obj, table_margin)
    clearance = max(cfg["VIAM_APPROACH_CLEARANCE_MM"], geom.approach_above_center_mm - geom.tcp_to_fingertip_mm)
    lateral, neighbor_hover, nearest_gap = neighbor_clearance_mm(
        obj, remaining_neighbors, pick_open,
    )
    neighbor_tops = [
        other.center_world.z + other.height_mm / 2.0 for other in remaining_neighbors
    ]
    hover = max(
        clearance,
        neighbor_hover,
        obj.height_mm + 40.0,
    )
    if neighbor_tops:
        hover = max(
            hover,
            max(neighbor_tops) - grasp_z + geom.tcp_to_fingertip_mm + 30.0,
        )
    floor = TABLE_TOP_Z_MM + table_margin
    # Translation-only clear-side standoff (same orientation) then center — no twist.
    approach_x, approach_y = approach_offset_xy(obj, remaining_neighbors, lateral)
    pregrasp = make_pose(
        obj.center_world.x, obj.center_world.y, grasp_z + hover, orientation,
    )
    side_pregrasp = make_pose(approach_x, approach_y, grasp_z + hover, orientation)
    grasp = make_pose(obj.center_world.x, obj.center_world.y, grasp_z, orientation)
    lift = make_pose(
        obj.center_world.x, obj.center_world.y,
        grasp_z + max(cfg["VIAM_LIFT_MM"], hover),
        orientation,
    )
    release_height = max(
        table_margin + 40.0,
        setting("VIAM_PLACE_RELEASE_HEIGHT_MM", 120.0),
    )
    place_z = max(destination.z + release_height, floor)
    place_approach_z = max(place_z + 30.0, destination.z + table_margin + clearance)
    step = max(10.0, cfg["VIAM_DESCENT_STEP_MM"])

    # Travel: keep distant depth keep-outs. Grasp descent: drop nearby source
    # neighbors — padded boxes always collide with open claws at ~70 mm gaps.
    # Lateral approach + reduced jaw open provide clearance instead.
    neighbor_pixels = {n.pixel for n in remaining_neighbors}
    travel_world = world_state_with_depth_obstacles(
        world_state, scene_obstacles, exclude=obj,
        pad_mm=setting("VIAM_DEPTH_OBSTACLE_PAD_MM", 20.0),
    )
    grasp_world = world_state_with_depth_obstacles(
        world_state, scene_obstacles, exclude=obj,
        exclude_pixels=neighbor_pixels,
        pad_mm=setting("VIAM_DEPTH_OBSTACLE_PAD_MM", 20.0),
    )

    logger.info(
        "Object at %s, height≈%.1f mm, gap≈%.1f mm, obstacles=%d (grasp excludes %d neighbors)",
        fmt_pose(obj.center_world), obj.height_mm, nearest_gap,
        max(0, len(scene_obstacles) - 1), len(neighbor_pixels),
    )
    logger.info("Destination %s", fmt_pose(destination))
    logger.info(
        "Simple pick (fixed wrist): open=%.0f, GRIPPER_LENGTH=%.0f, "
        "contact_z=%.1f top_z=%.1f hover=%.1f; pregrasp %s → grasp %s",
        pick_open, geom.tcp_to_fingertip_mm, contact_z, top_z, hover,
        fmt_pose(pregrasp), fmt_pose(grasp),
    )

    await open_gripper_for_pick(gripper, pick_open)
    await asyncio.sleep(geom.settle_s)

    # 1) Free plan to height (optionally via clear side — translation only).
    if lateral > 1.0:
        await move_gripper(motion, gripper_name, side_pregrasp, travel_world, linear=False,
                           timeout=timeout, label="clear-side pregrasp",
                           table_margin_mm=table_margin)
        await move_gripper(motion, gripper_name, pregrasp, travel_world, linear=True,
                           timeout=timeout, label="center above object",
                           table_margin_mm=table_margin)
    else:
        await move_gripper(motion, gripper_name, pregrasp, travel_world, linear=False,
                           timeout=timeout, label="pregrasp above object",
                           table_margin_mm=table_margin)

    # 2) Straight world-Z descent in small steps (same as pick_red_cube).
    # Use grasp_world so neighbor depth boxes do not block the claws.
    current_z = grasp_z + hover
    remaining = hover
    while remaining > 1e-3:
        delta = min(step, remaining)
        current_z -= delta
        remaining -= delta
        if current_z < floor:
            raise RuntimeError(
                f"Descent would reach Z={current_z:.1f} below table floor {floor:.1f}."
            )
        waypoint = make_pose(obj.center_world.x, obj.center_world.y, current_z, orientation)
        await move_gripper(motion, gripper_name, waypoint, grasp_world, linear=True,
                           timeout=timeout, label="descent step",
                           table_margin_mm=table_margin)

    from pick_red_cube import gentle_grab
    grip_speed = optional_int("VIAM_GRIPPER_SPEED", 0) or None
    grip_torque = optional_int("VIAM_GRIPPER_TORQUE_PCT", 0) or None
    if not await gentle_grab(gripper, speed=grip_speed, torque=grip_torque):
        await move_gripper(motion, gripper_name, pregrasp, grasp_world, linear=True,
                           timeout=timeout, label="retreat after missed grasp",
                           table_margin_mm=table_margin)
        await open_gripper_for_pick(gripper, pick_open)
        logger.warning("Missed grasp; will re-scan and retry remaining objects.")
        return "missed_grasp"
    await asyncio.sleep(geom.settle_s)
    closed_position = await gripper_position(gripper)

    await move_gripper(motion, gripper_name, lift, grasp_world, linear=True,
                       timeout=timeout, label="lift object", table_margin_mm=table_margin)
    holding = await with_timeout(gripper.is_holding_something(), 5.0, "verify hold after lift")
    if not holding.is_holding_something:
        await open_gripper_for_pick(gripper, pick_open)
        await move_gripper(motion, gripper_name, pregrasp, grasp_world, linear=True,
                           timeout=timeout, label="retreat after empty lift",
                           table_margin_mm=table_margin)
        logger.warning("Object dropped during lift; will re-scan and recover.")
        return "dropped"

    carry_world = travel_world
    hover_pose = await move_to_place_approach(
        motion, gripper_name, destination, orientation, place_approach_z,
        carry_world, table_margin, timeout,
    )
    holding = await with_timeout(gripper.is_holding_something(), 5.0, "verify hold after travel")
    if not holding.is_holding_something:
        await open_gripper_for_pick(gripper, pick_open)
        logger.warning("Object dropped in transit; will re-scan and recover.")
        return "dropped"

    place_above = make_pose(hover_pose.x, hover_pose.y, place_z, orientation)
    await move_gripper(motion, gripper_name, place_above, carry_world, linear=True,
                       timeout=timeout, label="lower into destination zone",
                       table_margin_mm=table_margin)
    await release_gripper_partial(gripper, closed_position)
    await move_gripper(motion, gripper_name, hover_pose, carry_world, linear=True,
                       timeout=timeout, label="retreat after release",
                       table_margin_mm=table_margin)
    holding = await with_timeout(gripper.is_holding_something(), 5.0, "verify release")
    if holding.is_holding_something:
        await open_gripper_for_pick(gripper, pick_open)
        logger.warning("Gripper still holding after release; will re-scan and recover.")
        return "dropped"
    await open_gripper_for_pick(gripper, pick_open)
    return "ok"


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
    # Transfer + recovery is the default. Preview-only when VIAM_PICK_ENABLE=0.
    enabled = os.getenv("VIAM_PICK_ENABLE", "1") != "0"
    teach_view_pose = os.getenv("VIAM_SAVE_VIEW_POSE", "0") == "1"
    # Transfer every detected object in the occupied square (no 3-object demo cap).
    max_objects = optional_int("VIAM_MAX_OBJECTS_PER_RUN", 40)
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

        # Overview once so both tape squares are in frame, then work from that
        # scene. Recovery rescans from the current pose (no climb unless needed).
        timeout = setting("VIAM_MOVE_TIMEOUT_SEC", 60.0)
        cfg = required_motion_settings()
        table_margin = cfg["VIAM_MIN_TCP_ABOVE_TABLE_MM"]
        if table_margin < 50:
            raise RuntimeError("VIAM_MIN_TCP_ABOVE_TABLE_MM must be >= 50 for table clearance.")
        if cfg["VIAM_GRASP_TCP_Z_OFFSET_MM"] < 50:
            raise RuntimeError(
                "VIAM_GRASP_TCP_Z_OFFSET_MM must be >= 50 (workshop GRIPPER_LENGTH). "
                "Pick height actually uses VIAM_GRIPPER_TCP_TO_TIP_MM."
            )
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
        await open_gripper_for_pick(gripper)
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
        source, destination_zone = choose_source_and_destination(scene.zones, None)
        locked_destination = destination_zone.center_px

        if not enabled:
            logger.info(
                "Preview (overview pose): source has %d object(s), destination has %d. "
                "Omit VIAM_PICK_ENABLE=0 to transfer with recovery.",
                len(source.objects), len(destination_zone.objects),
            )
            for idx, obj in enumerate(order_picks_outer_first(source.objects), 1):
                logger.info(
                    "Source object %d: color=%s pixel=%s height≈%.1f mm center=%s",
                    idx, obj.color_name, obj.pixel, obj.height_mm, fmt_pose(obj.center_world),
                )
            return

        recovery_passes = optional_int("VIAM_RECOVERY_PASSES", 6)
        transferred = 0
        placed_world: list[Pose] = []
        initial_count = len(source.objects)
        try:
            for attempt in range(1, recovery_passes + 1):
                await open_gripper_for_pick(gripper)
                if attempt > 1:
                    # Leftovers need a full overview — work-height views miss objects.
                    logger.info(
                        "Recovery pass %d/%d: overview rescan for leftovers "
                        "(%d moved so far, started with %d).",
                        attempt, recovery_passes, transferred, initial_count,
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
                    source, destination_zone = choose_source_and_destination(
                        scene.zones, locked_destination,
                    )

                pick_list = order_picks_outer_first(source.objects)
                if not pick_list:
                    logger.info(
                        "Source square is empty after %d successful transfer(s). Done.",
                        transferred,
                    )
                    return
                if len(pick_list) > max_objects:
                    logger.warning(
                        "Detected %d leftover objects; moving %d this pass.",
                        len(pick_list), max_objects,
                    )
                    pick_list = pick_list[:max_objects]

                logger.info(
                    "Pass %d/%d: source has %d object(s), destination already has %d.",
                    attempt, recovery_passes, len(pick_list), len(destination_zone.objects),
                )
                for idx, obj in enumerate(pick_list, 1):
                    logger.info(
                        "Source object %d: color=%s pixel=%s height≈%.1f mm center=%s",
                        idx, obj.color_name, obj.pixel, obj.height_mm, fmt_pose(obj.center_world),
                    )

                slot_pixels = drop_slot_pixels(
                    destination_zone,
                    pick_list,
                    prefer_toward=source.center_px,
                    occupied=destination_zone.objects,
                    tape_mask=scene.blue_tape,
                    depth=scene.depth,
                )
                destinations: list[Pose] = []
                for idx, slot in enumerate(slot_pixels, 1):
                    pose = await zone_surface_pose(
                        machine, scene, camera_name, destination_zone, slot,
                    )
                    assert_in_workspace(pose.x, pose.y, pose.z, workspace_bounds())
                    destinations.append(pose)
                    logger.info(
                        "Destination slot %d (interior+depth-clear): pixel=%s world=%s",
                        idx, slot, fmt_pose(pose),
                    )

                remaining = list(pick_list)
                need_rescan = False
                for index, (obj, target) in enumerate(zip(pick_list, destinations), 1):
                    neighbors = [other for other in remaining if other is not obj]
                    # All other depth objects in either square become planner obstacles.
                    scene_obstacles = (
                        [o for o in source.objects if o is not obj]
                        + list(destination_zone.objects)
                        + neighbors
                    )
                    # De-dupe by pixel while preserving order.
                    seen_px: set[tuple[int, int]] = set()
                    unique_obstacles: list[DetectedObject] = []
                    for o in scene_obstacles:
                        if o.pixel in seen_px:
                            continue
                        seen_px.add(o.pixel)
                        unique_obstacles.append(o)
                    result = await move_one_object(
                        obj, target, neighbors, unique_obstacles,
                        arm, gripper, motion, gripper_name,
                        world_state, cfg, table_margin, timeout,
                    )
                    if result == "ok":
                        transferred += 1
                        placed_world.append(target)
                        remaining = [other for other in remaining if other is not obj]
                        logger.info(
                            "Transferred object %d this pass (%d total so far).",
                            index, transferred,
                        )
                        continue
                    # One immediate retry from the same pose before overview recovery.
                    logger.warning(
                        "Transfer %s on object %d/%d; retrying once before overview recovery.",
                        result, index, len(pick_list),
                    )
                    result = await move_one_object(
                        obj, target, neighbors, unique_obstacles,
                        arm, gripper, motion, gripper_name,
                        world_state, cfg, table_margin, timeout,
                    )
                    if result == "ok":
                        transferred += 1
                        placed_world.append(target)
                        remaining = [other for other in remaining if other is not obj]
                        logger.info(
                            "Retry succeeded (%d total so far).",
                            transferred,
                        )
                        continue
                    logger.warning(
                        "Retry also %s; overview recovery for remaining leftovers.",
                        result,
                    )
                    need_rescan = True
                    break
                if need_rescan:
                    continue
                # Finished this pass's list — overview once to catch any leftover.
                if attempt < recovery_passes:
                    logger.info(
                        "Pass %d finished planned picks; overview check for leftovers.",
                        attempt,
                    )
                    continue
                break

            logger.info(
                "Transfer finished after %d pass(es): %d object(s) moved.",
                recovery_passes, transferred,
            )
        except Exception:
            try:
                await arm.stop()
            except Exception:  # noqa: BLE001
                logger.exception("Failed to stop arm after a transfer error")
            try:
                await open_gripper_for_pick(gripper)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to open gripper after a transfer error")
            raise


if __name__ == "__main__":
    asyncio.run(main())
