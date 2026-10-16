"""Soft self-toss: pick a colored cube, toss it up, track, and catch it.

Default is preview-only (detect + plan; never moves). Physical motion requires:

  VIAM_THROW_CATCH_ENABLE=1
  VIAM_SAFETY_ACK=I_UNDERSTAND_THROW_CATCH_RISKS

Hard caps (config may only lower speed/height, never raise above these):
  - arm speed ≤ 15 deg/s
  - toss apex ≤ 220 mm world Z
  - catch session ≤ 8 s
  - TCP stays inside a tight catch volume above the table

Soft toss only: people clear of the workcell; hand on e-stop.
This is not a sports juggle — small vertical release above the table.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from collections import deque
from typing import Any, Deque, Optional, Sequence, Tuple

from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.proto.common import Pose, PoseInFrame, WorldState
from viam.services.motion import MotionClient
from viam.services.vision import Detection, VisionClient

from connection import connect, env
from workcell_obstacles import (
    TABLE_TOP_Z_MM,
    assert_tcp_above_table,
    compute_grasp_tcp_z,
    workcell_world_state,
)

logger = logging.getLogger("throw_catch_cube")

SAFETY_ACK_PHRASE = "I_UNDERSTAND_THROW_CATCH_RISKS"
HARD_MAX_ARM_SPEED_DEGS = 15.0
HARD_MAX_TOSS_APEX_Z_MM = 220.0
HARD_MAX_SESSION_SEC = 8.0
HARD_MIN_TCP_ABOVE_TABLE_MM = 70.0
HARD_MAX_OBJECT_SPEED_MM_S = 900.0
HARD_MAX_LEAD_SEC = 0.45
HARD_MIN_TRACK_SAMPLES = 3

# Tight AABB for toss + catch (world mm). Keeps the demo over the table.
DEFAULT_CATCH_VOLUME = {
    "x_min": 150.0,
    "x_max": 450.0,
    "y_min": 250.0,
    "y_max": 550.0,
    "z_min": TABLE_TOP_Z_MM + HARD_MIN_TCP_ABOVE_TABLE_MM,
    "z_max": HARD_MAX_TOSS_APEX_Z_MM,
}


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
    return f"xyz=({p.x:.1f}, {p.y:.1f}, {p.z:.1f}) mm"


def make_pose(x: float, y: float, z: float, ox: float, oy: float, oz: float, theta: float) -> Pose:
    return Pose(x=x, y=y, z=z, o_x=ox, o_y=oy, o_z=oz, theta=theta)


async def with_timeout(awaitable, seconds: float, what: str):
    try:
        return await asyncio.wait_for(awaitable, timeout=seconds)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"Timed out after {seconds:.1f}s waiting for {what}") from exc


def filter_color_detections(
    detections: Sequence[Detection], label: str, min_conf: float
) -> list[Detection]:
    matched = [
        d
        for d in detections
        if (not label or d.class_name == label) and d.confidence >= min_conf
    ]
    if not matched:
        raise RuntimeError(
            f"No detection for label={label!r} (min_conf={min_conf}). "
            "Check lighting / color_detector / camera view."
        )

    def area(d: Detection) -> float:
        return max(0.0, (d.x_max - d.x_min) * (d.y_max - d.y_min))

    matched = sorted(matched, key=area, reverse=True)
    if len(matched) > 1 and area(matched[1]) > area(matched[0]) * 0.60:
        raise RuntimeError(
            f"Multiple similar detections for label={label!r}; refusing to guess."
        )
    return matched


def assert_in_volume(
    x: float, y: float, z: float, vol: dict[str, float], *, context: str
) -> None:
    if not (
        vol["x_min"] <= x <= vol["x_max"]
        and vol["y_min"] <= y <= vol["y_max"]
        and vol["z_min"] <= z <= vol["z_max"]
    ):
        raise RuntimeError(
            f"{context}: ({x:.1f},{y:.1f},{z:.1f}) outside catch volume "
            f"[({vol['x_min']:.0f},{vol['y_min']:.0f},{vol['z_min']:.0f})-"
            f"({vol['x_max']:.0f},{vol['y_max']:.0f},{vol['z_max']:.0f})]"
        )
    floor = TABLE_TOP_Z_MM + HARD_MIN_TCP_ABOVE_TABLE_MM
    if z < floor:
        raise RuntimeError(f"{context}: Z={z:.1f} below safety floor {floor:.1f}")


def clamp_to_volume(x: float, y: float, z: float, vol: dict[str, float]) -> Tuple[float, float, float]:
    return (
        min(max(x, vol["x_min"]), vol["x_max"]),
        min(max(y, vol["y_min"]), vol["y_max"]),
        min(max(z, vol["z_min"]), vol["z_max"]),
    )


async def localize_cube(
    machine,
    cam_name: str,
    detector: VisionClient,
    segmenter: VisionClient,
    *,
    label: str,
    min_conf: float,
    vision_timeout: float,
) -> dict[str, Any]:
    detections = await with_timeout(
        detector.get_detections_from_camera(cam_name),
        vision_timeout,
        "detections",
    )
    chosen = filter_color_detections(detections, label=label, min_conf=min_conf)[0]
    objects = await with_timeout(
        segmenter.get_object_point_clouds(cam_name),
        vision_timeout,
        "segments",
    )
    labeled = [
        o
        for o in objects
        if o.geometries.geometries and o.geometries.geometries[0].label == label
    ]
    if not labeled:
        raise RuntimeError(f"No 3D segment for label={label!r}; refusing to guess.")
    obj = max(labeled, key=lambda o: len(o.point_cloud))
    if len(obj.point_cloud) < 40 or not obj.geometries.geometries:
        raise RuntimeError("3D segment too small / missing geometry; refusing to move.")
    center_cam = obj.geometries.geometries[0].center
    if center_cam.z <= 0:
        raise RuntimeError(f"Invalid camera depth z={center_cam.z:.1f}")

    center_world = await with_timeout(
        machine.transform_pose(
            PoseInFrame(reference_frame=cam_name, pose=center_cam),
            "world",
        ),
        vision_timeout,
        "camera→world",
    )
    p = center_world.pose
    return {
        "label": chosen.class_name,
        "confidence": float(chosen.confidence),
        "x": float(p.x),
        "y": float(p.y),
        "z": float(p.z),
        "t": time.monotonic(),
        "cloud_bytes": len(obj.point_cloud),
    }


async def localize_optional(
    machine,
    cam_name: str,
    detector: VisionClient,
    segmenter: VisionClient,
    *,
    label: str,
    min_conf: float,
    vision_timeout: float,
) -> Optional[dict[str, Any]]:
    try:
        return await localize_cube(
            machine,
            cam_name,
            detector,
            segmenter,
            label=label,
            min_conf=min_conf,
            vision_timeout=vision_timeout,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("localize miss: %s", exc)
        return None


async def move_tcp(
    motion: MotionClient,
    gripper_name: str,
    pose: Pose,
    world_state: WorldState,
    *,
    timeout: float,
    label: str,
    table_margin_mm: float,
) -> None:
    assert_tcp_above_table(pose.z, margin_mm=table_margin_mm, context=label)
    ok = await with_timeout(
        motion.move(
            component_name=gripper_name,
            destination=PoseInFrame(reference_frame="world", pose=pose),
            world_state=world_state,
            timeout=timeout,
        ),
        timeout + 2.0,
        label,
    )
    if not ok:
        raise RuntimeError(f"Motion failed for {label}")


async def set_arm_speed(arm: Arm, deg_s: float) -> None:
    capped = min(deg_s, HARD_MAX_ARM_SPEED_DEGS)
    await with_timeout(arm.do_command({"set_speed": capped}), 10.0, "set_speed")


async def gentle_grab(gripper: Gripper, *, speed: int, torque: int) -> bool:
    if speed > 0:
        try:
            await with_timeout(
                gripper.do_command({"set_gripper_speed": float(speed)}),
                10.0,
                "set_gripper_speed",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("set_gripper_speed: %s", exc)
    if torque > 0:
        try:
            await with_timeout(
                gripper.do_command({"set_gripper_torque": int(torque)}),
                10.0,
                "set_gripper_torque",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("set_gripper_torque: %s", exc)
    grabbed = await with_timeout(gripper.grab(), 15.0, "gripper.grab")
    if grabbed:
        return True
    holding = await with_timeout(gripper.is_holding_something(), 10.0, "is_holding")
    return bool(holding.is_holding_something)


def estimate_velocity(
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


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cam_name = env("VIAM_CAMERA")
    arm_name = env("VIAM_ARM")
    gripper_name = env("VIAM_GRIPPER")
    detector_name = os.getenv("VIAM_DETECTOR", "red-object-detector")
    segmenter_name = os.getenv("VIAM_SEGMENTER", "red-object-segmenter")
    motion_name = os.getenv("VIAM_MOTION", "builtin")
    label = os.getenv("VIAM_DETECT_LABEL", "red")

    min_conf = setting("VIAM_MIN_DETECTION_CONFIDENCE", 0.55)
    vision_timeout = setting("VIAM_VISION_TIMEOUT_SEC", 8.0)
    move_timeout = setting("VIAM_MOVE_TIMEOUT_SEC", 20.0)
    poll_interval = max(0.05, setting("VIAM_POLL_INTERVAL_SEC", 0.08))

    arm_speed = min(setting("VIAM_ARM_SPEED_DEGS_PER_SEC", 10.0), HARD_MAX_ARM_SPEED_DEGS)
    grasp_z_offset = setting("VIAM_GRASP_TCP_Z_OFFSET_MM", 65.0)
    min_tcp = max(
        HARD_MIN_TCP_ABOVE_TABLE_MM,
        setting("VIAM_MIN_TCP_ABOVE_TABLE_MM", HARD_MIN_TCP_ABOVE_TABLE_MM),
    )
    gripper_speed = optional_int("VIAM_GRIPPER_SPEED", 800)
    gripper_torque = optional_int("VIAM_GRIPPER_TORQUE_PCT", 40)

    ox = setting("VIAM_GRASP_OX", 0.0)
    oy = setting("VIAM_GRASP_OY", 0.0)
    oz = setting("VIAM_GRASP_OZ", -1.0)
    theta = setting("VIAM_GRASP_THETA_DEG", 0.0)

    vol = {
        "x_min": setting("VIAM_CATCH_X_MIN_MM", DEFAULT_CATCH_VOLUME["x_min"]),
        "x_max": setting("VIAM_CATCH_X_MAX_MM", DEFAULT_CATCH_VOLUME["x_max"]),
        "y_min": setting("VIAM_CATCH_Y_MIN_MM", DEFAULT_CATCH_VOLUME["y_min"]),
        "y_max": setting("VIAM_CATCH_Y_MAX_MM", DEFAULT_CATCH_VOLUME["y_max"]),
        "z_min": max(
            TABLE_TOP_Z_MM + min_tcp,
            setting("VIAM_CATCH_Z_MIN_MM", DEFAULT_CATCH_VOLUME["z_min"]),
        ),
        "z_max": min(
            HARD_MAX_TOSS_APEX_Z_MM,
            setting("VIAM_CATCH_Z_MAX_MM", DEFAULT_CATCH_VOLUME["z_max"]),
        ),
    }
    if vol["z_max"] <= vol["z_min"]:
        raise RuntimeError("catch volume Z extent invalid")

    ready_x = setting("VIAM_READY_X_MM", (vol["x_min"] + vol["x_max"]) / 2)
    ready_y = setting("VIAM_READY_Y_MM", (vol["y_min"] + vol["y_max"]) / 2)
    ready_z = setting("VIAM_READY_Z_MM", (vol["z_min"] + vol["z_max"]) / 2)
    toss_release_z = setting("VIAM_TOSS_RELEASE_Z_MM", ready_z)
    toss_apex_z = min(
        HARD_MAX_TOSS_APEX_Z_MM,
        setting("VIAM_TOSS_APEX_Z_MM", min(vol["z_max"], toss_release_z + 80.0)),
    )
    if toss_apex_z <= toss_release_z:
        raise RuntimeError("VIAM_TOSS_APEX_Z_MM must be above VIAM_TOSS_RELEASE_Z_MM")

    close_radius = setting("VIAM_CLOSE_RADIUS_MM", 80.0)
    lead_sec = min(setting("VIAM_LEAD_TIME_SEC", 0.25), HARD_MAX_LEAD_SEC)
    max_obj_speed = min(
        setting("VIAM_MAX_OBJECT_SPEED_MM_S", 700.0), HARD_MAX_OBJECT_SPEED_MM_S
    )
    max_session = min(setting("VIAM_MAX_SESSION_SEC", 6.0), HARD_MAX_SESSION_SEC)
    approach_clearance = setting("VIAM_APPROACH_CLEARANCE_MM", 100.0)

    enabled = os.getenv("VIAM_THROW_CATCH_ENABLE", "0") == "1"
    ack = os.getenv("VIAM_SAFETY_ACK", "")

    for name, x, y, z in (
        ("ready", ready_x, ready_y, ready_z),
        ("toss_release", ready_x, ready_y, toss_release_z),
        ("toss_apex", ready_x, ready_y, toss_apex_z),
    ):
        assert_in_volume(x, y, z, vol, context=name)

    async with await connect() as machine:
        arm = Arm.from_robot(machine, arm_name)
        gripper = Gripper.from_robot(machine, gripper_name)
        detector = VisionClient.from_robot(machine, detector_name)
        segmenter = VisionClient.from_robot(machine, segmenter_name)
        motion = MotionClient.from_robot(machine, motion_name)
        ws = workcell_world_state(include_ceiling=True)

        sample = await localize_cube(
            machine,
            cam_name,
            detector,
            segmenter,
            label=label,
            min_conf=min_conf,
            vision_timeout=vision_timeout,
        )
        grasp_z = compute_grasp_tcp_z(
            sample["z"],
            offset_above_detection_mm=grasp_z_offset,
            table_margin_mm=min_tcp,
        )
        plan = {
            "cube": sample,
            "grasp_tcp_z": grasp_z,
            "ready": {"x": ready_x, "y": ready_y, "z": ready_z},
            "toss_release": {"x": ready_x, "y": ready_y, "z": toss_release_z},
            "toss_apex": {"x": ready_x, "y": ready_y, "z": toss_apex_z},
            "catch_volume": vol,
            "arm_speed_degs_per_sec": arm_speed,
            "enable": enabled,
            "note": (
                "preview never moves. Soft self-toss only; clear the workcell; "
                "hand on e-stop. Set VIAM_THROW_CATCH_ENABLE=1 and "
                f"VIAM_SAFETY_ACK={SAFETY_ACK_PHRASE}"
            ),
        }
        logger.info(
            "detected %s conf=%.2f at (%.1f, %.1f, %.1f) grasp_z=%.1f",
            sample["label"],
            sample["confidence"],
            sample["x"],
            sample["y"],
            sample["z"],
            grasp_z,
        )
        logger.info(
            "plan ready=(%.0f,%.0f,%.0f) release_z=%.0f apex_z=%.0f speed=%.1f deg/s",
            ready_x,
            ready_y,
            ready_z,
            toss_release_z,
            toss_apex_z,
            arm_speed,
        )

        if not enabled:
            logger.info("PREVIEW ONLY — no motion. Plan: %s", plan["note"])
            print(plan)
            return

        if ack != SAFETY_ACK_PHRASE:
            raise RuntimeError(
                f"Motion refused: set VIAM_SAFETY_ACK={SAFETY_ACK_PHRASE!r} "
                "(people clear, soft toss only, e-stop ready)"
            )

        # --- pick ---
        await set_arm_speed(arm, arm_speed)
        await with_timeout(gripper.open(), 15.0, "gripper.open")

        # Pick may be outside the toss/catch AABB (cube on table elsewhere).
        # Only enforce table clearance until we carry into the catch volume.
        approach_z = min(grasp_z + approach_clearance, vol["z_max"])
        approach = make_pose(sample["x"], sample["y"], approach_z, ox, oy, oz, theta)
        grasp = make_pose(sample["x"], sample["y"], grasp_z, ox, oy, oz, theta)
        assert_tcp_above_table(approach.z, margin_mm=min_tcp, context="pick approach")
        assert_tcp_above_table(grasp.z, margin_mm=min_tcp, context="pick grasp")

        logger.info("pick approach %s", fmt_pose(approach))
        await move_tcp(
            motion, gripper_name, approach, ws,
            timeout=move_timeout, label="pick approach", table_margin_mm=min_tcp,
        )
        logger.info("pick grasp %s", fmt_pose(grasp))
        await move_tcp(
            motion, gripper_name, grasp, ws,
            timeout=move_timeout, label="pick grasp", table_margin_mm=min_tcp,
        )
        if not await gentle_grab(gripper, speed=gripper_speed, torque=gripper_torque):
            raise RuntimeError("Failed to grab cube; aborting toss.")

        # Carry into catch volume before the toss.
        carry = make_pose(ready_x, ready_y, ready_z, ox, oy, oz, theta)
        assert_in_volume(carry.x, carry.y, carry.z, vol, context="carry to toss zone")
        await move_tcp(
            motion, gripper_name, carry, ws,
            timeout=move_timeout, label="carry to toss zone", table_margin_mm=min_tcp,
        )

        # --- toss: move to release, then flick up while opening ---
        release = make_pose(ready_x, ready_y, toss_release_z, ox, oy, oz, theta)
        apex = make_pose(ready_x, ready_y, toss_apex_z, ox, oy, oz, theta)
        logger.info("toss release %s → apex %s", fmt_pose(release), fmt_pose(apex))
        await move_tcp(
            motion, gripper_name, release, ws,
            timeout=move_timeout, label="toss release", table_margin_mm=min_tcp,
        )

        # Open while commanding the upward flick so the cube gets a soft loft.
        open_task = asyncio.create_task(with_timeout(gripper.open(), 15.0, "toss open"))
        try:
            await move_tcp(
                motion, gripper_name, apex, ws,
                timeout=move_timeout, label="toss apex flick", table_margin_mm=min_tcp,
            )
        finally:
            try:
                await open_task
            except Exception as exc:  # noqa: BLE001
                logger.warning("gripper.open during toss: %s", exc)

        # Retreat slightly so the falling cube is clear of the jaws.
        retreat = make_pose(ready_x, ready_y, ready_z, ox, oy, oz, theta)
        await move_tcp(
            motion, gripper_name, retreat, ws,
            timeout=move_timeout, label="post-toss retreat", table_margin_mm=min_tcp,
        )

        # --- catch: track + intercept ---
        track: Deque[dict[str, Any]] = deque(maxlen=8)
        started = time.monotonic()
        grabbed = False
        reason = "session_timeout"
        logger.info("catching… session≤%.1fs close≤%.0f mm", max_session, close_radius)

        try:
            while time.monotonic() - started < max_session:
                sample = await localize_optional(
                    machine,
                    cam_name,
                    detector,
                    segmenter,
                    label=label,
                    min_conf=min_conf,
                    vision_timeout=min(vision_timeout, 2.0),
                )
                if sample is None:
                    await asyncio.sleep(poll_interval)
                    continue

                track.append(sample)
                vel = estimate_velocity(track)
                if vel is None:
                    await asyncio.sleep(poll_interval)
                    continue

                _vx, _vy, _vz, speed = vel
                if speed > max_obj_speed:
                    reason = f"object_too_fast:{speed:.0f}mm/s"
                    await arm.stop()
                    break

                px = sample["x"] + _vx * lead_sec
                py = sample["y"] + _vy * lead_sec
                pz = sample["z"] + _vz * lead_sec
                if not (
                    vol["x_min"] <= px <= vol["x_max"]
                    and vol["y_min"] <= py <= vol["y_max"]
                    and vol["z_min"] <= pz <= vol["z_max"]
                ):
                    if not (
                        vol["x_min"] <= sample["x"] <= vol["x_max"]
                        and vol["y_min"] <= sample["y"] <= vol["y_max"]
                        and vol["z_min"] <= sample["z"] <= vol["z_max"]
                    ):
                        await asyncio.sleep(poll_interval)
                        continue
                    px, py, pz = sample["x"], sample["y"], sample["z"]

                cx, cy, cz = clamp_to_volume(px, py, pz, vol)
                target = make_pose(cx, cy, cz, ox, oy, oz, theta)
                dist = math.sqrt(
                    (sample["x"] - cx) ** 2
                    + (sample["y"] - cy) ** 2
                    + (sample["z"] - cz) ** 2
                )
                try:
                    await move_tcp(
                        motion, gripper_name, target, ws,
                        timeout=move_timeout, label="intercept", table_margin_mm=min_tcp,
                    )
                except Exception as exc:  # noqa: BLE001
                    reason = f"motion_failed:{exc}"
                    await arm.stop()
                    break

                if dist <= close_radius:
                    grabbed = await gentle_grab(
                        gripper, speed=gripper_speed, torque=gripper_torque
                    )
                    reason = "grabbed" if grabbed else "grab_missed"
                    break

                await asyncio.sleep(poll_interval)
        except Exception:
            await arm.stop()
            await with_timeout(gripper.open(), 15.0, "gripper.open after error")
            raise

        if not grabbed:
            await arm.stop()
            await with_timeout(gripper.open(), 15.0, "gripper.open after miss")
            try:
                await move_tcp(
                    motion, gripper_name, retreat, ws,
                    timeout=move_timeout, label="retreat after miss", table_margin_mm=min_tcp,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("retreat failed: %s", exc)
        else:
            hold = make_pose(ready_x, ready_y, vol["z_max"], ox, oy, oz, theta)
            try:
                await move_tcp(
                    motion, gripper_name, hold, ws,
                    timeout=move_timeout, label="hold after catch", table_margin_mm=min_tcp,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("hold failed: %s", exc)

        result = {
            "ok": grabbed,
            "reason": reason,
            "track_samples": len(track),
            "session_sec": round(time.monotonic() - started, 3),
            "last_sample": track[-1] if track else None,
        }
        logger.info("done: %s", result)
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
