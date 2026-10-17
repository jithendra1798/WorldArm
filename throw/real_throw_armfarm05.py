"""armfarm05 (xArm850): find the ball on the table, pick it up, throw it.

One command, operator at the E-stop:
  python scripts/real_throw_armfarm05.py                       # find, pick, max-speed throw
  python scripts/real_throw_armfarm05.py --check               # read-only: cameras, ball fixes, no motion
  python scripts/real_throw_armfarm05.py --calibrate-overview  # also teach armfarm03's overview camera

Finding the ball (no hand-loading):
- If armfarm03's overview camera is calibrated, one overview frame gives the
  ball's table position. Otherwise armfarm05's wrist camera scans from the
  WorldArm saved view pose and three other poses over the table.
- The wrist camera then looks again from straight above the estimate. Balls
  are found with the WorldArm pick_place_balls method (dark foam pixels +
  depth, table top z = -23 mm, 45 mm default height when foam reads flat),
  without the orange sheet, which is gone.
- --calibrate-overview: after the pick, the arm places the ball at six table
  points, photographs each from overview-cam, re-picks it, and fits a pixel ->
  table homography (results/real-throw/overview_calibration.json).

Motion: armfarm05's motion service currently rejects every plan
("cannot specify multiple geometries with the same name") because the
armfarm3-main remote brings armfarm03's gripper and obstacles into its frame
system. Pick moves therefore use local uf850 IK (URDF FK matches armfarm05's
GetEndPosition to ~1e-5 mm) along straight-line joint waypoints, with a table
floor check on every waypoint. Reorienting moves are only allowed high up.

Throw: one smooth streamed trajectory start -> second -> last (no stop at
second), time-scaled so the tool stays under --tcp-limit (the xArm faults
with "Linear speed exceeds limit in ServoJ mode" above ~1 m/s). The gripper
opens at peak ball speed while the arm is still moving, triggered from the
driver's per-setpoint acknowledgements; if the motion ends first it stays
closed. Arm speed/accel and gripper speed are restored on exit.

Only armfarm05 moves; armfarm03 is read for camera frames only.
armfarm05 runs collision_sensitivity 0 (collision detection off).
"""

from __future__ import annotations

import argparse
import asyncio
import threading
import io
import json
import math
import os
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pybullet as p
from grpclib.const import Status
from grpclib.exceptions import GRPCError
from PIL import Image
from scipy import ndimage
from scipy.interpolate import CubicSpline
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.proto.common import Pose, PoseInFrame
from viam.proto.component.arm import JointPositions
from viam.robot.client import RobotClient

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/real-throw"
CALIB_FILE = OUT / "overview_calibration.json"
MACHINE_ID = "25950d11-fdd4-48e8-bebf-ca5857103f46"  # armfarm05
PART_ID = "cf85a512-0675-455f-b52e-cdafd12c5c62"  # armfarm5-main
OVERVIEW_ID = "fe4d456c-ed89-47f9-a683-b98f44b51be0"  # armfarm03
OVERVIEW_PART_ID = "4c543eea-9982-40f9-a9a9-c143926ef486"  # armfarm3-main
OVERVIEW_ADDRESS = "armfarm3-main.310sld03v2.viam.cloud"
OVERVIEW_CAM = "overview-cam"
WRIST_CAM = "cam"

# Viam Control-page samples on armfarm05 (degrees).
POSES = {
    "start": [4.33, -27.34, -29.06, 77.01, -29.26, -165.54],
    "second": [39.60, -27.35, -44.68, 122.56, -41.74, -158.10],
    "last": [53.96, -27.35, -78.44, 122.15, -52.40, -148.37],
}
GRASP_OV = (0.0, 0.0, -1.0, 0.0)  # WorldArm VIAM_GRASP_O*/THETA: jaws straight down, fixed wrist
# Gripper-frame scan positions (mm), looking down. First is WorldArm's saved view pose.
SCAN = [(487.7, 495.1, 499.7), (330.0, 330.0, 500.0), (480.0, 220.0, 500.0), (300.0, 560.0, 500.0)]
LOOK_Z = 330.0  # gripper z for the close look from straight above the ball
RETRACT = (250.0, 150.0, 400.0)  # gripper parked near the base during overview calibration
CALIB_POINTS = [(300, 250), (450, 250), (500, 400), (400, 500), (250, 450), (300, 600)]
LANDING_ESTIMATE = (368.0, 765.0)  # local estimate at --tcp-limit 900, release at peak speed

# WorldArm-proven geometry for armfarm05 (WorldArm/.env, workcell_obstacles.py).
TABLE_Z = -23.0
GRIPPER_FRAME_MM = 105.0  # gripper frame [0, 0, 105] from the arm flange
TCP_TO_TIP_MM = 55.0
GRASP_LOWER_MM = 6.0
GRASP_RAISE_MM = 10.0  # operator: grip 10 mm higher than WorldArm's height
MIN_TCP_ABOVE_TABLE_MM = 55.0
BALL_DEFAULT_HEIGHT_MM = 45.0
HOVER_MM = 100.0
CAM_T = np.array([83.0, -14.0, 18.0])  # cam frame in the arm frame (config)
CAM_OV = (-0.030391, -0.003538, 0.999532, -97.731173)

CONFIGURED_SPEED, DEFAULT_ACCEL = 60.0, 381.67  # armfarm05 config / module default; restored
MAX_SPEED, MAX_ACCEL = 180.0, 1145.0  # viam:ufactory documented ceilings
PICK_SPEED = 20.0
GRIPPER_OPEN = 850  # viam:ufactory:gripper {"set": pos}: 0 closed, 850 open, units 0.1 mm
FAST_GRIPPER_SPEED = 5000
STEP_MM = 20.0
ARRIVE_TOL_DEG = 1.0
DARK = 60  # overview grey level below which a pixel counts as the black ball


def log(event, **kw):
    print(f"[{time.strftime('%H:%M:%S')}] {event} " + " ".join(f"{k}={v}" for k, v in kw.items()), flush=True)


async def connect(address, key, key_id):
    return await RobotClient.at_address(address, RobotClient.Options.with_api_key(api_key=key, api_key_id=key_id))


async def check_identity(machine, allowed, label):
    meta = await machine.get_cloud_metadata()
    ids = {getattr(meta, k, "") for k in ("machine_id", "machine_part_id", "robot_part_id")}
    if not ids & set(allowed):
        raise SystemExit(f"Connected machine is not {label}; refusing to continue.")
    log("connected", machine=label)


# ------------------------------------------------------------------ kinematics
def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def ov_to_matrix(ox, oy, oz, theta_deg):
    """Viam orientation vector -> rotation (ZYZ: Rz(lon) Ry(lat) Rz(theta))."""
    v = np.array([ox, oy, oz], float)
    v /= np.linalg.norm(v)
    lat = math.acos(np.clip(v[2], -1, 1))
    lon = math.atan2(v[1], v[0]) if 1 - abs(v[2]) > 1e-4 else 0.0
    return rz(lon) @ ry(lat) @ rz(math.radians(theta_deg))


def matrix_to_quat(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return [x, y, z, w]


def rot_deg(Ra, Rb):
    return math.degrees(math.acos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


class Kinematics:
    """Local uf850 FK/IK in mm and degrees (URDF FK matches armfarm05 GetEndPosition)."""

    def __init__(self):
        self.cid = p.connect(p.DIRECT)
        self.body = p.loadURDF(str(ROOT / "assets/uf850.urdf"), useFixedBase=True, physicsClientId=self.cid)
        info = [p.getJointInfo(self.body, j, physicsClientId=self.cid)
                for j in range(p.getNumJoints(self.body, physicsClientId=self.cid))]
        self.idx = [i[0] for i in info if i[2] == p.JOINT_REVOLUTE]
        self.lo = np.array([info[j][8] for j in self.idx])
        self.hi = np.array([info[j][9] for j in self.idx])

    def _set(self, q_rad):
        for j, v in zip(self.idx, q_rad):
            p.resetJointState(self.body, j, float(v), physicsClientId=self.cid)

    def flange(self, q_deg):
        self._set(np.radians(q_deg))
        s = p.getLinkState(self.body, self.idx[-1], computeForwardKinematics=True, physicsClientId=self.cid)
        return np.array(s[4]) * 1000.0, np.array(p.getMatrixFromQuaternion(s[5])).reshape(3, 3)

    def gripper(self, q_deg):
        pos, R = self.flange(q_deg)
        return pos + GRIPPER_FRAME_MM * R[:, 2], R

    def camera(self, q_deg):
        pos, R = self.flange(q_deg)
        return pos + R @ CAM_T, R @ ov_to_matrix(*CAM_OV)

    def ik(self, gripper_pos, R, seed_deg):
        gripper_pos = np.asarray(gripper_pos, float)
        target = ((gripper_pos - GRIPPER_FRAME_MM * R[:, 2]) / 1000.0).tolist()
        quat = matrix_to_quat(R)
        q = np.radians(np.asarray(seed_deg, float))
        for _ in range(8):
            self._set(q)
            q = np.array(p.calculateInverseKinematics(
                self.body, self.idx[-1], target, quat,
                lowerLimits=self.lo.tolist(), upperLimits=self.hi.tolist(),
                jointRanges=(self.hi - self.lo).tolist(), restPoses=q.tolist(),
                maxNumIterations=400, residualThreshold=1e-7, physicsClientId=self.cid))[:6]
        q_deg = np.degrees(q)
        pos, R2 = self.gripper(q_deg)
        ok = (np.linalg.norm(pos - gripper_pos) < 1.0 and rot_deg(R, R2) < 0.5
              and np.all(q >= self.lo - 1e-6) and np.all(q <= self.hi + 1e-6))
        return q_deg, bool(ok)


def reachable(x, y):
    return 150 <= math.hypot(x, y) <= 720 and x <= 640 and y >= -400


# ------------------------------------------------------------------ arm moves
async def joints(arm):
    return [round(v, 2) for v in (await arm.get_joint_positions()).values]


async def run_path(arm, path):
    positions = [JointPositions(values=[float(v) for v in q]) for q in path]
    if len(positions) > 1:
        try:
            await arm.move_through_joint_positions(positions)
            return
        except NotImplementedError:
            pass
        except GRPCError as exc:
            if exc.status != Status.UNIMPLEMENTED:
                raise
    for jp in positions:
        await arm.move_to_joint_positions(jp)


async def move_joints(arm, name):
    t0 = time.monotonic()
    await arm.move_to_joint_positions(JointPositions(values=POSES[name]))
    err = max(abs(a - b) for a, b in zip(await joints(arm), POSES[name]))
    log("arrived", pose=name, seconds=round(time.monotonic() - t0, 3), max_err_deg=round(err, 2))
    if err > ARRIVE_TOL_DEG:
        raise RuntimeError(f"{name}: arm is {err:.2f} deg from target")


class Mover:
    """Gripper-frame moves for armfarm05 using local IK (no motion service)."""

    def __init__(self, arm, kin):
        self.arm, self.kin = arm, kin

    async def q(self):
        return np.array(await joints(self.arm), float)

    def _line(self, p0, p1, R, seed, label):
        n = max(1, math.ceil(np.linalg.norm(p1 - p0) / STEP_MM))
        path = []
        for i in range(1, n + 1):
            pt = p0 + (p1 - p0) * i / n
            if pt[2] < TABLE_Z + MIN_TCP_ABOVE_TABLE_MM:
                raise RuntimeError(f"{label}: waypoint z {pt[2]:.0f} is below the table floor")
            q, ok = self.kin.ik(pt, R, seed)
            if not ok or np.max(np.abs(q - seed)) > 12.0:
                raise RuntimeError(f"{label}: IK path breaks near {pt.round(0).tolist()}")
            path.append(q)
            seed = q
        return path

    async def go(self, pos, R, label):
        """Straight line at orientation R; reorienting first happens above z = 300 mm."""
        pos = np.asarray(pos, float)
        if pos[2] < TABLE_Z + MIN_TCP_ABOVE_TABLE_MM:
            raise RuntimeError(f"{label}: gripper z {pos[2]:.0f} is below the table floor")
        q0 = await self.q()
        p0, R0 = self.kin.gripper(q0)
        if rot_deg(R0, R) > 3.0:
            if p0[2] < 300:  # lift straight up first, keeping the current orientation
                up = np.array([p0[0], p0[1], 320.0])
                await run_path(self.arm, self._line(p0, up, R0, q0, label + " (lift)"))
                q0 = await self.q()
                p0, R0 = self.kin.gripper(q0)
            high = np.array([pos[0], pos[1], max(pos[2], 320.0)])
            q, ok = self.kin.ik(high, R, q0)
            if not ok:
                raise RuntimeError(f"{label}: no IK solution at {high.round(0).tolist()}")
            await run_path(self.arm, [q])
            q0 = await self.q()
            p0, R0 = self.kin.gripper(q0)
        path = self._line(p0, pos, R, q0, label) if np.linalg.norm(pos - p0) > 1.0 else []
        if path:
            await run_path(self.arm, path)
            err = np.max(np.abs(await self.q() - path[-1]))
            if err > ARRIVE_TOL_DEG:
                raise RuntimeError(f"{label}: arm is {err:.2f} deg from the last waypoint")
        log(label, gripper=pos.round(0).tolist())


# ------------------------------------------------------------------ wrist camera
def decode_depth(data):
    """Viam image/vnd.viam.dep -> (H, W) depth in mm."""
    w = int.from_bytes(data[8:16], "big")
    h = int.from_bytes(data[16:24], "big")
    return np.frombuffer(data[24:], dtype=">u2").reshape(h, w).astype(np.float32)


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


async def wrist_frame(cam):
    images, _ = await cam.get_images()
    by = {im.name: im for im in images}
    if "color" not in by or "depth" not in by:
        raise RuntimeError("wrist camera must return 'color' and 'depth'")
    hsv = np.asarray(Image.open(io.BytesIO(by["color"].data)).convert("HSV"))
    depth = decode_depth(by["depth"].data)
    if hsv.shape[:2] != depth.shape:
        raise RuntimeError("wrist colour and depth are not aligned")
    return hsv, depth, by["color"].data


def detect_balls(hsv, depth, intr, cam_pos, cam_R):
    """Dark round foam blobs on the table, in armfarm05 world mm (WorldArm method)."""
    sat, val = hsv[..., 1].astype(np.int16), hsv[..., 2].astype(np.int16)
    dark = ndimage.binary_closing(ndimage.binary_opening((sat < 110) & (val < 100), disk(2)), disk(4))
    lab, _ = ndimage.label(dark)
    found = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        region = lab[sl] == i
        area = int(region.sum())
        h, w = region.shape
        if not 400 <= area <= 25000 or max(h, w) / max(1, min(h, w)) > 1.8 or area / (h * w) < 0.5:
            continue
        vals = val[sl][region].astype(np.float32)
        if np.mean(vals < 70) < 0.3:
            continue
        ys, xs = np.nonzero(region)
        weight = 255.0 - vals  # aim at the dark foam mass, not specular highlights
        u = sl[1].start + float((weight * xs).sum() / weight.sum())
        v = sl[0].start + float((weight * ys).sum() / weight.sum())
        d = depth[sl][region]
        d = d[d > 0]
        if d.size < 5:
            continue
        ray = cam_R @ np.array([(u - intr.center_x_px) / intr.focal_x_px,
                                (v - intr.center_y_px) / intr.focal_y_px, 1.0])
        if ray[2] > -0.1:
            continue
        top = cam_pos + ray * float(np.median(d))
        height = top[2] - TABLE_Z
        if not -15 <= height <= 130:
            continue
        if height < 8:  # soft foam reads nearly flat in depth
            height = BALL_DEFAULT_HEIGHT_MM
        cz = TABLE_Z + height / 2
        c = cam_pos + ray * ((cz - cam_pos[2]) / ray[2])
        found.append({"x": float(c[0]), "y": float(c[1]), "height": float(height), "u": round(u),
                      "v": round(v), "area": area, "darkness": float(np.median(vals))})
    return sorted(found, key=lambda b: b["darkness"])


async def wrist_balls(arm, kin, cam, intr, save=None):
    q = np.array(await joints(arm), float)
    hsv, depth, jpg = await wrist_frame(cam)
    if save:
        (OUT / save).write_bytes(jpg)
    return detect_balls(hsv, depth, intr, *kin.camera(q))


async def frame_check(machine, kin, arm):
    """Compare the local camera model with Viam's frame system (read-only)."""
    cam_pos, cam_R = kin.camera(np.array(await joints(arm), float))
    try:
        a = (await machine.transform_pose(PoseInFrame(reference_frame=WRIST_CAM, pose=Pose(o_z=1)), "world")).pose
        b = (await machine.transform_pose(PoseInFrame(reference_frame=WRIST_CAM, pose=Pose(z=500, o_z=1)), "world")).pose
    except Exception as exc:
        log("frame check skipped", reason=type(exc).__name__)
        return None
    da = math.dist((a.x, a.y, a.z), cam_pos)
    db = math.dist((b.x, b.y, b.z), cam_pos + cam_R @ [0, 0, 500])
    log("frame check (local camera model vs Viam)", origin_mm=round(da, 1), at_500mm=round(db, 1))
    return max(da, db)


async def look_closer(mover, kin, cam, intr, est, stamp):
    """Wrist camera straight above the estimate; refine the ball position."""
    R = ov_to_matrix(*GRASP_OV)
    off = R @ (CAM_T - [0, 0, GRIPPER_FRAME_MM])  # camera minus gripper frame, world
    pos = (est["x"] - off[0], est["y"] - off[1], LOOK_Z)
    if reachable(pos[0], pos[1]):
        await mover.go(pos, R, "look above ball")
        near = [b for b in await wrist_balls(mover.arm, kin, cam, intr, f"look-{stamp}.jpg")
                if math.hypot(b["x"] - est["x"], b["y"] - est["y"]) < 80]
        if near:
            best = min(near, key=lambda b: math.hypot(b["x"] - est["x"], b["y"] - est["y"]))
            log("close look", x=round(best["x"]), y=round(best["y"]), height=round(best["height"]),
                moved_mm=round(math.hypot(best["x"] - est["x"], best["y"] - est["y"]), 1))
            return best
    log("close look missed; using the first estimate")
    return est


# ------------------------------------------------------------------ gripper
async def set_gripper(gripper, pos):
    await gripper.do_command({"set": float(pos)})


async def close_on_ball(gripper, close_pos):
    await set_gripper(gripper, close_pos)
    pos = (await gripper.do_command({"get": True})).get("pos")
    try:
        holding = (await gripper.is_holding_something()).is_holding_something
    except Exception:
        holding = None
    # Jaws stopping >5 mm short of the target means a ball is between them.
    ok = bool(holding or (pos is not None and pos > close_pos + 50))
    log("grip", commanded=close_pos, reported=pos, holding=holding, ok=ok)
    return ok


def grasp_height(ball):
    """WorldArm: gripper frame TCP_TO_TIP above the ball centre, lowered GRASP_LOWER."""
    cz = TABLE_Z + ball["height"] / 2
    return max(cz - GRASP_LOWER_MM + GRASP_RAISE_MM + TCP_TO_TIP_MM, TABLE_Z + MIN_TCP_ABOVE_TABLE_MM)


async def pick(mover, gripper, ball, close_pos):
    R = ov_to_matrix(*GRASP_OV)
    x, y = ball["x"], ball["y"]
    if not reachable(x, y):
        raise SystemExit(f"Ball at ({x:.0f}, {y:.0f}) is outside armfarm05's safe reach.")
    gz = grasp_height(ball)
    hover = max(HOVER_MM, ball["height"] + 40)
    await set_gripper(gripper, GRIPPER_OPEN)
    await asyncio.sleep(0.45)
    await mover.go((x, y, gz + hover), R, "above ball")
    await mover.go((x, y, gz), R, "down to ball")
    ok = await close_on_ball(gripper, close_pos)
    await mover.go((x, y, max(gz + 150, 320.0)), R, "lift")
    if not ok:
        await set_gripper(gripper, GRIPPER_OPEN)
        raise SystemExit(f"Grip not confirmed at ({x:.0f}, {y:.0f}); no throw.")


async def place(mover, gripper, x, y, ball):
    R = ov_to_matrix(*GRASP_OV)
    gz = grasp_height(ball)
    await mover.go((x, y, gz + HOVER_MM), R, "above place point")
    await mover.go((x, y, gz), R, "down to place")
    await set_gripper(gripper, GRIPPER_OPEN)
    await asyncio.sleep(0.3)
    await mover.go((x, y, gz + HOVER_MM), R, "up from place")


# ------------------------------------------------------------------ overview camera
async def overview_frame(cam):
    images, _ = await cam.get_images()
    for im in images:
        if str(im.mime_type).removesuffix("+lazy") == "image/jpeg" and im.name != "depth":
            grey = np.asarray(Image.open(io.BytesIO(im.data)).convert("L"), dtype=np.int16)
            return grey, im.data
    raise RuntimeError("overview-cam returned no JPEG colour image")


def blobs(dark):
    """Round dark blobs; opening removes thin cables."""
    dark = ndimage.binary_opening(dark, iterations=2)
    lab, _ = ndimage.label(dark)
    found = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        region = lab[sl] == i
        area = int(region.sum())
        if area < 60 or not 0.6 < w / h < 1.6 or area / (w * h) < 0.55:
            continue
        cy, cx = ndimage.center_of_mass(region)
        found.append({"u": sl[1].start + cx, "v": sl[0].start + cy, "area": area})
    return found


def new_ball(ref, after):
    """Ball that appeared between two frames taken from the same arm pose."""
    cands = blobs((after < DARK) & (ref >= DARK + 25))
    return max(cands, key=lambda b: b["area"]) if cands else None


def fit_homography(px, mm):
    """Normalized DLT: pixel (u, v) -> table (x, y) mm."""
    def norm(pts):
        pts = np.asarray(pts, float)
        c, s = pts.mean(0), math.sqrt(2) / np.mean(np.linalg.norm(pts - pts.mean(0), axis=1))
        return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])
    Tp, Tm = norm(px), norm(mm)
    rows = []
    for (u, v), (x, y) in zip(px, mm):
        u, v, _ = Tp @ [u, v, 1]
        x, y, _ = Tm @ [x, y, 1]
        rows += [[-u, -v, -1, 0, 0, 0, x * u, x * v, x], [0, 0, 0, -u, -v, -1, y * u, y * v, y]]
    H = np.linalg.inv(Tm) @ np.linalg.svd(np.array(rows))[2][-1].reshape(3, 3) @ Tp
    return H / H[2, 2]


def apply_h(H, u, v):
    x, y, w = H @ [u, v, 1.0]
    return x / w, y / w


def overview_balls(grey, calib):
    """Balls on armfarm05's reachable table from one overview frame (saved homography)."""
    H = np.array(calib["H"])
    vv, uu = np.mgrid[0:grey.shape[0], 0:grey.shape[1]]
    w = H[2, 0] * uu + H[2, 1] * vv + H[2, 2]
    X = (H[0, 0] * uu + H[0, 1] * vv + H[0, 2]) / w
    Y = (H[1, 0] * uu + H[1, 1] * vv + H[1, 2]) / w
    reach = np.hypot(X, Y)
    mask = (reach >= 150) & (reach <= 720) & (X <= 640) & (Y >= -400)
    area = calib["ball_area_px"]
    cands = [b for b in blobs((grey < DARK) & mask) if 0.4 * area <= b["area"] <= 2.5 * area]
    for b in cands:
        b["x"], b["y"] = apply_h(H, b["u"], b["v"])
        b["height"] = BALL_DEFAULT_HEIGHT_MM
    return cands


async def calibrate_overview(mover, gripper, ocam, ball, close_pos):
    """Holding the ball: place it at known points, photograph, re-pick; fit pixel -> table."""
    R = ov_to_matrix(*GRASP_OV)
    px, mm, areas = [], [], []
    for x, y in CALIB_POINTS:
        await mover.go(RETRACT, R, "retract")
        ref, _ = await overview_frame(ocam)
        await place(mover, gripper, x, y, ball)
        await mover.go(RETRACT, R, "retract")
        await asyncio.sleep(0.5)  # let the ball settle
        after, jpg = await overview_frame(ocam)
        seen = new_ball(ref, after)
        if seen is None:
            (OUT / f"calib-miss-{x}-{y}.jpg").write_bytes(jpg)
            log("calibration point not seen", x=x, y=y)
        else:
            px.append((seen["u"], seen["v"]))
            mm.append((x, y))
            areas.append(seen["area"])
            log("calibration point", x=x, y=y, u=round(seen["u"], 1), v=round(seen["v"], 1), area=seen["area"])
        await pick(mover, gripper, {"x": x, "y": y, "height": ball["height"]}, close_pos)
    if len(px) < 4:
        log("overview calibration needs 4 seen points", seen=len(px))
        return None
    H = fit_homography(px, mm)
    err = [math.dist(apply_h(H, u, v), q) for (u, v), q in zip(px, mm)]
    rms = math.sqrt(sum(e * e for e in err) / len(err))
    log("overview calibration", points=len(px), rms_mm=round(rms, 1))
    if rms > 10:
        log("overview calibration not saved: residual too large")
        return None
    calib = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "H": H.tolist(), "points_px": px,
             "points_mm": mm, "residual_mm": [round(e, 1) for e in err], "rms_mm": round(rms, 1),
             "ball_area_px": float(np.median(areas))}
    CALIB_FILE.write_text(json.dumps(calib, indent=2))
    return calib


# ------------------------------------------------------------------ throw
def plan_throw(kin, q_start, joint_speed, joint_accel, tcp_limit, release_shift=0.0, dt=0.01):
    """One smooth joint spline start -> second -> last (no stop at second).

    Uniformly time-scaled so joint speed, joint acceleration and the tool speed
    (flange and a point 175 mm out, covering any controller TCP) stay under
    the limits; the xArm faults with "Linear speed exceeds limit in ServoJ
    mode" above its TCP ceiling. Release is at peak ball speed.
    """
    Q = np.array([q_start, POSES["second"], POSES["last"]], float)
    seg = np.max(np.abs(np.diff(Q, axis=0)), axis=1)
    knots = np.concatenate([[0.0], np.cumsum(seg)]) / 100.0
    point = lambda q, d: (lambda f: f[0] + d * f[1][:, 2])(kin.flange(q))
    for _ in range(6):
        cs = CubicSpline(knots, Q, bc_type="clamped")
        tt = np.linspace(0, knots[-1], 800)
        P = np.array([[point(cs(t), d) for d in (0.0, 175.0)] for t in tt])
        tool = np.linalg.norm(np.diff(P, axis=0), axis=2).max(1).max() / (tt[1] - tt[0])
        k = max(np.abs(cs(tt, 1)).max() / joint_speed, math.sqrt(np.abs(cs(tt, 2)).max() / joint_accel),
                tool / tcp_limit)
        knots = knots * k
    cs = CubicSpline(knots, Q, bc_type="clamped")
    times = np.append(np.arange(0.0, knots[-1] - dt / 2, dt), knots[-1])
    ball = np.array([point(cs(t), GRIPPER_FRAME_MM + TCP_TO_TIP_MM - GRASP_LOWER_MM) for t in times])
    vel = np.gradient(ball, times, axis=0)
    i = int(np.argmax(np.linalg.norm(vel, axis=1)))
    i = int(np.clip(i + round(release_shift / dt), 1, len(times) - 2))
    return {"times": times, "q": cs(times), "release_index": i, "release_t": float(times[i]),
            "ball_speed": float(np.linalg.norm(vel[i]) / 1000), "ball_vel": (vel[i] / 1000).round(2).tolist(),
            "ball_pos": ball[i].round(0).tolist(), "duration": float(times[-1]),
            "peak_joint_speed": float(np.abs(cs(times, 1)).max())}


def plan_flick(kin, q_start, tcp_limit, base_s=0.6, dt=0.01):
    """Wrist-flick variant: J5 winds back during the approach, then snaps at the
    full 180 deg/s (1145 deg/s^2 bang-bang) into the last pose. The other joints
    follow the smooth start -> second -> last spline over base_s, slowed if the
    flange would exceed tcp_limit. Only the flange is held under the cap: the
    gripper tip reaches ~1.5 m/s, which the controller accepts only if its TCP
    offset is the flange. Release is at peak wrist speed.
    """
    J, sign = 4, -1.0  # J5, snapping toward negative angles
    tf = MAX_SPEED / MAX_ACCEL
    Q = np.array([q_start, POSES["second"], POSES["last"]], float)
    end = POSES["last"][J]
    wind = end - sign * MAX_SPEED * tf
    seg = np.max(np.abs(np.diff(Q, axis=0)), axis=1)
    while True:
        knots = np.concatenate([[0.0], np.cumsum(seg)]) / seg.sum() * base_s
        cs = CubicSpline(knots, Q, bc_type="clamped")
        T = float(knots[-1])
        times = np.append(np.arange(0.0, T - dt / 2, dt), T)
        q = cs(times)
        t0 = T - 2 * tf
        pre = CubicSpline([0.0, t0], [q_start[J], wind], bc_type="clamped")
        for k, tk in enumerate(times):
            if tk <= t0:
                q[k, J] = pre(tk)
            elif tk - t0 <= tf:
                q[k, J] = wind + sign * 0.5 * MAX_ACCEL * (tk - t0) ** 2
            else:
                q[k, J] = end - sign * 0.5 * MAX_ACCEL * (T - tk) ** 2
        flange = np.array([kin.flange(x)[0] for x in q])
        fpeak = float(np.linalg.norm(np.gradient(flange, times, axis=0), axis=1).max())
        if fpeak <= tcp_limit:
            break
        base_s *= fpeak / tcp_limit * 1.02
    point = lambda x, d: (lambda f: f[0] + d * f[1][:, 2])(kin.flange(x))
    ball = np.array([point(x, GRIPPER_FRAME_MM + TCP_TO_TIP_MM - GRASP_LOWER_MM + GRASP_RAISE_MM) for x in q])
    tip = np.array([point(x, 175.0) for x in q])
    vel = np.gradient(ball, times, axis=0)
    i = int(np.argmin(np.abs(times - (T - tf))))
    qd = np.gradient(q, times, axis=0)
    return {"times": times, "q": q, "release_index": i, "release_t": float(times[i]),
            "ball_speed": float(np.linalg.norm(vel[i]) / 1000), "ball_vel": (vel[i] / 1000).round(2).tolist(),
            "ball_pos": ball[i].round(0).tolist(), "duration": T,
            "peak_joint_speed": float(np.abs(qd).max()),
            "joint_peaks": np.abs(qd).max(0).round(0).tolist(), "flange_peak_mm_s": round(fpeak),
            "tip175_peak_mm_s": round(float(np.linalg.norm(np.gradient(tip, times, axis=0), axis=1).max()))}


class Releaser(threading.Thread):
    """Opens the gripper on its own thread, event loop and Viam connection.

    The arm's stream acks reach the client only after the motion ends, so the
    release can't wait on them. This thread is armed with the stream's start
    time and fires `delay` seconds later on a wall-clock timer, independent of
    the stream. `cancel()` before the deadline keeps the gripper closed.
    """

    def __init__(self, address, key, key_id, delay):
        super().__init__(daemon=True)
        self.address, self.key, self.key_id, self.delay = address, key, key_id, delay
        self.ready, self.armed, self.cancelled = threading.Event(), threading.Event(), threading.Event()
        self.t_start = None
        self.result = {}

    def fire_from(self, t_start):
        self.t_start = t_start
        self.armed.set()

    def cancel(self):
        self.cancelled.set()
        self.armed.set()

    def run(self):
        asyncio.run(self._main())

    async def _main(self):
        machine = await connect(self.address, self.key, self.key_id)
        try:
            gripper = Gripper.from_robot(machine, "gripper")
            await gripper.do_command({"get": True})  # warm the connection
            self.ready.set()
            await asyncio.to_thread(self.armed.wait)
            if self.cancelled.is_set():
                return
            await asyncio.sleep(max(0.0, self.t_start + self.delay - time.monotonic()))
            if self.cancelled.is_set():
                return
            self.result["sent_s"] = round(time.monotonic() - self.t_start, 3)
            await gripper.open()
            self.result["open_returned_s"] = round(time.monotonic() - self.t_start, 3)
        except Exception as exc:  # reported, never raised into the throw
            self.result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            self.ready.set()
            await machine.close()


async def stream_throw(arm, plan, releaser):
    """Stream the planned trajectory while the releaser thread opens the gripper on time."""
    times, qs = plan["times"], plan["q"]

    async def batches():
        for t, q in zip(times, qs):
            yield [Arm.TrajectoryPoint(time=timedelta(seconds=float(t)), positions=[float(v) for v in q])]

    t0 = time.monotonic()
    releaser.fire_from(t0)
    try:
        async for _ in arm.move_through_joint_positions_streamed(batches()):
            pass
    except BaseException:
        releaser.cancel()  # a fault before the release keeps the ball held
        raise
    t_end = time.monotonic() - t0
    await asyncio.to_thread(releaser.join, 5.0)
    rel = dict(releaser.result)
    if "sent_s" in rel:
        rel["mid_motion"] = rel["sent_s"] < plan["duration"] and rel.get("open_returned_s", 1e9) < t_end
        rel["planned_release_s"] = round(plan["release_t"], 3)
    return {"stream_s": round(t_end, 3), "planned_s": round(plan["duration"], 3), "points": len(times),
            "release": rel or None}


# ------------------------------------------------------------------ main
async def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="read-only: no motion")
    ap.add_argument("--calibrate-overview", action="store_true",
                    help="after the pick, teach the overview camera (six automatic place/pick cycles)")
    ap.add_argument("--close-pct", type=float, default=67.0, help="gripper close, percent of travel")
    ap.add_argument("--jaw-s", type=float, default=0.03, help="assumed jaw-opening time (s)")
    ap.add_argument("--flick", action="store_true",
                    help="wrist-flick throw: J5 snaps at 180 deg/s; flange held under --tcp-limit")
    ap.add_argument("--release-shift", type=float, default=-0.2,
                    help="seconds to move the release relative to peak ball speed (negative = sooner)")
    ap.add_argument("--tcp-limit", type=float, default=1000.0,
                    help="tool speed ceiling, mm/s (xArm ServoJ faults above ~1000)")
    ap.add_argument("--joint-speed", type=float, default=0.9 * MAX_SPEED)
    ap.add_argument("--joint-accel", type=float, default=MAX_ACCEL)
    args = ap.parse_args()
    close_pos = round(GRIPPER_OPEN * (1 - args.close_pct / 100.0))
    assert 0 <= close_pos <= GRIPPER_OPEN and 100 <= args.tcp_limit <= 1000
    assert 3 <= args.joint_speed <= MAX_SPEED and 0 < args.joint_accel <= MAX_ACCEL

    key, key_id = os.environ["VIAM_API_KEY"], os.environ["VIAM_API_KEY_ID"]
    machine = await connect(os.environ["VIAM_ADDRESS"], key, key_id)
    try:
        overview = await connect(os.environ.get("OVERVIEW_ADDRESS", OVERVIEW_ADDRESS),
                                 os.environ.get("OVERVIEW_API_KEY", key), os.environ.get("OVERVIEW_API_KEY_ID", key_id))
    except Exception as exc:
        await machine.close()
        raise SystemExit(f"Could not connect to armfarm03 ({type(exc).__name__}). Add armfarm03's API key to "
                         ".env as OVERVIEW_API_KEY and OVERVIEW_API_KEY_ID (armfarm03 CONNECT tab).")
    arm = Arm.from_robot(machine, "arm")
    gripper = Gripper.from_robot(machine, "gripper")
    wcam = Camera.from_robot(machine, WRIST_CAM)
    ocam = Camera.from_robot(overview, OVERVIEW_CAM)
    kin = Kinematics()
    mover = Mover(arm, kin)
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    record = {"time": stamp, "args": vars(args), "close_pos": close_pos}
    moved = False
    gripper_speed = None
    try:
        await check_identity(machine, {MACHINE_ID, PART_ID}, "armfarm05")
        await check_identity(overview, {OVERVIEW_ID, OVERVIEW_PART_ID}, "armfarm03 (camera only)")
        intr = (await wcam.get_properties()).intrinsic_parameters
        log("armfarm05 joints", now=await joints(arm))
        mismatch = await frame_check(machine, kin, arm)
        if mismatch is not None and mismatch > 10:
            raise SystemExit(f"Local camera model is {mismatch:.0f} mm from Viam's frame system; not moving.")
        calib = json.loads(CALIB_FILE.read_text()) if CALIB_FILE.exists() else None
        log("plan", overview="calibrated" if calib else "not calibrated (wrist-camera scan)",
            close=f"{args.close_pct:.0f}% (pos {close_pos}/850)", tcp_limit_mm_s=args.tcp_limit,
            release="at peak ball speed, mid-motion")
        if calib:
            land = apply_h(np.linalg.inv(np.array(calib["H"])), *LANDING_ESTIMATE)
            log("expected landing near overview pixel", u=round(land[0]), v=round(land[1]))

        if args.check:
            for b in await wrist_balls(arm, kin, wcam, intr, f"check-wrist-{stamp}.jpg"):
                log("wrist ball candidate", x=round(b["x"]), y=round(b["y"]), height=round(b["height"]),
                    px=(b["u"], b["v"]), area=b["area"], darkness=round(b["darkness"]),
                    reachable=reachable(b["x"], b["y"]))
            grey, jpg = await overview_frame(ocam)
            (OUT / f"check-overview-{stamp}.jpg").write_bytes(jpg)
            if calib:
                for b in overview_balls(grey, calib):
                    log("overview ball candidate", x=round(b["x"]), y=round(b["y"]), px=(round(b["u"]), round(b["v"])))
            log("check only; no motion sent", frames=f"check-*-{stamp}.jpg")
            return

        if input("armfarm05 will find and pick the ball, then throw. "
                 "E-stop in hand, landing area in front of the arm clear? Type 'go': ").strip().lower() != "go":
            raise SystemExit("Cancelled; no motion sent.")
        moved = True
        gripper_speed = (await gripper.do_command({"get_gripper_speed": True})).get("gripper_speed")
        await arm.do_command({"set_speed": PICK_SPEED, "set_acceleration": DEFAULT_ACCEL})
        down = ov_to_matrix(*GRASP_OV)

        # 1. Find the ball: overview camera if calibrated, else wrist-camera scan.
        est = None
        if calib:
            grey, jpg = await overview_frame(ocam)
            (OUT / f"overview-{stamp}.jpg").write_bytes(jpg)
            cands = [b for b in overview_balls(grey, calib) if reachable(b["x"], b["y"])]
            log("overview", candidates=len(cands))
            if len(cands) == 1:
                est = cands[0]
        if est is None:
            for i, pos in enumerate(SCAN):
                await mover.go(pos, down, f"scan pose {i + 1}")
                cands = [b for b in await wrist_balls(arm, kin, wcam, intr, f"scan{i + 1}-{stamp}.jpg")
                         if reachable(b["x"], b["y"])]
                log("wrist scan", pose=i + 1, candidates=len(cands))
                if cands:
                    est = cands[0]  # darkest round blob
                    break
        if est is None:
            raise SystemExit("No ball found on armfarm05's table (frames saved in results/real-throw).")
        log("ball estimate", x=round(est["x"]), y=round(est["y"]))

        # 2. Look again from straight above, then pick.
        ball = await look_closer(mover, kin, wcam, intr, est, stamp)
        record["ball"] = ball
        await pick(mover, gripper, ball, close_pos)

        if args.calibrate_overview:
            calib = await calibrate_overview(mover, gripper, ocam, ball, close_pos) or calib

        # 3. Throw: one smooth streamed motion start -> second -> last, release mid-motion.
        await move_joints(arm, "start")
        if args.flick:
            plan = plan_flick(kin, await joints(arm), args.tcp_limit)
            log("flick speeds", joint_peaks_deg_s=plan["joint_peaks"], flange_peak_mm_s=plan["flange_peak_mm_s"],
                tip175_peak_mm_s=plan["tip175_peak_mm_s"])
        else:
            plan = plan_throw(kin, await joints(arm), args.joint_speed, args.joint_accel, args.tcp_limit,
                              args.release_shift)
        record["throw_plan"] = {k: v for k, v in plan.items() if k not in ("times", "q")}
        log("throw plan", seconds=round(plan["duration"], 2), release_t=round(plan["release_t"], 2),
            before_end_s=round(plan["duration"] - plan["release_t"], 2), ball_speed=round(plan["ball_speed"], 2),
            ball_vel=plan["ball_vel"], peak_joint_speed=round(plan["peak_joint_speed"]))
        rtts = []
        for _ in range(5):
            t = time.monotonic()
            await arm.get_joint_positions()
            rtts.append(time.monotonic() - t)
        rtt = float(np.median(rtts))
        await gripper.do_command({"set_gripper_speed": FAST_GRIPPER_SPEED})
        releaser = Releaser(os.environ["VIAM_ADDRESS"], key, key_id, plan["release_t"] - args.jaw_s)
        releaser.start()
        if not await asyncio.to_thread(releaser.ready.wait, 20.0) or "error" in releaser.result:
            releaser.cancel()
            raise SystemExit(f"Release thread could not connect ({releaser.result.get('error', 'timeout')}); no throw.")
        for n in (3, 2, 1):
            log(f"THROW in {n}")
            await asyncio.sleep(1)
        throw = await stream_throw(arm, plan, releaser)
        record["throw"] = throw
        rel = throw["release"]
        if rel is None:
            log("RELEASE NOT SENT (cancelled or thread error); gripper still closed",
                stream_s=throw["stream_s"], points=throw["points"])
        else:
            log("released", stream_s=throw["stream_s"], rtt_s=round(rtt, 3), **rel)
    except (KeyboardInterrupt, asyncio.CancelledError, Exception) as exc:
        # SystemExit (cancel, wrong machine, no ball, no grip) is not caught here.
        log("stopping arm", reason=f"{type(exc).__name__}: {exc}")
        if moved:
            await arm.stop()
        raise
    finally:
        if moved:
            await arm.do_command({"set_speed": CONFIGURED_SPEED, "set_acceleration": DEFAULT_ACCEL})
            if gripper_speed is not None:
                await gripper.do_command({"set_gripper_speed": gripper_speed})
            log("restored", speed=CONFIGURED_SPEED, accel=DEFAULT_ACCEL, gripper_speed=gripper_speed)
        (OUT / f"run-{stamp}.json").write_text(json.dumps(record, indent=2, default=str))
        await overview.close()
        await machine.close()


if __name__ == "__main__":
    asyncio.run(main())
