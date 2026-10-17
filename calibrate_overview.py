"""Calibrate the fixed overhead camera (overview-cam) against the arm. MOVES THE ARM (asks first).

A blue sticker on the top face of the gripper body is the target. The arm visits a list of poses
(gripper level or tipped 30 deg nose-down, several headings and heights), the sticker is found in
each overhead image, and one least-squares fit solves for
  - the camera's pose in the world frame (6 numbers), and
  - where the sticker sits on the flange (3 numbers), so nothing has to be measured.
Tipped poses are needed: with the gripper always level, the sticker's height above the flange
can't be told apart from the camera's height.

Prints the frame to paste into overview-cam's config and saves everything to state/.

    WORLDARM_ENV=.env.armfarm03 python calibrate_overview.py
"""

import asyncio
import json
import math
import os
import time

import cv2
import numpy as np
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.proto.common import Pose, PoseInFrame
from viam.services.motion import MotionClient

from worldarm.connection import connect

CAMERA = "overview-cam"
SPEED_DEG_S = 20.0
SETTLE_S = 1.0

# Flange poses: (x, y, z, heading deg, pitch-down deg). Heading 0 = gripper pointing +x.
# Kept over the free table toward -y, tip clear of the drawer box, gripper bottom >= ~20 cm up.
POSES = [
    (300, -250, 300, 0, 0),
    (300, -250, 260, -45, 0),
    (220, -300, 340, -90, 0),
    (300, -350, 280, -60, 0),
    (200, -400, 300, -90, 30),
    (280, -450, 340, -75, 0),
    (150, -450, 280, -110, 30),
    (150, -330, 300, -120, 0),
    (250, -300, 360, -30, 30),
    (250, -200, 320, -10, 0),
    (180, -250, 280, -60, 30),
    (220, -380, 360, -100, 30),
]

# Sticker colour (light blue), HSV in OpenCV units
HSV_LO, HSV_HI = (95, 55, 65), (122, 210, 210)
BLOB_AREA = (120, 4000)
STATIC_PX = 12            # a blob this close to the same spot in most images is background

# First guess for the sticker in the flange frame (mm): flange -x points up when theta = 0
STICKER_GUESS = (-40.0, 0.0, 55.0)

STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state")


# ---------------- geometry ----------------
def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def ov_to_R(ox, oy, oz, theta_deg):
    n = math.sqrt(ox * ox + oy * oy + oz * oz) or 1.0
    ox, oy, oz = ox / n, oy / n, oz / n
    lat = math.acos(max(-1.0, min(1.0, oz)))
    lon = math.atan2(oy, ox) if 1 - abs(oz) > 1e-4 else 0.0
    return _rz(lon) @ _ry(lat) @ _rz(math.radians(theta_deg))


def R_to_ov(R):
    o = R[:, 2]
    lat = math.acos(max(-1.0, min(1.0, o[2])))
    lon = math.atan2(o[1], o[0]) if 1 - abs(o[2]) > 1e-4 else 0.0
    M = (_rz(lon) @ _ry(lat)).T @ R
    return float(o[0]), float(o[1]), float(o[2]), math.degrees(math.atan2(M[1, 0], M[0, 0]))


def pose_dir(heading_deg, pitch_down_deg):
    h, p = math.radians(heading_deg), math.radians(pitch_down_deg)
    return math.cos(h) * math.cos(p), math.sin(h) * math.cos(p), -math.sin(p)


# ---------------- sticker detection ----------------
def blobs(jpeg: bytes):
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), HSV_LO, HSV_HI)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, st, cen = cv2.connectedComponentsWithStats(mask)
    return img, [(float(cen[i][0]), float(cen[i][1]), int(st[i, 4])) for i in range(1, n)
                 if BLOB_AREA[0] <= st[i, 4] <= BLOB_AREA[1]]


def pick_sticker(all_blobs):
    """Drop blobs that sit in the same place in most images (background); keep the biggest mover."""
    picks = []
    for bl in all_blobs:
        moving = []
        for b in bl:
            seen = sum(1 for other in all_blobs if any(math.hypot(b[0] - o[0], b[1] - o[1]) < STATIC_PX
                                                       for o in other))
            if seen < 0.6 * len(all_blobs):
                moving.append(b)
        picks.append(max(moving, key=lambda b: b[2]) if moving else None)
    return picks


# ---------------- fit ----------------
def project(params, F, Rf, K):
    rvec, t, m = params[:3], params[3:6], params[6:9]
    Rcw, _ = cv2.Rodrigues(rvec)                      # camera -> world
    P = F + np.einsum("nij,j->ni", Rf, m)             # sticker in world
    pc = (P - t) @ Rcw                                # world -> camera (rows: Rcw^T (P - t))
    return np.stack([K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2], K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]], 1), pc[:, 2]


def fit(F, Rf, uv, K, init):
    p = init.copy()
    lam = 1e-3
    def res(q):
        return (project(q, F, Rf, K)[0] - uv).ravel()
    r = res(p)
    for _ in range(200):
        J = np.empty((r.size, p.size))
        for j in range(p.size):
            d = np.zeros_like(p); d[j] = 1e-6 if j < 3 else 1e-3
            J[:, j] = (res(p + d) - r) / d[j]
        A, g = J.T @ J, J.T @ r
        step = np.linalg.solve(A + lam * np.diag(np.diag(A)), -g)
        r2 = res(p + step)
        if r2 @ r2 < r @ r:
            p, r, lam = p + step, r2, lam * 0.3
            if np.linalg.norm(step) < 1e-7:
                break
        else:
            lam *= 10
    return p, r


async def main():
    print(f"Calibrating {CAMERA}: {len(POSES)} poses, {SPEED_DEG_S:.0f} deg/s, motion service (plans around obstacles).")
    if input("This MOVES THE REAL ARM. Someone on the E-stop? Type 'yes': ").strip() != "yes":
        print("Cancelled.")
        return
    os.makedirs(STATE, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    shots = []
    async with await connect() as machine:
        arm = Arm.from_robot(machine, "arm")
        cam = Camera.from_robot(machine, CAMERA)
        motion = MotionClient.from_robot(machine, "builtin")
        ip = (await cam.get_properties()).intrinsic_parameters
        K = np.array([[ip.focal_x_px, 0, ip.center_x_px], [0, ip.focal_y_px, ip.center_y_px], [0, 0, 1.0]])
        await arm.do_command({"set_speed": SPEED_DEG_S})
        for i, (x, y, z, h, pitch) in enumerate(POSES):
            ox, oy, oz = pose_dir(h, pitch)
            try:
                await motion.move(component_name="arm", destination=PoseInFrame(
                    reference_frame="world", pose=Pose(x=x, y=y, z=z, o_x=ox, o_y=oy, o_z=oz, theta=0)))
            except Exception as e:
                print(f"  pose {i + 1:2d}: skipped (not reachable: {str(e)[:80]})")
                continue
            await asyncio.sleep(SETTLE_S)
            f = await arm.get_end_position()
            imgs, _ = await cam.get_images()
            jpeg = next(im for im in imgs if "jpeg" in im.mime_type).data
            img, bl = blobs(jpeg)
            shots.append({"i": i + 1, "F": [f.x, f.y, f.z], "ov": [f.o_x, f.o_y, f.o_z, f.theta], "blobs": bl,
                          "img": img})
            print(f"  pose {i + 1:2d}: flange ({f.x:.0f},{f.y:.0f},{f.z:.0f}) heading {h} pitch {pitch}: "
                  f"{len(bl)} blue blobs")
        # park level over the free table where it started
        try:
            await motion.move(component_name="arm", destination=PoseInFrame(
                reference_frame="world", pose=Pose(x=300, y=-250, z=300, o_x=1, o_y=0, o_z=0, theta=0)))
        except Exception:
            pass

    picks = pick_sticker([s["blobs"] for s in shots])
    used = [(s, p) for s, p in zip(shots, picks) if p is not None]
    for s, p in zip(shots, picks):
        vis = s["img"]
        if p:
            cv2.drawMarker(vis, (int(p[0]), int(p[1])), (0, 0, 255), cv2.MARKER_CROSS, 40, 3)
        cv2.imwrite(os.path.join(STATE, f"calib-{stamp}-pose{s['i']:02d}.jpg"), vis)
    print(f"Sticker found in {len(used)} of {len(shots)} images.")
    if len(used) < 7:
        print("Not enough views to calibrate; check the images in state/.")
        return

    F = np.array([s["F"] for s, _ in used], float)
    Rf = np.array([ov_to_R(*s["ov"]) for s, _ in used])
    uv = np.array([[p[0], p[1]] for _, p in used], float)
    m0 = np.array(STICKER_GUESS)
    P0 = F + np.einsum("nij,j->ni", Rf, m0)
    ok, rvec, tvec = cv2.solvePnP(P0, uv, K, None, flags=cv2.SOLVEPNP_EPNP)
    Rwc, _ = cv2.Rodrigues(rvec)                       # world -> camera
    Rcw = Rwc.T
    init = np.concatenate([cv2.Rodrigues(Rcw)[0].ravel(), (-Rcw @ tvec).ravel(), m0])
    params, r = fit(F, Rf, uv, K, init)
    _, depth = project(params, F, Rf, K)
    err_px = np.linalg.norm(r.reshape(-1, 2), axis=1)
    err_mm = err_px * depth / K[0, 0]

    # leave-one-out: predict each view from a fit without it
    loo = []
    for k in range(len(used)):
        keep = [j for j in range(len(used)) if j != k]
        pk, _ = fit(F[keep], Rf[keep], uv[keep], K, params)
        pred, dk = project(pk, F[k:k + 1], Rf[k:k + 1], K)
        loo.append(float(np.linalg.norm(pred[0] - uv[k]) * dk[0] / K[0, 0]))

    Rcw, _ = cv2.Rodrigues(params[:3])
    t, m = params[3:6], params[6:9]
    ox, oy, oz, th = R_to_ov(Rcw)
    assert np.allclose(ov_to_R(ox, oy, oz, th), Rcw, atol=1e-6)
    frame = {"parent": "world", "translation": {"x": round(t[0], 1), "y": round(t[1], 1), "z": round(t[2], 1)},
             "orientation": {"type": "ov_degrees",
                             "value": {"x": round(ox, 6), "y": round(oy, 6), "z": round(oz, 6), "th": round(th, 3)}}}
    print("\nPer-view fit error (mm at the sticker):", [round(v, 1) for v in err_mm])
    print(f"Fit error: mean {err_mm.mean():.1f} mm, max {err_mm.max():.1f} mm ({err_px.mean():.2f} px mean)")
    print(f"Leave-one-out (prediction on a view not used in the fit): mean {np.mean(loo):.1f} mm, "
          f"max {np.max(loo):.1f} mm")
    print(f"Camera at ({t[0]:.0f}, {t[1]:.0f}, {t[2]:.0f}) mm in world, looking along "
          f"({ox:.2f}, {oy:.2f}, {oz:.2f})")
    print(f"Sticker on the flange at ({m[0]:.0f}, {m[1]:.0f}, {m[2]:.0f}) mm (first guess {STICKER_GUESS})")
    print(f"\nPaste into {CAMERA}'s frame (CONFIGURE -> {CAMERA} -> Frame):")
    print(json.dumps(frame, indent=2))
    out = {"time": stamp, "camera": CAMERA, "frame": frame, "sticker_on_flange_mm": [round(v, 1) for v in m],
           "fit_err_mm": [round(v, 1) for v in err_mm], "loo_err_mm": [round(v, 1) for v in loo],
           "intrinsics": {"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2]},
           "views": [{"pose": s["i"], "flange": s["F"], "ov": s["ov"], "uv": [p[0], p[1]]} for s, p in used]}
    with open(os.path.join(STATE, f"overview-cam-calibration-{stamp}.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved state/overview-cam-calibration-{stamp}.json and the marked images.")


if __name__ == "__main__":
    asyncio.run(main())
