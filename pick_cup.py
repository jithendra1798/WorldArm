"""Find the paper cup with the overhead camera and pick it up with a level side grip. MOVES THE ARM.

  1. overview-cam (calibrated with calibrate_overview.py): the cup's inside is darker than the
     table, so it shows up as a dark round blob -> rough (x, y).
  2. Wrist RealSense straight above the cup: depth gives the exact center, rim height and diameter.
  3. Gripper pointing from the robot base at the cup, pitched down (PITCH_DEG), fingers closing side to
     side, camera on top: above the pre-grasp point -> straight down -> straight in along the tilted
     axis -> gentle grip -> check the fingers stopped on the cup -> straight up. (Level, the camera on
     top meets the forearm: Viam refused every IK solution, wrist_link vs cam_origin.)

    WORLDARM_ENV=.env.armfarm03 python pick_cup.py --look   # steps 1-2 only, never goes near the cup
    WORLDARM_ENV=.env.armfarm03 python pick_cup.py          # full pick
    WORLDARM_ENV=.env.armfarm03 python pick_cup.py --release
"""

import argparse
import asyncio
import glob
import json
import math
import os
import time

import cv2
import numpy as np
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.proto.common import Pose, PoseInFrame
from viam.proto.service.motion import Constraints, LinearConstraint
from viam.services.motion import MotionClient

from calibrate_overview import ov_to_R
from modules.drawer_service import CAM_R, CAM_T_MM, _decode_depth
from worldarm.cell.geometry import finger_axis_deg, other_wrist, rot_from_ov, theta_for_finger_axis
from worldarm.connection import connect

STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state")
TABLE_Z = 2.0
MAX_REACH_MM = 690.0
MAX_Y_MM = -30.0                  # never look or reach at y >= this: chargers and a power strip sit at y > 0

# overhead detection
RIM_Z_GUESS = 95.0                # only for the rough position; the wrist camera measures it
DARKER_BY = 30                    # cup (inside + shaded side) vs the local white of the table
BLOB_AREA_PX = (600, 4500)        # a cup is ~1000-2500 px from 1.7 m; drawer-front shadows are ~500
MIN_CIRCULARITY = 0.45            # inside + side wall merge into a rounded blob, not a perfect circle
MAX_SAT = 50                      # white cup; drops the blue sheet, blocks, knobs, cans' coloured sides
MAX_CANDIDATES = 4                # checked one by one with the wrist depth camera
ORANGE_LO, ORANGE_HI = (5, 90, 90), (24, 255, 255)   # orange insert inside the cup (measured H 15, S 183)
ORANGE_MIN_PX = 100
ORANGE_Z_GUESS = 50.0             # insert sits inside the cup, below the rim

# wrist look
LOOK_CAM_ABOVE_RIM_MM = 400.0     # RealSense needs > ~280 mm
SURVEY_CAM_Z = 600.0              # survey looks: camera 60 cm up sees ~80 x 45 cm of table
SURVEY_XY = [(330.0, -400.0), (330.0, -100.0)]
# what a cup looks like in depth: a round, hollow rim
CUP_RADIUS_MM = (25.0, 55.0)      # this cup measures 29-31 mm (6 mm height-map cells read a bit small)
CUP_HEIGHT_MM = (50.0, 200.0)
CUP_MAX_FILLED = 0.35             # fraction of the rim's inner half that is as high as the rim (solid top = cube/can)
RECENTER_MM = 50.0                # look again centred on the cup only if the first look was this far off
                                  # (an off-centre look still measures it to a few mm; the pinch takes +-19)

# approach clearance: the open gripper sweeps a corridor this wide from the pre-grasp point to just
# past the cup; other objects tall enough to reach the gripper's underside must stay outside it
CORRIDOR_HALF_WIDTH_MM = 60.0     # open fingers ~55 mm each side of the axis
CORRIDOR_MARGIN_MM = 10.0
TIP_PAST_FLANGE_MM = 170.0
APPROACH_OFFSETS_DEG = [0, 15, -15, 30, -30, 45, -45]   # tried in order, relative to straight from the base

# grip
SPEED_TRAVEL, SPEED_SLOW = 45.0, 20.0   # fast: was 25 / 10
REACH_BASE_MM = 110.0             # cup center = flange + (110 + cup radius) along the gripper: the
                                  # palm is ~105 mm out and the fingertips ~165 mm, so the fingers
                                  # straddle the widest part without the palm touching the cup
REACH_MAX_MM = 150.0
BACKOFF_MM = 100.0                # pre-grasp distance back from the grip point
ABOVE_MM = 250.0                  # flange height for the pre-grasp approach from above
GRIP_AXIS_MIN_Z = 50.0            # gripper axis height: body ~25 mm below the axis, keep clear of the table
GRIP_FRACTION = 0.5               # grip at this fraction of the cup height
OPEN_POS = 850                    # gripper fully open (8.5 cm)
GRIP_TORQUE_PCT = 15              # paper cup: gentle
GRIP_SPEED = 2500                 # fingers close fast; the torque cap still keeps it gentle
TAPER = 0.85                      # diameter at grip height vs rim (paper cups taper)
LIFT_MM = 100.0
PITCH_DEG = (20.0, 25.0, 30.0, 35.0)  # side grip pitched this far below level, tried in order until Viam plans it
TIP_ABOVE_TABLE_MM = 15.0         # the tilted fingertips reach below the grip point: keep them this high
PALM_MM, TIPS_MM = 105.0, 165.0   # along the gripper from the flange: the palm, the fingertips
PALM_CLEAR_MM = 6.0               # tilted, the palm rides higher, where the cup is widest: clear the rim by this
# rim pinch (the default): pointing straight down, one finger inside the cup and one outside its wall
PINCH_OPEN_MM = 50.0              # fingers open this far around the wall (+-19 mm of position error tolerated)
PINCH_DEPTH_MM = 20.0             # fingertips this far below the rim: the pads hold the top of the wall
PINCH_ABOVE_MM = 80.0             # start the slow straight descent this far above the pinch height
PINCH_TORQUE_PCT = 20
PINCH_MIN_WALL_MM = 0.8           # a pinch stops at least this much wider than closing on nothing
HOME_XYZ = (200.0, 200.0, 400.0)  # after leaving the cup, the arm parks here (flange, pointing down): high above
                                  # everything, so passing over the charger side (y > 0) is clear


def tilted_orientation(heading: float, pitch_deg: float) -> tuple[np.ndarray, float]:
    """Gripper axis along `heading`, pitched down; fingers closing horizontally across it; the wrist camera
    on the upper side. Returns (unit gripper axis, Viam theta), found with Viam's own orientation math."""
    p = math.radians(pitch_deg)
    a = np.array([math.cos(heading) * math.cos(p), math.sin(heading) * math.cos(p), -math.sin(p)])
    best = None
    for th in np.arange(-180.0, 180.0, 0.5):
        R = rot_from_ov(a[0], a[1], a[2], float(th))
        if (R @ np.array(CAM_T_MM))[2] <= 0:   # camera below the gripper: no
            continue
        level = abs(R[2, 1])                     # finger axis (flange y) out of the horizontal
        if best is None or level < best[0]:
            best = (level, float(th))
    return a, best[1]


def latest_calibration():
    files = sorted(glob.glob(os.path.join(STATE, "overview-cam-calibration-*.json")))
    if not files:
        raise RuntimeError("no overview-cam calibration in state/; run calibrate_overview.py first")
    c = json.load(open(files[-1]))
    t = np.array([c["frame"]["translation"][k] for k in "xyz"])
    o = c["frame"]["orientation"]["value"]
    return t, ov_to_R(o["x"], o["y"], o["z"], o["th"]), c["intrinsics"], os.path.basename(files[-1])


def pixel_to_world(u, v, z, cal):
    t, R, I, _ = cal
    d = R @ np.array([(u - I["cx"]) / I["fx"], (v - I["cy"]) / I["fy"], 1.0])
    return t + (z - t[2]) / d[2] * d


def detect_cup(img, cal):
    """Cup candidates: uncoloured, rounded, cup-sized blobs darker than the nearby table.
    White-on-white is ambiguous from 1.7 m, so these are only candidates; the wrist depth camera
    decides. Reachable ones first, biggest first."""
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1]
    h, w = gray.shape
    # reachable table area as an image mask
    us, vs = np.meshgrid(np.arange(0, w, 4), np.arange(0, h, 4))
    t, R, I, _ = cal
    d = np.stack([(us - I["cx"]) / I["fx"], (vs - I["cy"]) / I["fy"], np.ones_like(us, float)], -1) @ R.T
    s = (RIM_Z_GUESS - t[2]) / d[..., 2]
    X, Y = t[0] + s * d[..., 0], t[1] + s * d[..., 1]
    r = np.hypot(X, Y)
    # search a bit past the reach so a cup that's too far is still found (and reported as too far)
    # only the table in front of the robot (x > 80): never the arm itself or what's behind it
    # and only our side of the table, y < 0: chargers and a power strip are on the other
    reach = cv2.resize(((r > 200) & (r < MAX_REACH_MM + 400) & (X > 80) & (Y < MAX_Y_MM)).astype(np.uint8) * 255,
                       (w, h), interpolation=cv2.INTER_NEAREST)
    # local white level = brightest nearby pixel (robust next to the blue sheet or the floor)
    white = cv2.dilate(gray, np.ones((61, 61), np.uint8))
    table = float(np.median(white[reach > 0]))
    dark = ((gray.astype(int) < white.astype(int) - DARKER_BY) & (sat < MAX_SAT) & (white > 140)
            & (reach > 0)).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    out = []
    for c in cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]:
        area, per = cv2.contourArea(c), cv2.arcLength(c, True)
        if not (BLOB_AREA_PX[0] <= area <= BLOB_AREA_PX[1]) or per == 0:
            continue
        circ = 4 * math.pi * area / per ** 2
        if circ < MIN_CIRCULARITY:
            continue
        mo = cv2.moments(c)
        u, v = mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]
        rad = math.sqrt(area / math.pi)
        p = pixel_to_world(u, v, RIM_Z_GUESS, cal)
        dist = math.hypot(p[0], p[1])
        out.append({"px": [round(u), round(v)], "radius_px": round(rad, 1), "area_px": int(area),
                    "circularity": round(circ, 2), "xy": [round(float(p[0]), 1), round(float(p[1]), 1)],
                    "dist_mm": round(dist), "reachable": dist <= MAX_REACH_MM})
    # an orange insert in the cup is the strongest cue: orange blobs within reach go first
    orange = cv2.morphologyEx(cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), ORANGE_LO, ORANGE_HI) & reach,
                              cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, st, cen = cv2.connectedComponentsWithStats(orange)
    for i in range(1, n):
        if st[i, 4] < ORANGE_MIN_PX:
            continue
        u, v = cen[i]
        p = pixel_to_world(u, v, ORANGE_Z_GUESS, cal)
        dist = math.hypot(p[0], p[1])
        out.append({"px": [round(u), round(v)], "radius_px": round(math.sqrt(st[i, 4] / math.pi), 1),
                    "area_px": int(st[i, 4]), "orange": True, "xy": [round(float(p[0]), 1), round(float(p[1]), 1)],
                    "dist_mm": round(dist), "reachable": dist <= MAX_REACH_MM})
    out.sort(key=lambda c: (not c["reachable"], not c.get("orange", False), -c["area_px"]))
    return out, table


def find_objects(P, cell=6.0):
    """Objects standing on the table from wrist depth points (N, 3): a height map, connected blobs,
    then per blob its top ring and whether the middle is hollow (cup) or solid (cube, can)."""
    P = P[(P[:, 2] > TABLE_Z + 15) & (P[:, 2] < 320) & (np.hypot(P[:, 0], P[:, 1]) < MAX_REACH_MM + 150)]
    if len(P) < 30:
        return []
    x0, y0 = P[:, 0].min(), P[:, 1].min()
    ix = ((P[:, 0] - x0) / cell).astype(int)
    iy = ((P[:, 1] - y0) / cell).astype(int)
    H = np.zeros((iy.max() + 1, ix.max() + 1), np.float32)
    np.maximum.at(H, (iy, ix), P[:, 2])
    occ = cv2.morphologyEx((H > 0).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab = cv2.connectedComponents(occ)
    out = []
    for k in range(1, n):
        cells = lab == k
        if cells.sum() < 25:                      # < ~9 cm^2: noise
            continue
        hk = H[cells]
        top = float(np.percentile(hk[hk > 0], 95))
        ring_cells = cells & (H > top - 12)
        rv, rc = np.nonzero(ring_cells)
        rx, ry = x0 + (rc + 0.5) * cell, y0 + (rv + 0.5) * cell
        cx, cy, r = fit_circle(np.c_[rx, ry]) if len(rx) >= 6 else (rx.mean(), ry.mean(), 0.0)
        cv_, cc_ = (cy - y0) / cell, (cx - x0) / cell
        yy, xx = np.mgrid[0:H.shape[0], 0:H.shape[1]]
        inner = (np.hypot(yy - cv_, xx - cc_) < 0.5 * r / cell)
        filled = float(((H > top - 25) & inner).sum() / max(inner.sum(), 1)) if r > 0 else 1.0
        height = top - TABLE_Z
        why = []
        if not (CUP_RADIUS_MM[0] <= r <= CUP_RADIUS_MM[1]):
            why.append(f"rim radius {r:.0f} mm")
        if not (CUP_HEIGHT_MM[0] <= height <= CUP_HEIGHT_MM[1]):
            why.append(f"height {height:.0f} mm")
        if filled > CUP_MAX_FILLED:
            why.append(f"solid top ({filled:.0%} filled)")
        out.append({"x": round(float(cx), 1), "y": round(float(cy), 1), "rim_z": round(top, 1),
                    "rim_diameter": round(2 * r, 1), "height": round(height, 1), "filled": round(filled, 2),
                    "is_cup": not why, "why": ", ".join(why) or "round hollow rim"})
    return out


def fit_circle(xy):
    A = np.c_[2 * xy, np.ones(len(xy))]
    b = (xy ** 2).sum(1)
    cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
    return cx, cy, math.sqrt(max(c + cx * cx + cy * cy, 0.0))


class CupPicker:
    def __init__(self, machine):
        self.arm = Arm.from_robot(machine, "arm")
        self.gripper = Gripper.from_robot(machine, "gripper")
        self.cam = Camera.from_robot(machine, "cam")
        self.top = Camera.from_robot(machine, "overview-cam")
        self.motion = MotionClient.from_robot(machine, "builtin")
        self.log = {}

    async def speed(self, v):
        await self.arm.do_command({"set_speed": float(v)})

    async def flange_to(self, x, y, z, o, straight=False, theta=0.0):
        cons = Constraints(linear_constraint=[LinearConstraint(line_tolerance_mm=3, orientation_tolerance_degs=2)]) \
            if straight else None
        await self.motion.move(component_name="arm", constraints=cons, destination=PoseInFrame(
            reference_frame="world", pose=Pose(x=x, y=y, z=z, o_x=o[0], o_y=o[1], o_z=o[2], theta=theta)))

    async def fingers_stopped(self, moving_from=None, cap_s: float = 1.2):
        """Wait until the fingers stop moving (on the cup, or fully open/shut), not a fixed time. moving_from:
        where they were before the command; "stopped" only counts once they've left it (or after 0.7 s), so
        a gripper slow to start isn't read as already done."""
        t0, last = time.time(), None
        while time.time() - t0 < cap_s:
            await asyncio.sleep(0.08)
            p = await self.gpos()
            dt = time.time() - t0
            left = moving_from is None or p is None or abs(p - moving_from) > 20 or dt > 0.7
            if p is not None and last is not None and abs(p - last) < 3 and dt > 0.2 and left:
                return p
            last = p
        return await self.gpos()

    async def gpos(self):
        try:
            return float((await self.gripper.do_command({"get": True})).get("pos"))
        except Exception:
            return None

    async def find_overhead(self, cal):
        imgs, _ = await self.top.get_images()
        img = cv2.imdecode(np.frombuffer(next(i for i in imgs if "jpeg" in i.mime_type).data, np.uint8),
                           cv2.IMREAD_COLOR)
        cands, table = detect_cup(img, cal)
        vis = img.copy()
        for i, c in enumerate(cands):
            cv2.circle(vis, tuple(c["px"]), int(c["radius_px"]) + 4, (0, 0, 255) if i == 0 else (0, 200, 255), 2)
        cv2.imwrite(os.path.join(STATE, f"cup-overhead-{time.strftime('%H%M%S')}.jpg"), vis)
        self.log["overhead"] = {"candidates": cands[:5], "table_grey": float(table)}
        return cands

    async def look_down_at(self, x, y, cam_z):
        """Wrist camera straight above (x, y) at height cam_z; returns the depth points in world (N, 3).
        Tries the camera on each side of the gripper (wrist turned 0/90/180/-90): close to the base, some
        sides fold the arm into itself or the back wall."""
        o = (0.0, 0.0, -1.0)
        await self.speed(SPEED_TRAVEL)
        errors = []
        for th in (0.0, 90.0, 180.0, -90.0):
            off = ov_to_R(*o, th) @ np.array(CAM_T_MM)
            try:
                await self.flange_to(x - off[0], y - off[1], cam_z - off[2], o, theta=th)
                break
            except Exception as e:
                errors.append(f"wrist {th:.0f}: {str(e)[:120]}")
        else:
            raise RuntimeError(f"can't look straight down at ({x:.0f},{y:.0f}): " + "; ".join(errors))
        await asyncio.sleep(0.4)  # stop swaying before the depth picture
        return await self.depth_points()

    async def depth_points(self):
        ip = (await self.cam.get_properties()).intrinsic_parameters
        imgs, _ = await self.cam.get_images()
        dimg = next(i for i in imgs if "dep" in i.mime_type)
        w, h, px = _decode_depth(dimg.data)
        f = await self.arm.get_end_position()
        Rf, pf = ov_to_R(f.o_x, f.o_y, f.o_z, f.theta), np.array([f.x, f.y, f.z])
        D = np.array(px, float).reshape(h, w)[::3, ::3]
        vv, uu = np.mgrid[0:h:3, 0:w:3]
        ok = (D > 150) & (D < 1500)
        d = D[ok]
        pc = np.stack([(uu[ok] - ip.center_x_px) * d / ip.focal_x_px, (vv[ok] - ip.center_y_px) * d / ip.focal_y_px, d], 1)
        return pf + (np.array(CAM_T_MM) + pc @ np.array(CAM_R).T) @ Rf.T

    async def measure_with_wrist(self, x, y):
        """Wrist RealSense straight above (x, y): is there a cup here? Exact center, rim height, diameter."""
        try:
            P = await self.look_down_at(x, y, RIM_Z_GUESS + LOOK_CAM_ABOVE_RIM_MM)
        except Exception as e:  # can't get the camera there: skip this spot, nothing moved toward it
            return {"ok": False, "error": str(e)[:300]}
        objs = find_objects(P)
        if not objs:
            return {"ok": False, "error": f"no object above the table near ({x:.0f},{y:.0f})"}
        o = min(objs, key=lambda o: math.hypot(o["x"] - x, o["y"] - y))
        if math.hypot(o["x"] - x, o["y"] - y) > 120:
            return {"ok": False, "error": f"nothing within 12 cm of ({x:.0f},{y:.0f})", "objects": objs}
        if not o["is_cup"]:
            return {"ok": False, "error": f"object at ({o['x']:.0f},{o['y']:.0f}) is not a cup: {o['why']}",
                    "object": o}
        others = [q for q in objs if q is not o and math.hypot(q["x"] - o["x"], q["y"] - o["y"]) < 350]
        return {"ok": True, **o, "shift_from_candidate_mm": round(math.hypot(o["x"] - x, o["y"] - y), 1),
                "others": others}

    async def measure_centered(self, x, y):
        """Measure, then look again centred on the result while it moves more than RECENTER_MM."""
        m = await self.measure_with_wrist(x, y)
        for _ in range(2):
            if not m.get("ok") or m["shift_from_candidate_mm"] <= RECENTER_MM:
                break
            m = await self.measure_with_wrist(m["x"], m["y"])
        return m

    async def survey(self):
        """Look over the reachable table with the wrist camera; return every object, cups first."""
        found = []
        for (x, y) in SURVEY_XY:
            try:
                P = await self.look_down_at(x, y, SURVEY_CAM_Z)
            except Exception as e:
                found.append({"error": f"survey look at ({x},{y}) failed: {str(e)[:80]}"})
                continue
            for o in find_objects(P):
                if all(math.hypot(o["x"] - f.get("x", 1e9), o["y"] - f.get("y", 1e9)) > 40 for f in found):
                    found.append(o)
        objs = [f for f in found if "x" in f]
        return sorted(objs, key=lambda o: (not o["is_cup"], math.hypot(o["x"], o["y"])))

    @staticmethod
    def pick_heading(cup, reach, gz):
        """Most direct approach heading whose gripper corridor is clear of the other objects."""
        radial = math.degrees(math.atan2(cup["y"], cup["x"]))
        blockers = [q for q in cup.get("others", []) if q["rim_z"] > gz - 35]   # tall enough to touch the gripper
        tried = []
        for off in APPROACH_OFFSETS_DEG:
            h = math.radians(radial + off)
            u, n = (math.cos(h), math.sin(h)), (-math.sin(h), math.cos(h))
            s_min, s_max = -(reach + BACKOFF_MM), TIP_PAST_FLANGE_MM - reach      # along u, from the cup centre
            hit = None
            for q in blockers:
                dx, dy = q["x"] - cup["x"], q["y"] - cup["y"]
                s, t = dx * u[0] + dy * u[1], dx * n[0] + dy * n[1]
                rq = q["rim_diameter"] / 2 + CORRIDOR_MARGIN_MM
                ds = max(s_min - s, 0, s - s_max)
                dt = max(abs(t) - CORRIDOR_HALF_WIDTH_MM, 0)
                if math.hypot(ds, dt) < rq:
                    hit = q
                    break
            tried.append({"offset_deg": off, "blocked_by": [hit["x"], hit["y"]] if hit else None})
            if hit is None:
                return h, tried
        return None, tried

    async def grasp(self, cup):
        cx, cy, rim_z, rim_d = cup["x"], cup["y"], cup["rim_z"], cup["rim_diameter"]
        d_grip = TAPER * rim_d
        if d_grip > 80:
            return {"ok": False, "stage": "plan", "error": f"cup ~{d_grip:.0f} mm wide at grip height; gripper opens 85"}
        gz0 = max(GRIP_AXIS_MIN_Z, TABLE_Z + GRIP_FRACTION * (rim_z - TABLE_Z))   # grip point on the cup's axis
        heading, tried = self.pick_heading(cup, min(REACH_BASE_MM + d_grip / 2, REACH_MAX_MM), gz0)
        self.log["approach_headings_tried"] = tried
        if heading is None:
            return {"ok": False, "stage": "plan", "error": "every approach direction (straight +-45 deg) would sweep "
                    "the open gripper into a nearby object - give the cup ~10 cm of space. Nothing moved."}
        await self.gripper.do_command({"set": OPEN_POS})
        await self.speed(SPEED_TRAVEL)
        refused = []
        for pitch in PITCH_DEG:
            a, th = tilted_orientation(heading, pitch)
            o, p = (float(a[0]), float(a[1]), float(a[2])), math.radians(pitch)
            # the palm must stay off the cup at its widest (the rim): its horizontal distance from the cup's
            # axis is (reach - palm) cos(p); the fingertips must still reach the axis
            reach = PALM_MM + (rim_d / 2 + PALM_CLEAR_MM) / math.cos(p)
            if reach > TIPS_MM - 2.0:
                refused.append({"pitch_deg": pitch, "error": f"fingers too short at this tilt for a {rim_d:.0f} mm cup"})
                continue
            gz = gz0
            tip_z = gz - (TIP_PAST_FLANGE_MM - reach) * math.sin(p)
            if tip_z < TABLE_Z + TIP_ABOVE_TABLE_MM:
                gz += TABLE_Z + TIP_ABOVE_TABLE_MM - tip_z
            g = np.array([cx, cy, gz]) - reach * a                        # flange at the grip
            b = g - BACKOFF_MM * a                                         # flange at pre-grasp (back up the axis)
            above = max(ABOVE_MM, b[2])
            self.log["grasp_plan"] = {"heading_deg": round(math.degrees(heading), 1), "pitch_deg": pitch,
                                      "theta": th, "reach_mm": round(reach, 1), "grip_point_z": round(gz, 1),
                                      "flange_grip": np.round(g, 1).tolist(), "flange_pregrasp": np.round(b, 1).tolist(),
                                      "width_at_grip": round(d_grip, 1)}
            try:
                await self.flange_to(b[0], b[1], above, o, theta=th)      # above the pre-grasp point, tilted
                break
            except Exception as e:
                refused.append({"pitch_deg": pitch, "error": str(e)[:160]})
                self.log["pitches_refused"] = refused
        else:
            return {"ok": False, "stage": "plan", "error": f"Viam refused the side grip at every pitch {PITCH_DEG}. "
                    "Nothing moved toward the cup.", "refused": refused}
        await self.speed(SPEED_SLOW)
        if above > b[2] + 1.0:
            await self.flange_to(b[0], b[1], b[2], o, straight=True, theta=th)   # straight down beside the cup
        await self.flange_to(g[0], g[1], g[2], o, straight=True, theta=th)       # straight in along the axis
        target = max(0, int((d_grip - 15) * 10))
        await self.gripper.do_command({"grab_with_torque": {"position": target, "speed": GRIP_SPEED,
                                                            "torque": GRIP_TORQUE_PCT}})
        await asyncio.sleep(0.8)
        pos = await self.gpos()
        self.log["grip_pos"] = pos
        holding = pos is not None and target + 30 <= pos <= (d_grip + 8) * 10
        if not holding:
            await self.gripper.do_command({"set": OPEN_POS})
            await asyncio.sleep(0.5)
            await self.flange_to(b[0], b[1], b[2], o, straight=True, theta=th)
            await self.flange_to(b[0], b[1], above, o, straight=True, theta=th)
            return {"ok": False, "stage": "grip", "error": f"gripper at {pos} (target {target}); not holding the "
                                                           "cup - opened and backed out"}
        await self.flange_to(g[0], g[1], g[2] + LIFT_MM, o, straight=True, theta=th)
        await asyncio.sleep(0.5)
        pos2 = await self.gpos()
        self.log["after_lift_pos"] = pos2
        await self.speed(SPEED_TRAVEL)
        return {"ok": True, "holding_pos": pos2}

    async def pinch_rim(self, cup):
        """Pointing straight down at the near side of the rim, fingers along the line from the base: one goes
        inside the cup, one outside; pinch the top of the wall gently and lift. Works for any cup width."""
        cx, cy, rim_z, rim_r = cup["x"], cup["y"], cup["rim_z"], cup["rim_diameter"] / 2
        u = np.array([cx, cy]) / (math.hypot(cx, cy) or 1.0)          # from the base toward the cup
        px, py = np.array([cx, cy]) - rim_r * u                        # the near side of the rim
        if math.hypot(px, py) > MAX_REACH_MM:
            return {"ok": False, "stage": "plan", "error": "the rim is out of reach. Nothing moved."}
        o = (0.0, 0.0, -1.0)
        pinch_z = rim_z + TIPS_MM - PINCH_DEPTH_MM                     # flange height: fingertips 20 mm below the rim
        above_z = pinch_z + PINCH_ABOVE_MM
        axis = math.degrees(math.atan2(u[1], u[0]))
        th0 = theta_for_finger_axis(axis % 180.0, 0.0)
        open_counts = int(PINCH_OPEN_MM * 10)
        async def empty_close():  # what "closed on nothing" reads, in the air, while the arm moves over the rim
            before = await self.gpos()
            await self.gripper.do_command({"grab_with_torque": {"position": 0, "speed": GRIP_SPEED, "torque": PINCH_TORQUE_PCT}})
            e = await self.fingers_stopped(moving_from=before)
            await self.gripper.do_command({"set": open_counts})
            await self.fingers_stopped(moving_from=e)
            return e
        reference = asyncio.ensure_future(empty_close())
        self.log["pinch_plan"] = {"rim_point": [round(float(px), 1), round(float(py), 1)], "finger_axis_deg": round(axis, 1),
                                  "flange_pinch_z": round(pinch_z, 1)}
        await self.speed(SPEED_TRAVEL)
        th, refused = None, []
        for t in (th0, other_wrist(th0)):
            try:
                await self.flange_to(px, py, above_z, o, theta=t)          # above the rim
                th = t
                break
            except Exception as e:
                refused.append(str(e)[:160])
        empty = await reference                                             # fingers open again before going down
        self.log["pinch_plan"]["empty_close_pos"] = empty
        if th is None:
            return {"ok": False, "stage": "plan", "error": "Viam refused the pose above the rim. Nothing moved toward the cup.",
                    "refused": refused}
        self.log["pinch_plan"]["theta"] = th
        await self.speed(SPEED_SLOW)
        await self.flange_to(px, py, pinch_z, o, straight=True, theta=th)  # straight down around the wall
        await self.gripper.do_command({"grab_with_torque": {"position": 0, "speed": GRIP_SPEED, "torque": PINCH_TORQUE_PCT}})
        pos = await self.fingers_stopped(moving_from=open_counts)
        self.log["grip_pos"] = pos
        holding = pos is not None and empty is not None and pos >= empty + PINCH_MIN_WALL_MM * 10 and pos <= 150
        if not holding:
            await self.gripper.do_command({"set": open_counts})
            await asyncio.sleep(0.5)
            await self.flange_to(px, py, above_z, o, straight=True, theta=th)
            return {"ok": False, "stage": "grip", "error": f"fingers at {pos} (closed on nothing: {empty}); not on the "
                                                           "wall - opened and backed up"}
        await self.flange_to(px, py, pinch_z + LIFT_MM, o, straight=True, theta=th)
        self.log["after_lift_pos"] = await self.gpos()
        await self.speed(SPEED_TRAVEL)
        # how it's held, for placing: the cup's centre is rim_r along u from the fingers, at this wrist angle
        self.held = {"u": u, "rim_r": rim_r, "pinch_z": pinch_z, "theta": th, "o": o}
        return {"ok": True, "holding_pos": self.log["after_lift_pos"]}

    async def place_at(self, tx, ty):
        """Carry the pinched cup and stand it on the table with its centre at (tx, ty); let go, straight up.
        The wrist keeps its angle if Viam allows it there; else it turns 180, then +-90 (the cup swings round
        with it, so the fingers go where the cup's centre lands on the target). Refused everywhere: still held."""
        h = getattr(self, "held", None)
        if h is None:
            return {"ok": False, "stage": "place", "error": "not holding a cup"}
        if ty >= MAX_Y_MM:
            return {"ok": False, "stage": "place", "error": f"({tx:.0f},{ty:.0f}) is on the charger side (y >= {MAX_Y_MM:.0f})"}
        o, pz, th0 = h["o"], h["pinch_z"], h["theta"]
        carry_z = pz + LIFT_MM + 50.0                                    # cup's bottom 15 cm up while carrying
        await self.speed(SPEED_TRAVEL)
        tried, th = [], None
        for turn in (0.0, 180.0, 90.0, -90.0):
            t = (th0 + turn + 180.0) % 360.0 - 180.0
            yaw = math.radians(finger_axis_deg(t) - finger_axis_deg(th0))   # how far the hanging cup swings round
            u = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]) @ h["u"]
            fx, fy = np.array([tx, ty]) - h["rim_r"] * u                 # the fingers, over the cup's near rim
            if math.hypot(fx, fy) > MAX_REACH_MM:
                tried.append({"wrist": t, "error": "out of reach"})
                continue
            try:
                await self.flange_to(fx, fy, carry_z, o, theta=t)         # carry it over
                th = t
                break
            except Exception as e:
                tried.append({"wrist": t, "error": str(e)[:140]})
        self.log["place_tried"] = tried
        if th is None:
            return {"ok": False, "stage": "place", "error": "Viam refused carrying it there at every wrist angle; the cup "
                    "is still held (pick_cup.py --release sets it back down)", "tried": tried}
        self.log["place_plan"] = {"cup_centre": [round(tx, 1), round(ty, 1)], "flange": [round(float(fx), 1), round(float(fy), 1)],
                                  "wrist": th, "carry_flange_z": round(carry_z, 1), "set_down_flange_z": round(pz + 3.0, 1)}
        await self.speed(SPEED_SLOW)
        await self.flange_to(fx, fy, pz + 3.0, o, straight=True, theta=th)   # straight down: standing on the table
        before = await self.gpos()
        await self.gripper.do_command({"set": int(PINCH_OPEN_MM * 10)})
        await self.fingers_stopped(moving_from=before)
        await self.flange_to(fx, fy, pz + PINCH_ABOVE_MM, o, straight=True, theta=th)
        await self.speed(SPEED_TRAVEL)
        self.held = None
        return {"ok": True, "placed_at": [round(tx, 1), round(ty, 1)]}

    async def go_home(self) -> dict:
        """Park at HOME_XYZ, pointing down (a planned move, like every other); the wrist camera on any side."""
        await self.speed(SPEED_TRAVEL)
        errors = []
        for th in (0.0, 90.0, 180.0, -90.0):
            try:
                await self.flange_to(*HOME_XYZ, (0.0, 0.0, -1.0), theta=th)
                return {"ok": True, "at": list(HOME_XYZ)}
            except Exception as e:
                errors.append(f"wrist {th:.0f}: {str(e)[:120]}")
        return {"ok": False, "error": "couldn't park at " + str(HOME_XYZ) + ": " + "; ".join(errors)}

    async def release(self):
        f = await self.arm.get_end_position()
        if f.o_z < -0.9:  # pinched from above: set the cup back down, let go, straight up
            o = (f.o_x, f.o_y, f.o_z)
            await self.speed(SPEED_SLOW)
            await self.flange_to(f.x, f.y, f.z - LIFT_MM, o, straight=True, theta=f.theta)
            await self.gripper.do_command({"set": int(PINCH_OPEN_MM * 10)})
            await asyncio.sleep(0.5)
            await self.flange_to(f.x, f.y, f.z, o, straight=True, theta=f.theta)
            await self.speed(SPEED_TRAVEL)
            return
        await self.gripper.do_command({"set": OPEN_POS})
        await asyncio.sleep(0.5)
        await self.speed(SPEED_SLOW)
        o = (f.o_x, f.o_y, f.o_z)
        n = math.hypot(o[0], o[1]) or 1.0
        await self.flange_to(f.x - 80 * o[0] / n, f.y - 80 * o[1] / n, f.z, o, straight=True)
        await self.speed(SPEED_TRAVEL)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--look", action="store_true", help="find and measure only; never goes near the cup")
    ap.add_argument("--release", action="store_true", help="set the cup back down and let go")
    ap.add_argument("--side", action="store_true", help="tilted side grip instead of pinching the rim from above")
    ap.add_argument("--to", help="after picking, stand the cup with its centre here: x,y in armfarm03's frame (mm), "
                                 "e.g. 340,-589 (680 mm out, 60 deg to the right)")
    a = ap.parse_args()
    os.makedirs(STATE, exist_ok=True)
    cal = latest_calibration()
    if input("This MOVES THE REAL ARM. Someone on the E-stop? Type 'yes': ").strip() != "yes":
        print("Cancelled.")
        return
    async with await connect() as machine:
        p = CupPicker(machine)
        if a.release:
            await p.release()
            print("released")
            return
        result = {"calibration": cal[3]}
        cands = await p.find_overhead(cal)
        near = [c for c in cands if c["reachable"] and c["xy"][1] < MAX_Y_MM][:MAX_CANDIDATES]
        if True:
            # the wrist depth camera decides which candidate is the cup
            cup, checks = None, []
            for c in near:
                m = await p.measure_centered(*c["xy"])
                checks.append({"candidate_px": c["px"], **m})
                if m.get("ok") and math.hypot(m["x"], m["y"]) <= MAX_REACH_MM:
                    cup = m
                    break
            p.log["wrist_checks"] = checks
            if cup is None:
                # top camera couldn't point at it: search the reachable table with the wrist camera
                objs = await p.survey()
                p.log["survey"] = objs
                cups = [o for o in objs if o["is_cup"] and math.hypot(o["x"], o["y"]) <= MAX_REACH_MM
                        and o["y"] < MAX_Y_MM]
                if cups:
                    cup = await p.measure_centered(cups[0]["x"], cups[0]["y"])   # centred look for precision
                    checks.append(cup)
                    cup = cup if cup.get("ok") else None
            if cup is None:
                result.update(ok=False, stage="find cup", error="no cup found by the top camera or the wrist survey")
            elif a.look:
                result.update(ok=True, stage="measured (look only)", cup=cup)
            else:
                result.update(cup=cup, **await (p.grasp(cup) if a.side else p.pinch_rim(cup)))
                if a.to and result.get("ok") and not a.side:
                    tx, ty = (float(v) for v in a.to.split(","))
                    placed = await p.place_at(tx, ty)
                    result.update(place=placed, ok=placed["ok"], stage="placed" if placed["ok"] else placed["stage"])
                    if placed["ok"]:
                        result["home"] = await p.go_home()
        result.update(p.log)
        print(json.dumps(result, indent=2))
        with open(os.path.join(STATE, f"pick-cup-{time.strftime('%Y%m%d-%H%M%S')}.json"), "w") as fh:
            json.dump(result, fh, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
