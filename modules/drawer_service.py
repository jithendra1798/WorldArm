# Inline module for armfarm03: open and close a drawer of the 3-knob organizer.
#
# This reproduces, step by step and with the same numbers, the run that worked on the real arm
# (16 Oct, drawer 1): find knobs from straight above -> hover over the knob -> careful descent ->
# gentle grip -> open 12 cm -> close 12 cm -> release -> lift -> back to the overhead view -> check.
#
# Module attributes (CONFIGURE tab), names of existing resources:
#   {"arm": "arm", "gripper": "gripper", "camera": "cam", "knob_segmenter": "knob-segmenter"}
#
# Commands (CONTROL tab -> this service -> DoCommand):
#   {}  or  {"action": "all_drawers"}           -> open + close drawers 1, 2, 3 in one go
#   {"action": "all_drawers", "order": [3,2,1]} -> same, in another order
#   {"action": "drawer_run", "drawer": 2}       -> one drawer only (1 = leftmost facing the fronts)
#   {"action": "locate"}                        -> find + close-up correction + hover over knob 1, no touching
#   add "refine": true to any of these to use the slower two-look (180 deg) correction instead
#   {"action": "find_knobs"}                    -> overhead look only: knob positions, open direction
#   {"action": "release"}                       -> open the gripper to 3 cm and lift 6 cm
#   {"action": "open", "drawer": 2}             -> open one drawer 12 cm and leave it open; returns a drop point
#   {"action": "close", "drawer": 2}            -> close a drawer this service opened (no new look)
#   {"action": "status"}                        -> which drawers this service has open; never moves
#   {"action": "forget"}                        -> clear the remembered open drawers (after fixing by hand)

import asyncio
import json
import math
import os
import struct
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple

from typing_extensions import Self
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import Pose, PoseInFrame, ResourceName
from viam.proto.service.motion import Constraints, LinearConstraint
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic
from viam.services.motion import MotionClient
from viam.services.vision import VisionClient
from viam.utils import ValueTypes, struct_to_dict

# ---------------- numbers from the successful run ----------------
LOOK_CAMERA_HEIGHT_MM = 450.0     # camera height above the knobs for the overhead look (depth camera
                                  # needs > ~280 mm; 450 gave clean, consistent readings)
LOOK_WRISTS_DEG = [0.0, 90.0, -90.0, 180.0]  # wrist angles tried for the overhead look
GRASP_OUTWARD_MM = 8.0            # grasp point = knob + 8 mm along the open direction (5 mm brushed the box)
GRASP_Z_MM = 94.5                 # gripper reference height when gripping (knob ~40 mm above the table)
DESCENT_Z_MM = [214.5]            # hover height above the grasp point (then one smooth move down to GRASP_Z_MM)
DESCENT_WINDOW_S = 0.6            # smooth descent: trend of each joint load over this window
# Stop if a joint load is this far off its trend, for 2 readings in a row. Joint 2 (shoulder) carries
# most of the arm's weight and wobbles up to ~1.6 while moving in free air, so it gets a wider limit;
# a real push on the fingers also shows on joints 3 and 5. A ~7 N push on the fingers reaches these.
DESCENT_JOINT_LIMITS = [1.5, 3.0, 1.5, 1.5, 1.5, 1.5]
DESCENT_CONFIRM_POLLS = 2
DESCENT_WATCH_BELOW_MM = 145.0    # only judge contact below this gripper height (knob tops are ~60 mm)
DESCENT_WARMUP_S = 0.6            # ignore the start of the move (acceleration shows up in the loads)
DESCENT_POLL_S = 0.1
APPROACH_OPEN_POS = 350.0         # fingers open 3.5 cm on the way down (knob is 2.3 cm)
GRIP_TORQUE_PCT = 25              # gentle grip
GRIP_SPEED = 1500
HOLD_MIN_POS, HOLD_MAX_POS = 150.0, 300.0   # holding the knob read 213-215
TRAVEL_MM = 120.0                 # open / close distance
KNOB_PROTRUSION_MM = 20.0         # knob centre in front of the drawer face (measure it; used for the drop point)
DROP_FROM_FACE_MM = 70.0          # drop point: this far out from the organizer face, inside the open drawer
RELEASE_POS = 300.0               # open to 3 cm to let go
LIFT_MM = 60.0                    # lift after releasing
SLOW_DEG_S = 10.0                 # speed near the box
TRAVEL_DEG_S = 30.0               # speed for moves in free space
CLOSE_LOAD_TOLERANCE = 2.0        # after closing, loads should match the pre-pull loads within this
SETTLE_S = 0.5

# Close-up refinement of the knob position (self-correcting, no manual offsets).
# From a hover above the knob the camera looks at it twice, with the wrist turned 180 deg between
# the looks. Any error in the camera mounting (or in the knob height) shifts the two readings in
# opposite directions, so their average is the true knob position. Repeat until it stops moving.
REFINE_Z_MM = 320.0               # gripper height for the close-up looks (camera ~380 mm above the knobs,
                                  # so the whole row of three is in view from both wrist directions)
REFINE_MAX_ROUNDS = 1             # extra rounds centred on the knob (a 3rd round only ever moved 0.1-0.5 mm)
REFINE_DONE_MM = 2.0              # stop when the estimate moves less than this between rounds
REFINE_MAX_SHIFT_MM = 45.0        # refuse a correction larger than this (something is wrong)
KNOB_DIAMETER_MM = 23.0

# Measured camera correction for the overhead look (world mm to ADD to every knob position).
# From 7 runs on 5 box positions: overhead estimate vs. the two-look (180 deg) self-correction.
#   look wrist 0:  +3.9, +24.0  (spread +-0.5 mm over 5 runs)
#   look wrist 90: +27.0, +4.0  (spread +-1 mm over 2 runs)
# With these the overhead look alone is accurate, so the 180 deg two-look correction (several wrist
# turns) is only used when the overhead look had to use another wrist, or when asked ("refine": true).
OVERHEAD_CORRECTION = {0.0: (3.9, 24.0), 90.0: (27.0, 4.0)}
MAX_REACH_MM = 690.0              # grasp points further than this from the base can't be reached
                                  # (662 mm worked, 738 mm was unreachable)

# Knob row shape checks
KNOB_SPACING_MIN_MM, KNOB_SPACING_MAX_MM = 50.0, 120.0
KNOB_LINE_TOL_MM = 15.0

# Search grid for finding the box when it isn't in view (flange x, y for the overhead look)
SEARCH_XY = [(380.0, -184.0), (233.0, -360.0), (380.0, -360.0), (300.0, -480.0), (420.0, 0.0), (300.0, 120.0)]
SEARCH_Z_MM = 504.0

# Camera mounting relative to the flange (from armfarm03's `cam` frame; matches Viam's frame system)
CAM_T_MM = (-83.0, 14.0, 18.0)
CAM_OV = (0.030391, 0.003538, 0.999532, -97.731173)


# ---------------- 3D helpers ----------------
def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]


def _mm(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _ap(r, v):
    return [sum(r[i][k] * v[k] for k in range(3)) for i in range(3)]


def _ov(ox, oy, oz, theta_deg):
    n = math.sqrt(ox * ox + oy * oy + oz * oz) or 1.0
    ox, oy, oz = ox / n, oy / n, oz / n
    lat = math.acos(max(-1.0, min(1.0, oz)))
    lon = math.atan2(oy, ox) if 1 - abs(oz) > 1e-4 else 0.0
    return _mm(_mm(_rz(lon), _ry(lat)), _rz(math.radians(theta_deg)))


CAM_R = _ov(*CAM_OV)


def _wrap(deg: float) -> float:
    """Angle in (-180, 180]."""
    d = (deg + 180.0) % 360.0 - 180.0
    return 180.0 if d == -180.0 else d


def _decode_depth(data: bytes):
    if data[:8] != b"DEPTHMAP":
        raise ValueError("unexpected depth format")
    w = struct.unpack(">Q", data[8:16])[0]
    h = struct.unpack(">Q", data[16:24])[0]
    body = data[24:24 + 2 * w * h]
    big = struct.unpack(f">{w * h}H", body)
    little = struct.unpack(f"<{w * h}H", body)
    ok = lambda arr: sum(1 for d in arr[:: max(1, len(arr) // 2000)] if 100 < d < 3000)
    return int(w), int(h), (big if ok(big) >= ok(little) else little)


class MyGenericService(Generic, EasyResource):
    # Must match what the Viam app generated for this inline module.
    MODEL: ClassVar[Model] = Model(
        ModelFamily("hackathons", "ef05651f-50f0-4eb1-bed8-af81c318e62c"), "generic-service")

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        self = super().new(config, dependencies)
        a = struct_to_dict(config.attributes)
        self.gripper_name = a.get("gripper", "gripper")
        self.camera_name = a.get("camera", "cam")
        self.arm = dependencies[Arm.get_resource_name(a.get("arm", "arm"))]
        self.gripper = dependencies[Gripper.get_resource_name(self.gripper_name)]
        self.camera = dependencies[Camera.get_resource_name(self.camera_name)]
        self.knob_seg = dependencies[VisionClient.get_resource_name(a.get("knob_segmenter", "knob-segmenter"))]
        self.knob_det = dependencies[VisionClient.get_resource_name(a.get("knob_detector", "knob-color"))]
        self.motion = dependencies[MotionClient.get_resource_name("builtin")]
        self.state_file = os.path.join(os.environ.get("VIAM_MODULE_DATA", "/tmp"), "drawer_last_look.json")
        self.open_state_file = os.path.join(os.environ.get("VIAM_MODULE_DATA", "/tmp"), "drawer_open_state.json")
        self.open_drawers = self._load_open_state()
        return self

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> Tuple[Sequence[str], Sequence[str]]:
        a = struct_to_dict(config.attributes)
        return [a.get("arm", "arm"), a.get("gripper", "gripper"), a.get("camera", "cam"),
                a.get("knob_segmenter", "knob-segmenter"), a.get("knob_detector", "knob-color"), "builtin"], []

    # ---------------- low-level helpers ----------------
    async def _speed(self, deg_s: float):
        try:
            await self.arm.do_command({"set_speed": float(deg_s)})
        except Exception:
            pass

    async def _loads(self) -> Optional[List[float]]:
        try:
            r = await self.arm.do_command({"load": ""})
            return [float(v) for v in r.get("load")]
        except Exception:
            return None

    async def _watched_move(self, x, y, z, theta) -> Dict[str, Any]:
        """Smooth straight move to (x, y, z) while watching the joint loads.
        While the arm moves, gravity makes each joint load drift steadily, so every reading is
        compared with the straight-line trend of the readings from the previous DESCENT_WINDOW_S.
        Pressing on something breaks that trend. Watching starts once the gripper is below
        DESCENT_WATCH_BELOW_MM (above that nothing can be touched)."""
        loop = asyncio.get_event_loop()
        task = asyncio.ensure_future(self._gripper_to(x, y, z, theta))
        t0 = loop.time()
        hist: List[Tuple[float, List[float]]] = []
        worst, gz, over = 0.0, None, 0
        while not task.done():
            await asyncio.sleep(DESCENT_POLL_S)
            now = loop.time()
            cur = await self._loads()
            try:
                gz = (await self.arm.get_end_position()).z - 105.0
            except Exception:
                pass
            if not cur or now - t0 < DESCENT_WARMUP_S:
                continue
            past = [h for h in hist if now - h[0] <= DESCENT_WINDOW_S]
            hist.append((now, cur))
            if gz is None or gz > DESCENT_WATCH_BELOW_MM or len(past) < 4 or now - past[0][0] < 0.6 * DESCENT_WINDOW_S:
                continue
            # per-joint straight-line fit over the window, extrapolated to now
            ts = [h[0] for h in past]
            tm = sum(ts) / len(ts)
            stt = sum((t - tm) ** 2 for t in ts) or 1e-9
            dev = []
            for j in range(len(cur)):
                vs = [h[1][j] for h in past]
                vm = sum(vs) / len(vs)
                slope = sum((t - tm) * (v - vm) for t, v in zip(ts, vs)) / stt
                dev.append(cur[j] - (vm + slope * (now - tm)))
            ratio = [abs(v) / lim for v, lim in zip(dev, DESCENT_JOINT_LIMITS)]
            d = max(ratio)
            worst = max(worst, d)
            over = over + 1 if d > 1.0 else 0
            if over >= DESCENT_CONFIRM_POLLS and not task.done():
                try:
                    await self.arm.stop()
                except Exception:
                    pass
                task.cancel()
                try:
                    await task
                except BaseException:
                    pass
                return {"contact": True, "z": round(gz, 1), "change": round(d, 2), "limits": DESCENT_JOINT_LIMITS,
                        "per_joint": [round(v, 2) for v in dev]}
        await task          # re-raise a planning error, if any
        return {"contact": False, "max_off_trend_vs_limit": round(worst, 2), "seconds": round(loop.time() - t0, 1)}

    async def _settled_loads(self) -> Optional[List[float]]:
        """Joint loads after the arm has settled, averaged over 3 readings."""
        await asyncio.sleep(0.8)
        reads = []
        for _ in range(3):
            r = await self._loads()
            if r:
                reads.append(r)
            await asyncio.sleep(0.1)
        if not reads:
            return None
        return [sum(v) / len(reads) for v in zip(*reads)]

    async def _gpos(self) -> Optional[float]:
        try:
            return float((await self.gripper.do_command({"get": True})).get("pos"))
        except Exception:
            return None

    async def _gripper_to(self, x, y, z, theta, straight=True):
        dest = PoseInFrame(reference_frame="world", pose=Pose(x=x, y=y, z=z, o_x=0, o_y=0, o_z=-1, theta=theta))
        cons = (Constraints(linear_constraint=[LinearConstraint(line_tolerance_mm=3, orientation_tolerance_degs=2)])
                if straight else None)
        await self.motion.move(component_name=self.gripper_name, destination=dest, constraints=cons)

    async def _cam_to_world(self):
        f = await self.arm.get_end_position()
        rf, pf = _ov(f.o_x, f.o_y, f.o_z, f.theta), [f.x, f.y, f.z]

        def tw(p):
            q = [a + b for a, b in zip(CAM_T_MM, _ap(CAM_R, p))]
            return [a + b for a, b in zip(pf, _ap(rf, q))]
        return tw, rf

    # ---------------- seeing the knobs ----------------
    async def _knobs_here(self) -> List[Dict[str, float]]:
        """Knob centers in world coordinates from Viam's 3D knob segmenter (current view)."""
        objs = await self.knob_seg.get_object_point_clouds(self.camera_name)
        tw, _ = await self._cam_to_world()
        out = []
        for o in objs:
            frame = o.geometries.reference_frame or "world"
            for g in o.geometries.geometries:
                c = [g.center.x, g.center.y, g.center.z]
                w = c if frame == "world" else tw(c)
                out.append({"x": w[0], "y": w[1], "z": w[2], "points": len(o.point_cloud)})
        # merge pieces of the same ball (e.g. a ball and its stem tab split in two)
        merged: List[Dict[str, float]] = []
        for k in sorted(out, key=lambda k: -k["points"]):
            if all(math.hypot(k["x"] - m["x"], k["y"] - m["y"]) > 25 for m in merged):
                merged.append(k)
        return merged

    @staticmethod
    def _row(knobs: List[Dict[str, float]]):
        """Pick 3 knobs in a straight, evenly spaced row. Returns them ordered along the row, or None."""
        import itertools
        best, score = None, 1e9
        for tri in itertools.combinations(knobs, 3):
            pts = sorted(tri, key=lambda k: k["y"])
            for perm in (pts, sorted(tri, key=lambda k: k["x"])):
                a, m, b = perm
                L = math.hypot(b["x"] - a["x"], b["y"] - a["y"])
                if L < 1e-6:
                    continue
                u = [(b["x"] - a["x"]) / L, (b["y"] - a["y"]) / L]
                along = (m["x"] - a["x"]) * u[0] + (m["y"] - a["y"]) * u[1]
                off = abs(-(m["x"] - a["x"]) * u[1] + (m["y"] - a["y"]) * u[0])
                d1, d2 = along, L - along
                if not (KNOB_SPACING_MIN_MM <= d1 <= KNOB_SPACING_MAX_MM and
                        KNOB_SPACING_MIN_MM <= d2 <= KNOB_SPACING_MAX_MM) or off > KNOB_LINE_TOL_MM:
                    continue
                s = off + abs(d1 - d2)
                if s < score:
                    best, score = ([a, m, b], u), s
        return best

    async def _box_side_sign(self, mid, n) -> Optional[float]:
        """Which side of the knob row is the box body? Compare surface heights 7 cm to each side
        (depth image). Returns +1 if the box is on the +n side, -1 if on the -n side, None if unsure."""
        try:
            props = await self.camera.get_properties()
            ip = props.intrinsic_parameters
            res = await self.camera.get_images()
            imgs = res[0] if isinstance(res, tuple) else res
            dimg = next(im for im in imgs if "dep" in (getattr(im, "mime_type", "") or "")
                        or "depth" in (getattr(im, "name", "") or "").lower())
            w, h, px = _decode_depth(dimg.data)
            tw, rf = await self._cam_to_world()
            f = await self.arm.get_end_position()
            pf = [f.x, f.y, f.z]
            rft = [[rf[j][i] for j in range(3)] for i in range(3)]
            crt = [[CAM_R[j][i] for j in range(3)] for i in range(3)]

            def height(x, y):
                q = _ap(rft, [x - pf[0], y - pf[1], mid[2] - pf[2]])
                c = _ap(crt, [q[i] - CAM_T_MM[i] for i in range(3)])
                if c[2] <= 1:
                    return None
                u = int(ip.focal_x_px * c[0] / c[2] + ip.center_x_px)
                v = int(ip.focal_y_px * c[1] / c[2] + ip.center_y_px)
                vals = []
                for dv in range(-4, 5, 2):
                    for du in range(-4, 5, 2):
                        uu, vv = u + du, v + dv
                        if 0 <= uu < w and 0 <= vv < h and px[vv * w + uu] > 0:
                            d = px[vv * w + uu]
                            p = [(uu - ip.center_x_px) * d / ip.focal_x_px, (vv - ip.center_y_px) * d / ip.focal_y_px, d]
                            vals.append(tw(p)[2])
                return sorted(vals)[len(vals) // 2] if vals else None
            hp = height(mid[0] + n[0] * 70, mid[1] + n[1] * 70)
            hm = height(mid[0] - n[0] * 70, mid[1] - n[1] * 70)
            if hp is None or hm is None or abs(hp - hm) < 15:
                return None
            return 1.0 if hp > hm else -1.0
        except Exception:
            return None

    async def _look_at(self, target) -> Optional[float]:
        """Put the camera straight above `target`, LOOK_CAMERA_HEIGHT_MM up. Returns the wrist used."""
        for th in LOOK_WRISTS_DEG:
            off = _ap(_ov(0, 0, -1, th), list(CAM_T_MM))
            fx, fy = target[0] - off[0], target[1] - off[1]
            fz = target[2] + LOOK_CAMERA_HEIGHT_MM - off[2]
            try:
                await self.arm.move_to_position(Pose(x=fx, y=fy, z=fz, o_x=0, o_y=0, o_z=-1, theta=th))
                await asyncio.sleep(SETTLE_S + 0.3)
                return th
            except Exception:
                continue
        return None

    async def find_knobs(self) -> Dict[str, Any]:
        """Overhead look: find the 3-knob row, center the camera on it, read it again, work out the
        open direction and drawer numbering."""
        await self._speed(TRAVEL_DEG_S)
        row = self._row(await self._knobs_here())
        tried = ["current view"]
        if row is None:
            spots = []
            try:
                with open(self.state_file) as f:
                    spots.append(tuple(json.load(f)["mid"]))
            except Exception:
                pass
            for spot in spots:
                if await self._look_at([spot[0], spot[1], 40.0]) is not None:
                    tried.append("last known position")
                    row = self._row(await self._knobs_here())
                    if row:
                        break
            if row is None:
                for (x, y) in SEARCH_XY:
                    try:
                        await self.arm.move_to_position(Pose(x=x, y=y, z=SEARCH_Z_MM, o_x=0, o_y=0, o_z=-1, theta=0))
                        await asyncio.sleep(SETTLE_S + 0.3)
                    except Exception:
                        continue
                    tried.append(f"search ({x:.0f},{y:.0f})")
                    row = self._row(await self._knobs_here())
                    if row:
                        break
        if row is None:
            return {"found": False, "views_tried": tried}
        # center the camera straight above the middle knob and read again (clean, consistent view)
        mid0 = row[0][1]
        wrist = await self._look_at([mid0["x"], mid0["y"], mid0["z"]])
        row2 = self._row(await self._knobs_here()) if wrist is not None else None
        if row2:
            row = row2
        knobs, u = row
        mid = [sum(k[c] for k in knobs) / 3 for c in ("x", "y", "z")]
        n = [-u[1], u[0]]
        sign = await self._box_side_sign(mid, n)
        if sign is None:   # fallback: drawers face the robot -> open toward the robot base
            sign = 1.0 if (n[0] * -mid[0] + n[1] * -mid[1]) < 0 else -1.0
            pull_from = "fallback (toward robot)"
        else:
            pull_from = "depth"
        box_dir = [n[0] * sign, n[1] * sign]
        pull = [-box_dir[0], -box_dir[1]]
        # drawer 1 = leftmost for a person facing the drawer fronts (looking along -pull)
        left = [pull[1], -pull[0]]   # left of the facing direction (-pull), rotated +90 deg
        ordered = sorted(knobs, key=lambda k: -(k["x"] * left[0] + k["y"] * left[1]))
        try:
            with open(self.state_file, "w") as f:
                json.dump({"mid": mid[:2]}, f)
        except Exception:
            pass
        return {"found": True, "views_tried": tried, "look_wrist": wrist,
                "knobs": [{"drawer": i + 1, "x": round(k["x"], 1), "y": round(k["y"], 1), "z": round(k["z"], 1)}
                          for i, k in enumerate(ordered)],
                "row_dir": [round(u[0], 4), round(u[1], 4)], "open_dir": [round(pull[0], 4), round(pull[1], 4)],
                "open_dir_from": pull_from, "mid": [round(v, 1) for v in mid]}

    # ---------------- the drawer run ----------------
    @staticmethod
    def _wrist_for(u, box_dir) -> Tuple[float, float]:
        """Wrist angle with the fingers closing along the knob row and the camera on the box side,
        plus the half-turn alternative (fingers still along the row, camera on the knob side).
        The fingers close along the wrist camera's image x axis."""
        cands = []
        t = -180.0
        while t < 180.0:
            rf = _ov(0, 0, -1, t - 180.0)                 # flange orientation for gripper wrist t
            c = _ap(rf, _ap(CAM_R, [1.0, 0.0, 0.0]))
            n = math.hypot(c[0], c[1]) or 1.0
            align = abs((c[0] * u[0] + c[1] * u[1]) / n)
            cam = _ap(rf, list(CAM_T_MM))
            side = cam[0] * box_dir[0] + cam[1] * box_dir[1]
            cands.append((align, side, t))
            t += 0.5
        top = max(c[0] for c in cands)
        aligned = [c for c in cands if c[0] >= top - 0.0005]       # fingers along the row
        t = max(aligned, key=lambda c: c[1])[2]                     # camera on the box side
        return t, (t + 180.0 if t <= 0 else t - 180.0)

    async def _knobs_in_view(self, knob_z, box_dir) -> List[Dict[str, float]]:
        """All knobs the colour detector sees in the current view, each turned into a world (x, y)
        on the plane z = knob_z. Small red tabs are ignored; balls cut by the image edge or partly
        hidden behind a drawer front are centered from their uncut side."""
        props = await self.camera.get_properties()
        ip = props.intrinsic_parameters
        W, H = ip.width_px or 1280, ip.height_px or 720
        tw, rf = await self._cam_to_world()
        o = tw([0.0, 0.0, 0.0])
        depth = max(o[2] - knob_z, 1.0)
        dexp = KNOB_DIAMETER_MM * ip.focal_x_px / depth
        # which way is the box in the image? (drawer fronts hide the box-side part of each ball)
        rft = [[rf[j][i] for j in range(3)] for i in range(3)]
        crt = [[CAM_R[j][i] for j in range(3)] for i in range(3)]
        box_img = _ap(crt, _ap(rft, [box_dir[0], box_dir[1], 0.0]))
        box_up = box_img[1] < 0
        out = []
        for d in await self.knob_det.get_detections_from_camera(self.camera_name):
            w, h = d.x_max - d.x_min, d.y_max - d.y_min
            if max(w, h) < 0.6 * dexp:
                continue
            dia = w if (d.x_min > 1 and d.x_max < W - 2) else (h if (d.y_min > 1 and d.y_max < H - 2) else dexp)
            if d.x_min <= 1:
                u = d.x_max - dia / 2
            elif d.x_max >= W - 2:
                u = d.x_min + dia / 2
            else:
                u = (d.x_min + d.x_max) / 2
            if d.y_min <= 1:
                v = d.y_max - dia / 2
            elif d.y_max >= H - 2:
                v = d.y_min + dia / 2
            elif h < 0.85 * dia:
                v = d.y_max - dia / 2 if box_up else d.y_min + dia / 2
            else:
                v = (d.y_min + d.y_max) / 2
            q = tw([(u - ip.center_x_px) / ip.focal_x_px, (v - ip.center_y_px) / ip.focal_y_px, 1.0])
            dv = [q[i] - o[i] for i in range(3)]
            if abs(dv[2]) < 1e-6:
                continue
            t = (knob_z - o[2]) / dv[2]
            out.append({"x": o[0] + t * dv[0], "y": o[1] + t * dv[1], "px": [round(u), round(v)]})
        return out

    @staticmethod
    def _label_row(seen, L):
        """Given the knobs seen in one view and the drawer-1 -> drawer-3 direction L, return the three
        knobs in drawer order, or None unless exactly one clean row of three is visible."""
        if len(seen) < 3:
            return None
        best = None
        import itertools
        for tri in itertools.combinations(seen, 3):
            s = sorted(tri, key=lambda k: k["x"] * L[0] + k["y"] * L[1])
            d1 = math.hypot(s[1]["x"] - s[0]["x"], s[1]["y"] - s[0]["y"])
            d2 = math.hypot(s[2]["x"] - s[1]["x"], s[2]["y"] - s[1]["y"])
            if not (KNOB_SPACING_MIN_MM <= d1 <= KNOB_SPACING_MAX_MM and KNOB_SPACING_MIN_MM <= d2 <= KNOB_SPACING_MAX_MM):
                continue
            if best is None or abs(d1 - d2) < best[0]:
                best = (abs(d1 - d2), s)
        return best[1] if best else None

    async def _two_looks(self, gx, gy, t_main, t_alt, knob_z, box_dir, read):
        """Look from (gx, gy, REFINE_Z_MM) with the wrist at t_alt, then at t_main (180 deg apart).
        `read(seen)` turns the knobs seen in one view into the result for that view (or None).
        Returns {"alt": ..., "main": ...} or the reason it failed."""
        out = {}
        for name, t in (("alt", t_alt), ("main", t_main)):
            try:
                await self._gripper_to(gx, gy, REFINE_Z_MM, t, straight=False)
            except Exception as e:
                return {"error": f"wrist {t:.0f}: {str(e)[:90]}"}
            await asyncio.sleep(SETTLE_S + 0.3)
            res = read(await self._knobs_in_view(knob_z, box_dir))
            if res is None:
                return {"error": f"wrist {t:.0f}: knobs not clearly in view"}
            out[name] = res
        return out

    async def _refine_knob(self, fk, drawer, pull, wrist, wrist_alt) -> Dict[str, Any]:
        """Close-up, self-correcting knob position (no manual offsets).
        Two looks with the wrist turned 180 deg between them: a camera-mounting error pushes the two
        reads in opposite directions, so their average is the true position.
          Round 1: hover over the middle knob so the whole row is in view, number the knobs, average.
                   The difference between each view and the average is that view's error.
          Round 2+: hover over the chosen knob itself (most precise), remove the known per-view error
                   only to pick the right ball, average the raw reads again. Repeat until it settles."""
        box_dir = [-pull[0], -pull[1]]
        ks = {k["drawer"]: k for k in fk["knobs"]}
        n = math.hypot(ks[3]["x"] - ks[1]["x"], ks[3]["y"] - ks[1]["y"]) or 1.0
        L = [(ks[3]["x"] - ks[1]["x"]) / n, (ks[3]["y"] - ks[1]["y"]) / n]
        kz = ks[drawer]["z"]
        rounds: List[Dict[str, Any]] = []

        # Round 1 - whole row, centered on the middle knob
        mx, my = ks[2]["x"] + pull[0] * GRASP_OUTWARD_MM, ks[2]["y"] + pull[1] * GRASP_OUTWARD_MM
        looks, pair = None, None
        for off in (-45.0, 0.0, 90.0, 45.0):     # -45 is the pair that has always been reachable
            pair = (_wrap(wrist + off), _wrap(wrist_alt + off))
            res = await self._two_looks(mx, my, pair[0], pair[1], kz, box_dir,
                                        lambda seen: self._label_row(seen, L))
            if "error" not in res:
                looks = res
                break
            rounds.append({"round": 1, "wrists": list(pair), "error": res["error"]})
        if looks is None:
            return {"ok": False, "rounds": rounds, "error": "could not get two opposite views of the knob row"}
        a, m = looks["alt"][drawer - 1], looks["main"][drawer - 1]
        est = [(a["x"] + m["x"]) / 2, (a["y"] + m["y"]) / 2]
        bias = {"alt": [a["x"] - est[0], a["y"] - est[1]], "main": [m["x"] - est[0], m["y"] - est[1]]}
        rounds.append({"round": 1, "wrists": list(pair), "estimate": [round(est[0], 1), round(est[1], 1)],
                       "view_disagreement_mm": round(math.hypot(a["x"] - m["x"], a["y"] - m["y"]), 1),
                       "row_spacing_mm": [round(math.hypot(looks["main"][i + 1]["x"] - looks["main"][i]["x"],
                                                           looks["main"][i + 1]["y"] - looks["main"][i]["y"]), 1)
                                          for i in range(2)]})

        # Round 2+ - centered on the chosen knob
        for r in range(2, REFINE_MAX_ROUNDS + 2):
            gx, gy = est[0] + pull[0] * GRASP_OUTWARD_MM, est[1] + pull[1] * GRASP_OUTWARD_MM

            def pick(name):
                def f(seen):
                    if not seen:
                        return None
                    c = min(seen, key=lambda k: math.hypot(k["x"] - bias[name][0] - est[0],
                                                           k["y"] - bias[name][1] - est[1]))
                    d = math.hypot(c["x"] - bias[name][0] - est[0], c["y"] - bias[name][1] - est[1])
                    return c if d < 25.0 else None
                return f
            out = {}
            for name, t in (("alt", pair[1]), ("main", pair[0])):
                try:
                    await self._gripper_to(gx, gy, REFINE_Z_MM, t, straight=False)
                except Exception as e:
                    out = {"error": f"wrist {t:.0f}: {str(e)[:90]}"}
                    break
                await asyncio.sleep(SETTLE_S + 0.3)
                c = pick(name)(await self._knobs_in_view(kz, box_dir))
                if c is None:
                    out = {"error": f"wrist {t:.0f}: knob not found where expected"}
                    break
                out[name] = c
            if "error" in out:
                rounds.append({"round": r, "error": out["error"], "note": "kept the round-1 estimate"})
                break
            a, m = out["alt"], out["main"]
            new = [(a["x"] + m["x"]) / 2, (a["y"] + m["y"]) / 2]
            step = math.hypot(new[0] - est[0], new[1] - est[1])
            rounds.append({"round": r, "estimate": [round(new[0], 1), round(new[1], 1)], "moved_mm": round(step, 1),
                           "view_disagreement_mm": round(math.hypot(a["x"] - m["x"], a["y"] - m["y"]), 1)})
            if step > 15.0:      # should only be a few mm; don't trust a jump
                rounds[-1]["note"] = "jump too big, kept the previous estimate"
                break
            est = new
            if step < REFINE_DONE_MM:
                break

        shift = math.hypot(est[0] - ks[drawer]["x"], est[1] - ks[drawer]["y"])
        if shift > REFINE_MAX_SHIFT_MM:
            return {"ok": False, "rounds": rounds,
                    "error": f"close-up says the knob is {shift:.0f} mm from the overhead estimate - not trusting it"}
        return {"ok": True, "x": est[0], "y": est[1], "shift_from_overhead_mm": round(shift, 1), "rounds": rounds}

    async def _prepare(self, drawer: int, log: Dict[str, Any]):
        """Overhead look + close-up self-correction for `drawer`. Returns (fk, ref, wrist, wrist_alt) or None."""
        fk = await self.find_knobs()
        log["find"] = fk
        if not fk.get("found"):
            log.update({"success": False, "stage": "find knobs"})
            return None
        if not any(k["drawer"] == drawer for k in fk["knobs"]):
            log.update({"success": False, "stage": "find knobs", "error": f"no drawer {drawer}"})
            return None
        u, pull = fk["row_dir"], fk["open_dir"]
        wrist, wrist_alt = self._wrist_for(u, [-pull[0], -pull[1]])
        await self.gripper.do_command({"set": APPROACH_OPEN_POS})
        await self._speed(TRAVEL_DEG_S)
        corr = OVERHEAD_CORRECTION.get(float(fk.get("look_wrist") or 0.0)) if fk.get("look_wrist") is not None else None
        if corr is not None and not getattr(self, "force_refine", False):
            k = next(k for k in fk["knobs"] if k["drawer"] == drawer)
            ref = {"ok": True, "x": k["x"] + corr[0], "y": k["y"] + corr[1],
                   "from": f"overhead look (wrist {fk['look_wrist']:.0f}) + measured camera correction {list(corr)}"}
        else:
            ref = await self._refine_knob(fk, drawer, pull, wrist, wrist_alt)
        log["refine"] = ref
        if not ref.get("ok"):
            log.update({"success": False, "stage": "refine knob position"})
            return None
        return fk, ref, wrist, wrist_alt

    async def _operate(self, knob_xy, pull, wrist, wrist_alt, log: Dict[str, Any],
                       locate_only: bool = False, from_hover: bool = False) -> Dict[str, Any]:
        """Hover over the knob -> smooth watched descent -> gentle grip -> open 12 cm -> close 12 cm ->
        release -> straight up to the hover height. Returns {"ok": ..., "final_knob": [x, y], "wrist": ...}.
        from_hover: the gripper is already at hover height (next knob of the same box) - slide over
        to the new hover in a straight line instead of planning a free move."""
        gx = knob_xy[0] + pull[0] * GRASP_OUTWARD_MM
        gy = knob_xy[1] + pull[1] * GRASP_OUTWARD_MM
        hz = DESCENT_Z_MM[0]

        # 1. hover above the grasp point, fingers open 3.5 cm
        await self.gripper.do_command({"set": APPROACH_OPEN_POS})
        await self._speed(TRAVEL_DEG_S)
        used = None
        for t in (wrist, wrist_alt):
            for straight in ((True, False) if from_hover else (False,)):
                try:
                    await self._gripper_to(gx, gy, hz, t, straight=straight)
                    used = t
                    break
                except Exception as e:
                    log.setdefault("hover_errors", []).append(str(e)[:120])
            if used is not None:
                break
        if used is None:
            return {"ok": False, "stage": "hover"}
        log["grasp"] = {"x": round(gx, 1), "y": round(gy, 1), "z": GRASP_Z_MM, "wrist": used,
                        "camera_on_box_side": used == wrist}
        if locate_only:
            return {"ok": True, "stage": "hovering above the grasp point", "wrist": used}

        # 2. careful descent: one smooth, slow, straight move from the hover down to the grip height,
        #    joint loads watched the whole way; stops and backs off on contact.
        await self._speed(SLOW_DEG_S)
        mon = await self._watched_move(gx, gy, GRASP_Z_MM, used)
        log["descent"] = mon
        if mon.get("contact"):
            try:
                await self.arm.do_command({"clear_error": True})
            except Exception:
                pass
            await self._gripper_to(gx, gy, hz, used)
            return {"ok": False, "stage": "descent", "error": f"contact felt near z={mon.get('z')} "
                    f"(load off trend {mon.get('change')}x the limit) - stopped and backed off, nothing forced"}

        # 3. gentle grip and check
        await self.gripper.do_command({"grab_with_torque": {"position": 0, "speed": GRIP_SPEED,
                                                            "torque": GRIP_TORQUE_PCT}})
        await asyncio.sleep(SETTLE_S)
        pos = await self._gpos()
        log["grip_pos"] = pos
        if pos is None or not (HOLD_MIN_POS <= pos <= HOLD_MAX_POS):
            await self.gripper.do_command({"set": APPROACH_OPEN_POS})
            await self._gripper_to(gx, gy, hz, used)
            return {"ok": False, "stage": "grip", "error": "fingers did not close on the knob"}
        loads_closed = await self._loads()

        # 4. open the drawer 12 cm (one straight move along the open direction)
        ox, oy = gx + pull[0] * TRAVEL_MM, gy + pull[1] * TRAVEL_MM
        await self._gripper_to(ox, oy, GRASP_Z_MM, used)
        pos = await self._gpos()
        log["opened"] = {"to": [round(ox, 1), round(oy, 1)], "grip_pos": pos}
        if pos is None or not (HOLD_MIN_POS <= pos <= HOLD_MAX_POS):
            await self.gripper.do_command({"set": RELEASE_POS})
            await self._gripper_to(ox, oy, GRASP_Z_MM + LIFT_MM, used)
            return {"ok": False, "stage": "open", "error": "lost the grip while opening - released "
                    "and lifted; drawer may be partly open"}
        await asyncio.sleep(SETTLE_S)

        # 5. close the drawer 12 cm (straight back to the grasp point)
        await self._gripper_to(gx, gy, GRASP_Z_MM, used)
        loads_after = await self._loads()
        if loads_closed and loads_after:
            diff = max(abs(a - b) for a, b in zip(loads_after, loads_closed))
            log["closed"] = {"load_diff_vs_before": round(diff, 2), "fully_closed": diff <= CLOSE_LOAD_TOLERANCE}
        # where the knob really is now (the fingers are holding it, drawer closed)
        f = await self.arm.get_end_position()
        final_knob = [f.x - pull[0] * GRASP_OUTWARD_MM, f.y - pull[1] * GRASP_OUTWARD_MM]
        log["final_knob"] = [round(final_knob[0], 1), round(final_knob[1], 1)]

        # 6. release and go straight up to the hover height
        await self.gripper.do_command({"set": RELEASE_POS})
        await asyncio.sleep(SETTLE_S)
        await self._speed(SLOW_DEG_S)
        await self._gripper_to(f.x, f.y, GRASP_Z_MM + LIFT_MM, used)
        await self._speed(TRAVEL_DEG_S)
        await self._gripper_to(f.x, f.y, hz, used)
        return {"ok": True, "final_knob": final_knob, "wrist": used}

    async def drawer_run(self, drawer: int, locate_only: bool = False) -> Dict[str, Any]:
        log: Dict[str, Any] = {}
        prep = await self._prepare(drawer, log)
        if prep is None:
            return log
        fk, ref, wrist, wrist_alt = prep
        res = await self._operate([ref["x"], ref["y"]], fk["open_dir"], wrist, wrist_alt, log, locate_only)
        if not res["ok"]:
            return {"success": False, "stage": res.get("stage"), "error": res.get("error"), **log}
        if locate_only:
            return {"success": True, "drawer": drawer, "stopped": res["stage"], **log}
        return {"success": True, "drawer": drawer, **log}

    async def all_drawers(self, order: Sequence[int] = (1, 2, 3)) -> Dict[str, Any]:
        """Open and close every drawer in turn, with a fresh overhead look before each one.
        Closing a drawer can push the box back 1-2 cm, so a knob position from an earlier look is
        stale (that's why drawer 2 used to miss). Each drawer's log has "box_moved_mm" against the
        first look."""
        out: Dict[str, Any] = {"order": list(order), "drawers": []}
        first = order[0]
        prep = await self._prepare(first, out)
        if prep is None:
            return out
        fk, ref, wrist, wrist_alt = prep
        pull = fk["open_dir"]
        overhead = {k["drawer"]: [k["x"], k["y"]] for k in fk["knobs"]}
        # reach check up front: every grasp point must be within the arm's reach
        shift = [ref["x"] - overhead[first][0], ref["y"] - overhead[first][1]]
        far = {}
        for d in order:
            gx = overhead[d][0] + shift[0] + pull[0] * GRASP_OUTWARD_MM
            gy = overhead[d][1] + shift[1] + pull[1] * GRASP_OUTWARD_MM
            dist = math.hypot(gx, gy)
            if dist > MAX_REACH_MM:
                far[d] = round(dist)
        if far:
            worst = max(far.values())
            return {"success": False, "stage": "reach check",
                    "error": f"drawer(s) {sorted(far)} are {far} mm from the robot base; the arm reaches about "
                             f"{MAX_REACH_MM:.0f} mm. Move the box about {(worst - MAX_REACH_MM) / 10 + 2:.0f} cm "
                             "closer to the robot and run again - nothing was touched.", **out}
        for i, d in enumerate(order):
            log: Dict[str, Any] = {"drawer": d}
            if i > 0:
                # Closing a drawer can shove the whole box a centimetre or two, so look again before
                # every drawer instead of reusing the first look (the "final position" of the previous
                # knob is only the commanded grasp point, so it can't see the box move).
                prep = await self._prepare(d, log)
                if prep is None:
                    out["drawers"].append(log)
                    return {"success": False, "stopped_at_drawer": d, "stage": log.get("stage"),
                            "error": log.get("error"), **out}
                fk, ref, wrist, wrist_alt = prep
                pull = fk["open_dir"]
                log["box_moved_mm"] = round(math.hypot(fk["mid"][0] - out["find"]["mid"][0],
                                                       fk["mid"][1] - out["find"]["mid"][1]), 1)
            log["knob_used"] = [round(ref["x"], 1), round(ref["y"], 1)]
            res = await self._operate([ref["x"], ref["y"]], pull, wrist, wrist_alt, log)
            out["drawers"].append(log)
            if not res["ok"]:
                return {"success": False, "stopped_at_drawer": d, "stage": res.get("stage"),
                        "error": res.get("error"), **out}
        return {"success": True, **out}

    async def release(self) -> Dict[str, Any]:
        await self.gripper.do_command({"set": RELEASE_POS})
        await asyncio.sleep(SETTLE_S)
        pose = await self.motion.get_pose(component_name=self.gripper_name, destination_frame="world")
        p = pose.pose
        await self._speed(SLOW_DEG_S)
        await self._gripper_to(p.x, p.y, p.z + LIFT_MM, p.theta)
        await self._speed(TRAVEL_DEG_S)
        return {"released": True}

    # ---------------- open and close as separate steps ----------------
    # For sorting: open a drawer, leave it open while a block is dropped in, then close it. Same
    # moves and checks as drawer_run, split at the open position. Where the knob was left is
    # remembered (also on disk), so `close` needs no new look.

    def _load_open_state(self) -> Dict[int, Dict[str, Any]]:
        try:
            with open(self.open_state_file) as f:
                return {int(k): v for k, v in json.load(f).items()}
        except Exception:
            return {}

    def _save_open_state(self) -> None:
        try:
            with open(self.open_state_file, "w") as f:
                json.dump({str(k): v for k, v in self.open_drawers.items()}, f)
        except Exception as e:
            self.logger.warning(f"could not save the open-drawer state: {e}")

    async def _grip_at(self, gx, gy, wrists, log: Dict[str, Any]) -> Dict[str, Any]:
        """Steps 1-3 of _operate: hover above the grasp point, watched descent, gentle grip and check."""
        hz = DESCENT_Z_MM[0]
        await self.gripper.do_command({"set": APPROACH_OPEN_POS})
        await self._speed(TRAVEL_DEG_S)
        used = None
        for t in wrists:
            try:
                await self._gripper_to(gx, gy, hz, t, straight=False)
                used = t
                break
            except Exception as e:
                log.setdefault("hover_errors", []).append(str(e)[:120])
        if used is None:
            return {"ok": False, "stage": "hover"}
        await self._speed(SLOW_DEG_S)
        mon = await self._watched_move(gx, gy, GRASP_Z_MM, used)
        log["descent"] = mon
        if mon.get("contact"):
            try:
                await self.arm.do_command({"clear_error": True})
            except Exception:
                pass
            await self._gripper_to(gx, gy, hz, used)
            return {"ok": False, "stage": "descent", "error": f"contact felt near z={mon.get('z')} "
                    f"(load off trend {mon.get('change')}x the limit) - stopped and backed off, nothing forced"}
        await self.gripper.do_command({"grab_with_torque": {"position": 0, "speed": GRIP_SPEED,
                                                            "torque": GRIP_TORQUE_PCT}})
        await asyncio.sleep(SETTLE_S)
        pos = await self._gpos()
        log["grip_pos"] = pos
        if pos is None or not (HOLD_MIN_POS <= pos <= HOLD_MAX_POS):
            await self.gripper.do_command({"set": APPROACH_OPEN_POS})
            await self._gripper_to(gx, gy, hz, used)
            return {"ok": False, "stage": "grip", "error": "fingers did not close on the knob"}
        return {"ok": True, "wrist": used, "loads": await self._loads()}

    async def _let_go_and_lift(self, wrist) -> List[float]:
        """Step 6 of _operate: release, straight up, then up to the hover height. Returns the release x, y."""
        f = await self.arm.get_end_position()
        await self.gripper.do_command({"set": RELEASE_POS})
        await asyncio.sleep(SETTLE_S)
        await self._speed(SLOW_DEG_S)
        await self._gripper_to(f.x, f.y, GRASP_Z_MM + LIFT_MM, wrist)
        await self._speed(TRAVEL_DEG_S)
        await self._gripper_to(f.x, f.y, DESCENT_Z_MM[0], wrist)
        return [f.x, f.y]

    async def open_drawer(self, drawer: int) -> Dict[str, Any]:
        """Open `drawer` TRAVEL_MM and leave it open. Returns where to drop a block into it."""
        if self.open_drawers:
            return {"success": False, "stage": "open",
                    "error": f"drawer {sorted(self.open_drawers)[0]} is open; close it first (one open at a time)"}
        log: Dict[str, Any] = {}
        prep = await self._prepare(drawer, log)
        if prep is None:
            return {"success": False, **log}
        fk, ref, wrist, wrist_alt = prep
        pull = fk["open_dir"]
        knob_z = next(k["z"] for k in fk["knobs"] if k["drawer"] == drawer)
        gx, gy = ref["x"] + pull[0] * GRASP_OUTWARD_MM, ref["y"] + pull[1] * GRASP_OUTWARD_MM
        grip = await self._grip_at(gx, gy, (wrist, wrist_alt), log)
        if not grip["ok"]:
            return {"success": False, "stage": grip["stage"], "error": grip.get("error"), **log}
        used = grip["wrist"]
        ox, oy = gx + pull[0] * TRAVEL_MM, gy + pull[1] * TRAVEL_MM
        await self._gripper_to(ox, oy, GRASP_Z_MM, used)
        pos = await self._gpos()
        log["opened"] = {"to": [round(ox, 1), round(oy, 1)], "grip_pos": pos}
        if pos is None or not (HOLD_MIN_POS <= pos <= HOLD_MAX_POS):
            await self.gripper.do_command({"set": RELEASE_POS})
            await self._gripper_to(ox, oy, GRASP_Z_MM + LIFT_MM, used)
            return {"success": False, "stage": "open", "error": "lost the grip while opening - released "
                    "and lifted; drawer may be partly open", **log}
        await asyncio.sleep(SETTLE_S)
        rx, ry = await self._let_go_and_lift(used)
        # The knob is where the fingers let go; the drawer face is KNOB_PROTRUSION_MM behind the knob,
        # and the organizer face is TRAVEL_MM behind that. Drop DROP_FROM_FACE_MM out from the organizer.
        knob = [rx - pull[0] * GRASP_OUTWARD_MM, ry - pull[1] * GRASP_OUTWARD_MM]
        back = TRAVEL_MM + KNOB_PROTRUSION_MM - DROP_FROM_FACE_MM
        drop = {"x": round(knob[0] - pull[0] * back, 1), "y": round(knob[1] - pull[1] * back, 1),
                "z": round(knob_z, 1)}
        self.open_drawers[drawer] = {"open_grasp": [rx, ry], "closed_grasp": [gx, gy], "pull": pull,
                                     "wrist": used, "loads_closed": grip["loads"], "knob_z": knob_z}
        self._save_open_state()
        return {"success": True, "drawer": drawer, "open_knob": [round(knob[0], 1), round(knob[1], 1)],
                "drop_point": drop, "open_dir": pull, **log}

    async def close_drawer(self, drawer: int) -> Dict[str, Any]:
        """Close a drawer this service opened: grip the knob where it was left, push it back, check."""
        st = self.open_drawers.get(drawer)
        if st is None:
            return {"success": False, "stage": "close", "error": f"no open drawer {drawer} remembered; open it "
                    "with this service first, or close it by hand"}
        log: Dict[str, Any] = {}
        ox, oy = st["open_grasp"]
        grip = await self._grip_at(ox, oy, (st["wrist"],), log)
        if not grip["ok"]:
            return {"success": False, "stage": grip["stage"], "error": grip.get("error"), **log}
        gx, gy = st["closed_grasp"]
        await self._gripper_to(gx, gy, GRASP_Z_MM, grip["wrist"])
        loads_after = await self._loads()
        fully_closed = None
        if st.get("loads_closed") and loads_after:
            diff = max(abs(a - b) for a, b in zip(loads_after, st["loads_closed"]))
            fully_closed = diff <= CLOSE_LOAD_TOLERANCE
            log["closed"] = {"load_diff_vs_before": round(diff, 2), "fully_closed": fully_closed}
        await self._let_go_and_lift(grip["wrist"])
        del self.open_drawers[drawer]
        self._save_open_state()
        return {"success": True, "drawer": drawer, "fully_closed": fully_closed, **log}

    async def do_command(self, command: Mapping[str, ValueTypes], *, timeout: Optional[float] = None,
                         **kwargs) -> Mapping[str, ValueTypes]:
        action = command.get("action") or "all_drawers"
        self.force_refine = bool(command.get("refine", False))
        try:
            if action == "all_drawers":
                order = command.get("order") or [1, 2, 3]
                return await self.all_drawers([int(v) for v in order])
            if action == "drawer_run":
                return await self.drawer_run(int(command.get("drawer", 1)))
            if action == "locate":
                return await self.drawer_run(int(command.get("drawer", 1)), locate_only=True)
            if action == "find_knobs":
                return await self.find_knobs()
            if action == "release":
                return await self.release()
            if action == "open":
                return await self.open_drawer(int(command.get("drawer", 1)))
            if action == "close":
                return await self.close_drawer(int(command.get("drawer", 1)))
            if action == "status":
                return {"open_drawers": {str(k): v for k, v in self.open_drawers.items()}}
            if action == "forget":
                self.open_drawers = {}
                self._save_open_state()
                return {"forgot": True}
            return {"error": f"unknown action '{action}'. Use all_drawers, drawer_run, locate, find_knobs, "
                    "release, open, close, status or forget."}
        except Exception as e:
            self.logger.error(f"{action} failed: {e}")
            return {"success": False, "error": str(e)}
