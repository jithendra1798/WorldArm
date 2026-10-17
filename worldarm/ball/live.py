"""The goalie on the robot: overhead camera -> ball -> table position -> rolling model -> paddle on the lane.

The same RollSession as the simulator does the thinking; this file only does the I/O. The paddle can
only go to points on the taught lane (between its two ends, at the taught height and wrist): the lane
ends are flange poses read from the arm itself, so no frame conventions are re-derived here. The one
exception is approach(), which brings the paddle onto the lane from nearby by way of a safe height.

Times are the event loop's clock (time.monotonic() on the robot), so tests can run it on a simulated clock.
"""

from __future__ import annotations

import asyncio
import math
from collections import deque

import cv2
import numpy as np
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.proto.common import Pose

from worldarm.ball.config import BallCfg
from worldarm.ball.detect import find_ball, find_balls
from worldarm.ball.intercept import Goalie, Lane
from worldarm.ball.model import RollModel
from worldarm.ball.session import RollLog, RollSession
from worldarm.ball.table import TableMap
from worldarm.ball.track import Tracker
from worldarm.perception import decode_color

NEAR_MARGIN_MM = 50.0     # a frame's travel: the last sighting before the ball vanishes at the paddle can be this far out
CAMERA_FAILS = 20         # this many frames in a row not read: the camera is gone, stop
MOVE_TIMEOUT_S = 30.0     # no lane move takes this long
SAFE_ABOVE_MM = 80.0      # approach(): the paddle travels this high above its lane height
POLL_S = 0.03             # how often the paddle's position is read while defending
FRESH_S = 0.15            # a paddle reading this old still says where it is


def now() -> float:
    return asyncio.get_running_loop().time()


def _captured_at(meta) -> float | None:
    """The robot's time (s) the frame was taken, if the camera stamped it."""
    ts = getattr(meta, "captured_at", None)
    if ts is None or (ts.seconds == 0 and ts.nanos == 0):
        return None
    return ts.seconds + ts.nanos * 1e-9


class Overhead:
    """The overhead camera, color only (Riyaj measured 23 fps this way). A frame is timed by when it was
    taken, when the camera says so: the robot's stamp moved onto this computer's clock by the median offset
    over the last frames, so a frame that arrives late (the network) still gets the time it was taken.
    Without stamps, by when it arrived (sightings() then takes off cfg.camera_latency_s either way)."""

    def __init__(self, machine, name: str, cam=None):
        self.cam = cam if cam is not None else Camera.from_robot(machine, name)
        self._offsets: deque[float] = deque(maxlen=30)
        self._last_stamp: float | None = None
        self._repeats = 0
        self._stamps = True

    async def frame(self) -> tuple[float, np.ndarray | None]:
        """(time, BGR image), or (time, None) if no new color image came back."""
        images, meta = await self.cam.get_images(filter_source_names=["color"], timeout=2.0)
        t = now()
        stamp = _captured_at(meta) if self._stamps else None
        if stamp is not None:
            if self._last_stamp is not None and stamp <= self._last_stamp:
                self._repeats += 1
                if self._repeats < 5:
                    return t, None                         # the same frame again: nothing new
                self._stamps = False                       # stamps that never move: go by arrival times
            else:
                self._repeats, self._last_stamp = 0, stamp
                if self._offsets and abs(t - stamp - float(np.median(self._offsets))) > 1.0:
                    self._offsets.clear()                  # the robot's clock jumped
                self._offsets.append(t - stamp)
                t = stamp + float(np.median(self._offsets))
        img = next((i for i in images if i.name == "color"), images[0] if images else None)
        return t, (decode_color(img.data) if img is not None else None)


def _quiet(task: asyncio.Future) -> None:
    """Done callback for an interrupted move: its error (if any) is expected, not news."""
    if not task.cancelled():
        task.exception()


class LaneArm:
    """The arm, confined to the lane: every target is a point between the lane's two ends at the taught
    flange height and orientation, at a capped speed. A new target interrupts the move in progress: stop,
    clear the arm's error state (as the drawer service and the rig do after a stop), then the new move."""

    def __init__(self, arm: Arm, lane: Lane, speed_deg_s: float, max_speed_deg_s: float = 90.0):
        self.arm, self.lane = arm, lane
        self.speed = float(min(speed_deg_s, max_speed_deg_s))
        self._task: asyncio.Task | None = None
        self.target: float | None = None
        self._span: tuple[float, float] | None = None   # where along the lane the paddle may be right now
        self.seen: tuple[float, float] | None = None    # (time, position along the lane) last measured (watch())

    def _pose_at(self, x: float, y: float, z: float) -> Pose:
        o = self.lane.orientation
        return Pose(x=float(x), y=float(y), z=float(z), o_x=o[0], o_y=o[1], o_z=o[2], theta=float(self.lane.theta))

    def pose(self, s: float) -> Pose:
        x, y = self.lane.point(self.lane.clamp(s))
        return self._pose_at(x, y, self.lane.z)

    @property
    def idle(self) -> bool:
        return self._task is None or self._task.done()

    async def _clear_error(self) -> None:
        try:
            await self.arm.do_command({"clear_error": True})
        except Exception:
            pass                                           # tolerated, as in the proven stop paths

    async def start(self) -> None:
        await self._clear_error()
        await self.arm.do_command({"set_speed": self.speed})   # fails loudly: never move at an unknown speed

    def _check(self, p: Pose) -> tuple[str | None, bool]:
        """(why the paddle mustn't go onto the lane from pose p at all, or None; whether p is on the lane as
        taught already - position, height, tilt and wrist)."""
        xy = np.array([p.x, p.y])
        off = float(np.linalg.norm(xy - self.lane.point(self.lane.clamp(self.lane.along(xy)))))
        dz = float(p.z) - self.lane.z
        o, q = np.array([p.o_x, p.o_y, p.o_z]), np.array(self.lane.orientation)
        tilt = math.degrees(math.acos(float(np.clip(o @ q / ((np.linalg.norm(o) * np.linalg.norm(q)) or 1.0), -1, 1))))
        turn = abs((float(p.theta) - self.lane.theta + 180.0) % 360.0 - 180.0)
        if off > 150.0 or not -20.0 <= dz <= 200.0 or tilt > 20.0:
            return (f"The paddle is {off:.0f} mm from the lane, {dz:+.0f} mm from its height and tilted {tilt:.0f} deg "
                    "from the taught pose: bring it onto the lane first (e.g. where you taught end b)."), False
        return None, off <= 10.0 and abs(dz) <= 10.0 and tilt <= 5.0 and turn <= 5.0

    async def off_lane(self) -> str | None:
        """Why the paddle mustn't go onto the lane from where the arm is now, or None if it can."""
        return self._check(await self.arm.get_end_position())[0]

    async def approach(self, s: float, say=print) -> None:
        """Onto the lane at s from where the arm is. Straight there if it's on the lane as taught already;
        otherwise first straight up to SAFE_ABOVE_MM over the lane height, then over to s with the wrist
        turned and levelled up there, then down - never sweeping the paddle low with the wrist turning.
        Refuses (SystemExit) if the arm is far from the lane."""
        p = await self.arm.get_end_position()
        problem, on_lane = self._check(p)
        if problem:
            raise SystemExit(problem)
        target = self.lane.clamp(s)
        if not on_lane:
            z = max(float(p.z), self.lane.z + SAFE_ABOVE_MM)
            say(f"  onto the lane by way of flange z {z:.0f} (the lane is at {self.lane.z:.0f})")
            if z - p.z > 1.0:
                await self._move(Pose(x=p.x, y=p.y, z=z, o_x=p.o_x, o_y=p.o_y, o_z=p.o_z, theta=p.theta))
            x, y = self.lane.point(target)
            await self._move(self._pose_at(x, y, z))
        await self.go(target)
        await self.settle()

    async def _move(self, pose: Pose) -> None:
        self._raise_failed()
        self._task = asyncio.ensure_future(self.arm.move_to_position(pose))
        await self.settle()

    def _raise_failed(self) -> None:
        t = self._task
        if t is not None and t.done() and not t.cancelled() and t.exception() is not None:
            self._task = None
            raise RuntimeError(f"the paddle's last move failed: {t.exception()!r}") from t.exception()

    async def go(self, s: float) -> None:
        """Head for point s on the lane, now (doesn't wait for the move to finish)."""
        self._raise_failed()
        target = self.lane.clamp(s)                        # a NaN target raises here, before anything is sent
        if self._task is not None and not self._task.done():
            await self.arm.stop()
            self._task.cancel()
            self._task.add_done_callback(_quiet)
            await self._clear_error()
        self.target = target
        lo, hi = self._span or (target, target)
        self._span = (min(lo, target), max(hi, target))
        self._task = asyncio.ensure_future(self.arm.move_to_position(self.pose(target)))

    async def where(self) -> float:
        p = await self.arm.get_end_position()
        s = self.lane.along((p.x, p.y))
        self.seen = (now(), s)
        lo, hi = (s, s) if self.idle or self._span is None else self._span
        self._span = (min(lo, s), max(hi, s))
        return s

    def span(self) -> tuple[float, float] | None:
        """Where along the lane the paddle may be right now: anywhere it has been since it was last at rest."""
        if self._task is not None and self._task.done() and not self._task.cancelled() \
                and self._task.exception() is None and self.target is not None:
            self._span = (self.target, self.target)
        return self._span

    async def watch(self, period_s: float = POLL_S) -> None:
        """Read where the paddle is, over and over (a task while defending: cancel it to stop)."""
        while True:
            try:
                p = await self.arm.get_end_position()
                self.seen = (now(), self.lane.along((p.x, p.y)))
            except Exception:
                pass                                       # keep the last reading: it ages, the zone widens
            await asyncio.sleep(period_s)

    def paddle(self, t: float, speed_mm_s: float) -> tuple[float, float, bool]:
        """(lo, hi, measured): where along the lane the paddle is at time t - its last reading, give or take
        how far it can have moved since - or, with no reading, anywhere it may be; measured if the reading
        is fresh."""
        if self.seen is None:
            span = self.span()
            return (*(span if span is not None else (0.0, self.lane.length)), False)
        t_seen, s = self.seen
        slack = 5.0 + 1.5 * speed_mm_s * max(0.0, t - t_seen)
        return s - slack, s + slack, t - t_seen <= FRESH_S

    async def settle(self, timeout_s: float = MOVE_TIMEOUT_S) -> None:
        """Wait for the move in progress to finish. Raises if it failed, or took too long (the arm is stopped)."""
        if self._task is None:
            return
        done, _ = await asyncio.wait([self._task], timeout=timeout_s)
        if not done:
            await self.stop()
            raise RuntimeError(f"the paddle's move didn't finish in {timeout_s:.0f} s; arm stopped")
        self._raise_failed()

    async def stop(self) -> None:
        """Stop the arm and drop the move in progress. Doesn't raise (it runs on the way out of errors)."""
        task, self._task = self._task, None
        try:
            await self.arm.stop()
        except Exception as e:
            print(f"!!! COULD NOT STOP THE ARM ({type(e).__name__}: {e}): use the E-stop")
        finally:
            if task is not None:
                task.cancel()
                task.add_done_callback(_quiet)


def setup(cfg: BallCfg, mode: str) -> tuple[TableMap, Lane, RollModel, Goalie]:
    table, lane = cfg.table(), cfg.get_lane()
    if table is None:
        raise SystemExit("No table map yet: run `python ball.py calibrate`.")
    if lane is None:
        raise SystemExit("No lane yet: run `python ball.py lane --teach a` and `--teach b`.")
    model = RollModel(cfg.decel_mm_s2)
    return table, lane, model, Goalie(lane, model, cfg.arm_model(), cfg.paddle_half_width_mm, mode,
                                      ball_radius_mm=cfg.ball_radius_mm)


def field_px(table: TableMap, lane: Lane | None, cfg: BallCfg) -> np.ndarray | None:
    """Where the ball can be, in pixels: the strip of table from just behind the lane out to the far end.
    Nothing outside it (people, the arm's base, the controller) can be taken for the ball."""
    if lane is None:
        return np.array(cfg.table_px) if cfg.table_px else None
    m = cfg.field_margin_mm
    corners = [lane.a - lane.unit * m + lane.normal * m, lane.b + lane.unit * m + lane.normal * m,
               lane.b + lane.unit * m - lane.normal * cfg.field_length_mm,
               lane.a - lane.unit * m - lane.normal * cfg.field_length_mm]
    return np.array([table.to_pixel(c) for c in corners])


def paddle_zone_px(table: TableMap, lane: Lane, along: tuple[float, float] | None, cfg: BallCfg) -> list[np.ndarray]:
    """Where the ball isn't looked for, as pixel polygons: cfg.zone_front_mm in front of the lane (the paddle's
    thickness) to cfg.zone_back_mm behind it (the gripper, wrist and arm above and behind the paddle - the
    higher they are, the further out the camera sees them), over `along` (where the paddle is, along the
    lane) widened by the paddle's half width + cfg.zone_side_mm. Thin in front, so the ball is seen up to
    the paddle and coming back off it."""
    if along is None:
        return []
    w = cfg.paddle_half_width_mm + cfg.zone_side_mm
    s0, s1 = along[0] - w, along[1] + w
    corners = [lane.point(s0) - lane.normal * cfg.zone_front_mm, lane.point(s1) - lane.normal * cfg.zone_front_mm,
               lane.point(s1) + lane.normal * cfg.zone_back_mm, lane.point(s0) + lane.normal * cfg.zone_back_mm]
    return [np.array([table.to_pixel(c) for c in corners])]


def _blank(img: np.ndarray, zones) -> np.ndarray:
    """The picture with the zones painted white: nothing dark there to be taken for the ball."""
    for poly in zones:
        cv2.fillPoly(img, [np.round(np.asarray(poly, float)).astype(np.int32)], (255, 255, 255))
    return img


async def sightings(overhead, table: TableMap, cfg: BallCfg, avoid=(), lane: Lane | None = None):
    """Endless stream of (t, world xy or None, pixel sighting or None). `avoid` (pixel polygons, see
    paddle_zone_px) may be updated by the caller as the paddle moves. A frame the camera fails to give
    counts as a miss."""
    field = field_px(table, lane, cfg)
    fails = 0
    while True:
        try:
            t, img = await overhead.frame()
            fails = 0
        except Exception as e:
            fails += 1
            if fails >= CAMERA_FAILS:
                raise RuntimeError(f"the overhead camera failed {fails} times in a row: {e!r}") from e
            t, img = now(), None
        s = find_ball(_blank(img, list(avoid)), cfg.ball_radius_px, field) if img is not None else None
        xy = table.to_world((s.u, s.v)) if s else None
        yield t - cfg.camera_latency_s, xy, s


async def look_for_lookalikes(cam, table: TableMap, lane: Lane, cfg: BallCfg, along: tuple[float, float],
                              frames: int = 8) -> list[np.ndarray]:
    """With the ball off the table: the table position of everything in the field, outside the paddle zone,
    that would pass for the ball. The camera sees the gripper and wrist further out the higher they are,
    so this is how to know the zone covers them wherever the paddle goes."""
    field, zone, found = field_px(table, lane, cfg), paddle_zone_px(table, lane, along, cfg), []
    for _ in range(frames):
        _, img = await cam.frame()
        if img is not None:
            found += [table.to_world((s.u, s.v)) for s in find_balls(_blank(img, zone), cfg.ball_radius_px, field)]
    return found


async def wait_for_ready(stream, lane: Lane, min_gap_mm: float = 350.0, still_s: float = 0.5, say=print) -> None:
    """Until the ball sits still, far enough out from the lane: then it's someone's turn to roll it."""
    say("Put the ball at the far end and hold still...")
    still_since, last = None, None
    async for t, xy, _ in stream:
        if xy is None:
            if last is not None and t - last[0] > 0.3:     # lost it for a while: start over
                still_since, last = None, None
            continue
        if float((lane.a - xy) @ lane.normal) < min_gap_mm:
            still_since, last = None, None
            continue
        if last is not None and float(np.linalg.norm(xy - last[1])) < 6.0:
            still_since = last[0] if still_since is None else still_since
            if t - still_since >= still_s:
                say("ROLL IT!")
                return
        else:
            still_since = None
        last = (t, xy)


async def defend(stream, lane: Lane, goalie: Goalie, lane_arm: LaneArm, cfg: BallCfg, timeout_s: float = 8.0,
                 say=print, avoid: list | None = None, table: TableMap | None = None,
                 wait_s: float = 30.0) -> RollLog:
    """One roll: track it, move the paddle when the goalie decides, until it's blocked, scored or stopped -
    or nobody rolls it within wait_s, it's lost from sight, or it isn't over timeout_s after it started rolling.

    The paddle's position is read throughout (judging needs where it is, not only where it was sent). With
    `avoid` and `table`, the camera ignores the paddle's surroundings (kept up to date here), so the ball
    vanishes as it reaches the paddle: "near the lane" is set to match."""
    goalie.reset()
    hide = avoid is not None and table is not None
    near = cfg.zone_front_mm + cfg.ball_radius_mm + NEAR_MARGIN_MM if hide else None
    session = RollSession(lane, goalie, Tracker(decel=cfg.decel_mm_s2), cfg.ball_radius_mm, near)
    speed = cfg.arm_model().speed_mm_s
    s_sent = await lane_arm.where()
    watcher = asyncio.ensure_future(lane_arm.watch())
    try:
        if hide:
            avoid[:] = paddle_zone_px(table, lane, lane_arm.paddle(now(), speed)[:2], cfg)
        t_first = None
        async for t, xy, _ in stream:
            t_first = t if t_first is None else t_first
            lo, hi, measured = lane_arm.paddle(now(), speed)
            dec = session.on_sighting(t, xy, (lo + hi) / 2 if measured else s_sent, measured)
            if dec is not None:
                await lane_arm.go(dec.s)
                s_sent = dec.s
                late = "" if dec.in_time in (None, True) else " (late)"
                when = f", arrives in {dec.t_cross - t:.2f} s" if dec.t_cross is not None else ""
                say(f"  paddle -> {dec.s:.0f} mm along the lane{when}{late}")
            if hide:
                avoid[:] = paddle_zone_px(table, lane, lane_arm.paddle(now(), speed)[:2], cfg)
            if session.log.outcome is not None:
                break
            if session.t_start is None and t - t_first > wait_s:
                say("  nobody rolled it")
                break
            if session.lost(t):
                say("  lost sight of it")
                break
            if session.t_start is not None and t - session.t_start > timeout_s:
                break
    finally:
        watcher.cancel()
        watcher.add_done_callback(_quiet)
    say(f"  {session.log.outcome or 'not judged'}")
    return session.log
