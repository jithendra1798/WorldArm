"""The ball.py drills, end to end, against fakes: calibrate on the simulated rig (tests/fake_rig.py) under a
simulated overhead camera; lane-test, record + fit, and goalie + report with an arm and a camera simulated
here. Time is simulated too (an event loop whose clock jumps to the next timer instead of waiting), so a
minute of frames and moves takes a second or two, the same way every run. Nothing touches the robot."""

import asyncio
import selectors

import cv2
import numpy as np
import pytest
from viam.proto.common import Pose

import ball
from tests.fake_rig import FakeRig, FakeWorld, FBlock
from tests.synth import Box
from worldarm.ball.config import BallCfg
from worldarm.ball.intercept import Lane
from worldarm.ball.model import Roll, RollModel
from worldarm.ball.table import TableMap

pytestmark = pytest.mark.sim  # rendered frames, many of them: `pytest -m sim`

LANE = Lane(np.array([450.0, -250.0]), np.array([450.0, 250.0]), 200.0, 0.0, home=np.array([0.0, 0.0]))
TABLE = TableMap(np.array([[0.0, -2.0, 1300.0], [2.0, 0.0, -360.0], [0.0, 0.0, 1.0]]))     # 2 mm per pixel
TILTED = TableMap(np.array([[0.0, -2.0, 1300.0], [2.0, 0.0, -360.0], [2e-5, -1e-5, 1.0]]))  # a camera not quite overhead


class SimClock(asyncio.SelectorEventLoop):
    """An event loop whose clock jumps straight to the next timer instead of waiting for it."""

    def __init__(self):
        self._t = 0.0
        sel = selectors.DefaultSelector()
        real = sel.select

        def select(timeout=None):
            ready = real(0)
            if ready or timeout == 0:
                return ready
            if timeout is None:
                return real(None)          # nothing scheduled: only another thread could wake the loop
            self._t += timeout
            return []

        sel.select = select
        super().__init__(sel)

    def time(self) -> float:
        return self._t


def run(coro):
    return asyncio.run(coro, loop_factory=SimClock)


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setattr(BallCfg, "path", property(lambda self: tmp_path / "ball.json"))
    monkeypatch.setattr(ball, "RUNS", tmp_path / "runs")
    return BallCfg(table_map=TABLE.to_json(), lane=LANE.to_json(), camera_latency_s=0.0, decel_mm_s2=120.0,
                   arm={"latency_s": 0.2, "speed_mm_s": 150.0})


class FakeArm:
    """The Viam Arm calls the goalie makes. A move starts after `latency` and goes straight at `speed`; a
    cancelled move stops where it is. `fail_on`: the n-th move raises (an unreachable pose, say)."""

    def __init__(self, xy=None, latency=0.2, speed=150.0, fail_on=None, z=LANE.z, theta=0.0):
        self.p = np.array([*(LANE.b if xy is None else xy), z])
        self.o = [0.0, 0.0, -1.0, theta]                   # o_x, o_y, o_z, theta: as last commanded
        self.latency, self.speed, self.fail_on = latency, speed, fail_on
        self.calls: list = []

    async def do_command(self, cmd):
        self.calls.append(("do_command", cmd))
        return {}

    async def move_to_position(self, pose: Pose):
        self.calls.append(("move", pose))
        if self.fail_on == sum(c[0] == "move" for c in self.calls):
            raise RuntimeError("no IK solution")
        self.o = [pose.o_x, pose.o_y, pose.o_z, pose.theta]
        await asyncio.sleep(self.latency)
        goal = np.array([pose.x, pose.y, pose.z])
        while np.linalg.norm(goal - self.p) > 0.5:
            await asyncio.sleep(0.01)
            step = goal - self.p
            self.p = self.p + step * min(1.0, self.speed * 0.01 / np.linalg.norm(step))

    async def get_end_position(self):
        return Pose(x=float(self.p[0]), y=float(self.p[1]), z=float(self.p[2]), o_x=self.o[0], o_y=self.o[1],
                    o_z=self.o[2], theta=self.o[3])

    async def stop(self):
        self.calls.append(("stop", None))

    def moves(self) -> list[Pose]:
        return [c[1] for c in self.calls if c[0] == "move"]


class Rolls:
    """The overhead camera while people roll the ball, one roll after another: put down at the far end and
    held still, rolled (slowing at `decel`; back off the paddle if it covers the ball where it crosses the
    lane), then picked up. With an arm, the gripper above the paddle is drawn too, as round and dark as the
    ball. The schedule starts at the first frame asked for."""

    def __init__(self, rolls, decel, arm=None, still_s=1.0, roll_s=4.0, away_s=2.5):
        self.rolls, self.model, self.arm = rolls, RollModel(decel), arm
        self.still_s, self.roll_s, self.cycle = still_s, roll_s, still_s + roll_s + away_s
        self.t0 = None
        self.covered = [None] * len(rolls)

    def ball_xy(self, t):
        i, tau = int((t - self.t0) // self.cycle), (t - self.t0) % self.cycle
        if i >= len(self.rolls) or tau > self.still_s + self.roll_s:
            return None                                          # picked up
        start, aim, v0 = (np.asarray(v, float) if k < 2 else v for k, v in enumerate(self.rolls[i]))
        if tau < self.still_s:
            return start
        roll = Roll(self.t0 + i * self.cycle + self.still_s, start, (aim - start) / np.linalg.norm(aim - start), v0)
        hit = self.model.crossing(roll, LANE.a, LANE.normal)
        if hit is not None and t >= hit[1]:
            if self.covered[i] is None:
                self.covered[i] = self.arm is not None and abs(LANE.along(self.arm.p[:2]) - LANE.along(hit[0])) <= 80.0
            if self.covered[i]:
                return self.model.at(Roll(hit[1], hit[0], -roll.u, 200.0), t)
        return self.model.at(roll, t)

    async def frame(self):
        await asyncio.sleep(1 / 23)
        t = asyncio.get_running_loop().time()
        self.t0 = t if self.t0 is None else self.t0
        img = np.full((720, 1280, 3), 225, np.uint8)
        for xy in ([] if self.arm is None else [self.arm.p[:2]]) + [self.ball_xy(t)]:
            if xy is not None:
                u, v = TABLE.to_pixel(xy)
                cv2.circle(img, (int(round(u)), int(round(v))), 13, (20, 20, 20), -1)
        return t, img


# ---- calibrate -------------------------------------------------------------------------------------------

class TableCam:
    """The overhead camera over the simulated rig's table: the ball (a black block in the fake world) drawn
    where it is, through a tilted camera's homography."""

    def __init__(self, world: FakeWorld):
        self.world = world

    async def frame(self):
        await asyncio.sleep(1 / 23)
        img = np.full((720, 1280, 3), 225, np.uint8)
        for b in self.world.blocks:
            u, v = TILTED.to_pixel((b.x, b.y))
            cv2.circle(img, (int(round(u * 16)), int(round(v * 16))), 16 * 16, (20, 20, 20), -1, cv2.LINE_8, 4)
        return 0.0, img


def calibrate_on(lane: Lane, tmp_path, monkeypatch):
    monkeypatch.setattr(BallCfg, "path", property(lambda self: tmp_path / "ball.json"))
    c = BallCfg(lane=lane.to_json())
    rig = FakeRig(FakeWorld(Box(), []))
    # the person puts the ball between the open fingers: in the pads' reach, below the gripper
    ball_blk = FBlock("black", rig.x, rig.y, rig.z - rig.cfg.gripper.tip_below_frame_mm - 15.0, 50.0, 50.0, 50.0, 90.0)
    rig.world.blocks.append(ball_blk)
    prompts = []
    res = run(ball.calibrate(rig, TableCam(rig.world), c, ask=lambda q: prompts.append(q) or "",
                             say=lambda *_: None))
    return res, c, rig, ball_blk, prompts


def map_error_mm(tm: TableMap, lane: Lane, depth: float) -> float:
    pts = [lane.point(s) - lane.normal * d for s in np.linspace(0, lane.length, 9) for d in np.linspace(0, depth, 7)]
    return max(float(np.linalg.norm(tm.to_world(TILTED.to_pixel(p)) - p)) for p in pts)


def test_calibrate_maps_the_camera_to_the_table(tmp_path, monkeypatch):
    lane = Lane(np.array([300.0, -200.0]), np.array([300.0, 200.0]), 200.0, 0.0, home=np.array([0.0, 0.0]))
    res, c, rig, ball_blk, prompts = calibrate_on(lane, tmp_path, monkeypatch)
    assert res["points"] == 9 and map_error_mm(BallCfg.load().table(), lane, 240.0) < 3.0
    assert abs(ball_blk.base) < 0.5 and rig.held is None                   # the ball left on the table
    assert np.hypot(ball_blk.x - lane.point(0.9 * lane.length)[0] - 240.0, ball_blk.y - 160.0) < 1.0
    assert abs(rig.x - 300.0) < 1.0 and abs(rig.y) < 1.0                   # waiting above the lane's middle
    assert len(prompts) == 3 and "paddle" in prompts[-1]


def test_calibrate_shrinks_the_grid_to_the_work_area(tmp_path, monkeypatch):
    lane = Lane(np.array([500.0, -200.0]), np.array([500.0, 200.0]), 200.0, 0.0, home=np.array([0.0, 0.0]))
    res, c, rig, _, _ = calibrate_on(lane, tmp_path, monkeypatch)            # 240 or 180 mm out: past x / reach
    assert res["points"] == 9 and map_error_mm(BallCfg.load().table(), lane, 120.0) < 3.0
    assert max(float((lane.a - p) @ lane.normal) for p in res["world"]) == pytest.approx(120.0)   # 3 rows, 120 deep


# ---- lane-test -------------------------------------------------------------------------------------------

def test_lane_test_measures_latency_and_speed(cfg):
    arm = FakeArm(latency=0.25, speed=200.0)
    res = run(ball.lane_test(arm, cfg, times=1, say=lambda *_: None))
    assert abs(res["latency_s"] - 0.25) < 0.05 and abs(res["speed_mm_s"] - 200.0) < 40.0, res
    saved = BallCfg.load()
    assert saved.arm == {"latency_s": round(res["latency_s"], 3), "speed_mm_s": round(res["speed_mm_s"], 1)}
    assert saved.arm_speed_deg_s == 30.0 and ("do_command", {"set_speed": 30.0}) in arm.calls
    assert res["stray_mm"] < 1.0 and res["height_mm"] < 1.0 and arm.calls[-1][0] == "stop"
    assert abs(res["retarget_s"] - 0.25) < 0.05 and saved.arm_retarget_s == round(res["retarget_s"], 3)
    assert_clears_after_every_stop(arm)


def assert_clears_after_every_stop(arm):
    """Every move that interrupts another goes out after a stop and a clear_error, as in the proven stop paths."""
    for i, c in enumerate(arm.calls[:-1]):
        if c[0] == "stop":
            assert arm.calls[i + 1] == ("do_command", {"clear_error": True}) and arm.calls[i + 2][0] == "move"


def test_the_paddle_comes_onto_the_lane_from_above():
    """Off the lane as taught (the wrist turned, a bit high): straight up, over with the wrist turned up
    there, then down - never turning the wrist low over the table."""
    from worldarm.ball.live import SAFE_ABOVE_MM, LaneArm
    start = LANE.point(150.0) - LANE.normal * 40.0
    arm = FakeArm(xy=start, z=LANE.z + 30.0, theta=90.0)
    run(LaneArm(arm, LANE, 30.0).approach(250.0, say=lambda *_: None))
    up, over, down = arm.moves()
    assert (up.z, up.theta) == (LANE.z + SAFE_ABOVE_MM, 90.0) and np.allclose([up.x, up.y], start)
    assert (over.z, over.theta) == (LANE.z + SAFE_ABOVE_MM, LANE.theta) and np.allclose([over.x, over.y], LANE.point(250.0))
    assert (down.z, down.theta) == (LANE.z, LANE.theta) and np.allclose([down.x, down.y], LANE.point(250.0))
    assert np.allclose(arm.p, [*LANE.point(250.0), LANE.z], atol=1.0)
    straight = FakeArm(xy=LANE.point(400.0))
    run(LaneArm(straight, LANE, 30.0).approach(250.0, say=lambda *_: None))
    assert len(straight.moves()) == 1                                     # on the lane already: just along it


class GripperCam:
    """No ball on the table; the gripper drawn as a dark disc where the camera would see it: shifted by
    `shift_mm` from the paddle (what height does, off the camera's axis)."""

    def __init__(self, arm, shift_mm):
        self.arm, self.shift = arm, np.asarray(shift_mm, float)

    async def frame(self):
        await asyncio.sleep(1 / 23)
        img = np.full((720, 1280, 3), 225, np.uint8)
        u, v = TABLE.to_pixel(self.arm.p[:2] + self.shift)
        cv2.circle(img, (int(round(u)), int(round(v))), 13, (20, 20, 20), -1)
        return asyncio.get_running_loop().time(), img


def test_lane_test_finds_what_the_paddle_zone_misses(cfg):
    behind = run(ball.lane_test(arm := FakeArm(), cfg, times=1, cam=GripperCam(arm, LANE.normal * 70.0),
                                say=lambda *_: None))
    assert behind["lookalikes"] == []                                     # seen behind the lane: in the zone
    lines = []
    ahead = run(ball.lane_test(arm := FakeArm(), cfg, times=1, cam=GripperCam(arm, -LANE.normal * 70.0),
                               say=lines.append))
    ends = sorted({round(s) for s, _ in ahead["lookalikes"]})
    assert ends == [0, 500] and any("passes for the ball" in line for line in lines)


def test_lane_test_refuses_to_start_off_the_lane(cfg):
    arm = FakeArm(xy=(750.0, 0.0))                                           # 300 mm out from the lane
    with pytest.raises(SystemExit, match="bring it onto the lane"):
        run(ball.lane_test(arm, cfg, say=lambda *_: None))
    assert arm.calls == []                                                   # not even the speed was set


def test_lane_test_stops_the_arm_when_a_move_fails(cfg):
    arm = FakeArm(fail_on=2)
    with pytest.raises(RuntimeError, match="no IK solution"):
        run(ball.lane_test(arm, cfg, say=lambda *_: None))
    assert arm.calls[-1][0] == "stop"


def test_lane_test_stops_the_arm_on_ctrl_c(cfg):
    arm = FakeArm()

    async def interrupted():
        task = asyncio.ensure_future(ball.lane_test(arm, cfg, say=lambda *_: None))
        await asyncio.sleep(2.5)                                             # mid-move
        task.cancel()                                                        # what Ctrl-C does to asyncio.run's task
        with pytest.raises(asyncio.CancelledError):
            await task
        where = arm.p.copy()
        await asyncio.sleep(1.0)
        return where

    where = run(interrupted())
    assert arm.calls[-1][0] == "stop" and np.allclose(arm.p, where)         # stopped, and stays stopped


# ---- record + fit ----------------------------------------------------------------------------------------

def test_record_then_fit_learns_the_slowdown(cfg, tmp_path):
    rolls = [((1250.0, -100.0), (450.0, -70.0), 420.0), ((1250.0, 60.0), (450.0, 20.0), 470.0),
             ((1250.0, 0.0), (450.0, 20.0), 440.0), ((1250.0, 120.0), (450.0, 60.0), 480.0)]
    tracks = run(ball.record(Rolls(rolls, decel=150.0, still_s=1.0, roll_s=3.5, away_s=1.0), cfg, len(rolls),
                             out=tmp_path / "runs" / "20261017-090000-ball-record", say=lambda *_: None))
    assert len(tracks) == 4 and all(len(t) > 40 for t in tracks)
    res = ball.fit(BallCfg.load(), runs=tmp_path / "runs", say=lambda *_: None)
    assert res["rolls"] == 4 and abs(res["decel"] - 150.0) < 0.15 * 150.0, res
    assert abs(BallCfg.load().decel_mm_s2 - res["decel"]) < 0.1


def test_rolling_part_drops_the_wait_the_bounce_and_what_follows_a_loss():
    m = RollModel(150.0)
    roll = Roll(1.0, np.array([0.0, 0.0]), np.array([1.0, 0.0]), 500.0)
    t = np.arange(0.0, 3.0, 1 / 23)
    xy = np.array([m.at(roll, ti) if ti < 2.0 else m.at(Roll(2.0, m.at(roll, 2.0), -roll.u, 150.0), ti) for ti in t])
    track = [(ti, x, y) for ti, (x, y) in zip(t, xy)]
    tt, pts = ball.rolling_part(track)
    assert 1.0 <= tt[0] <= 1.15 and 1.9 <= tt[-1] <= 2.0 and np.all(np.diff(pts[:, 0]) > 0)
    lost = [p for p in track if not 1.5 < p[0] < 1.8]                       # out of sight for 0.3 s
    assert ball.rolling_part(lost)[0][-1] < 1.5


# ---- goalie + report -------------------------------------------------------------------------------------

GOALIE_ROLLS = [((1250.0, -50.0), (450.0, -50.0), 800.0),     # straight at 200 mm along the lane: blocked
                ((1250.0, 150.0), (450.0, -100.0), 700.0),    # diagonal, from 400 to 150: blocked
                ((1250.0, -220.0), (450.0, -220.0), 900.0)]   # fast, at 30, too far for the paddle: a goal


def test_goalie_defends_and_report_counts_it(cfg, tmp_path):
    arm = FakeArm()
    cam = Rolls(GOALIE_ROLLS, decel=120.0, arm=arm)
    answers = iter(["", "g", ""])                  # the operator agrees, overrules the second call, agrees
    res = run(ball.goalie(arm, cam, cfg, "predict", 3, out=tmp_path / "runs" / "20261017-100000-ball-goalie-predict",
                          say=lambda *_: None, ask=lambda q: next(answers)))
    assert [r["camera"] for r in res["rolls"]] == ["blocked", "blocked", "goal"], res["rolls"]
    assert [r["human"] for r in res["rolls"]] == [r["outcome"] for r in res["rolls"]] == ["blocked", "goal", "goal"]
    assert cam.covered == [True, True, False]
    for i, (r, (start, aim, v0)) in enumerate(zip(res["rolls"], GOALIE_ROLLS)):
        start, aim = np.array(start), np.array(aim)
        roll = Roll(cam.t0 + i * cam.cycle + cam.still_s, start, (aim - start) / np.linalg.norm(aim - start), v0)
        t_cross = RollModel(120.0).crossing(roll, LANE.a, LANE.normal)[1]
        assert abs(r["decisions"][0][2] - t_cross) < 0.25          # the wait before the roll isn't fitted as slow rolling
    for p in arm.moves():                          # only ever to points on the lane, at its height and wrist
        assert abs(float((np.array([p.x, p.y]) - LANE.a) @ LANE.normal)) < 1e-6
        assert -1e-6 <= LANE.along((p.x, p.y)) <= LANE.length + 1e-6
        assert (p.z, p.o_x, p.o_y, p.o_z, p.theta) == (LANE.z, *LANE.orientation, LANE.theta)
    assert ("do_command", {"set_speed": 30.0}) in arm.calls and arm.calls[-1][0] == "stop"
    assert_clears_after_every_stop(arm)
    rep = ball.report(BallCfg.load(), runs=tmp_path / "runs", say=lambda *_: None)
    assert rep["H5"] == {"predict": {"blocked": 1, "goal": 2, "short": 0, "not_judged": 0}}   # the operator's calls
    assert rep["camera_agrees"] == {"predict": (2, 3)}


def test_goalie_stops_the_arm_on_ctrl_c_and_keeps_the_rolls_so_far(cfg, tmp_path):
    arm = FakeArm()
    out = tmp_path / "runs" / "20261017-110000-ball-goalie-predict"

    async def interrupted():
        cam = Rolls(GOALIE_ROLLS, decel=120.0, arm=arm)
        task = asyncio.ensure_future(ball.goalie(arm, cam, cfg, "predict", 3, out=out, say=lambda *_: None))
        loop = asyncio.get_running_loop()
        while cam.t0 is None or loop.time() < cam.t0 + cam.cycle + cam.still_s + 0.5:
            await asyncio.sleep(0.05)              # until the second roll is half a second under way
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run(interrupted())
    assert arm.calls[-1][0] == "stop"
    rep = ball.report(BallCfg.load(), runs=tmp_path / "runs", say=lambda *_: None)
    assert rep["H5"]["predict"]["blocked"] == 1


# ---- no motion: teach, check, the camera ------------------------------------------------------------------

def test_teach_keeps_the_higher_end_and_saves(cfg):
    cfg.lane = None
    arm = FakeArm(xy=(450.0, -250.0))
    lines = []
    run(ball.teach(arm, cfg, "a", say=lines.append))
    arm.p = np.array([450.0, 250.0, LANE.z + 8.0])
    run(ball.teach(arm, cfg, "b", say=lines.append))
    lane = BallCfg.load().get_lane()
    assert lane.length == pytest.approx(500.0) and lane.z == pytest.approx(LANE.z + 8.0)
    assert np.allclose(lane.normal, [-1.0, 0.0]) and any("apart in height" in line for line in lines)


class TableView:
    """A grey floor, the white table on it, and the ball on the table."""

    async def frame(self):
        await asyncio.sleep(1 / 23)
        img = np.full((720, 1280, 3), 90, np.uint8)
        cv2.rectangle(img, (200, 100), (1100, 650), (230, 230, 230), -1)
        cv2.circle(img, (640, 380), 15, (20, 20, 20), -1)
        return asyncio.get_running_loop().time(), img


def test_check_finds_the_table_and_the_ball(cfg, tmp_path):
    cfg.table_px = None
    res = run(ball.check(TableView(), cfg, frames=10, out=tmp_path / "captures", say=lambda *_: None))
    assert res["seen"] == 10 and np.allclose(res["ball_px"], (640, 380), atol=1.0)
    corners = np.array(BallCfg.load().table_px)
    assert corners[:, 0].min() > 195 and corners[:, 0].max() < 1105 and corners[:, 1].min() > 95
    assert (tmp_path / "captures" / "ball_check.jpg").exists()


def test_overhead_reads_the_color_image_by_name():
    from viam.media.video import CameraMimeType, NamedImage
    from worldarm.ball.live import Overhead
    img = np.full((72, 128, 3), 200, np.uint8)
    img[10:20, 30:40] = (0, 0, 255)
    png = cv2.imencode(".png", img)[1].tobytes()

    class Cam:
        async def get_images(self, *, filter_source_names=None, extra=None, timeout=None):
            self.asked = (filter_source_names, timeout)
            return [NamedImage("depth", b"DEPTHMAP", CameraMimeType.VIAM_RAW_DEPTH),
                    NamedImage("color", png, CameraMimeType.PNG)], None

    oh = Overhead(None, "overview-cam", cam=Cam())
    t, got = run(oh.frame())
    assert oh.cam.asked == (["color"], 2.0) and got.shape == img.shape and np.array_equal(got, img)


def test_overhead_times_frames_by_when_they_were_taken():
    """A frame that arrives late (the network) keeps the time it was taken: the robot's stamp, moved onto this
    clock by the usual offset. The same frame twice is not news; stamps that never move aren't trusted."""
    from google.protobuf.timestamp_pb2 import Timestamp
    from viam.media.video import CameraMimeType, NamedImage
    from viam.proto.common import ResponseMetadata
    from worldarm.ball.live import Overhead
    png = cv2.imencode(".png", np.full((36, 64, 3), 200, np.uint8))[1].tobytes()

    class Cam:                                      # the robot's clock reads 1000 s ahead of this one
        def __init__(self, delays, stamps=None):
            self.delays, self.stamps, self.taken = list(delays), stamps, []

        async def get_images(self, *, filter_source_names=None, extra=None, timeout=None):
            t = asyncio.get_running_loop().time()
            self.taken.append(t)
            robot = self.stamps.pop(0) if self.stamps else 1000.0 + t
            await asyncio.sleep(self.delays.pop(0))
            stamp = Timestamp(seconds=int(robot), nanos=int(round(robot % 1 * 1e9)))
            return [NamedImage("color", png, CameraMimeType.PNG)], ResponseMetadata(captured_at=stamp)

    async def frames(cam, n):
        oh = Overhead(None, "overview-cam", cam=cam)
        return [await oh.frame() for _ in range(n)]

    cam = Cam([0.05, 0.05, 0.40, 0.05, 0.06, 0.05])
    got = run(frames(cam, 6))
    assert all(abs(t - (taken + 0.05)) < 0.006 for (t, _), taken in zip(got, cam.taken))   # the late one too
    same = run(frames(Cam([0.05] * 8, stamps=[1000.0, 1000.5] + [1000.5] * 6), 8))
    assert [img is None for _, img in same] == [False, False, True, True, True, True, False, False]


# ---- the command line ------------------------------------------------------------------------------------

def test_moving_commands_ask_first_and_check_the_machine(cfg, monkeypatch):
    from worldarm import connection

    async def no_robot():
        raise AssertionError("connected to the robot")

    monkeypatch.setattr(connection, "connect", no_robot)
    monkeypatch.setattr("builtins.input", lambda prompt: "no")
    monkeypatch.setattr(connection, "env", lambda name: "armfarm03-main.abc123.viam.cloud")
    for argv in (["goalie", "--rolls", "1"], ["lane-test"], ["calibrate"]):
        with pytest.raises(SystemExit, match="Cancelled"):
            asyncio.run(ball.main(argv))
    monkeypatch.setattr(connection, "env", lambda name: "armfarm05-main.abc123.viam.cloud")
    with pytest.raises(SystemExit, match="points at armfarm05"):
        asyncio.run(ball.main(["goalie", "--yes"]))


def test_offline_commands_never_connect(cfg, monkeypatch, capsys):
    from worldarm import connection

    async def no_robot():
        raise AssertionError("connected to the robot")

    monkeypatch.setattr(connection, "connect", no_robot)
    asyncio.run(ball.main(["sim", "--n", "4"]))
    asyncio.run(ball.main(["report"]))
    with pytest.raises(SystemExit, match="No usable recorded rolls"):
        asyncio.run(ball.main(["fit"]))
    assert "predict" in capsys.readouterr().out
