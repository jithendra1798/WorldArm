"""toss.py's drills end to end, against the physical fakes in tests/toss_arm.py on a simulated clock (seconds
of swings and camera frames take a moment, the same way every run). Nothing touches the robot."""

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
from viam.proto.common import Pose

import toss
from tests.toss_arm import ARMS, CHAIN, SWING, TABLE, TOOL, XARMISH, FakeArm, World, run
from worldarm.toss.config import TossCfg
from worldarm.toss.landing import ArmsFrame

pytestmark = pytest.mark.sim  # simulated swings, gripper and camera frames: `pytest -m sim`


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setattr(TossCfg, "path", property(lambda self: tmp_path / "toss.json"))
    monkeypatch.setattr(toss, "RUNS", tmp_path / "runs")
    kin = tmp_path / "kinematics.json"
    kin.write_text(json.dumps(XARMISH))
    c = TossCfg(kinematics=str(kin), start_deg=list(SWING["start_deg"]), signs=list(SWING["signs"]),
                joints=list(SWING["joints"]), tool_mm=TOOL, target_mm=(800.0, 0.0), catch_z_mm=25.0, hold_mm=46.0,
                thrower_in_catcher=ARMS.to_json(), throw_mm=None,   # aim at a point; this test arm throws over its base
                clear_base_mm=0.0, walls=[], ceiling_z_mm=1300.0)
    c.save()
    return c


def quiet(*_):
    return ""


def test_probe_saves_the_arms_kinematics_and_checks_our_model_against_the_arm(cfg):
    lines = []
    arm = FakeArm(q=[0.0] * 6)
    res = run(toss.probe(arm, World(arm).gripper, cfg, say=lines.append))
    assert res["joints"] == 6 and res["model_error_mm"] < 0.01 and res["model_error_deg"] < 0.01
    assert res["do_command"]["set_acceleration"].startswith("ok") and Path(res["kinematics"]).exists()
    assert not any("WARNING" in l for l in lines)

    wrong = json.loads(json.dumps(XARMISH))
    wrong["links"][3]["translation"]["x"] = 456.0           # a forearm 30 mm longer than the arm's
    arm = FakeArm(q=[0.0] * 6, takes_accel=False)

    async def kin(**kw):
        return 1, json.dumps(wrong).encode(), {}
    arm.get_kinematics = kin
    lines = []
    res = run(toss.probe(arm, World(arm).gripper, cfg, say=lines.append))
    assert res["model_error_mm"] > 20.0 and any("WARNING" in l for l in lines)
    assert res["do_command"]["set_acceleration"].startswith("refused")


def test_plan_shows_the_throw_and_how_forgiving_it_is(cfg):
    lines = []
    res = toss.plan_cmd(cfg, CHAIN, say=lines.append)
    assert res["plan"]["ok"] and np.allclose(res["plan"]["landing_m"], (0.8, 0.0), atol=0.003)
    assert abs(res["forgiveness"]["late_10ms_mm"]) < 3.0          # the release angle that forgives lateness
    assert any("how forgiving" in l for l in lines) and any("armfarm03's" in l for l in lines)
    far = toss.plan_cmd(cfg, CHAIN, target=(4000.0, 0.0), say=lines.append)
    assert not far["plan"]["ok"]


def test_windup_refuses_a_way_through_the_table_and_otherwise_goes_slowly(cfg):
    arm = FakeArm(q=[0.0, 90.0, -10.0, 0.0, 30.0, 0.0])        # gripper down on the table
    with pytest.raises(SystemExit, match="Refused"):
        run(toss.windup(arm, CHAIN, cfg, say=quiet))
    assert not any(c[0] == "move" for c in arm.calls)
    arm = FakeArm(q=[0.0] * 6)
    q = run(toss.windup(arm, CHAIN, cfg, say=quiet))
    assert ("do_command", {"set_speed": 20.0}) in arm.calls and np.allclose(arm.q, q, atol=1e-6)


def test_dry_swings_measure_the_start_delay_speed_and_acceleration(cfg):
    arm = FakeArm(start_latency=0.08, jitter=(0.0, 0.012, -0.012), rtt=0.02)
    lines = []
    res = run(toss.dry(arm, CHAIN, cfg, repeat=3, say=lines.append))
    assert res["start_latency_s"] == pytest.approx(0.08, abs=0.005)
    assert res["start_latency_sd_s"] == pytest.approx(0.012, abs=0.004)       # the timed trigger's jitter
    assert res["speed_ratio"] == pytest.approx(1.0, abs=0.01) and res["accel_deg_s2"] == pytest.approx(600.0, rel=0.1)
    assert TossCfg.load().start_latency_s == pytest.approx(0.08, abs=0.005)
    arm = FakeArm(takes_accel=False, accel_default=900.0)
    lines = []
    run(toss.dry(arm, CHAIN, cfg, repeat=1, say=lines.append))
    assert any("accelerated at 9" in l for l in lines)                       # it says what the arm really does


def test_grip_teach_finds_the_opening_that_holds_and_measures_the_open_delay(cfg):
    world = World(FakeArm())
    cfg.pick_ball_height_mm = 50.0                                           # the fake ball's size
    answers = iter(["", "", "", "y"])                                        # holds at 2 mm under ball + 2
    res = run(toss.grip(world.gripper, cfg, teach=True, repeat=5, say=quiet, ask=lambda *_: next(answers)))
    assert res["hold_mm"] == 50.0 and TossCfg.load().hold_mm == 50.0
    assert 0.04 < res["median_s"] < 0.09                                     # 10 ms there + 30 ms + 3 mm of opening
    assert TossCfg.load().jaw_s == pytest.approx(res["median_s"], abs=1e-3)  # the throw's lead, saved


@pytest.mark.parametrize("trigger", ["predict", "timed"])
def test_a_throw_lands_where_the_plan_says_and_is_logged(cfg, trigger):
    cfg.trigger, cfg.start_latency_s = trigger, 0.08
    world = World(FakeArm(start_latency=0.08, rtt=0.02))
    lines = []
    log = run(toss.throw(world.arm, world.gripper, world, TABLE, cfg, CHAIN, repeat=2, say=lines.append, ask=quiet))
    assert len(log) == 2
    for t in log:
        assert np.linalg.norm(np.asarray(t["landing_catcher_mm"]) - t["planned_catcher_mm"]) < 15.0, t["sent"]
        assert t["sent_progress_deg"] == pytest.approx(t["trigger_deg"], abs=0.6)
    saved = json.loads(next((toss.RUNS).glob("*-toss-throw/throws.json")).read_text())
    assert len(saved) == 2 and world.arm.calls[-1][0] == "stop"


def test_ctrl_c_mid_throw_stops_the_arm(cfg):
    world = World(FakeArm())

    async def interrupted():
        task = asyncio.ensure_future(toss.throw(world.arm, world.gripper, world, TABLE, cfg, CHAIN, say=quiet, ask=quiet))
        for _ in range(2000):                                   # until the swing is under way (10 s at most)
            if world.arm.motion is not None:
                break
            await asyncio.sleep(0.005)
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run(interrupted())
    assert world.arm.calls[-1][0] == "stop" and world.arm.stopped


def test_calibration_from_throws_finds_the_true_constants_and_then_throws_land_closer(cfg):
    true_arms = ArmsFrame(1795.0, 25.0, 178.0)                       # the tape measure was a little off
    world = World(FakeArm(rtt=0.02), speed_factor=0.97, arms=true_arms)
    world.gripper.open_delay = 0.045                                 # the real release delay: 10 + 45 + 10 ms
    for target in [(650.0, 0.0), (800.0, 0.0), (700.0, 200.0), (800.0, -150.0)]:
        run(toss.throw(world.arm, world.gripper, world, TABLE, cfg, CHAIN, repeat=2, target=target, say=quiet, ask=quiet))
    before = toss.load_throws()
    miss_before = np.median([np.linalg.norm(np.asarray(t["landing_catcher_mm"]) - t["planned_catcher_mm"]) for t in before])
    res = toss.calibrate(cfg, CHAIN, say=quiet)
    assert res["values"]["speed_factor"] == pytest.approx(0.97, abs=0.01)
    assert res["values"]["release_delay_s"] == pytest.approx(0.065, abs=0.012)
    assert res["values"]["x_mm"] == pytest.approx(1795.0, abs=12.0) and res["values"]["yaw_deg"] == pytest.approx(178.0, abs=0.8)
    cfg = TossCfg.load()
    after = run(toss.throw(world.arm, world.gripper, world, TABLE, cfg, CHAIN, repeat=2, say=quiet, ask=quiet,
                           out=toss.RUNS / "after-toss-throw"))
    miss_after = np.median([np.linalg.norm(np.asarray(t["landing_catcher_mm"]) - t["planned_catcher_mm"]) for t in after])
    assert miss_before > 40.0 and miss_after < 15.0, (miss_before, miss_after)


class CatcherArm:
    """armfarm03 holding the bowl: records its moves."""

    def __init__(self):
        self.pose = Pose(x=300.0, y=0.0, z=300.0, o_x=0.0, o_y=0.0, o_z=-1.0, theta=0.0)
        self.calls: list = []

    async def do_command(self, cmd):
        self.calls.append(("do_command", cmd))
        return {}

    async def get_end_position(self, **kw):
        return self.pose

    async def move_to_position(self, pose, **kw):
        self.calls.append(("move", pose))
        await asyncio.sleep(0.5)
        self.pose = pose

    async def stop(self, **kw):
        self.calls.append(("stop", None))


def test_catch_puts_the_bowl_where_the_throw_will_land(cfg):
    cfg.bowl_hold = {"dx": -60.0, "dy": 0.0, "z": 220.0, "o_x": 1.0, "o_y": 0.0, "o_z": 0.0, "theta": 0.0}
    cfg.thrower_in_catcher = {"x_mm": 1300.0, "y_mm": 100.0, "yaw_deg": 180.0}   # arms moved closer: in reach
    catcher, world = CatcherArm(), World(FakeArm())
    res = run(toss.catch(catcher, world.arm, world.gripper, world, TABLE, cfg, CHAIN, dry_run=True, say=quiet))
    moves = [c[1] for c in catcher.calls if c[0] == "move"]
    assert np.allclose(res["bowl_at_mm"], (500.0, 100.0), atol=3.0)          # 800 mm ahead of armfarm05
    assert (moves[-1].x, moves[-1].y, moves[-1].z) == pytest.approx((res["bowl_at_mm"][0] - 60.0, 100.0, 220.0), abs=3.0)
    assert (moves[0].x, moves[0].y, moves[0].z) == pytest.approx((300.0, 0.0, 380.0))       # straight up first
    assert catcher.calls[-1][0] == "stop"
    cfg.thrower_in_catcher = ARMS.to_json()                                  # 1.78 m apart: out of reach
    with pytest.raises(SystemExit, match="reach"):
        run(toss.catch(CatcherArm(), world.arm, world.gripper, world, TABLE, cfg, CHAIN, dry_run=True, say=quiet))


def test_report_gives_the_spread_of_the_landings_and_how_far_off_the_plan(cfg, tmp_path):
    rng = np.random.default_rng(0)
    throws = [{"speed_deg_s": 118.4, "base_deg": 0.0, "trigger": "predict", "planned_catcher_mm": [980.0, 0.0],
               "landing_catcher_mm": (np.array([990.0, 5.0]) + rng.normal(0, 8.0, 2)).tolist(),
               "sent_progress_deg": 26.4 + rng.normal(0, 0.12), "trigger_deg": 26.4} for _ in range(12)]
    out = toss.RUNS / "20261017-100000-toss-throw"
    out.mkdir(parents=True)
    (out / "throws.json").write_text(json.dumps(throws))
    lines = []
    res = toss.report(cfg, say=lines.append)
    (r,) = res.values()
    assert r["n"] == 12 and 4.0 < r["sd_mm"][0] < 14.0 and 5.0 < np.linalg.norm(r["off_plan_mm"]) < 20.0
    assert r["release_ms_sd"] == pytest.approx(1.0, abs=0.6)                 # 0.12 deg at 118 deg/s


def test_a_release_that_never_fires_keeps_the_ball_and_says_so(cfg):
    world = World(FakeArm())
    calls = {"n": 0}
    real = world.arm.get_joint_positions

    async def flaky(**kw):                                      # readings fail once the swing is commanded
        calls["n"] += 1
        if any(c[0] == "move" for c in world.arm.calls):
            raise RuntimeError("timeout")
        return await real(**kw)
    world.arm.get_joint_positions = flaky
    lines = []
    log = run(toss.throw(world.arm, world.gripper, world, TABLE, cfg, CHAIN, say=lines.append, ask=quiet))
    assert log[0]["released"] is False and any("never fired" in l for l in lines)
    sets = [c for c in world.gripper.calls if "set" in c]
    assert sets[-1] == {"set": cfg.counts(cfg.hold_mm)}                  # never told to open
    assert world.released == [] and world.arm.calls[-1][0] == "stop"


def test_throw_with_pick_finds_the_ball_picks_it_and_throws(cfg, monkeypatch):
    chain = load_real_chain()
    real = TossCfg(kinematics="config/kinematics_armfarm05.urdf", start_deg=[4.2, -27.0, -57.0, 0.0, -59.0, 0.0],
                   joints=[0, 1, 4], signs=[1, 1, 1], amplitude_deg=40.0, accel_deg_s2=1000.0, elevation_deg=45.0,
                   throw_mm=219.0, min_throw_mm=200.0, hold_mm=46.0, speed_factor=1.0,
                   walls=[[[711.0, -1500.0, -500.0], [811.0, 1500.0, 1100.0]], [[-1500.0, -1395.0, -500.0], [1500.0, -1295.0, 1100.0]]],
                   ceiling_z_mm=1000.0, thrower_in_catcher={"x_mm": 0.0, "y_mm": -1905.0, "yaw_deg": 0.0})
    world = World(FakeArm(q=[0.0, 0.0, -60.0, 0.0, -60.0, 0.0]))
    looked = []

    async def ball_on_table(machine, camera, c, chain, q, save=None):     # what the wrist camera reports
        if save is not None and save.name == "after-lift.jpg":
            return []                                                      # lifted: gone from the table
        looked.append(camera)
        return [{"x": 400.0, "y": 300.0, "height": 45.0, "u": 640, "v": 360, "area": 3000, "darkness": 30.0}]
    monkeypatch.setattr(toss, "wrist_look", ball_on_table)
    picker = lambda seen=None, grab=True: toss.pick_ball(world.arm, world.gripper, None, chain, real, "cam", say=quiet,
                                                         seen=seen, grab=grab)
    log = run(toss.throw(world.arm, world.gripper, None, TABLE, real, chain, manual=True, say=quiet, ask=lambda *_: "",
                         picker=picker))
    assert log and log[0].get("sent") is not None and world.released                           # picked, then thrown
    assert len(looked) == 4                       # 2 looks to pick it, 2 more after the throw to find where it landed
    assert log[0]["landing_catcher_mm"] is not None and "wrist camera" in log[0]["landing_how"]
    assert world.arm.calls[-1][0] == "stop"


def load_real_chain():
    from worldarm.toss.kinematics import load_chain
    return load_chain("config/kinematics_armfarm05.urdf")
