"""Task C's pieces on their own: the swing's speed profile and its fit, the whole-arm safety check, the release
angle that forgives a late release, the swing search, the camera's resting-ball finder, the calibration."""

import asyncio
import math

import numpy as np
import pytest

from tests.toss_arm import ARMS, CHAIN, SWING, TABLE, TOOL, run
from worldarm.toss.calibrate import fit_constants, predicted
from worldarm.toss.landing import ArmsFrame, resting_ball
from worldarm.toss.profile import Trapezoid, fit_trapezoid
from worldarm.toss.sweep import KeepOut, motion_problem, pose_problem
from worldarm.toss.swing import (SwingSpec, choose_elevation, design_swing, end_pose, forgiveness, plan_throw,
                                 pose_at)

SPEC = SwingSpec(SWING["start_deg"], SWING["joints"], SWING["signs"], SWING["amplitude_deg"], SWING["accel_deg_s2"],
                 TOOL, 0.0, 80.0)
KEEP = KeepOut()


def test_trapezoid_progress_and_its_inverse():
    prof = Trapezoid(90.0, 120.0, 600.0, t0=0.1)
    assert prof.cruise == pytest.approx((12.0, 78.0))
    for p in (0.5, 5.0, 12.0, 40.0, 80.0, 89.9):
        assert prof.progress(prof.time_at(p)) == pytest.approx(p, abs=1e-9)
    assert prof.progress(0.05) == 0.0 and prof.progress(10.0) == 90.0
    short = Trapezoid(10.0, 120.0, 600.0)                  # too short to reach 120 deg/s: a triangle
    assert short.peak == pytest.approx(math.sqrt(6000.0)) and short.cruise[0] == pytest.approx(short.cruise[1])


def test_profile_fit_recovers_when_the_swing_started_and_how_fast():
    true = Trapezoid(90.0, 110.0, 700.0, t0=0.083)
    rng = np.random.default_rng(1)
    t = np.arange(0.0, 1.3, 0.021) + rng.normal(0, 0.002, 62)          # readings every ~21 ms, 2 ms timing noise
    p = np.array([true.progress(ti) for ti in t]) + rng.normal(0, 0.05, len(t))
    fit, sd, rms = fit_trapezoid(t, p, 90.0, 120.0, 600.0)              # starts from the planned values
    assert fit.t0 == pytest.approx(0.083, abs=0.004) and fit.w == pytest.approx(110.0, rel=0.01)
    assert fit.a == pytest.approx(700.0, rel=0.1) and rms < 0.5


def test_safety_check_refuses_the_table_a_wall_and_a_joint_limit():
    low = [0.0, 90.0, -10.0, 0.0, 30.0, 0.0]               # leaning forward, the gripper on the table
    assert "table" in pose_problem(CHAIN, low, TOOL, KEEP)
    up = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert pose_problem(CHAIN, up, TOOL, KEEP) is None
    wall = KeepOut(walls=[[[300.0, -500.0, 0.0], [600.0, 500.0, 2000.0]]])
    assert "keep-out" in pose_problem(CHAIN, up, TOOL, wall)
    assert "limit" in pose_problem(CHAIN, [0.0, 130.0, 0.0, 0.0, 0.0, 0.0], TOOL, KEEP)
    why = motion_problem(CHAIN, up, low, TOOL, KEEP)       # fine at the start, not on the way
    assert why is not None and "% of the way" in why
    assert motion_problem(CHAIN, SPEC.start_deg, end_pose(SPEC, 0.0), TOOL, KEEP) is None


def test_the_release_angle_is_chosen_so_a_late_release_lands_in_the_same_place():
    at45 = plan_throw(CHAIN, SPEC, (800.0, 0.0), 25.0, 45.0)
    f45 = forgiveness(CHAIN, SPEC, at45, 25.0)
    plan, f = choose_elevation(CHAIN, SPEC, (800.0, 0.0), 25.0, 180.0)
    assert at45.ok and plan.ok and abs(f.late_10ms_mm) < 3.0 < abs(f45.late_10ms_mm)
    assert np.allclose(plan.landing.xy * 1000.0, (800.0, 0.0), atol=2.0)


@pytest.mark.sim                                           # a few seconds of search
def test_swing_search_finds_safe_swings_that_land_on_target():
    found = design_swing(CHAIN, [1, 2, 4], 90.0, 600.0, (800.0, 0.0), 25.0, KEEP, [0.0] * 6, TOOL, 180.0, step_deg=40.0)
    assert found
    score, spec, plan, f = found[0]
    assert plan.ok and np.allclose(plan.landing.xy * 1000.0, (800.0, 0.0), atol=3.0)
    assert motion_problem(CHAIN, pose_at(spec, 0.0, plan.base_deg), end_pose(spec, plan.base_deg), TOOL, KEEP) is None
    assert [s for s, *_ in found] == sorted(s for s, *_ in found)


def test_arms_frame_round_trip():
    arms = ArmsFrame(1780.0, 40.0, 177.0)
    xy = np.array([812.0, -33.0])
    assert np.allclose(arms.to_thrower(arms.to_catcher(xy)), xy)
    assert np.allclose(ARMS.to_catcher((800.0, 0.0)), (980.0, 0.0))     # 80 cm ahead of armfarm05, toward armfarm03


def test_resting_ball_waits_until_the_ball_is_still():
    pts = [(900.0, 100.0), (930.0, 80.0), (950.0, 70.0)] + [(960.0, 65.0)] * 6   # bounces, rolls, stops

    def render(xy):
        import cv2
        img = np.full((720, 1280, 3), 235, np.uint8)
        u, v = TABLE.to_pixel(xy)
        cv2.circle(img, (int(round(u)), int(round(v))), 12, (25, 25, 25), -1)
        return img

    frames = iter([None] + [render(p) for p in pts])

    async def frame():
        await asyncio.sleep(0.05)
        return 0.0, next(frames)

    xy, why = run(resting_ball(frame, TABLE))
    assert why == "at rest" and np.allclose(xy, (960.0, 65.0), atol=2.5)


def test_calibration_recovers_speed_factor_release_delay_and_where_the_arms_stand():
    true = {"speed_factor": 0.97, "release_delay_s": 0.065, "x_mm": 1795.0, "y_mm": 25.0, "yaw_deg": 178.0}
    arms = ArmsFrame(true["x_mm"], true["y_mm"], true["yaw_deg"])
    rng = np.random.default_rng(3)
    throws = []
    for target in [(600.0, 0.0), (800.0, 0.0), (650.0, 250.0), (800.0, -200.0)]:
        plan = plan_throw(CHAIN, SPEC, target, 25.0, 40.0)
        for _ in range(2):
            t = {"spec": {**SPEC.__dict__, "tool_mm": list(TOOL)}, "base_deg": plan.base_deg,
                 "speed_deg_s": plan.speed_deg_s, "swing_w_deg_s": plan.speed_deg_s, "catch_z_mm": 25.0,
                 "sent_progress_deg": plan.trigger_progress_deg + rng.normal(0, 0.3)}
            land = predicted(CHAIN, t, true["speed_factor"], true["release_delay_s"])
            t["landing_catcher_mm"] = arms.to_catcher(land) + rng.normal(0, 4.0, 2)
            throws.append(t)
    prior = {"speed_factor": 1.0, "release_delay_s": 0.05, "x_mm": 1780.0, "y_mm": 0.0, "yaw_deg": 180.0}
    cal = fit_constants(CHAIN, throws, prior)
    assert cal.n == 8 and cal.rms_mm < 8.0 and not cal.note
    assert cal.values["speed_factor"] == pytest.approx(0.97, abs=0.01)
    assert cal.values["release_delay_s"] == pytest.approx(0.065, abs=0.012)
    assert cal.values["yaw_deg"] == pytest.approx(178.0, abs=0.7)
    one_speed = fit_constants(CHAIN, throws[:2], prior)     # the two throws at one target
    assert "can't be told apart" in one_speed.note


def test_ball_finder_places_a_dark_ball_from_colour_and_depth():
    """Waj's method: dark foam blob, depth to its top, centre on the plane half its height up."""
    import cv2
    from types import SimpleNamespace
    import toss
    from worldarm.toss.config import TossCfg
    cfg = TossCfg()
    intr = SimpleNamespace(focal_x_px=908.0, focal_y_px=907.0, center_x_px=640.0, center_y_px=360.0)
    cam_pos, cam_R = np.array([400.0, 300.0, 600.0]), np.diag([1.0, -1.0, -1.0])   # looking straight down
    ball = np.array([430.0, 270.0, cfg.pick_table_z_mm + 22.5])                   # a 45 mm ball's centre
    top = ball + [0.0, 0.0, 22.5]
    u = 640.0 + 908.0 * (top[0] - cam_pos[0]) / (cam_pos[2] - top[2])
    v = 360.0 + 907.0 * -(top[1] - cam_pos[1]) / (cam_pos[2] - top[2])
    bgr = np.full((720, 1280, 3), 200, np.uint8)
    cv2.circle(bgr, (int(round(u)), int(round(v))), 40, (30, 30, 30), -1)
    depth = np.full((720, 1280), cam_pos[2] - cfg.pick_table_z_mm, np.float32)    # the table, straight below
    cv2.circle(depth, (int(round(u)), int(round(v))), 40, float(np.linalg.norm(top - cam_pos)), -1)
    cv2.circle(bgr, (600, 715), 18, (30, 30, 30), -1)                           # a fingertip, cut off at the bottom edge
    cv2.circle(depth, (600, 715), 18, 300.0, -1)
    cv2.circle(bgr, (950, 650), 26, (30, 30, 30), -1)                           # far too small for a ball that close
    cv2.circle(depth, (950, 650), 26, 200.0, -1)
    cv2.circle(bgr, (200, 200), 9, (30, 30, 30), -1)                            # a dark speck on the table
    cv2.circle(depth, (200, 200), 9, 620.0, -1)
    found = toss.detect_balls(bgr, depth, intr, cam_pos, cam_R, cfg)
    assert len(found) == 1                                                      # only the ball
    assert found[0]["x"] == pytest.approx(430.0, abs=4.0) and found[0]["y"] == pytest.approx(270.0, abs=4.0)
    assert found[0]["height"] == pytest.approx(45.0, abs=6.0)
    assert found[0]["table_z"] == pytest.approx(cfg.pick_table_z_mm, abs=1.0)   # the table measured around it


def test_taught_throw_plan_keeps_every_limit_and_releases_near_peak_speed():
    """Waj's throw on armfarm05's model: the spline passes its poses, joints and the tool stay under the caps."""
    from worldarm.toss import taught
    from worldarm.toss.kinematics import load_chain
    chain = load_chain("config/kinematics_armfarm05.urdf")
    p = taught.plan(chain, taught.POSES["start"], tcp_limit=1000.0, release_shift_s=0.0)
    assert np.allclose(p["q"][0], taught.POSES["start"]) and np.allclose(p["q"][-1], taught.POSES["last"], atol=1e-6)
    assert p["peak_joint_speed"] <= 0.9 * taught.MAX_SPEED + 1.0
    tool = np.array([taught.point(chain, q, 175.0) for q in p["q"]])
    assert np.linalg.norm(np.diff(tool, axis=0), axis=1).max() / 0.01 <= 1000.0 * 1.03     # the xArm's tool cap
    assert 0 < p["release_t"] < p["duration"] and p["ball_speed"] > 0.5
