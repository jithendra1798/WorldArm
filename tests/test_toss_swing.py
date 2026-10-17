"""The throw planner on a known arm: release at 45 degrees in the cruise, distance from speed alone, aim from
the base joint alone - and the predicted landing is where the ball really goes."""

import numpy as np
import pytest

from tests.test_toss_kinematics import PLANAR
from worldarm.toss.kinematics import Chain
from worldarm.toss.swing import SwingSpec, elevation_deg, find_release, plan_throw, unit_velocity

CHAIN = Chain.from_sva(PLANAR)
# an overhand catapult: wound back over the shoulder, the three pitch joints whip forward together and the
# ball leaves while the arm still points up and back (heading 45 degrees up)
SPEC = SwingSpec(start_deg=[0.0, 120.0, 40.0, 40.0], joints=[1, 2, 3], signs=[-1, -1, -1], amplitude_deg=80.0,
                 accel_deg_s2=1500.0, table_z_mm=0.0, clearance_mm=60.0, tool_mm=(0.0, 0.0, 0.0))


def test_release_point_is_where_the_ball_heads_45_degrees_up():
    p = find_release(CHAIN, SPEC, 45.0)
    assert p is not None
    assert elevation_deg(unit_velocity(CHAIN, SPEC, p, 0.0)) == pytest.approx(45.0, abs=0.05)


@pytest.mark.parametrize("target", [(900.0, 0.0), (1100.0, 300.0), (800.0, -400.0)])
def test_planned_throw_lands_on_target(target):
    plan = plan_throw(CHAIN, SPEC, target, catch_z_mm=100.0, max_speed_deg_s=400.0)
    assert plan.landing is not None
    assert np.allclose(plan.landing.xy * 1000.0, target, atol=2.0), plan.to_json()


def test_distance_comes_from_speed_alone():
    near = plan_throw(CHAIN, SPEC, (800.0, 0.0), 100.0, max_speed_deg_s=400.0)
    far = plan_throw(CHAIN, SPEC, (1200.0, 0.0), 100.0, max_speed_deg_s=400.0)
    assert far.speed_deg_s > near.speed_deg_s
    assert far.release_progress_deg == pytest.approx(near.release_progress_deg, abs=1e-6)  # same release point


def test_a_throw_beyond_the_speed_limit_is_refused():
    plan = plan_throw(CHAIN, SPEC, (5000.0, 0.0), 100.0, max_speed_deg_s=180.0)
    assert not plan.ok and "deg/s" in plan.reason


def test_release_delay_moves_the_trigger_earlier_not_the_release():
    a = plan_throw(CHAIN, SPEC, (900.0, 0.0), 100.0, max_speed_deg_s=400.0)
    b = plan_throw(CHAIN, SPEC, (900.0, 0.0), 100.0, max_speed_deg_s=400.0, release_delay_s=0.02)
    assert b.release_progress_deg == pytest.approx(a.release_progress_deg)
    assert a.trigger_progress_deg - b.trigger_progress_deg == pytest.approx(a.speed_deg_s * 0.02, rel=1e-6)


def test_a_swing_that_dips_toward_the_table_is_refused():
    low = SwingSpec(start_deg=[0.0, 10.0, -20.0, -20.0], joints=[1, 2, 3], signs=[-1, -1, -1], amplitude_deg=80.0,
                    accel_deg_s2=1500.0, table_z_mm=0.0, clearance_mm=60.0, tool_mm=(0.0, 0.0, 0.0))
    plan = plan_throw(CHAIN, low, (900.0, 0.0), 100.0, max_speed_deg_s=400.0)
    assert not plan.ok
