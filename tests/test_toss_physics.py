"""Throw physics: the numbers the plan rests on."""

import math

import numpy as np
import pytest

from worldarm.toss.physics import Release, drag_k, fly, sensitivity, speed_for, speed_for_drag


def test_the_planned_throw_lands_at_178_cm():
    land = fly(Release.aimed([0, 0, 0.3], 4.18, 45.0, 0.0), 0.3)
    assert land.xy[0] == pytest.approx(1.78, abs=0.005) and land.t == pytest.approx(0.60, abs=0.01)


def test_speed_for_is_the_inverse_of_fly():
    for dist, el, drop in ((1.78, 45, 0.0), (0.9, 45, 0.25), (1.2, 35, -0.1)):
        v = speed_for(dist, el, drop)
        land = fly(Release.aimed([0, 0, drop], v, el, 0.0), 0.0)
        assert land.xy[0] == pytest.approx(dist, abs=1e-6)


def test_aiming_by_azimuth():
    land = fly(Release.aimed([0.1, -0.2, 0.0], 3.0, 45.0, 30.0), 0.0)
    d = land.xy - np.array([0.1, -0.2])
    assert math.degrees(math.atan2(d[1], d[0])) == pytest.approx(30.0, abs=1e-6)


def test_drag_is_a_small_shortfall_and_speed_for_drag_compensates():
    k = drag_k()
    no_drag = fly(Release.aimed([0, 0, 0], 4.18, 45.0, 0.0), 0.0).xy[0]
    with_drag = fly(Release.aimed([0, 0, 0], 4.18, 45.0, 0.0), 0.0, k).xy[0]
    assert 0.0 < no_drag - with_drag < 0.08                      # a few cm short
    v = speed_for_drag(1.78, 45.0, 0.0, k)
    assert fly(Release.aimed([0, 0, 0], v, 45.0, 0.0), 0.0, k).xy[0] == pytest.approx(1.78, abs=0.003)


def test_45_degrees_is_forgiving_on_angle_but_not_on_speed():
    s = sensitivity(4.18, 45.0, 0.0, end_rate_deg_s=180.0)
    assert abs(s.per_deg_elevation_m) < 0.002                     # 1 degree off: under 2 mm
    assert 0.03 < s.per_pct_speed_m < 0.04                        # 1% speed: ~3.6 cm
    assert s.per_ms_late_m > 0.002                                # every ms late moves it millimetres
