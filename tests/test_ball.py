"""Chapter 2 core, no robot: the rolling model, learning its slowdown, the tracker, the table map, and the
goalie in simulation (predicting must beat chasing, and the learned model must beat constant velocity)."""

import math

import numpy as np
import pytest

from worldarm.ball.intercept import ArmModel, Goalie, Lane
from worldarm.ball.model import Roll, RollModel, fit_deceleration
from worldarm.ball.holdout import holdout_errors
from worldarm.ball.sim import SimCfg, intercept_rate, paddle_step, prediction_error, stop_error
from worldarm.ball.table import fit_table_map, leave_one_out_mm
from worldarm.ball.track import Tracker

LANE = Lane(np.array([450.0, -250.0]), np.array([450.0, 250.0]), 120.0, 0.0, home=np.array([0.0, 0.0]))


def test_rolling_ball_slows_and_stops():
    m = RollModel(100.0)
    r = Roll(0.0, np.array([0.0, 0.0]), np.array([1.0, 0.0]), 400.0)
    assert m.stop_time(r) == pytest.approx(4.0)
    assert m.at(r, 10.0)[0] == pytest.approx(800.0)          # v^2 / 2a, then it stays put
    assert m.at(r, 2.0)[0] == pytest.approx(400 * 2 - 50 * 4)


def test_crossing_point_and_time():
    m = RollModel(100.0)
    r = Roll(0.0, np.array([0.0, 0.0]), np.array([math.cos(0.3), math.sin(0.3)]), 600.0)
    xy, t = m.crossing(r, np.array([500.0, 0.0]), np.array([1.0, 0.0]))
    assert xy[0] == pytest.approx(500.0) and np.allclose(m.at(r, t), xy, atol=1e-6)
    assert m.crossing(r, np.array([5000.0, 0.0]), np.array([1.0, 0.0])) is None   # stops short
    assert m.crossing(r, np.array([-100.0, 0.0]), np.array([1.0, 0.0])) is None   # behind it


def test_slowdown_is_learned_from_recorded_rolls():
    rng = np.random.default_rng(0)
    m, tracks = RollModel(140.0), []
    for _ in range(12):
        r = Roll(0.0, rng.uniform(-50, 50, 2), np.array([1.0, 0.2]) / np.hypot(1, 0.2), rng.uniform(400, 800))
        t = np.arange(0, 2.5, 1 / 23)
        tracks.append((t, np.array([m.at(r, ti) for ti in t]) + rng.normal(0, 4, (len(t), 2))))
    a, spread, n = fit_deceleration(tracks)
    assert n == 12 and abs(a - 140.0) < 15.0



def test_slowdown_fit_shrugs_off_misdetections():
    rng = np.random.default_rng(2)
    m, tracks = RollModel(140.0), []
    for _ in range(9):
        r = Roll(0.0, np.zeros(2), np.array([1.0, 0.0]), rng.uniform(500, 800))
        t = np.arange(0, 2.5, 1 / 23)
        xy = np.array([m.at(r, ti) for ti in t]) + rng.normal(0, 4, (len(t), 2))
        xy[-4] += (-90.0, 0.0)                               # two wrong sightings late in each roll
        xy[-9] += (-70.0, 0.0)                               # (a plain fit says 153)
        tracks.append((t, xy))
    a, _, n = fit_deceleration(tracks)
    assert n == 9 and abs(a - 140.0) < 6.0


def _recorded_rolls(n: int, decel: float = 140.0, seed: int = 0) -> list:
    """Rolls as `record` keeps them: from the push until the ball has nearly stopped, 23 fps, 4 mm noise."""
    rng = np.random.default_rng(seed)
    m, tracks = RollModel(decel), []
    for _ in range(n):
        r = Roll(0.0, rng.uniform(-50, 50, 2), np.array([1.0, 0.2]) / np.hypot(1, 0.2), rng.uniform(400, 800))
        t = np.arange(0, m.stop_time(r) - 0.05, 1 / 23)
        tracks.append((t, np.array([m.at(r, ti) for ti in t]) + rng.normal(0, 4, (len(t), 2))))
    return tracks


def test_h4_on_held_out_rolls():
    """Each roll predicted with the slowdown learned from the others: arrival time and resting point."""
    tracks = _recorded_rolls(12)
    lost = (tracks[0][0][:20], tracks[0][1][:20])          # lost from view while still rolling fast
    h = holdout_errors(tracks + [lost])
    s = h.summary()
    for horizon in (1.0, 0.6):
        a = s["arrival"][horizon]
        assert a["learned"]["median"] < 0.3 * a["constant"]["median"] and a["learned"]["failed"] == 0, a
        assert a["learned_closer"]["closer"] == a["learned_closer"]["n"] >= 10
    assert np.nanmax(np.abs(h.decels - 140.0)) < 10.0
    assert np.isnan(h.stop_learned[-1])                     # where a lost ball stopped is unknown: not judged
    assert s["stop"]["n"] >= 9 and s["stop"]["median"] < 40.0, s["stop"]
    rng = np.random.default_rng(1)
    m = RollModel(120.0)
    r = Roll(0.0, np.array([0.0, 0.0]), np.array([0.6, 0.8]), 700.0)
    tr = Tracker(decel=120.0)
    for t in np.arange(0, 0.4, 1 / 23):
        tr.add(t, m.at(r, t) + rng.normal(0, 4, 2))
    est = tr.estimate()
    t_last = est.roll.t0
    assert np.linalg.norm(est.roll.p0 - m.at(r, t_last)) < 8.0
    assert abs(est.roll.v0 - (700.0 - 120.0 * t_last)) < 40.0
    assert float(est.roll.u @ np.array([0.6, 0.8])) > 0.995


def test_tracker_starts_over_on_a_new_roll():
    tr = Tracker()
    for t in np.arange(0, 0.3, 0.05):
        tr.add(t, (t * 500.0, 0.0))
    tr.add(1.0, (0.0, 300.0))                                  # long gap: a new roll
    assert len(tr.seen) == 1


def test_table_map_from_a_few_points():
    # a tilted camera: a true homography, then fit it back from 6 noisy points
    H = np.array([[1.4, 0.12, -300.0], [-0.08, 1.5, -500.0], [1e-5, 2e-5, 1.0]])
    uv = np.array([[u, v] for u in (200.0, 640.0, 1080.0) for v in (150.0, 380.0, 610.0)])  # a 3 x 3 grid
    w = np.c_[uv, np.ones(len(uv))] @ H.T
    xy = w[:, :2] / w[:, 2:3] + np.random.default_rng(2).normal(0, 1.0, (len(uv), 2))
    m = fit_table_map(uv, xy)
    assert m.rms_mm < 2.0 and max(leave_one_out_mm(uv, xy)) < 6.0
    assert np.allclose(m.to_pixel(m.to_world([640, 360])), [640, 360], atol=1e-6)


def test_goalie_commits_once_and_only_retargets_on_real_changes():
    g = Goalie(LANE, RollModel(120.0), ArmModel(), 90.0, "predict")
    tr = Tracker(decel=120.0)
    r = Roll(0.0, np.array([1300.0, 40.0]), np.array([-1.0, 0.0]), 700.0)   # from the far end toward the arm
    cmds = []
    for t in np.arange(0, 0.8, 1 / 23):
        tr.add(t, RollModel(120.0).at(r, t))
        d = g.decide(t, tr.estimate(), LANE.length / 2)
        if d:
            cmds.append(d)
    assert len(cmds) == 1 and abs(LANE.point(cmds[0].s)[1] - 40.0) < 10.0 and cmds[0].in_time


SIM = SimCfg()
ARM = ArmModel(latency_s=0.45, speed_mm_s=250.0)   # a realistic arm: slow to start, 25 cm/s along the lane


def test_predicting_beats_chasing_in_simulation():
    """H5 in simulation, a 12 cm paddle against diagonal rolls (the sweep: 10-30 points ahead everywhere)."""
    pred = intercept_rate(LANE, "predict", SIM.true_decel, ARM, SIM, n=100, seed=3, half_width_mm=60.0)
    chase = intercept_rate(LANE, "chase", SIM.true_decel, ARM, SIM, n=100, seed=3, half_width_mm=60.0)
    assert pred["rate"] > 0.85 and pred["rate"] > chase["rate"] + 0.15, (pred, chase)


def test_learned_slowdown_predicts_arrival_time_better_than_constant_velocity():
    learned = prediction_error(SIM.true_decel, SIM, LANE, n=80, seed=4)
    constant = prediction_error(0.0, SIM, LANE, n=80, seed=4)
    assert learned[1.0][0] < 0.5 * constant[1.0][0], (learned, constant)   # a second before arrival


def test_learned_slowdown_predicts_where_it_stops():
    err, n = stop_error(SIM.true_decel, SIM, n=80, seed=5)
    assert n > 60 and err < 40.0
    assert stop_error(0.0, SIM)[0] == float("inf")            # constant velocity never stops



def test_a_prediction_that_it_stops_short_counts_as_a_failure():
    wrong = prediction_error(4 * SIM.true_decel, SIM, LANE, n=40, seed=4)   # believes it slows 4x faster
    assert math.isinf(wrong[1.0][0])


def test_paddle_speeds_up_and_brakes_onto_its_target():
    s, v, t = 0.0, 0.0, 0.0
    while (s, v) != (200.0, 0.0) and t < 3.0:
        s, v = paddle_step(s, v, 200.0, 250.0, 1000.0, 0.01)
        assert s <= 200.0                                    # never past it
        t += 0.01
    assert t == pytest.approx(1.05, abs=0.05)                # 0.25 s to full speed, 0.55 s cruising, 0.25 s braking
    s, v = 100.0, 250.0
    for _ in range(30):                                      # stopped: it brakes to rest, 31 mm on
        s, v = paddle_step(s, v, None, 250.0, 1000.0, 0.01)
    assert v == 0.0 and s == pytest.approx(100.0 + 250.0 ** 2 / 2000.0, abs=3.0)


def test_the_realistic_robot_is_harder_but_predicting_still_beats_chasing():
    real = SimCfg.realistic()
    rate = {m: intercept_rate(LANE, m, real.true_decel, ARM, real, n=40, seed=3, half_width_mm=60.0)["rate"]
            for m in ("predict", "chase")}
    ideal = intercept_rate(LANE, "predict", SIM.true_decel, ARM, SIM, n=40, seed=3, half_width_mm=60.0)["rate"]
    assert rate["chase"] < rate["predict"] < ideal, (rate, ideal)
    """The session calls each roll from what the camera sees; in simulation we know the truth."""
    import numpy as np
    from worldarm.ball.sim import roll_once
    rng = np.random.default_rng(9)
    agree = n = 0
    for mode in ("predict", "chase"):
        goalie = Goalie(LANE, RollModel(SIM.true_decel), ARM, 60.0, mode)
        for _ in range(40):
            o = roll_once(LANE, goalie, ARM, Tracker(decel=SIM.true_decel), SIM, rng)
            if not o.reached or o.s_cross is None or not (-60 <= o.s_cross <= LANE.length + 60):
                continue
            n += 1
            agree += (o.judged == "blocked") == o.defended
    assert n > 50 and agree / n > 0.95, (agree, n)


def test_ball_detector_finds_the_ball_and_ignores_cables_and_off_table_dark_things():
    import cv2
    from worldarm.ball.detect import find_ball
    img = np.full((720, 1280, 3), 230, np.uint8)
    cv2.circle(img, (700, 400), 18, (20, 20, 20), -1)                     # the ball
    cv2.line(img, (300, 200), (900, 210), (25, 25, 25), 4)                # a cable on the table
    cv2.circle(img, (100, 100), 18, (20, 20, 20), -1)                     # dark round thing off the table
    table = np.array([[250, 150], [1200, 150], [1200, 650], [250, 650]])
    s = find_ball(img, table_px=table)
    assert s is not None and abs(s.u - 700) < 1.0 and abs(s.v - 400) < 1.0
    assert find_ball(img, table_px=table, avoid_px=[(700, 400, 40)]) is None


def test_ball_config_round_trips(tmp_path, monkeypatch):
    from worldarm.ball.config import BallCfg
    monkeypatch.setattr(BallCfg, "path", property(lambda self: tmp_path / "ball.json"))
    c = BallCfg()
    c.lane, c.arm, c.decel_mm_s2 = LANE.to_json(), {"latency_s": 0.4, "speed_mm_s": 220.0}, 130.0
    c.save()
    d = BallCfg.load()
    assert d.get_lane().length == LANE.length and d.arm_model().speed_mm_s == 220.0 and d.decel_mm_s2 == 130.0


# ---- judging a roll from what the camera sees (session.py) -------------------------------------------------

def _at(gap, s=250.0):
    """The point `gap` mm out from LANE (toward the far end), `s` mm along it."""
    return LANE.point(s) - LANE.normal * gap


def _judge(gaps, s_paddle=250.0, near_mm=None, s=250.0, keep=False, known=True):
    """Feed one sighting per frame (23 fps): a gap from the lane, or None for a missed frame."""
    from worldarm.ball.session import RollSession
    rs = RollSession(LANE, Goalie(LANE, RollModel(120.0), ArmModel(), 60.0), Tracker(decel=120.0), 25.0, near_mm)
    for i, g in enumerate(gaps):
        rs.on_sighting(i / 23, None if g is None else _at(g, s), s_paddle, known)
    rs.t_end = (len(gaps) - 1) / 23
    return rs.log if not keep else rs


def test_session_blocked_when_it_comes_back_from_the_paddle():
    log = _judge(list(range(600, 0, -30)) + list(range(0, 200, 10)))
    assert log.outcome == "blocked" and abs(log.s_cross - 250.0) < 1.0


def test_session_goal_when_seen_past_the_lane():
    log = _judge(list(range(600, -200, -30)))
    assert log.outcome == "goal" and abs(log.s_cross - 250.0) < 1.0


def test_session_short_when_it_stops_before_the_lane():
    gaps = [600 - 300 * t + 150 * t * t for t in np.arange(0, 1.0, 1 / 23)] + [450.0] * 15   # slows, stops at 450
    assert _judge(gaps).outcome == "short"


def test_session_waits_for_the_roll_a_ball_at_rest_at_the_far_end_is_not_short():
    log = _judge([800.0] * 40)
    assert log.outcome is None and log.t_start is None
    log = _judge([800.0] * 30 + list(range(800, 0, -30)) + list(range(0, 200, 10)))
    assert log.outcome == "blocked" and log.t_start > 1.2


def test_session_blocked_when_it_reappears_going_back_after_hiding_next_to_the_paddle():
    """Live, the camera ignores the paddle's surroundings: the ball vanishes ~150 mm out, bounces unseen and
    reappears going back. Only the widened 'near' (paddle zone + ball + a frame's travel) calls it."""
    gaps = list(range(600, 150, -30)) + [None] * 12 + [145, 150, 160, 170, 180, 190, 195, 198, 200] + [200] * 12
    assert _judge(gaps, near_mm=90.0 + 25.0 + 50.0).outcome == "blocked"
    assert _judge(gaps).outcome == "short"            # the sim's 'near' never saw it close enough


def test_session_at_rest_against_the_paddle_is_blocked_elsewhere_short():
    gaps = [400 - 300 * t + 100 * t * t for t in np.arange(0, 1.5, 1 / 23)] + [30.0] * 15     # rolls up to 30 mm
    assert _judge(gaps, s_paddle=250.0).outcome == "blocked"
    assert _judge(gaps, s_paddle=100.0).outcome == "short"


def test_session_ignores_a_sighting_too_far_from_the_ball_to_be_it():
    gaps = list(range(600, 300, -30)) + [-100] + list(range(300, 150, -30)) + [150.0] * 15   # one stray dark spot
    log = _judge(gaps)
    assert log.outcome == "short" and any(abs(x - _at(-100)[0]) < 1.0 for _, x, _ in log.sightings)  # logged, not a goal


def test_session_gives_up_on_a_ball_out_of_sight_for_a_while():
    """Hidden next to the paddle for good (or picked up): seen again only at the far end, it proves nothing."""
    rs = _judge(list(range(600, 150, -30)) + [None] * 50, near_mm=165.0, keep=True)
    assert rs.lost(rs.t_end) and rs.log.outcome is None
    assert not _judge([800.0] * 80, keep=True).lost(80 / 23)      # never rolled: waiting, not lost


def test_session_blocked_when_it_vanishes_into_the_paddle_that_is_there():
    """Live, the ball can vanish right at the paddle and stay hidden (resting against it, or a weak bounce).
    Heading in, the paddle measured where it would cross, not seen past the lane a second later: blocked."""
    gaps = list(range(600, 50, -30)) + [None] * 40              # last seen 60 mm out, at ~690 mm/s
    log = _judge(gaps, s_paddle=250.0)
    assert log.outcome == "blocked" and abs(log.s_cross - 250.0) < 5.0
    rs = _judge(gaps, s_paddle=100.0, keep=True)               # the paddle wasn't there: can't say
    assert rs.log.outcome is None
    assert _judge(gaps, known=False).outcome is None            # where the paddle was isn't known: can't say
    assert _judge(list(range(600, 50, -30)) + [None] * 10 + [-60, -90]).outcome == "goal"   # seen past in time: goal


def test_session_a_stray_sighting_does_not_reach_the_tracker():
    rs = _judge(list(range(600, 300, -30)) + [-100, 300], keep=True)
    assert len(rs.tracker.seen) == 11                          # the roll so far, not restarted by the stray


def test_session_back_out_of_sight_of_the_paddle_and_at_rest_is_blocked():
    """A weak bounce: it vanishes at the paddle, comes back out and stops just in front of it (too close to
    have gone back 15 mm, too far to be touching). It went in and came out: blocked. Never hidden: short."""
    gaps = list(range(600, 50, -30)) + [None] * 6 + [52.0, 56.0, 58.0] + [60.0] * 14
    assert _judge(gaps, near_mm=95.0).outcome == "blocked"
    assert _judge(list(range(600, 50, -30)) + [55.0, 58.0] + [60.0] * 14, near_mm=95.0).outcome == "short"
