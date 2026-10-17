"""Continuous contact watch: gravity drift and wrist ringing are not contact; a push on J3 is."""

from worldarm.act.watch import ContactWatch
from worldarm.cell.config import ContactCfg

CFG = ContactCfg()
DT = 0.1


def loads(t, push=0.0, j6=0.0, j2_blip=0.0):
    """Six joint loads during a slow descent: smooth drift, plus optional events."""
    return [0.2, -4.3 + 0.4 * t + j2_blip, -15.4 - 0.5 * t + push, 0.1, -1.9 + 0.2 * t, j6]


def run(events, active_below=None, z_of=lambda t: 200.0 - 40.0 * t, seconds=4.0):
    w = ContactWatch(CFG, active_below)
    t = 0.0
    while t <= seconds:
        r = w.feed(t, z_of(t), loads(t, **events(t)))
        if r is not None:
            return r, w
        t = round(t + DT, 3)
    return None, w


def test_smooth_descent_is_not_contact():
    r, w = run(lambda t: {})
    assert r is None and w.worst < 0.5


def test_a_push_on_j3_is_contact():
    r, _ = run(lambda t: {"push": 2.5 if t >= 2.0 else 0.0})
    assert r is not None and 2.0 <= r.seconds <= 2.25 and r.contact


def test_joint_6_ringing_is_ignored():
    r, _ = run(lambda t: {"j6": 3.0 * (-1) ** int(t * 10) if 1.0 <= t <= 2.0 else 0.0})
    assert r is None


def test_one_odd_reading_is_not_contact():
    r, _ = run(lambda t: {"j2_blip": 6.0 if abs(t - 1.5) < 1e-6 else 0.0})
    assert r is None


def test_nothing_is_judged_above_reach():
    # reach is z <= 100, i.e. from t = 2.5. A blip up high, over well before then, is ignored...
    r, _ = run(lambda t: {"push": 2.5 if 1.0 <= t < 1.4 else 0.0}, active_below=100.0)
    assert r is None
    # ...and so is a lasting shift that starts up high (the trend takes it in before judging starts)
    r, _ = run(lambda t: {"push": 2.5 if t >= 1.5 else 0.0}, active_below=100.0)
    assert r is None
    # the same push inside reach stops the move
    r, _ = run(lambda t: {"push": 2.5 if t >= 3.0 else 0.0}, active_below=100.0)
    assert r is not None and r.z <= 100.0


def test_nothing_is_judged_during_warmup():
    r, _ = run(lambda t: {"push": 2.5 if t < 0.5 else 0.0})
    assert r is None
