"""Drawer skill on the simulated arm: open to the working opening, close without shoving the box,
and report a missed knob instead of pulling on nothing."""

import asyncio
import dataclasses

import numpy as np
import pytest

from tests.fake_rig import FakeRig, FakeWorld
from tests.synth import Box
from worldarm.act.drawer import BLOCKED, CLOSED, MISSED_KNOB, OPENED, ClientDrawers
from worldarm.see.scene import observe

pytestmark = pytest.mark.sim  # simulated arm + rendered frames: `pytest -m sim`


def setup(openings=(0.0, 0.0, 0.0)):
    w = FakeWorld(Box(openings=openings))
    rig = FakeRig(w)
    org = observe(asyncio.run(rig.look()), rig.cfg).organizer
    return w, rig, org


def slot_of(org, w, k):
    knob = org.drawer(k).knob[:2]
    return min(range(3), key=lambda i: np.linalg.norm(w.knob(i)[:2] - knob))


def test_open_a_shut_drawer_to_the_working_opening():
    w, rig, org = setup()
    slot = slot_of(org, w, 2)
    r = asyncio.run(ClientDrawers(rig).open(org, 2))
    assert r.status == OPENED and abs(w.openings[slot] - rig.cfg.organizer.open_target_mm) < 4


def test_open_a_partly_open_drawer_the_rest_of_the_way():
    w, rig, org = setup((0.0, 0.0, 55.0))
    k = next(d.index for d in org.drawers if d.state == "partial")
    r = asyncio.run(ClientDrawers(rig).open(org, k))
    assert r.status == OPENED and abs(max(w.openings) - rig.cfg.organizer.open_target_mm) < 4


def test_close_seats_the_drawer_without_shoving_the_box():
    w, rig, org = setup((0.0, 120.0, 0.0))
    k = next(d.index for d in org.drawers if d.state == "open")
    r = asyncio.run(ClientDrawers(rig).close(org, k))
    assert r.status == CLOSED and max(w.openings) < 1.0
    assert r.detail["seated"]["felt_shut"] and getattr(rig, "pressing", 0.0) <= 4.0


def test_close_with_the_opening_overestimated_still_stops_at_shut():
    """What slid the real box: the push went a fixed distance past the measured opening."""
    w, rig, org = setup((0.0, 120.0, 0.0))
    k = next(d.index for d in org.drawers if d.state == "open")
    drawers = [dataclasses.replace(d, opening_mm=d.opening_mm + 7.0) if d.index == k else d for d in org.drawers]
    biased = dataclasses.replace(org, drawers=drawers)
    r = asyncio.run(ClientDrawers(rig).close(biased, k))
    assert r.status == CLOSED and max(w.openings) < 1.0 and getattr(rig, "pressing", 0.0) <= 4.0


def knob_off(org, k, mm):
    d = org.drawer(k)
    off = dataclasses.replace(d, knob=d.knob + np.append(org.row * mm, 0.0))
    return dataclasses.replace(org, drawers=[off if x.index == k else x for x in org.drawers])


def test_the_close_look_fixes_a_knob_the_overview_saw_in_the_wrong_place():
    w, rig, org = setup()
    r = asyncio.run(ClientDrawers(rig).open(knob_off(org, 1, 16.0), 1))
    assert r.status == OPENED and r.detail["knob"]["from"] == "close look"
    assert abs(r.detail["knob"]["off_mm"]["1"] - 16.0) < 3.0


def test_a_knob_seen_in_the_wrong_place_is_a_miss_not_a_pull_on_nothing():
    """Without the close look (look_mm 0), a wrong knob position must still fail safely."""
    w, rig, org = setup()
    rig.cfg.camera.look_mm = 0.0
    r = asyncio.run(ClientDrawers(rig).open(knob_off(org, 1, 16.0), 1))
    assert r.status in (MISSED_KNOB, BLOCKED) and w.openings == [0.0, 0.0, 0.0]


def test_after_a_close_the_next_knob_is_already_measured():
    """Closing drawer 2 looks at the row on the way up; opening drawer 3 next grips where that look saw it,
    with no look of its own."""
    w, rig, org = setup((0.0, 120.0, 0.0))
    drawers = ClientDrawers(rig)
    k = next(d.index for d in org.drawers if d.state == "open")
    r = asyncio.run(drawers.close(org, k))
    assert r.status == CLOSED and r.detail["look_after"]["status"] == "seen"
    assert set(drawers.fresh) == {1, 2, 3}
    org2 = observe(asyncio.run(rig.look()), rig.cfg).organizer
    nxt = 3 if k != 3 else 1
    looks = rig.looks
    r = asyncio.run(drawers.open(org2, nxt))
    assert r.status == OPENED and r.detail["knob"]["from"].startswith("seen just before")
    assert rig.looks == looks                                    # no extra look before gripping


def test_knobs_are_gripped_gently():
    w, rig, org = setup()
    asyncio.run(ClientDrawers(rig).open(org, 2))
    g = rig.cfg.gripper
    assert rig.grabs[-1] == (g.knob_torque_pct, g.knob_grip_speed) and g.knob_torque_pct < 25


def test_close_with_the_opening_underestimated_still_shuts_it():
    w, rig, org = setup((0.0, 120.0, 0.0))
    k = next(d.index for d in org.drawers if d.state == "open")
    drawers = [dataclasses.replace(d, opening_mm=d.opening_mm - 7.0) if d.index == k else d for d in org.drawers]
    r = asyncio.run(ClientDrawers(rig).close(dataclasses.replace(org, drawers=drawers), k))
    assert r.status == CLOSED and max(w.openings) < 3.0
