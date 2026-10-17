"""Every drill, run against the simulated arm: the same code the morning's real drills run."""

import asyncio

import numpy as np
import pytest

from tests.fake_rig import FakeRig, FakeWorld, FBlock
from tests.synth import Box
from worldarm import drills
from worldarm.act.drawer import ClientDrawers

pytestmark = pytest.mark.sim  # simulated arm + rendered frames: `pytest -m sim`


def world(openings=(0.0, 0.0, 0.0)):
    return FakeWorld(Box(openings=openings), [FBlock("yellow", 400.0, -280.0, 0.0, 60.0, axis=35.0),
                                             FBlock("green", 330.0, -230.0, 0.0, 35.0, 30.0, 60.0, axis=-20.0)])


def go(coro):
    return asyncio.run(coro)


def test_check_passes_with_everything_readable():
    r = go(drills.check(FakeRig(world())))
    assert r["pass"], r["problems"]


def test_teach_view_saves_a_good_view(tmp_path, monkeypatch):
    rig = FakeRig(world())
    monkeypatch.setattr(type(rig.cfg), "path", property(lambda self: tmp_path / "cfg.json"))
    r = go(drills.teach_view(rig))
    assert r["pass"] and (tmp_path / "cfg.json").exists()


def test_scan_reports_and_is_repeatable(tmp_path):
    r = go(drills.scan(FakeRig(world()), times=2, out=tmp_path))
    assert r["pass"] and any("yellow" in line for line in r["lines"])
    assert all(v < 1.0 for v in r["repeatability_mm"].values())
    assert (tmp_path / "scan-1.jpg").exists() and (tmp_path / "scan-1.npz").exists()


def test_scan_replays_a_saved_frame(tmp_path):
    from worldarm.see.frame import Frame
    rig = FakeRig(world())
    go(drills.scan(rig, out=tmp_path))
    f = Frame.load(tmp_path / "scan-1.npz")
    r = drills.policy_dry(frame=f, cfg=rig.cfg)
    r = go(r)
    assert "open drawer 1" in r["next"]


def test_pick_test_picks_and_puts_back():
    w = world()
    start = (w.blocks[0].x, w.blocks[0].y)
    r = go(drills.pick_test(FakeRig(w), color="yellow", times=2))
    assert r["pass"], r
    assert np.hypot(w.blocks[0].x - start[0], w.blocks[0].y - start[1]) < 5.0  # back where it was


def test_drawer_cycle():
    rig = FakeRig(world())
    r = go(drills.drawer_test(rig, ClientDrawers(rig), 2, "cycle", times=2))
    assert r["pass"], r
    assert rig.world.openings == [0.0, 0.0, 0.0]


def test_drop_then_unload():
    rig = FakeRig(world())
    r = go(drills.drop_test(rig, ClientDrawers(rig), 1, "yellow"))
    assert r["pass"], r
    r = go(drills.unload_test(rig, ClientDrawers(rig), 1, "yellow"))
    assert r["pass"], r


def test_touch_measures_the_fingertip_length():
    rig = FakeRig(world())
    r = go(drills.touch(rig))
    assert r["pass"] and abs(r["tip_below_frame_mm"] - rig.cfg.gripper.tip_below_frame_mm) < 1.0


def test_calibrate_runs_and_finds_nothing_to_fix_in_a_perfect_camera():
    r = go(drills.calibrate(FakeRig(world())))
    assert r["pass"], r
    assert r["report"]["rms_after_mm"] < 1.0
