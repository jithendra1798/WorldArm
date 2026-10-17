"""Pick skill on the simulated arm: a clean grip, a finger landing on the block (position error), and
nothing there - each reported as what it is, never pushing into anything."""

import asyncio
import dataclasses

import numpy as np
import pytest

from tests.fake_rig import FakeRig, FakeWorld, FBlock
from tests.synth import Box
from worldarm.act.grasp_plan import plan_grasp
from worldarm.act.pick import BLOCKED, HELD, MISSED, pick
from worldarm.cell.geometry import vec_of
from worldarm.see.scene import observe

pytestmark = pytest.mark.sim  # simulated arm + rendered frames: `pytest -m sim`


def setup(blocks):
    w = FakeWorld(Box(), blocks)
    rig = FakeRig(w)
    sc = observe(asyncio.run(rig.look()), rig.cfg)
    return w, rig, sc


def test_clean_grip_holds_with_the_expected_width():
    w, rig, sc = setup([FBlock("yellow", 400.0, -280.0, 0.0, 60.0, axis=35.0)])
    b = sc.table_blocks[0]
    r = asyncio.run(pick(rig, plan_grasp(b, rig.cfg, sc.table_blocks)))
    assert r.status == HELD and abs(r.width_mm - 33.0) < 3.0
    assert rig.held is w.blocks[0] and w.blocks[0].base > 50.0              # lifted


def test_a_finger_landing_on_the_block_stops_and_backs_off():
    """A 25 mm error along the finger axis (what an uncalibrated camera does) puts a finger on top."""
    w, rig, sc = setup([FBlock("yellow", 400.0, -280.0, 0.0, 60.0, axis=35.0)])
    p = plan_grasp(sc.table_blocks[0], rig.cfg, sc.table_blocks)
    shift = vec_of(p.finger_axis_deg) * 25.0
    off = dataclasses.replace(p, x=p.x + shift[0], y=p.y + shift[1])
    r = asyncio.run(pick(rig, off))
    assert r.status == BLOCKED and rig.held is None and w.blocks[0].base == 0.0


def test_nothing_there_is_a_miss():
    w, rig, sc = setup([FBlock("yellow", 400.0, -280.0, 0.0, 60.0, axis=35.0)])
    p = plan_grasp(sc.table_blocks[0], rig.cfg, sc.table_blocks)
    w.blocks.clear()                                                       # someone took it
    r = asyncio.run(pick(rig, p))
    assert r.status == MISSED and r.width_mm < 3.0


def test_short_block_is_gripped_without_the_fingertips_touching_the_table():
    w, rig, sc = setup([FBlock("green", 380.0, -260.0, 0.0, 18.0, 30.0, 40.0, axis=0.0)])
    b = sc.table_blocks[0]
    r = asyncio.run(pick(rig, plan_grasp(b, rig.cfg, sc.table_blocks)))
    assert r.status == HELD
    assert r.grip_frame_z - rig.cfg.gripper.tip_below_frame_mm >= rig.cfg.grasp.tip_clearance_mm - 0.5
