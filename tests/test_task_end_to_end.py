"""The whole task in simulation: real perception on rendered frames, the real policy, belief and skills,
and a fake arm with simple physics. Each scenario must end sorted, with every drawer shut."""

import asyncio

import numpy as np
import pytest

from tests.fake_rig import FakeRig, FakeWorld, FBlock
from tests.synth import Box
from worldarm.act.drawer import ClientDrawers
from worldarm.act.grasp_plan import plan_grasp
from worldarm.act.pick import pick
from worldarm.act.place import drawer_target, place
from worldarm.see.scene import observe
from worldarm.task.runner import run_task

pytestmark = pytest.mark.sim  # simulated arm + rendered frames: `pytest -m sim`


def run(world, **kw):
    rig = FakeRig(world)
    lines = []
    result = asyncio.run(run_task(rig, ClientDrawers(rig, say=lines.append), say=lines.append, **kw))
    return result, rig, lines


def sorted_world(world):
    """Table clear, every drawer single-colored; with up to two colors they're in drawers 1 and 2 and
    drawer 3 is left open and empty; with three colors every drawer ends shut."""
    drawers = world.drawer_colors()
    colors = {c for cs in drawers.values() for c in cs}
    n = len(world.openings)
    if world.table_colors() or not all(len(set(c)) <= 1 for c in drawers.values()):
        return False
    if len(colors) <= n - 1:
        return all(o < 12 for o in world.openings[:-1]) and world.openings[-1] > 60 and not drawers[n - 1]
    return all(o < 12 for o in world.openings)


def test_pick_and_drop_into_an_open_drawer():
    world = FakeWorld(Box(openings=(0.0, 120.0, 0.0)), [FBlock("yellow", 400.0, -280.0, 0.0, 60.0, axis=35.0)])
    rig = FakeRig(world)

    async def go():
        frame = await rig.look()
        sc = observe(frame, rig.cfg)
        b = sc.table_blocks[0]
        held = await pick(rig, plan_grasp(b, rig.cfg, sc.table_blocks))
        assert held.ok, held.to_json()
        k = next(d.index for d in sc.organizer.drawers if d.state == "open")
        target, why = drawer_target(sc.organizer, k, held, sc.contents.get(k, []), rig.cfg)
        assert target is not None, why
        res = await place(rig, held, target)
        assert res.ok, res.to_json()
    asyncio.run(go())
    assert world.drawer_colors()[1] == ["yellow"] and world.table_colors() == []


def test_sort_from_all_shut_and_unknown():
    world = FakeWorld(Box(), [FBlock("yellow", 400.0, -280.0, 0.0, 60.0, axis=35.0),
                              FBlock("green", 330.0, -230.0, 0.0, 35.0, 30.0, 60.0, axis=-20.0),
                              FBlock("yellow", 470.0, -330.0, 0.0, 45.0, 30.0, 36.0, axis=80.0)])
    result, rig, lines = run(world)
    assert result["success"], "\n".join(lines)
    assert sorted_world(world), (world.drawer_colors(), world.openings)


def test_a_drawer_left_open_with_mixed_colors_is_fixed():
    """Drawer 2 left open with a yellow against the box face (can't be gripped) and a green mid-drawer.
    Keeping green (alphabetical tie) would need the yellow out, which can't be done: so it keeps yellow,
    takes the green out, and sorts the rest."""
    world = FakeWorld(Box(openings=(0.0, 120.0, 0.0)), [FBlock("green", 380.0, -250.0, 0.0, 40.0, axis=10.0),
                                                       FBlock("yellow", 470.0, -320.0, 0.0, 50.0, axis=60.0)])
    b = world.box
    for color, along, s in (("yellow", 0.0, 12.0), ("green", 0.0, 62.0)):
        c = np.asarray(b.front) + b.slot_along(1) * b.row + along * b.row + s * b.pull
        world.blocks.append(FBlock(color, float(c[0]), float(c[1]), b.floor, 40.0, 26.0, 30.0, axis=150.0, slot=1))
    result, rig, lines = run(world)
    assert result["success"], "\n".join(lines)
    assert sorted_world(world), (world.drawer_colors(), world.table_colors(), world.openings)
    assert world.drawer_colors()[1] == ["yellow", "yellow"]


def test_partly_open_drawer_start():
    world = FakeWorld(Box(openings=(50.0, 0.0, 0.0)), [FBlock("blue", 400.0, -280.0, 0.0, 50.0, axis=0.0)])
    result, rig, lines = run(world)
    assert result["success"], "\n".join(lines)
    assert sorted_world(world)


def test_nothing_to_do_stops_after_looking_into_every_drawer():
    world = FakeWorld(Box(), [])
    result, rig, lines = run(world)
    assert result["success"], "\n".join(lines)
    assert sum(1 for line in lines if "open drawer" in line) == 3   # looked into each once


def test_dry_run_only_decides():
    world = FakeWorld(Box(), [FBlock("yellow", 400.0, -280.0, 0.0, 60.0)])
    result, rig, lines = run(world, dry_run=True)
    assert result["status"] == "dry run" and world.openings == [0.0, 0.0, 0.0]


def test_a_block_that_cant_be_reached_in_a_drawer_ends_cleanly_unsorted():
    """A green right against the box face can't be gripped (the fingers would go under the box top)."""
    world = FakeWorld(Box(openings=(0.0, 120.0, 0.0)), [])
    b = world.box
    for color, along, s in (("yellow", 0.0, 60.0), ("green", 0.0, 13.0)):
        c = np.asarray(b.front) + b.slot_along(1) * b.row + along * b.row + s * b.pull
        world.blocks.append(FBlock(color, float(c[0]), float(c[1]), b.floor, 40.0, 26.0, 30.0, axis=150.0, slot=1))
    result, rig, lines = run(world)
    assert result["status"] in ("done", "no progress") and not result["success"], "\n".join(lines)
    assert len(lines) < 80                     # no endless open/close loop
