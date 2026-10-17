"""Organizer from a synthetic frame: drawer states, numbering, insides, contents, and table blocks."""

import numpy as np
import pytest

from tests.synth import Box, scene
from worldarm.blocks import measure_surface
from worldarm.cell.config import CellConfig
from worldarm.see.organizer import find_organizer
from worldarm.see.scene import free_spot, observe

CFG = CellConfig()


def surf_of(s):
    f = s.frame()
    return measure_surface(f.color, f.depth, f.intr, f.rot, f.trans)


def by_slot(org, box):
    """Map each physical slot (0..2) to the drawer the organizer reports there."""
    out = {}
    for i in range(3):
        k = box.knob(i)
        out[i] = min(org.drawers, key=lambda d: np.linalg.norm(d.knob[:2] - k[:2]))
    return out


def test_all_shut():
    box = Box()
    org, why = find_organizer(surf_of(scene(box)), CFG.organizer)
    assert org is not None, why
    assert org.how == "all shut"
    assert all(d.state == "closed" and d.opening_mm < 5 for d in org.drawers)
    assert float(org.pull @ box.pull) > 0.99           # drawers come out away from the box
    assert [d.index for d in sorted(org.drawers, key=lambda d: d.along_mm)] == [1, 2, 3]


def test_knob_centres_are_found_in_3d():
    box = Box()
    org, _ = find_organizer(surf_of(scene(box)), CFG.organizer)
    for i, d in by_slot(org, box).items():
        assert np.linalg.norm(d.knob - box.knob(i)) < 4.0


def test_numbering_matches_the_drawer_service():
    """Drawer 1 is leftmost for someone facing the fronts (looking against `pull`)."""
    box = Box()
    org, _ = find_organizer(surf_of(scene(box)), CFG.organizer)
    left = np.array([box.pull[1], -box.pull[0]])
    expected = sorted(range(3), key=lambda i: -(box.knob(i)[:2] @ left))
    slots = by_slot(org, box)
    assert [slots[i].index for i in expected] == [1, 2, 3]


@pytest.mark.parametrize("slot", [0, 1, 2])
def test_one_drawer_out(slot):
    openings = [0.0, 0.0, 0.0]
    openings[slot] = 120.0
    box = Box(openings=tuple(openings))
    org, why = find_organizer(surf_of(scene(box)), CFG.organizer)
    assert org is not None, why
    slots = by_slot(org, box)
    assert slots[slot].state == "open" and abs(slots[slot].opening_mm - 120) < 6
    assert all(slots[i].state == "closed" for i in range(3) if i != slot)
    d = slots[slot]
    assert d.inside is not None and abs(d.floor_z - box.floor) < 4 and abs(d.rim_z - box.rim) < 8
    assert float(org.pull @ box.pull) > 0.99


def test_partly_open_drawer():
    box = Box(openings=(0.0, 0.0, 55.0))
    org, why = find_organizer(surf_of(scene(box)), CFG.organizer)
    assert org is not None, why
    d = by_slot(org, box)[2]
    assert d.state == "partial" and abs(d.opening_mm - 55) < 6


def test_two_out_needs_the_remembered_row():
    shut = Box()
    org0, _ = find_organizer(surf_of(scene(shut)), CFG.organizer)
    two = Box(openings=(120.0, 0.0, 60.0))
    alone, _ = find_organizer(surf_of(scene(two)), CFG.organizer)
    assert alone is None or alone.how != "all shut"
    org, why = find_organizer(surf_of(scene(two)), CFG.organizer, prior=org0.model())
    assert org is not None, why
    slots = by_slot(org, two)
    assert abs(slots[0].opening_mm - 120) < 6 and abs(slots[2].opening_mm - 60) < 6 and slots[1].state == "closed"


def test_contents_one_color_and_mixed():
    box = Box(openings=(0.0, 120.0, 0.0),
              contents={1: [("yellow", (-12.0, 25.0), (30.0, 34.0), 45.0), ("green", (14.0, 70.0), (28.0, 30.0), 40.0)]})
    sc = observe(scene(box).frame(), CFG)
    assert sc.organizer is not None
    k = by_slot(sc.organizer, box)[1].index
    colors = sorted(b.color for b in sc.contents[k])
    assert colors == ["green", "yellow"]
    assert all(abs(b.table_z_mm - box.floor) < 4 for b in sc.contents[k])  # standing on the drawer floor
    assert sc.table_blocks == []                                          # not mistaken for table blocks


def test_empty_open_drawer_is_empty_and_its_floor_is_not_a_block():
    box = Box(openings=(120.0, 0.0, 0.0))
    sc = observe(scene(box).frame(), CFG)
    k = by_slot(sc.organizer, box)[0].index
    assert sc.contents[k] == [] and sc.table_blocks == []


def test_table_blocks_and_organizer_together():
    box = Box()
    s = scene(box, blocks=[("yellow", (440.0, -300.0)), ("green", (330.0, -250.0), (30.0, 60.0), 70.0, 35.0)])
    sc = observe(s.frame(), CFG)
    assert sorted(b.color for b in sc.table_blocks) == ["green", "yellow"]
    assert sc.organizer is not None and sc.organizer.how == "all shut"


def test_free_spot_is_clear_of_blocks_and_the_organizer():
    box = Box()
    s = scene(box, blocks=[("yellow", (440.0, -300.0)), ("green", (330.0, -250.0))])
    f = s.frame()
    sc = observe(f, CFG)
    spot = free_spot(sc, f, CFG, near_xy=(300.0, -250.0))
    assert spot is not None
    for b in sc.table_blocks:
        assert np.linalg.norm(spot - b.center[:2]) > 85
    for d in sc.organizer.drawers:
        assert np.linalg.norm(spot - d.knob[:2]) > CFG.organizer.clear_mm


def test_a_knob_cut_off_by_the_image_edge_is_flagged():
    box = Box()
    s = scene(box, cam_xy=(300.0, -300.0))   # this framing cuts one knob at the image edge
    org, why = find_organizer(surf_of(s), CFG.organizer)
    msgs = (org.warnings if org else []) + why
    assert any("cut off" in m for m in msgs)
