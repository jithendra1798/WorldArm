"""Grasp geometry for blocks of different size, shape, height, and with neighbours."""

import numpy as np
import pytest

from worldarm.act.grasp_plan import plan_grasp
from worldarm.blocks import Block
from worldarm.cell.config import CellConfig
from worldarm.cell.geometry import axis_diff, finger_axis_deg

CFG = CellConfig()
G = CFG.gripper


def blk(x=300.0, y=-200.0, short=33.0, long=41.0, h=60.0, axis=30.0, base=2.0, color="yellow"):
    return Block(color, (0, 0), np.zeros((2, 2), np.uint8), np.array([x, y, base + h / 2]), h, short, long, axis, base)


def test_fingers_close_across_the_short_side_and_the_wrist_points_them_there():
    p = plan_grasp(blk(axis=37.0), CFG)
    assert p.feasible and axis_diff(p.finger_axis_deg, 37.0) < 0.5
    assert axis_diff(finger_axis_deg(p.theta), 37.0) < 1.0 and axis_diff(finger_axis_deg(p.alt_theta), 37.0) < 1.0
    assert p.open_mm == pytest.approx(33.0 + 2 * CFG.grasp.approach_clearance_mm)
    assert p.expected_width_mm == pytest.approx(33.0)


@pytest.mark.parametrize("h", [60.0, 35.0, 20.0, 15.0])
def test_grip_height_centres_the_pads_but_keeps_the_fingertips_off_the_table(h):
    b = blk(h=h)
    p = plan_grasp(b, CFG)
    tip = p.frame_z_grip - G.tip_below_frame_mm
    assert tip >= b.table_z_mm + CFG.grasp.tip_clearance_mm - 1e-6        # never into the table
    pad_lo, pad_hi = tip, tip + G.pad_length_mm
    overlap = min(pad_hi, b.table_z_mm + h) - max(pad_lo, b.table_z_mm)
    assert overlap >= min(G.pad_length_mm, h - CFG.grasp.tip_clearance_mm) - 1e-6  # as much pad on it as fits
    if h >= G.pad_length_mm + 2 * CFG.grasp.tip_clearance_mm:
        assert abs((pad_lo + pad_hi) / 2 - (b.table_z_mm + h / 2)) < 1e-6    # tall: pads centred on its middle


def test_block_in_a_drawer_is_gripped_from_the_drawer_floor():
    p = plan_grasp(blk(h=40.0, base=12.0), CFG)
    assert p.base_z == 12.0 and p.frame_z_grip - G.tip_below_frame_mm >= 12.0 + CFG.grasp.tip_clearance_mm - 1e-6
    assert p.bottom_below_frame_mm == pytest.approx(p.frame_z_grip - 12.0)


def test_contact_is_judged_only_near_the_block_top():
    b = blk(h=60.0)
    p = plan_grasp(b, CFG)
    assert p.reach_z == pytest.approx(b.table_z_mm + 60 + G.tip_below_frame_mm + CFG.contact.reach_margin_mm)
    assert p.frame_z_hover > p.reach_z > p.frame_z_grip


def test_a_wide_block_is_gripped_across_its_long_side_if_that_still_fits():
    p = plan_grasp(blk(short=30.0, long=70.0, axis=0.0), CFG)
    assert p.feasible and axis_diff(p.finger_axis_deg, 0.0) < 0.5   # short side still first choice


def test_too_wide_both_ways_is_refused():
    p = plan_grasp(blk(short=80.0, long=90.0), CFG)
    assert not p.feasible and "wider than the gripper" in p.reason


def test_a_neighbour_beside_the_finger_narrows_the_opening():
    b = blk(x=300, y=-200, axis=0.0)                                   # fingers along world x
    nb = blk(x=300 + 62, y=-200, short=30, long=30, h=40, color="green")
    p = plan_grasp(b, CFG, [b, nb])
    assert p.feasible and axis_diff(p.finger_axis_deg, 0.0) < 0.5        # still across the short side
    assert 33.0 + 10.0 <= p.open_mm < 33.0 + 2 * CFG.grasp.approach_clearance_mm


def test_a_neighbour_too_close_on_the_short_axis_turns_the_grip_to_the_long_side():
    b = blk(x=300, y=-200, short=30.0, long=40.0, axis=0.0)
    nb = blk(x=300 + 36, y=-200, short=26, long=26, h=40, color="green")
    p = plan_grasp(b, CFG, [b, nb])
    assert p.feasible and axis_diff(p.finger_axis_deg, 90.0) < 0.5 and p.expected_width_mm == pytest.approx(40.0)


def test_hover_clears_a_taller_neighbour():
    b = blk(h=30.0)
    nb = blk(x=380, y=-200, h=70.0, color="green")
    p = plan_grasp(b, CFG, [b, nb])
    assert p.frame_z_hover - G.tip_below_frame_mm >= nb.table_z_mm + 70.0 + CFG.grasp.hover_above_top_mm - 1e-6


def test_in_a_drawer_the_fingers_must_fit_between_the_walls():
    """Drawer inside: 63 mm across the row (x), 107 mm along the pull (y). A 33 x 41 block lying with
    its short side along x: gripping across x would need ~81 mm of finger span -> grip across y."""
    inside = np.array([[268.5, -253.5], [331.5, -253.5], [331.5, -146.5], [268.5, -146.5]])
    b = blk(x=300, y=-200, short=33.0, long=41.0, axis=0.0, base=12.0, h=40.0)
    p = plan_grasp(b, CFG, [b], inside=inside, clear_z=82.0)
    assert p.feasible and axis_diff(p.finger_axis_deg, 90.0) < 0.5
    assert p.frame_z_hover - G.tip_below_frame_mm >= 82.0 + CFG.grasp.hover_above_top_mm - 1e-6


def test_no_room_between_the_walls_is_refused():
    inside = np.array([[285.0, -230.0], [315.0, -230.0], [315.0, -170.0], [285.0, -170.0]])  # 30 x 60 mm
    p = plan_grasp(blk(x=300, y=-200, short=28.0, long=40.0, axis=0.0, base=12.0), CFG, inside=inside)
    assert not p.feasible and "a wall" in p.reason
