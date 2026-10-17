"""Block finder on a synthetic top-down frame: blocks found by height, tall things skipped."""

import math
from types import SimpleNamespace

import cv2
import numpy as np

from worldarm.blocks import color_fraction, find_blocks, knob_row

INTR = SimpleNamespace(focal_x_px=900.0, focal_y_px=900.0, center_x_px=640.0, center_y_px=360.0)
ROT = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])  # camera looking straight down
TRANS = np.array([0.0, 0.0, 400.0])                        # 400 mm above the table (world z = 0)


def project(x, y, z):
    cam = ROT.T @ (np.array([x, y, z]) - TRANS)
    return [INTR.focal_x_px * cam[0] / cam[2] + INTR.center_x_px, INTR.focal_y_px * cam[1] / cam[2] + INTR.center_y_px]


def scene(objects):
    color = np.full((720, 1280, 3), 225, np.uint8)
    depth = np.full((720, 1280), 400, np.uint16)
    for (cx, cy), (width, length), long_deg, height, bgr in objects:
        a = math.radians(long_deg)
        along, across = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
        corners = [np.array([cx, cy]) + sl * length / 2 * along + sw * width / 2 * across
                   for sl, sw in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
        poly = np.array([project(p[0], p[1], height) for p in corners], np.int32)
        cv2.fillPoly(depth, [poly], int(400 - height))
        cv2.fillPoly(color, [poly], bgr)
    return color, depth


GREEN_BAR = ((60, 30), (35, 70), 30, 25, (40, 160, 40))       # long side along 30 deg
BLACK_CUBE = ((-50, 80), (35, 35), 10, 30, (15, 15, 15))
DRAWER_BOX = ((-120, -80), (100, 160), 0, 90, (200, 220, 230))  # too tall to be a block


def axis_err(a, b):
    d = (a - b) % 180.0
    return min(d, 180.0 - d)


def test_finds_blocks_by_height_and_skips_the_tall_box():
    blocks, table = find_blocks(*scene([GREEN_BAR, BLACK_CUBE, DRAWER_BOX]), INTR, ROT, TRANS)
    assert sorted(b.color for b in blocks) == ["black", "green"]
    assert abs(table[2]) < 1.0 and abs(table[0]) < 1e-3 and abs(table[1]) < 1e-3


def test_green_bar_size_position_height_and_short_side():
    blocks, _ = find_blocks(*scene([GREEN_BAR]), INTR, ROT, TRANS)
    (b,) = blocks
    assert abs(b.short_mm - 35) < 4 and abs(b.long_mm - 70) < 5
    assert abs(b.center[0] - 60) < 3 and abs(b.center[1] - 30) < 3
    assert abs(b.height_mm - 25) < 3 and abs(b.center[2] - 12.5) < 3
    assert axis_err(b.short_axis_deg, 30 + 90) < 3  # short side is across the long one


def test_no_blocks_on_an_empty_table():
    blocks, _ = find_blocks(*scene([]), INTR, ROT, TRANS)
    assert blocks == []


def test_color_fraction_in_a_window():
    color, _ = scene([GREEN_BAR])
    u, v = (int(round(c)) for c in project(60, 30, 25))
    assert color_fraction(color, (u, v), 20, "green") > 0.9
    assert color_fraction(color, (u, v), 20, "yellow") < 0.05


def test_black_block_without_depth_is_found_by_color_and_fingertips_are_not():
    color, depth = scene([GREEN_BAR, ((-50, 80), (35, 50), 20, 30, (15, 15, 15))])
    poly = np.array([project(p[0], p[1], 30) for p in
                     [(-50 + dx, 80 + dy) for dx, dy in ((30, 30), (30, -30), (-30, -30), (-30, 30))]], np.int32)
    cv2.fillPoly(depth, [poly], 0)  # black absorbs the IR: no depth at all over the black block
    cv2.rectangle(color, (600, 690), (680, 719), (20, 20, 20), -1)  # dark fingertips at the frame edge
    depth[690:, 600:681] = 0
    blocks, _ = find_blocks(color, depth, INTR, ROT, TRANS)
    black = [b for b in blocks if b.color == "black"]
    assert len(black) == 1 and sorted(b.color for b in blocks) == ["black", "green"]
    (b,) = black
    assert abs(b.center[0] + 50) < 5 and abs(b.center[1] - 80) < 5
    assert abs(b.short_mm - 35) < 6 and abs(b.long_mm - 50) < 7
    assert axis_err(b.short_axis_deg, 20 + 90) < 5


CREAM = (46, 78, 100)        # the drawer box's plastic in shadow, as measured: yellow's hue, saturation 138
YELLOW = (30, 200, 230)
KNOB = (30, 30, 200)


def test_cream_plastic_and_a_low_drawer_floor_are_not_blocks():
    """Run 20261016-204103 picked an empty open drawer: its floor read as a 10 mm 'yellow' block."""
    cream = ((60, 30), (60, 60), 0, 25, CREAM)            # block-sized and block-tall, but not paint
    floor = ((-60, -40), (50, 50), 0, 10, YELLOW)          # yellow, but only 10 mm up
    blocks, _ = find_blocks(*scene([cream, floor]), INTR, ROT, TRANS)
    assert blocks == []


def test_nothing_near_a_drawer_knob_is_a_block():
    knob = ((0, 0), (25, 25), 0, 40, KNOB)
    inside = ((90, 0), (35, 45), 0, 25, YELLOW)            # 9 cm from the knob: in the open drawer
    table = ((-200, 60), (35, 45), 0, 25, YELLOW)          # 20 cm away: on the table
    report = {}
    blocks, _ = find_blocks(*scene([knob, inside, table]), INTR, ROT, TRANS, report=report)
    assert [b.color for b in blocks] == ["yellow"] and abs(blocks[0].center[0] + 200) < 5
    (k,) = report["knobs"]
    assert abs(k[0]) < 5 and abs(k[1]) < 5


def test_a_block_cut_off_by_the_image_edge_is_reported_not_picked():
    cut = ((270, 0), (35, 70), 0, 25, (40, 160, 40))       # green bar running off the right edge
    report = {}
    blocks, _ = find_blocks(*scene([GREEN_BAR, cut]), INTR, ROT, TRANS, report=report)
    assert [b.color for b in blocks] == ["green"] and abs(blocks[0].center[0] - 60) < 3
    (e,) = report["edge_cut"]
    assert e["color"] == "green" and 230 < e["xy"][0] < 280 and abs(e["xy"][1]) < 10


def test_the_drawer_check_does_not_count_cream_plastic_as_yellow():
    color, _ = scene([((0, 0), (80, 80), 0, 25, CREAM)])
    u, v = (int(round(c)) for c in project(0, 0, 25))
    assert color_fraction(color, (u, v), 30, "yellow") < 0.05


def test_knob_row_tells_shut_from_open():
    shut = [[0, 0], [110, 0], [220, 0]]
    assert knob_row(shut)[0] is True
    assert knob_row([[0, 0], [110, 3], [220, -2]])[0] is True     # a few mm of measuring noise
    assert knob_row([[0, 0], [110, 120], [220, 0]])[0] is False   # middle drawer pulled 12 cm
    assert knob_row([[0, 0], [110, 0], [220, 120]])[0] is False   # end drawer pulled 12 cm
    assert knob_row(shut[:2])[0] is None


def test_a_cable_loop_is_neither_a_block_nor_a_block_at_the_edge():
    """A black cable lying in a U spans a block-sized rectangle but fills little of it."""
    color, depth = scene([GREEN_BAR])
    in_view = [(-200, -50), (-200, -120), (-100, -120), (-100, -50)]
    at_edge = [(0, -190), (0, -110), (100, -110), (100, -190)]  # the image's bottom edge is at y = -152 here
    for loop in (in_view, at_edge):
        line = np.array([project(x, y, 20) for x, y in loop], np.int32)
        cv2.polylines(color, [line], False, (15, 15, 15), 19)
        cv2.polylines(depth, [line], False, 380, 19)  # 20 mm up where it loops over itself
    report = {}
    blocks, _ = find_blocks(color, depth, INTR, ROT, TRANS, report=report)
    assert [b.color for b in blocks] == ["green"] and report["edge_cut"] == []
