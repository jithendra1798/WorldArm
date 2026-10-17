"""Block finder on a synthetic top-down frame: blocks found by height, tall things skipped."""

import math
from types import SimpleNamespace

import cv2
import numpy as np

from worldarm.blocks import color_fraction, find_blocks

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
