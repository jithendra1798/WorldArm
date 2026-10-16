"""Waj's tape-square finder on synthetic images: one square for sorting, two by default."""

import cv2
import numpy as np
import pytest

import pick_place_cubes as ppc

SQUARE_A = ((300, 120), (700, 520))   # larger square
SQUARE_B = ((820, 160), (1140, 480))  # smaller square


def scene(*squares):
    color = np.full((720, 1280, 3), 235, np.uint8)  # white table
    for top_left, bottom_right in squares:
        cv2.rectangle(color, top_left, bottom_right, (255, 0, 0), 12)  # blue tape (BGR)
    depth = np.full((720, 1280), 400, np.uint16)  # flat table 400 mm away
    return color, depth


def test_one_square_when_sorting_into_drawers(monkeypatch):
    monkeypatch.setenv("VIAM_TAPE_ZONES", "1")
    _, zones = ppc.build_zones(*scene(SQUARE_A))
    assert len(zones) == 1


def test_a_second_square_is_ignored_when_one_is_expected(monkeypatch):
    monkeypatch.setenv("VIAM_TAPE_ZONES", "1")
    _, zones = ppc.build_zones(*scene(SQUARE_A, SQUARE_B))
    assert len(zones) == 1
    assert abs(zones[0].center_px[0] - 500) < 30  # the larger square wins


def test_default_still_needs_two_squares(monkeypatch):
    monkeypatch.delenv("VIAM_TAPE_ZONES", raising=False)
    with pytest.raises(RuntimeError, match="Expected exactly 2"):
        ppc.build_zones(*scene(SQUARE_A))
    _, zones = ppc.build_zones(*scene(SQUARE_A, SQUARE_B))
    assert len(zones) == 2
