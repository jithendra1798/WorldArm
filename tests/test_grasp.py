"""Block orientation -> gripper wrist angle (no arm needed)."""

import asyncio
import math

import cv2
import numpy as np
import pytest

from worldarm.grasp import (block_grasp_theta, finger_axis_deg, place_theta_for_drawer, short_axis_px,
                            theta_for_finger_axis)


def axis_err(a: float, b: float) -> float:
    d = (a - b) % 180.0
    return min(d, 180.0 - d)


def test_finger_axis_from_viam_math():
    assert axis_err(finger_axis_deg(0), 90) < 0.5   # theta 0: fingers along world +y
    assert axis_err(finger_axis_deg(90), 0) < 0.5   # theta 90: fingers along world +x


@pytest.mark.parametrize("axis", [0, 25, 60, 90, 135, 170])
def test_theta_puts_the_fingers_on_the_requested_axis_near_the_base(axis):
    theta = theta_for_finger_axis(axis, base_theta_deg=0)
    assert axis_err(finger_axis_deg(theta), axis) < 0.6
    assert -90 <= theta <= 90


def rotated_block(angle_deg: float, w: int = 120, h: int = 40) -> np.ndarray:
    mask = np.zeros((720, 1280), np.uint8)
    box = cv2.boxPoints(((640, 360), (w, h), angle_deg)).astype(np.int32)
    cv2.fillConvexPoly(mask, box, 255)
    return mask


@pytest.mark.parametrize("angle", [0, 20, 45, 70, 110])
def test_short_axis_of_a_rotated_bar(angle):
    _, (du, dv), short_px, long_px, fill = short_axis_px(rotated_block(angle))
    # the bar's long side is along `angle`, so its short side is along angle + 90
    assert axis_err(math.degrees(math.atan2(dv, du)), angle + 90) < 2.0
    assert 35 < short_px < 45 and 115 < long_px < 125 and fill > 0.9


async def pixel_is_world(u, v, z):
    return float(u), float(v)  # identity mapping: image axes = world axes


@pytest.mark.parametrize("angle", [0, 30, 75, 120])
def test_fingers_close_across_the_short_side(angle):
    depth = np.full((720, 1280), 400, np.uint16)
    theta, info = asyncio.run(block_grasp_theta(rotated_block(angle), depth, pixel_is_world, base_theta_deg=0))
    assert info["shape"] == "box"
    assert axis_err(finger_axis_deg(theta), angle + 90) < 2.0  # finger axis = short side


def test_round_block_is_labelled_round_and_gripped_near_the_default_wrist():
    mask = np.zeros((720, 1280), np.uint8)
    cv2.circle(mask, (640, 360), 40, 255, -1)
    theta, info = asyncio.run(block_grasp_theta(mask, np.full((720, 1280), 400, np.uint16), pixel_is_world, 15.0))
    assert info["shape"] == "round" and abs(theta - 15.0) <= 90  # any angle grips a disc


@pytest.mark.parametrize("pull", [(1, 0), (0, 1), (0.6, -0.8)])
def test_place_puts_the_long_side_along_the_pull(pull):
    theta = place_theta_for_drawer(pull, pick_theta_deg=10)
    pull_deg = math.degrees(math.atan2(pull[1], pull[0]))
    assert axis_err(finger_axis_deg(theta), pull_deg + 90) < 0.6  # fingers across the pull
    assert abs(theta - 10) <= 90


def test_a_box_with_depth_holes_is_still_a_box():
    """Black rubber returns patchy depth; the outline is full of holes but still a rotated box."""
    mask = rotated_block(30, w=90, h=70)
    rng = np.random.default_rng(0)
    mask[(rng.random(mask.shape) < 0.35) & (mask > 0)] = 0  # knock out 35% of the pixels
    theta, info = asyncio.run(block_grasp_theta(mask, np.full((720, 1280), 400, np.uint16), pixel_is_world, 0.0))
    assert info["shape"] == "box"
    assert axis_err(finger_axis_deg(theta), 30 + 90) < 3.0
