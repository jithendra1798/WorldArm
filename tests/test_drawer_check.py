import numpy as np
import cv2
from worldarm.drawer_check import colors_in


def blank():
    return np.full((200, 200, 3), 255, dtype=np.uint8)   # plain white image


def test_empty_image_finds_nothing():
    assert colors_in(blank()) == []


def test_finds_green_square():
    img = blank()
    cv2.rectangle(img, (50, 50), (120, 120), (0, 160, 0), -1)   # solid green square (BGR)
    assert colors_in(img) == ["green"]


def test_ignores_red_knob():
    img = blank()
    cv2.circle(img, (100, 100), 40, (0, 0, 200), -1)          # solid red circle, like a knob
    assert colors_in(img) == []