"""Read-only: find which block colors are visible in a color image. Moves nothing."""

import cv2
import numpy as np

COLORS = {
    "yellow": ([20, 100, 100], [35, 255, 255]),
    "green":  ([40, 80, 50],   [85, 255, 255]),
    "blue":   ([95, 120, 50],  [130, 255, 255]),
}
MIN_AREA = 500  # pixels; smaller blobs are noise


def colors_in(img: np.ndarray) -> list[str]:
    """Return the block colors seen in a BGR image, e.g. ["green", "blue"]."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    found = []
    for name, (low, high) in COLORS.items():
        mask = cv2.inRange(hsv, np.array(low), np.array(high))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if any(cv2.contourArea(c) >= MIN_AREA for c in contours):
            found.append(name)
    return found