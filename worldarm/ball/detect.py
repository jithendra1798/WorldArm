"""The black ball in an overhead frame (Riyaj's detector: dark, round, the right size).

Only the table is searched (a polygon in pixels, from the table map), so people, clothes and cables off
the table can't be mistaken for the ball; round-enough dark blobs of the ball's size are kept, and the
best one's centre is refined from its moments.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DARK_V = 60            # HSV brightness: the ball is darker than this (Riyaj: 60)
MIN_ROUNDNESS = 0.6    # area / enclosing-circle area (cables, shadows score lower)
MIN_CORNERS = 7        # simplified outline (2% of the perimeter): a ball has 8, even blurred 30 px; a block 4-6
MIN_CIRCULARITY = 0.75  # 4 pi area / perimeter^2: disc ~0.86, square 0.785, a block 21x27 0.67


@dataclass
class Sighting:
    u: float
    v: float
    radius_px: float
    roundness: float


def find_ball(frame: np.ndarray, radius_px: tuple[float, float] = (8.0, 30.0),
              table_px: np.ndarray | None = None, avoid_px: list[tuple[float, float, float]] = (),
              expected_radius_px: float | None = None) -> Sighting | None:
    """The most ball-like dark round blob, or None (see find_balls)."""
    found = find_balls(frame, radius_px, table_px, avoid_px, expected_radius_px)
    return found[0] if found else None


def find_balls(frame: np.ndarray, radius_px: tuple[float, float] = (8.0, 30.0),
               table_px: np.ndarray | None = None, avoid_px: list[tuple[float, float, float]] = (),
               expected_radius_px: float | None = None) -> list[Sighting]:
    """Every ball-like dark blob, roundest first. table_px: polygon to search inside; avoid_px: (u, v, r)
    circles to ignore (e.g. where the paddle and gripper are); expected_radius_px: the ball's measured size,
    which narrows the radius gate to +-25%. A black block passes "dark and roundish" but not the corner
    count: its outline simplifies to 4-6 corners, a ball's to 8."""
    if expected_radius_px is not None:
        radius_px = (0.75 * expected_radius_px, 1.25 * expected_radius_px)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 2] <= DARK_V).astype(np.uint8)
    if table_px is not None:
        keep = np.zeros_like(mask)
        cv2.fillPoly(keep, [np.asarray(table_px, np.int32)], 1)
        mask &= keep
    for u, v, r in avoid_px:
        cv2.circle(mask, (int(u), int(v)), int(r), 0, -1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    found = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < np.pi * radius_px[0] ** 2 * 0.5:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        if not (radius_px[0] <= r <= radius_px[1]):
            continue
        roundness = area / (np.pi * r * r)
        perim = cv2.arcLength(c, True)
        if roundness < MIN_ROUNDNESS or perim <= 0 or 4 * np.pi * area / (perim * perim) < MIN_CIRCULARITY:
            continue
        if len(cv2.approxPolyDP(c, 0.02 * perim, True)) < MIN_CORNERS:
            continue                                   # straight sides: a block, not the ball
        m = cv2.moments(c)
        if m["m00"] > 0:
            x, y = m["m10"] / m["m00"], m["m01"] / m["m00"]
        found.append(Sighting(float(x), float(y), float(r), float(roundness)))
    return sorted(found, key=lambda s: -s.roundness)
