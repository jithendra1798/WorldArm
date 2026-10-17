"""What's inside an open drawer: painted blocks standing on its floor, with the same geometry as table
blocks (so one can be picked back out). Only the part of the drawer that's out of the box is visible."""

from __future__ import annotations

import cv2
import numpy as np

from worldarm.blocks import DARK_V, PAINT_S, Block, Surface, _short_side, name_color
from worldarm.cell.geometry import point_in_polygon

MIN_ABOVE_FLOOR_MM = 12.0   # taller than this above the drawer floor: something standing in it
MIN_PIXELS = 120


def drawer_contents(surf: Surface, inside: np.ndarray, floor_z: float, shrink_mm: float = 3.0) -> list[Block]:
    """Blocks inside the drawer region `inside` ((4, 2) world corners) whose floor is at world z floor_z."""
    poly = _shrink(inside, shrink_mm)
    in_poly = point_in_polygon(surf.world[:, :2], poly)
    if not in_poly.any():
        return []
    table = surf.coef[0] * surf.world[:, 0] + surf.coef[1] * surf.world[:, 1] + surf.coef[2]
    floor_h = floor_z - table                    # floor height above the table plane, per point
    h = surf.above - floor_h                     # height above the drawer floor
    pick = in_poly & (h > MIN_ABOVE_FLOOR_MM) & (h < 110.0)
    mask = np.zeros(surf.valid.shape, np.uint8)
    v, u = np.nonzero(surf.valid)
    mask[v[pick], u[pick]] = 1
    hsv = surf.hsv
    paint = (hsv[:, :, 1] >= PAINT_S) | (hsv[:, :, 2] < DARK_V)   # block paint or a black block
    mask &= paint.astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    out: list[Block] = []
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] < MIN_PIXELS:
            continue
        comp = labels == i
        name = name_color(hsv[comp])
        if name in ("white", "red"):
            continue
        idx = surf.index[comp & surf.valid]
        if idx.size < 40:
            continue
        hh = h[idx]
        h_mm = float(np.percentile(hh, 75))
        top = idx[hh >= 0.5 * h_mm]
        pts = surf.world[top if top.size >= 20 else idx]
        short_mm, long_mm, short_deg, center_xy = _short_side(pts[:, :2])
        base = float(floor_z)
        out.append(Block(color=name, pixel=(int(centroids[i][0]), int(centroids[i][1])), mask=comp.astype(np.uint8) * 255,
                         center=np.array([center_xy[0], center_xy[1], base + h_mm / 2.0]), height_mm=h_mm,
                         short_mm=short_mm, long_mm=long_mm, short_axis_deg=short_deg, table_z_mm=base))
    return out


def _shrink(poly: np.ndarray, mm: float) -> np.ndarray:
    c = poly.mean(axis=0)
    out = []
    for p in poly:
        d = p - c
        n = float(np.linalg.norm(d)) or 1.0
        out.append(p - d / n * mm)
    return np.array(out)
