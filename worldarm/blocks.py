"""Find blocks on the table from one aligned color + depth frame. Read-only.

Anything standing MIN_HEIGHT_MM..MAX_HEIGHT_MM above the fitted table plane, with a block-sized
footprint, is a block; it is named by its color. No tape square or mat is needed. Taller things
(the drawer box, the arm) and anything attached to them are skipped.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MIN_HEIGHT_MM = 8.0           # above the table plane: blocks, not table texture or depth noise
MAX_HEIGHT_MM = 70.0          # taller things are not blocks (drawer box, arm, people)
MIN_SIDE_MM, MAX_SIDE_MM = 22.0, 130.0  # our blocks are >= 28 mm across; curled tape is smaller
MIN_DEPTH_MM, MAX_DEPTH_MM = 200, 1500
DARK_V = 40                   # HSV brightness below this is a dark (black) block ...
DARK_HEIGHT_MM = 30.0         # ... which often returns no depth (black absorbs the IR): assume this height
BORDER_PX = 4                 # anything touching the image edge is cut off (or is the gripper's fingertips)


@dataclass
class Block:
    color: str
    pixel: tuple[int, int]        # image centroid
    mask: np.ndarray              # uint8 full-image mask (255 = block)
    center: np.ndarray            # world x, y, z (mm); z is mid-height
    height_mm: float
    short_mm: float
    long_mm: float
    short_axis_deg: float         # world angle (deg, from +x toward +y) of the block's short side
    table_z_mm: float             # table height under the block


def name_color(hsv: np.ndarray) -> str:
    """Name a block from its pixels' median HSV (OpenCV ranges: H 0-179, S and V 0-255)."""
    h, s, v = (float(x) for x in np.median(hsv.reshape(-1, 3), axis=0))
    if v < 35:
        return "black"   # brightness alone: dark pixels have noisy hue and saturation
    if s < 60:
        return "white"
    if h <= 4 or h >= 170:
        return "red"
    if h < 15:
        return "orange"
    if h < 42:
        return "yellow"
    if h < 95:
        return "green"
    if h < 135:
        return "blue"
    return "purple"


def fit_table(points: np.ndarray) -> np.ndarray:
    """Plane z = a*x + b*y + c through the dominant flat surface (the table), robustly."""
    z = points[:, 2]
    if z.size < 500:
        raise RuntimeError("too little depth to find the table")
    hist, edges = np.histogram(z, bins=np.arange(z.min() - 2.5, z.max() + 7.5, 5.0))  # always >= 2 bins
    peak = edges[np.argmax(hist)] + 2.5
    keep = np.abs(z - peak) < 15.0
    coef = np.array([0.0, 0.0, peak])
    for _ in range(3):
        if keep.sum() < 500:
            raise RuntimeError("too little flat surface in view to find the table")
        a = np.c_[points[keep, 0], points[keep, 1], np.ones(keep.sum())]
        coef, *_ = np.linalg.lstsq(a, z[keep], rcond=None)
        resid = z - (coef[0] * points[:, 0] + coef[1] * points[:, 1] + coef[2])
        keep = np.abs(resid) < 6.0
    return coef


def _short_side(xy: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """(short_mm, long_mm, short-side angle in deg, center xy) of points' minimum-area rectangle."""
    rect = cv2.minAreaRect(xy.astype(np.float32))
    box = cv2.boxPoints(rect)
    e1, e2 = box[1] - box[0], box[2] - box[1]
    n1, n2 = float(np.hypot(*e1)), float(np.hypot(*e2))
    short_vec, short_len, long_len = (e1, n1, n2) if n1 <= n2 else (e2, n2, n1)
    return short_len, long_len, float(np.degrees(np.arctan2(short_vec[1], short_vec[0]))), np.array(rect[0])


def find_blocks(color: np.ndarray, depth: np.ndarray, intr, rot: np.ndarray, trans: np.ndarray):
    """Blocks in one aligned frame. `rot`, `trans` map camera mm to world mm (frames.frame_to_world).

    Returns (blocks, table_coef) with table_coef = (a, b, c) of the world table plane.
    """
    hgt, wid = depth.shape
    valid = (depth > MIN_DEPTH_MM) & (depth < MAX_DEPTH_MM)
    v, u = np.nonzero(valid)
    z = depth[v, u].astype(np.float32)
    cam = np.stack(((u - intr.center_x_px) * z / intr.focal_x_px,
                    (v - intr.center_y_px) * z / intr.focal_y_px, z), axis=1)
    world = cam @ rot.T + trans
    coef = fit_table(world[:: max(1, len(world) // 60000)])
    above = world[:, 2] - (coef[0] * world[:, 0] + coef[1] * world[:, 1] + coef[2])

    height = np.full((hgt, wid), np.nan, np.float32)
    height[v, u] = above
    index = np.full((hgt, wid), -1, np.int64)
    index[v, u] = np.arange(len(world))

    blockish = ((height > MIN_HEIGHT_MM) & (height < MAX_HEIGHT_MM)).astype(np.uint8)
    blockish = cv2.morphologyEx(blockish, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    blockish = cv2.morphologyEx(blockish, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    tall = cv2.dilate((height >= MAX_HEIGHT_MM).astype(np.uint8), np.ones((15, 15), np.uint8))
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

    blocks = []
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(blockish)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] < 150 or _touches_border(stats[i], wid, hgt):
            continue
        comp = labels == i
        if np.any(comp & (tall > 0)):
            continue  # part of something tall (the drawer box, its knobs)
        idx = index[comp & valid]
        if idx.size < 100:
            continue
        h_mm = float(np.percentile(above[idx], 75))  # robust to edge pixels that land between top and table
        top = idx[above[idx] >= 0.5 * h_mm]          # the footprint from the top face, not the edge skirt
        pts = world[top if top.size >= 30 else idx]
        short_mm, long_mm, short_deg, center_xy = _short_side(pts[:, :2])
        if not (MIN_SIDE_MM <= short_mm and long_mm <= MAX_SIDE_MM):
            continue
        inner = cv2.erode(comp.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        name = name_color(hsv[inner] if inner.sum() > 30 else hsv[comp])
        if name == "white":
            continue  # table texture, tape, cables seen against the table: not a block color
        table_z = float(coef[0] * center_xy[0] + coef[1] * center_xy[1] + coef[2])
        blocks.append(Block(
            color=name, pixel=(int(centroids[i][0]), int(centroids[i][1])),
            mask=(comp.astype(np.uint8) * 255), center=np.array([center_xy[0], center_xy[1], table_z + h_mm / 2.0]),
            height_mm=h_mm, short_mm=short_mm, long_mm=long_mm, short_axis_deg=short_deg, table_z_mm=table_z))
    found = np.zeros_like(blockish)
    for b in blocks:
        found |= (b.mask > 0).astype(np.uint8)
    near_found = cv2.dilate(found, np.ones((31, 31), np.uint8))  # a dark patch beside a block is that block
    blocks += _dark_blocks(hsv, height, near_found, tall, intr, rot, trans, coef)
    return blocks, coef


def _touches_border(stat, wid: int, hgt: int) -> bool:
    x, y, w, h = stat[0], stat[1], stat[2], stat[3]
    return x < BORDER_PX or y < BORDER_PX or x + w > wid - BORDER_PX or y + h > hgt - BORDER_PX


def _dark_blocks(hsv, height, found, tall, intr, rot, trans, coef) -> list[Block]:
    """Black blocks found by color where depth found nothing, placed on the table plane.

    Height comes from whatever depth they do return, else DARK_HEIGHT_MM. The footprint is the dark
    pixels' rays meeting the plane at that height."""
    hgt, wid = height.shape
    dark = ((hsv[:, :, 2] < DARK_V) & (found == 0) & (tall == 0)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    out = []
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(dark)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] < 400 or _touches_border(stats[i], wid, hgt):
            continue
        comp = labels == i
        seen = height[comp]
        seen = seen[np.isfinite(seen) & (seen > MIN_HEIGHT_MM) & (seen < MAX_HEIGHT_MM)]
        h_mm = float(np.percentile(seen, 90)) if seen.size >= 50 else DARK_HEIGHT_MM
        v, u = np.nonzero(comp)
        rays = np.stack(((u - intr.center_x_px) / intr.focal_x_px, (v - intr.center_y_px) / intr.focal_y_px,
                         np.ones(u.size)), axis=1) @ rot.T
        a, b, c = coef
        t = (a * trans[0] + b * trans[1] + c + h_mm - trans[2]) / (rays[:, 2] - a * rays[:, 0] - b * rays[:, 1])
        xy = trans[:2] + t[:, None] * rays[:, :2]
        short_mm, long_mm, short_deg, center_xy = _short_side(xy)
        if not (MIN_SIDE_MM <= short_mm and long_mm <= MAX_SIDE_MM):
            continue  # cables, gaps and shadows are too thin or too long
        table_z = float(a * center_xy[0] + b * center_xy[1] + c)
        out.append(Block(color="black", pixel=(int(centroids[i][0]), int(centroids[i][1])),
                         mask=comp.astype(np.uint8) * 255,
                         center=np.array([center_xy[0], center_xy[1], table_z + h_mm / 2.0]), height_mm=h_mm,
                         short_mm=short_mm, long_mm=long_mm, short_axis_deg=short_deg, table_z_mm=table_z))
    return out


def color_fraction(color: np.ndarray, center_px: tuple[int, int], half_px: int, name: str) -> float:
    """Share of pixels in a square window that name_color would call `name` (per-pixel version)."""
    x, y = center_px
    h, w = color.shape[:2]
    win = color[max(0, y - half_px):min(h, y + half_px), max(0, x - half_px):min(w, x + half_px)]
    if win.size == 0:
        return 0.0
    hsv = cv2.cvtColor(win, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    hh, ss, vv = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    tests = {
        "black": vv < 35,
        "red": (vv >= 35) & (ss >= 60) & ((hh <= 4) | (hh >= 170)),
        "orange": (vv >= 35) & (ss >= 60) & (hh > 4) & (hh < 15),
        "yellow": (vv >= 35) & (ss >= 60) & (hh >= 15) & (hh < 42),
        "green": (vv >= 35) & (ss >= 60) & (hh >= 42) & (hh < 95),
        "blue": (vv >= 35) & (ss >= 60) & (hh >= 95) & (hh < 135),
    }
    return float(np.mean(tests[name])) if name in tests else 0.0
