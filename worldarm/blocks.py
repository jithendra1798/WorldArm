"""Find blocks on the table from one aligned color + depth frame. Read-only.

Anything standing MIN_HEIGHT_MM..MAX_HEIGHT_MM above the fitted table plane, with a block-sized
footprint and a painted color, is a block. No tape square or mat is needed. Skipped: taller things (the
drawer box, the arm) and anything attached to them, unpainted things (table, tape, the box's cream
plastic), and anything near the box's red knobs (an open drawer's floor). The knobs are reported too,
so a caller can tell whether every drawer is shut (knob_row).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MIN_HEIGHT_MM = 15.0          # above the table plane: blocks, not depth noise or an open drawer's floor (~10 mm)
MAX_HEIGHT_MM = 70.0          # taller things are not blocks (drawer box, arm, people)
MIN_SIDE_MM, MAX_SIDE_MM = 22.0, 130.0  # our blocks are >= 28 mm across; curled tape is smaller
MIN_DEPTH_MM, MAX_DEPTH_MM = 200, 1500
DARK_V = 40                   # HSV brightness below this is a dark (black) block ...
DARK_HEIGHT_MM = 30.0         # ... which often returns no depth (black absorbs the IR): assume this height
BORDER_PX = 4                 # anything touching the image edge is cut off (or is the gripper's fingertips)
KNOB_CLEAR_MM = 160.0         # nothing within this of a drawer knob is a block (the box is behind the knobs)
PAINT_S = 165                 # HSV saturation of block paint. Measured: blocks 205-238; the box's cream plastic
                              # up to 138 in shadow (yellow's hue), table 4
MIN_FILL = 0.5                # a block's outline fills its bounding rectangle; a cable loop's doesn't (~0.3)
ROW_TOLERANCE_MM = 12.0       # shut drawers' knobs lie on one line within this; an open drawer's is ~120 mm out


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
    if s < PAINT_S:
        return "white"   # not paint: table, tape, the drawer box's cream plastic
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


@dataclass
class Surface:
    """One aligned frame measured against the table: world points and height above the table."""
    color: np.ndarray
    hsv: np.ndarray
    depth: np.ndarray
    intr: object
    rot: np.ndarray
    trans: np.ndarray
    valid: np.ndarray          # (H, W) pixels with usable depth
    world: np.ndarray          # (N, 3) world points of the valid pixels
    index: np.ndarray          # (H, W) row of each pixel in `world`, -1 where invalid
    above: np.ndarray          # (N,) height of each world point above the table plane
    height: np.ndarray         # (H, W) the same as an image, NaN where invalid
    coef: np.ndarray           # table plane z = a*x + b*y + c

    def table_z(self, x: float, y: float) -> float:
        return float(self.coef[0] * x + self.coef[1] * y + self.coef[2])

    def to_pixel(self, xyz) -> tuple[float, float]:
        cam = self.rot.T @ (np.asarray(xyz, float) - self.trans)
        return (float(self.intr.focal_x_px * cam[0] / cam[2] + self.intr.center_x_px),
                float(self.intr.focal_y_px * cam[1] / cam[2] + self.intr.center_y_px))

    def mask_xy(self, inside) -> np.ndarray:
        """(H, W) mask of valid pixels whose world x, y satisfy `inside(xy (N, 2)) -> bool (N,)`."""
        m = np.zeros(self.valid.shape, bool)
        keep = inside(self.world[:, :2])
        v, u = np.nonzero(self.valid)
        m[v[keep], u[keep]] = True
        return m


def measure_surface(color: np.ndarray, depth: np.ndarray, intr, rot: np.ndarray, trans: np.ndarray) -> Surface:
    """Back-project the depth to world, fit the table plane, and measure every pixel's height above it."""
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
    return Surface(color, cv2.cvtColor(color, cv2.COLOR_BGR2HSV), depth, intr, rot, trans, valid, world,
                   index, above, height, coef)


def find_blocks(color: np.ndarray, depth: np.ndarray, intr, rot: np.ndarray, trans: np.ndarray,
                report: dict | None = None):
    """Blocks in one aligned frame. `rot`, `trans` map camera mm to world mm (frames.frame_to_world).

    Returns (blocks, table_coef) with table_coef = (a, b, c) of the world table plane. If `report` is
    given it gets "edge_cut": [{"color", "xy"}] for painted things cut off by the image edge (world x, y
    of the part in view), and "knobs": world [x, y] of each red knob seen.
    """
    return blocks_on_surface(measure_surface(color, depth, intr, rot, trans), report)


def blocks_on_surface(surf: Surface, report: dict | None = None, clear_mm: float = KNOB_CLEAR_MM):
    """find_blocks on an already measured surface (see find_blocks)."""
    hsv, height, index, world, above, valid, coef = (surf.hsv, surf.height, surf.index, surf.world, surf.above,
                                                     surf.valid, surf.coef)
    hgt, wid = height.shape
    blockish = ((height > MIN_HEIGHT_MM) & (height < MAX_HEIGHT_MM)).astype(np.uint8)
    blockish = cv2.morphologyEx(blockish, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    blockish = cv2.morphologyEx(blockish, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    tall = cv2.dilate((height >= MAX_HEIGHT_MM).astype(np.uint8), np.ones((15, 15), np.uint8))

    knobs = _knob_points(hsv, index, world)

    def near_knob(xy) -> bool:  # on or in the drawer box: an open drawer's floor, the box's edges
        return len(knobs) > 0 and float(np.min(np.linalg.norm(knobs - np.asarray(xy)[:2], axis=1))) < clear_mm

    blocks, edge_cut = [], []
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(blockish)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] < 150:
            continue
        comp = labels == i
        if np.any(comp & (tall > 0)):
            continue  # part of something tall (the drawer box, its knobs)
        idx = index[comp & valid]
        if idx.size < 100:
            continue
        if _touches_border(stats[i], wid, hgt):  # cut off: report it, so the caller can look again
            name, xy = name_color(hsv[comp]), np.median(world[idx, :2], axis=0)
            short_mm, long_mm, _, _ = _short_side(world[idx, :2])
            if (name != "white" and not near_knob(xy) and _fill(comp) >= MIN_FILL
                    and MIN_SIDE_MM <= short_mm and long_mm <= MAX_SIDE_MM):
                edge_cut.append({"color": name, "xy": [round(float(xy[0]), 1), round(float(xy[1]), 1)]})
            continue
        h_mm = float(np.percentile(above[idx], 75))  # robust to edge pixels that land between top and table
        top = idx[above[idx] >= 0.5 * h_mm]          # the footprint from the top face, not the edge skirt
        pts = world[top if top.size >= 30 else idx]
        short_mm, long_mm, short_deg, center_xy = _short_side(pts[:, :2])
        if not (MIN_SIDE_MM <= short_mm and long_mm <= MAX_SIDE_MM) or _fill(comp) < MIN_FILL:
            continue
        inner = cv2.erode(comp.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        name = name_color(hsv[inner] if inner.sum() > 30 else hsv[comp])
        if name == "white" or near_knob(center_xy):
            continue  # not paint (table, tape, cream plastic), or part of the drawer box
        table_z = float(coef[0] * center_xy[0] + coef[1] * center_xy[1] + coef[2])
        blocks.append(Block(
            color=name, pixel=(int(centroids[i][0]), int(centroids[i][1])),
            mask=(comp.astype(np.uint8) * 255), center=np.array([center_xy[0], center_xy[1], table_z + h_mm / 2.0]),
            height_mm=h_mm, short_mm=short_mm, long_mm=long_mm, short_axis_deg=short_deg, table_z_mm=table_z))
    found = np.zeros_like(blockish)
    for b in blocks:
        found |= (b.mask > 0).astype(np.uint8)
    near_found = cv2.dilate(found, np.ones((31, 31), np.uint8))  # a dark patch beside a block is that block
    blocks += [b for b in _dark_blocks(hsv, height, near_found, tall, surf.intr, surf.rot, surf.trans, coef)
               if not near_knob(b.center)]
    if report is not None:
        report.update(edge_cut=edge_cut, knobs=[[round(float(x), 1), round(float(y), 1)] for x, y in knobs])
    return blocks, coef


def knob_centers(surf: Surface, diameter_mm: float = 23.0, with_edge: bool = False):
    """World centres (N, 3) of the red knob balls. The camera sees the near half of each ball; its pixels'
    centroid sits 2/3 of a radius in front of the centre (pixels sample the projected disc evenly), so step
    that far further along the view ray. with_edge=True also returns which balls touch the image edge
    (cut off: their centre is unreliable)."""
    h, s, v = surf.hsv[:, :, 0], surf.hsv[:, :, 1], surf.hsv[:, :, 2]
    red = (((h <= 6) | (h >= 170)) & (s >= 120) & (v >= 60)).astype(np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(red)
    out, edge = [], []
    hgt, wid = surf.valid.shape
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] < 200:
            continue
        idx = surf.index[(labels == i) & (surf.index >= 0)]
        if idx.size < 20:
            continue
        cap = surf.world[idx]
        mid = cap.mean(axis=0)
        ray = mid - surf.trans
        ray = ray / (np.linalg.norm(ray) or 1.0)
        out.append(mid + ray * diameter_mm / 3.0)
        edge.append(_touches_border(stats[i], wid, hgt))
    centers = np.array(out).reshape(-1, 3)
    return (centers, np.array(edge, bool)) if with_edge else centers


def _knob_points(hsv: np.ndarray, index: np.ndarray, world: np.ndarray) -> np.ndarray:
    """World x, y of the drawer box's red knobs (saturated red blobs with depth)."""
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    red = (((h <= 6) | (h >= 170)) & (s >= 120) & (v >= 60)).astype(np.uint8)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(red)
    points = []
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] < 200:
            continue
        idx = index[(labels == i) & (index >= 0)]
        if idx.size >= 20:
            points.append(np.median(world[idx, :2], axis=0))
    return np.array(points)


def knob_row(knobs) -> tuple[bool | None, float]:
    """Are all drawers shut? Their knobs then lie on one straight line. Returns (in a row, the farthest
    knob's distance from the best-fit line in mm); (None, nan) unless exactly three knobs were seen."""
    pts = np.asarray(knobs, float).reshape(-1, 2)
    if len(pts) != 3:
        return None, float("nan")
    centered = pts - pts.mean(axis=0)
    normal = np.linalg.svd(centered)[2][1]  # the direction the knobs spread least in
    off = float(np.max(np.abs(centered @ normal)))
    return off <= ROW_TOLERANCE_MM, off


def _fill(comp: np.ndarray) -> float:
    """Share of its minimum-area rectangle that a mask covers (pixels)."""
    ys, xs = np.nonzero(comp)
    (_, _), (w, h), _ = cv2.minAreaRect(np.column_stack((xs, ys)).astype(np.float32))
    return xs.size / max(w * h, 1.0)


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
        if not (MIN_SIDE_MM <= short_mm and long_mm <= MAX_SIDE_MM) or _fill(comp) < MIN_FILL:
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
        "red": (vv >= 35) & (ss >= PAINT_S) & ((hh <= 4) | (hh >= 170)),
        "orange": (vv >= 35) & (ss >= PAINT_S) & (hh > 4) & (hh < 15),
        "yellow": (vv >= 35) & (ss >= PAINT_S) & (hh >= 15) & (hh < 42),
        "green": (vv >= 35) & (ss >= PAINT_S) & (hh >= 42) & (hh < 95),
        "blue": (vv >= 35) & (ss >= PAINT_S) & (hh >= 95) & (hh < 135),
    }
    return float(np.mean(tests[name])) if name in tests else 0.0
