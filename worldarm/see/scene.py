"""One call from a frame to everything the task needs: table blocks, the organizer, each drawer's state
and contents, and free table spots. This is the observation the belief (task/belief.py) is updated from."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from worldarm.blocks import Block, Surface, blocks_on_surface, measure_surface
from worldarm.cell.config import CellConfig
from worldarm.cell.geometry import point_in_polygon
from worldarm.see.contents import drawer_contents
from worldarm.see.frame import Frame
from worldarm.see.organizer import Organizer, RowModel, find_organizer


@dataclass
class Scene:
    stamp: float
    table_blocks: list[Block]
    organizer: Organizer | None
    contents: dict[int, list[Block]]        # drawer index -> blocks seen inside (open drawers only)
    table: np.ndarray                       # plane coefficients
    edge_cut: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    surface: Surface | None = None          # kept for drawing and free-spot search; not logged

    def blocks_of(self, color: str) -> list[Block]:
        return [b for b in self.table_blocks if b.color == color]

    def to_json(self) -> dict:
        return {
            "table_blocks": [block_json(b) for b in self.table_blocks],
            "organizer": self.organizer.to_json() if self.organizer else None,
            "contents": {str(k): [block_json(b) for b in v] for k, v in self.contents.items()},
            "table_plane": [round(float(c), 5) for c in self.table],
            "edge_cut": self.edge_cut, "warnings": self.warnings,
        }


def block_json(b: Block) -> dict:
    return {"color": b.color, "center": [round(float(v), 1) for v in b.center], "height_mm": round(b.height_mm, 1),
            "short_mm": round(b.short_mm, 1), "long_mm": round(b.long_mm, 1),
            "short_axis_deg": round(b.short_axis_deg, 1), "base_z": round(b.table_z_mm, 1)}


def observe(frame: Frame, cfg: CellConfig, prior: RowModel | None = None) -> Scene:
    """Everything in one frame. Read-only."""
    surf = measure_surface(frame.color, frame.depth, frame.intr, frame.rot, frame.trans)
    report: dict = {}
    organizer, warnings = find_organizer(surf, cfg.organizer, prior)
    # with the organizer found, its box and open drawers are kept out by their outlines (below), so only a
    # knob's own surroundings are; without it, the wide knob rule keeps the box's parts from being blocks
    clear = cfg.organizer.knob_keep_out_mm if organizer is not None else cfg.organizer.clear_mm
    blocks, coef = blocks_on_surface(surf, report, clear_mm=clear)
    contents: dict[int, list[Block]] = {}
    if organizer is not None:
        warnings += organizer.warnings
        keep_out = _organizer_keep_out(organizer, margin=30.0)
        blocks = [b for b in blocks if not any(point_in_polygon(b.center[:2], poly)[0] for poly in keep_out)]
        for d in organizer.drawers:
            if d.inside is not None and d.floor_z is not None:
                contents[d.index] = drawer_contents(surf, d.inside, d.floor_z)
    return Scene(frame.stamp, blocks, organizer, contents, coef, report.get("edge_cut", []), warnings, surf)


def _organizer_keep_out(org: Organizer, margin: float) -> list[np.ndarray]:
    """Footprints (with a margin) where nothing is a table block: the box, and every drawer that's out."""
    polys = [_grow(org.box, margin)]
    for d in org.drawers:
        if d.inside is not None:
            polys.append(_grow(d.inside, margin))
    return polys


def _grow(poly: np.ndarray, mm: float) -> np.ndarray:
    c = poly.mean(axis=0)
    return np.array([p + (p - c) / (np.linalg.norm(p - c) or 1.0) * mm for p in poly])


def visible_table(frame: Frame, table: np.ndarray, inset_px: float = 90.0) -> np.ndarray:
    """World x, y polygon of the table the camera sees, inset from the image edge."""
    w, h = frame.intr.width_px, frame.intr.height_px
    corners = [(inset_px, inset_px), (w - inset_px, inset_px), (w - inset_px, h - inset_px), (inset_px, h - inset_px)]
    out = []
    for u, v in corners:
        ray = frame.rot @ np.array([(u - frame.intr.center_x_px) / frame.intr.focal_x_px,
                                    (v - frame.intr.center_y_px) / frame.intr.focal_y_px, 1.0])
        a, b, c = table
        t = (a * frame.trans[0] + b * frame.trans[1] + c - frame.trans[2]) / (ray[2] - a * ray[0] - b * ray[1])
        out.append(frame.trans[:2] + t * ray[:2])
    return np.array(out)


def free_spot(scene: Scene, frame: Frame, cfg: CellConfig, near_xy=None, clearance_mm: float = 85.0,
              avoid: list[np.ndarray] | None = None) -> np.ndarray | None:
    """A clear table spot in view and in reach for setting a block down: away from blocks and the
    organizer, outside the knob keep-out (so it stays visible as a table block). Nearest to `near_xy`."""
    view = visible_table(frame, scene.table)
    lo, hi = view.min(axis=0), view.max(axis=0)
    xs, ys = np.meshgrid(np.arange(lo[0], hi[0], 15.0), np.arange(lo[1], hi[1], 15.0))
    cand = np.stack([xs.ravel(), ys.ravel()], axis=1)
    cand = cand[point_in_polygon(cand, view)]
    wc = cfg.workcell
    ax0, ax1, ay0, ay1 = cfg.task.sort_area_xy  # stay on our side of the table, where blocks count (y < 0)
    cand = cand[(cand[:, 0] >= max(wc.x_min_mm, ax0) + 40) & (cand[:, 0] <= min(wc.x_max_mm, ax1) - 40)
                & (cand[:, 1] >= max(wc.y_min_mm, ay0) + 40) & (cand[:, 1] <= min(wc.y_max_mm, ay1) - 40)
                & (np.linalg.norm(cand, axis=1) <= wc.reach_mm - 60) & (np.linalg.norm(cand, axis=1) >= 200)]
    if not len(cand):
        return None
    keep = np.ones(len(cand), bool)
    for b in scene.table_blocks:
        keep &= np.linalg.norm(cand - b.center[:2], axis=1) >= clearance_mm + 0.5 * math.hypot(b.short_mm, b.long_mm)
    for p in avoid or []:
        keep &= np.linalg.norm(cand - np.asarray(p)[:2], axis=1) >= clearance_mm
    if scene.organizer is not None:
        org = scene.organizer
        for poly in _organizer_keep_out(org, margin=60.0):
            keep &= ~point_in_polygon(cand, poly)
        for d in org.drawers:
            keep &= np.linalg.norm(cand - d.knob[:2], axis=1) >= cfg.organizer.clear_mm + 40
    cand = cand[keep]
    if not len(cand):
        return None
    if near_xy is None:
        near_xy = view.mean(axis=0)
    return cand[int(np.argmin(np.linalg.norm(cand - np.asarray(near_xy)[:2], axis=1)))]
