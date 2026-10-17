"""The drawer organizer, from one frame: its knobs, which drawer is which, how far each drawer is out,
where the inside of an open drawer is (to look into and drop into), and the box (an obstacle).

Drawers slide along `pull`, perpendicular to the knob row, so a shut drawer's knob sits on one line and
an open drawer's knob sits `opening` in front of it. The three knobs are explained by one of:
  all shut      - the three knobs are on one line;
  one out       - two knobs make the line, the third is off it, in front, at the next slot along the row;
  remembered    - the row line from an earlier look (the box hasn't moved), for two drawers out at once.
Numbering matches the drawer service: drawer 1 is leftmost for someone facing the drawer fronts.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from worldarm.blocks import Surface, knob_centers
from worldarm.cell.config import OrganizerCfg
from worldarm.cell.geometry import perp, unit

NEAR_MM = 450.0           # only look at points this close to the knobs (speed)


@dataclass
class DrawerView:
    index: int                       # 1..3
    knob: np.ndarray                 # world x, y, z of the knob centre now
    opening_mm: float                # how far it is out (0 = shut)
    state: str                       # "closed" | "partial" | "open"
    along_mm: float                  # its place along the row (row coordinate of the knob)
    inside: np.ndarray | None = None  # (4, 2) world x, y corners of the visible inside, if out
    floor_z: float | None = None     # world z of the floor inside (measured)
    rim_z: float | None = None       # world z of the top of its side walls (measured)

    def to_json(self) -> dict:
        return {"index": self.index, "state": self.state, "opening_mm": round(self.opening_mm, 1),
                "knob": [round(float(v), 1) for v in self.knob],
                "floor_z": None if self.floor_z is None else round(self.floor_z, 1),
                "rim_z": None if self.rim_z is None else round(self.rim_z, 1),
                "inside": None if self.inside is None else np.round(self.inside, 1).tolist()}


@dataclass
class RowModel:
    """What to remember between looks: the row line of shut knobs (fixed while the box stays put)."""
    pull: np.ndarray
    row: np.ndarray
    closed_line: float               # pull coordinate of a shut drawer's knob
    along: list[float]               # row coordinate of each drawer's knob, drawer 1..3

    def to_json(self) -> dict:
        return {"pull": np.round(self.pull, 5).tolist(), "row": np.round(self.row, 5).tolist(),
                "closed_line": round(self.closed_line, 2), "along": [round(a, 1) for a in self.along]}

    @classmethod
    def from_json(cls, d: dict) -> RowModel:
        return cls(np.array(d["pull"], float), np.array(d["row"], float), float(d["closed_line"]),
                   [float(a) for a in d["along"]])


@dataclass
class Organizer:
    pull: np.ndarray                 # unit (2,): the way a drawer comes out
    row: np.ndarray                  # unit (2,): from drawer 1 toward drawer 3
    closed_line: float
    drawers: list[DrawerView]
    box: np.ndarray                  # (4, 2) world x, y footprint of the box body
    box_top_z: float
    how: str                         # which explanation fitted the knobs
    warnings: list[str] = field(default_factory=list)

    def drawer(self, k: int) -> DrawerView:
        return next(d for d in self.drawers if d.index == k)

    def open_drawers(self) -> list[DrawerView]:
        return [d for d in self.drawers if d.state != "closed"]

    def model(self) -> RowModel:
        return RowModel(self.pull, self.row, self.closed_line, [d.along_mm for d in self.drawers])

    def to_json(self) -> dict:
        return {"how": self.how, "pull": np.round(self.pull, 4).tolist(), "row": np.round(self.row, 4).tolist(),
                "box": np.round(self.box, 1).tolist(), "box_top_z": round(self.box_top_z, 1),
                "drawers": [d.to_json() for d in self.drawers], "warnings": self.warnings}


@dataclass
class _Fit:
    error: float
    how: str
    u: np.ndarray                    # row direction (sign not yet meaningful)
    pull: np.ndarray | None          # None: not decided by the knobs alone
    closed_line: float | None


def find_organizer(surf: Surface, cfg: OrganizerCfg, prior: RowModel | None = None,
                   knobs: np.ndarray | None = None) -> tuple[Organizer | None, list[str]]:
    """The organizer in this frame, or None with the reason."""
    edge_warn: list[str] = []
    if knobs is None:
        knobs, at_edge = knob_centers(surf, cfg.knob_diameter_mm, with_edge=True)
        if at_edge.any():
            edge_warn.append(f"{int(at_edge.sum())} knob(s) cut off by the edge of the view: their position is "
                             "uncertain; re-teach the overview so the whole organizer is in view")
    knobs = np.asarray(knobs, float)
    n = cfg.n_drawers
    if len(knobs) < n:
        return None, [f"{len(knobs)} knob(s) in view; the organizer has {n}"] + edge_warn
    near = _near(surf, knobs)
    best: tuple[_Fit, np.ndarray] | None = None
    for tri in itertools.combinations(range(len(knobs)), n):
        k = knobs[list(tri)]
        fit = _explain(k, cfg, prior, near)
        if fit is not None and (best is None or fit.error < best[0].error):
            best = (fit, k)
    if best is None:
        return None, ["the knobs don't fit the organizer (more than one drawer out, and no earlier look to go by)"] \
            + edge_warn
    fit, k = best
    org = _build(fit, k, cfg, near, surf)
    org.warnings += edge_warn
    return org, []


class _Near:
    """World points within NEAR_MM of the knobs, with heights: the organizer's neighbourhood."""

    CELL = 5.0

    def __init__(self, surf: Surface, knobs: np.ndarray):
        c = knobs[:, :2].mean(axis=0)
        d = surf.world[:, :2] - c
        keep = (np.abs(d[:, 0]) < NEAR_MM) & (np.abs(d[:, 1]) < NEAR_MM)
        keep[keep] = np.einsum("ij,ij->i", d[keep], d[keep]) < NEAR_MM * NEAR_MM
        self.xy = surf.world[keep, :2]
        self.h = surf.above[keep]
        self.surf = surf
        # mean height per 5 mm cell, for fast point lookups
        self.origin = c - NEAR_MM
        n = int(2 * NEAR_MM / self.CELL) + 1
        ij = np.clip(((self.xy - self.origin) / self.CELL).astype(int), 0, n - 1)
        flat = ij[:, 0] * n + ij[:, 1]
        self.grid_sum = np.bincount(flat, weights=self.h, minlength=n * n).reshape(n, n)
        self.grid_n = np.bincount(flat, minlength=n * n).reshape(n, n).astype(float)
        self._basis: tuple | None = None

    def heights(self, row: np.ndarray, pull: np.ndarray, t_lo, t_hi, s_lo, s_hi) -> np.ndarray:
        key = (tuple(np.round(row, 9)), tuple(np.round(pull, 9)))
        if self._basis is None or self._basis[0] != key:  # the same drawer frame is queried many times
            self._basis = (key, self.xy @ row, self.xy @ pull)
        t, s = self._basis[1], self._basis[2]
        return self.h[(t >= t_lo) & (t <= t_hi) & (s >= s_lo) & (s <= s_hi)]

    def height_near(self, xy, r: float = 10.0) -> float | None:
        """Mean height within about r of a world point (None where nothing was measured)."""
        n = self.grid_n.shape[0]
        i, j = ((np.asarray(xy)[:2] - self.origin) / self.CELL).astype(int)
        k = max(1, int(round(r / self.CELL)))
        if not (0 <= i < n and 0 <= j < n):
            return None
        sl = (slice(max(0, i - k), i + k + 1), slice(max(0, j - k), j + k + 1))
        cnt = float(self.grid_n[sl].sum())
        return float(self.grid_sum[sl].sum() / cnt) if cnt >= 5 else None


def _near(surf: Surface, knobs: np.ndarray) -> _Near:
    return _Near(surf, knobs)


def _explain(k: np.ndarray, cfg: OrganizerCfg, prior: RowModel | None, near: _Near) -> _Fit | None:
    p = k[:, :2]
    S, tol = cfg.knob_spacing_mm, cfg.closed_below_mm
    fits: list[_Fit] = []

    # all shut: one line, two even gaps
    c = p.mean(axis=0)
    u0 = np.linalg.svd(p - c)[2][0]
    off = float(np.max(np.abs((p - c) @ perp(u0))))
    gaps = np.diff(np.sort((p - c) @ u0))
    if off <= tol and all(0.65 * S <= g <= 1.4 * S for g in gaps):
        fits.append(_Fit(off / tol + abs(gaps[0] - gaps[1]) / S, "all shut", u0, None, None))

    # one drawer out: the other two make the line; the third is in front, at the next slot
    for out in range(3):
        a, b = [i for i in range(3) if i != out]
        L = float(np.linalg.norm(p[b] - p[a]))
        if L < 1e-6:
            continue
        u = (p[b] - p[a]) / L
        t = float((p[out] - p[a]) @ u)
        d = float((p[out] - p[a]) @ perp(u))
        if abs(d) < tol:
            continue
        if 0.65 * S <= L <= 1.4 * S:
            slots = (-L, 2 * L)
        elif 1.3 * S <= L <= 2.7 * S:
            slots = (L / 2,)
        else:
            continue
        err = min(abs(t - e) for e in slots)
        if err > 0.35 * S:
            continue
        pull = np.sign(d) * perp(u)
        box_pull = _pull_from_box(p[[a, b]], u, near)
        if box_pull is not None and float(box_pull @ pull) < 0:
            continue  # the "open" knob is behind the line: not a drawer out
        fits.append(_Fit(1.0 + err / S, f"drawer at slot {out} out", u, pull, float(p[a] @ pull)))

    # remembered row line: works for any number out, as long as the box hasn't moved
    if prior is not None:
        s = p @ prior.pull - prior.closed_line
        t = p @ prior.row
        slot_err = [min(abs(ti - a) for a in prior.along) for ti in t]
        if min(s) >= -tol and max(slot_err) <= 0.35 * S:
            fits.append(_Fit(2.0 + max(slot_err) / S, "remembered row", prior.row, prior.pull, prior.closed_line))

    return min(fits, key=lambda f: f.error) if fits else None


def _pull_from_box(p: np.ndarray, u: np.ndarray, near: _Near, probe_mm: float = 60.0) -> np.ndarray | None:
    """Which side of the knob line is the box? Its top is high; the other side is table or drawer floor."""
    n = perp(u)
    plus, minus = [], []
    for q in p:
        hp, hm = near.height_near(q + probe_mm * n), near.height_near(q - probe_mm * n)
        if hp is not None and hm is not None:
            plus.append(hp)
            minus.append(hm)
    if not plus:
        return None
    diff = float(np.mean(plus) - np.mean(minus))
    if abs(diff) < 15.0:
        return None
    return -n if diff > 0 else n  # drawers come out away from the box


def _build(fit: _Fit, k: np.ndarray, cfg: OrganizerCfg, near: _Near, surf: Surface) -> Organizer:
    warnings: list[str] = []
    p = k[:, :2]
    pull = fit.pull
    if pull is None:
        pull = _pull_from_box(p, fit.u, near)
        if pull is None:
            toward_base = -p.mean(axis=0)
            pull = perp(fit.u) * (1.0 if float(perp(fit.u) @ toward_base) > 0 else -1.0)
            warnings.append("couldn't tell the box side from depth; assumed the drawers face the arm")
    pull = unit(pull)
    left = np.array([pull[1], -pull[0]])
    order = np.argsort(-(p @ left))              # drawer 1 = largest along `left` (service numbering)
    row = -left
    s_all = p @ pull
    closed_line = fit.closed_line if fit.closed_line is not None else float(np.mean(s_all))
    S = cfg.knob_spacing_mm
    drawers: list[DrawerView] = []
    for idx, i in enumerate(order, start=1):
        opening = float(s_all[i] - closed_line)
        if opening < cfg.closed_below_mm:
            opening = max(0.0, opening)
            state = "closed"
        else:
            state = "open" if opening >= cfg.usable_above_mm else "partial"
        d = DrawerView(idx, k[i].copy(), opening, state, float(p[i] @ row))
        if state != "closed":
            _measure_inside(d, pull, row, closed_line, cfg, near, surf)
        drawers.append(d)

    along = [d.along_mm for d in drawers]
    face = closed_line - cfg.knob_protrusion_mm
    t_lo, t_hi = min(along) - S / 2 - 8, max(along) + S / 2 + 8
    behind = near.heights(row, pull, t_lo, t_hi, face - 400.0, face - 10.0)
    top_h = float(np.percentile(behind, 90)) if behind.size >= 50 else 95.0
    depth = 200.0
    if behind.size >= 50:
        xy = near.xy
        t, s = xy @ row, xy @ pull
        sel = (t >= t_lo) & (t <= t_hi) & (s >= face - 400.0) & (s <= face - 10.0) & (near.h > 0.6 * top_h)
        if sel.any():
            depth = float(np.clip(face - float(s[sel].min()), 80.0, 400.0))
    corners = np.array([t_lo * row + face * pull, t_hi * row + face * pull,
                        t_hi * row + (face - depth) * pull, t_lo * row + (face - depth) * pull])
    centre = corners.mean(axis=0)
    return Organizer(pull, row, closed_line, drawers, corners, surf.table_z(*centre) + top_h, fit.how, warnings)


def _measure_inside(d: DrawerView, pull: np.ndarray, row: np.ndarray, closed_line: float, cfg: OrganizerCfg,
                    near: _Near, surf: Surface) -> None:
    """Corners of the visible inside, and the floor and wall-top heights measured there."""
    half = cfg.knob_spacing_mm / 2.0 - cfg.drawer_wall_mm
    s_lo = closed_line - cfg.knob_protrusion_mm + 3.0              # the box front
    s_hi = closed_line + d.opening_mm - cfg.knob_protrusion_mm - cfg.face_mm  # inner side of the drawer front
    if s_hi - s_lo < 15.0:
        return
    t = d.along_mm
    d.inside = np.array([(t - half) * row + s_lo * pull, (t + half) * row + s_lo * pull,
                         (t + half) * row + s_hi * pull, (t - half) * row + s_hi * pull])
    cx, cy = d.inside.mean(axis=0)
    base = surf.table_z(cx, cy)
    floor = near.heights(row, pull, t - half + 3, t + half - 3, s_lo + 3, s_hi - 3)
    d.floor_z = base + (float(np.percentile(floor, 15)) if floor.size >= 30 else cfg.floor_above_table_mm)
    w = cfg.drawer_wall_mm
    walls = np.concatenate([near.heights(row, pull, t - half - w - 3, t - half + 1, s_lo, s_hi),   # left wall
                            near.heights(row, pull, t + half - 1, t + half + w + 3, s_lo, s_hi)])  # right wall
    d.rim_z = base + (float(np.percentile(walls, 85)) if walls.size >= 20 else cfg.rim_above_table_mm)
