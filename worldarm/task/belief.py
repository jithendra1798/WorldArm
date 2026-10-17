"""The world model for sorting: what the robot believes is where, kept between looks.

Each look replaces what's visible (table blocks, drawer states, the inside of open drawers). What isn't
visible is remembered: a closed drawer keeps the contents last seen in it. Each action leaves an
expectation; the next look checks it, and a mismatch is a Surprise (logged, and the policy re-plans
from what was actually seen). Blocks or drawers that keep failing are given up on, so the run ends.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from worldarm.blocks import Block
from worldarm.cell.config import CellConfig
from worldarm.see.organizer import Organizer, RowModel
from worldarm.see.scene import Scene

BOX_SLID_MM = 20.0


@dataclass
class DrawerBelief:
    index: int
    state: str = "unknown"                 # closed | partial | open | unknown
    opening_mm: float | None = None
    contents: Counter | None = None        # color -> count; None = never seen inside
    blocks: list[Block] = field(default_factory=list)  # last seen inside (with positions)
    seen_step: int | None = None
    held_color: str | None = None          # the color we've been sorting into it

    @property
    def known(self) -> bool:
        return self.contents is not None

    @property
    def colors(self) -> set[str]:
        return {c for c, n in (self.contents or {}).items() if n > 0}

    @property
    def empty(self) -> bool:
        return self.known and not self.colors

    @property
    def pure(self) -> bool:
        return self.known and len(self.colors) == 1

    @property
    def mixed(self) -> bool:
        return self.known and len(self.colors) > 1

    @property
    def color(self) -> str | None:
        """Its color if it holds exactly one."""
        return next(iter(self.colors)) if self.pure else None

    def keep_color(self, on_table: Counter | None = None) -> str | None:
        """For a mixed drawer: the color to keep - the one it has most of; ties: the one it held, then the
        one with more blocks still on the table (they'll join it), then alphabetical."""
        if not self.colors:
            return None
        counts = self.contents
        best = max(counts[c] for c in self.colors)
        tied = sorted(c for c in self.colors if counts[c] == best)
        if self.held_color in tied:
            return self.held_color
        return max(tied, key=lambda c: ((on_table or Counter())[c], -ord(c[0])))

    def to_json(self) -> dict:
        return {"state": self.state, "opening_mm": None if self.opening_mm is None else round(self.opening_mm, 1),
                "contents": None if self.contents is None else dict(self.contents), "seen_step": self.seen_step,
                "held_color": self.held_color}


@dataclass
class Surprise:
    step: int
    what: str
    expected: str
    seen: str

    def __str__(self) -> str:
        return f"{self.what}: expected {self.expected}, saw {self.seen}"


class SortBelief:
    def __init__(self, cfg: CellConfig):
        self.cfg = cfg
        self.step = 0
        self.table: list[Block] = []
        self.drawers = {k: DrawerBelief(k) for k in range(1, cfg.organizer.n_drawers + 1)}
        self.organizer: Organizer | None = None
        self.row_model: RowModel | None = None
        self.box_ref: np.ndarray | None = None
        self.surprises: list[Surprise] = []
        self.expect: list[tuple[str, object]] = []
        self.fails: list[tuple[str, np.ndarray, int]] = []   # (color, xy, times) for blocks
        self.drawer_fails: Counter = Counter()

    # ---- observing --------------------------------------------------------------------------------
    def observe(self, scene: Scene, step: int) -> list[Surprise]:
        self.step = step
        new: list[Surprise] = []
        self.table = [b for b in scene.table_blocks if b.color in self.cfg.task.colors]
        org = scene.organizer
        self.organizer = org
        if org is None:
            for d in self.drawers.values():
                d.state = "unknown"
        else:
            self.row_model = org.model()
            centre = org.box.mean(axis=0)
            if self.box_ref is None:
                self.box_ref = centre
            elif float(np.linalg.norm(centre - self.box_ref)) > BOX_SLID_MM:
                new.append(Surprise(step, "drawer box", "where it was",
                                    f"moved {np.linalg.norm(centre - self.box_ref):.0f} mm (tape it down)"))
                self.box_ref = centre
            for dv in org.drawers:
                d = self.drawers[dv.index]
                d.state, d.opening_mm = dv.state, dv.opening_mm
                if dv.index in scene.contents:
                    inside = [b for b in scene.contents[dv.index] if b.color in self.cfg.task.colors]
                    d.contents = Counter(b.color for b in inside)
                    d.blocks = inside
                    d.seen_step = step
        new += self._check(scene)
        self.surprises += new
        return new

    def _check(self, scene: Scene) -> list[Surprise]:
        out = []
        for kind, data in self.expect:
            if kind == "in_drawer":
                k, color, before = data
                d = self.drawers[k]
                if d.seen_step == self.step and d.contents is not None and d.contents[color] <= before:
                    out.append(Surprise(self.step, f"drawer {k}", f"one more {color} inside", f"{d.contents[color]}"))
            elif kind == "state":
                k, want = data
                st = self.drawers[k].state
                if (want == "closed") != (st == "closed") and st != "unknown":
                    out.append(Surprise(self.step, f"drawer {k}", want, f"{st} ({self.drawers[k].opening_mm or 0:.0f} mm)"))
        self.expect = []
        return out

    # ---- acting -------------------------------------------------------------------------------------
    def after(self, kind: str, ok: bool, *, drawer: int | None = None, color: str | None = None,
              block: Block | None = None) -> None:
        """Record what an action should have changed, for the next look to check."""
        if kind == "transfer" and drawer is not None:
            if ok:
                d = self.drawers[drawer]
                before = (d.contents or Counter())[color]
                self.expect.append(("in_drawer", (drawer, color, before)))
                d.held_color = d.held_color or color
            elif block is not None:
                self._failed(block)
        elif kind == "unload" and not ok and block is not None:
            self._failed(block)
        elif kind in ("open", "close") and drawer is not None:
            if ok:
                self.expect.append(("state", (drawer, "open" if kind == "open" else "closed")))
            else:
                self.drawer_fails[drawer] += 1

    def _failed(self, block: Block) -> None:
        xy = np.asarray(block.center[:2], float)
        for i, (c, p, n) in enumerate(self.fails):
            if c == block.color and float(np.linalg.norm(p - xy)) < 30.0:
                self.fails[i] = (c, xy, n + 1)
                return
        self.fails.append((block.color, xy, 1))

    def give_up(self, block: Block) -> None:
        """No point trying again (e.g. it can't be gripped where it is)."""
        for _ in range(self.cfg.task.max_tries_per_block):
            self._failed(block)

    def gave_up(self, block: Block) -> bool:
        xy = np.asarray(block.center[:2], float)
        return any(c == block.color and float(np.linalg.norm(p - xy)) < 30.0 and n >= self.cfg.task.max_tries_per_block
                   for c, p, n in self.fails)

    def drawer_stuck(self, k: int) -> bool:
        return self.drawer_fails[k] >= self.cfg.task.max_tries_per_drawer_move

    def to_json(self) -> dict:
        return {"step": self.step, "table": [f"{b.color}@({b.center[0]:.0f},{b.center[1]:.0f})" for b in self.table],
                "drawers": {k: d.to_json() for k, d in self.drawers.items()},
                "gave_up": [f"{c}@({p[0]:.0f},{p[1]:.0f})" for c, p, n in self.fails if n >= self.cfg.task.max_tries_per_block],
                "stuck_drawers": [k for k in self.drawers if self.drawer_stuck(k)]}
