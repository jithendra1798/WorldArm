"""The sorting rules: what to do next, from the belief alone. Pure: easy to test, easy to change.

  one drawer open at a time   two or more open -> close one (keep the one with work to do)
  partly open                 pull it out to the working opening, so we can look in and drop
  open + mixed colors         keep the color it has most of; take the others out onto the table
  open + one color            put that color's table blocks in; none left -> close it
  open + empty                put in the color with the most table blocks that has no drawer yet;
                              every table color already has a drawer -> close it
  none open                   open, in this order: a drawer known to be mixed (fix it) -> the drawer
                              holding a color that's on the table -> a drawer not yet looked into ->
                              an empty drawer for a color that has none
  stop                        nothing on the table and every drawer seen and single-colored (sorted);
                              or nothing more can be done (organizer not in view, no free drawer,
                              blocks or drawers that keep failing)
  drawer 3 stays free         with up to two colors, the colors go into drawers 1 and 2 only; drawer 3
                              is never given a color, anything already in it is taken out (and sorted
                              into 1 and 2), and once everything is sorted it is opened and left open.
                              Three colors need every drawer: then every drawer ends shut, as before
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from worldarm.blocks import Block
from worldarm.task.belief import DrawerBelief, SortBelief


@dataclass(eq=False)
class Action:
    kind: str                       # open | close | transfer | unload | stop
    drawer: int | None = None
    block: Block | None = None
    color: str | None = None
    reason: str = ""
    success: bool | None = None     # for stop: sorted or not

    def __str__(self) -> str:
        if self.kind == "stop":
            return f"STOP ({'sorted' if self.success else 'not sorted'}): {self.reason}"
        what = {"open": f"open drawer {self.drawer}", "close": f"close drawer {self.drawer}",
                "transfer": f"{self.color} -> drawer {self.drawer}",
                "unload": f"take {self.color} out of drawer {self.drawer}"}[self.kind]
        return f"{what}  ({self.reason})"

    def to_json(self) -> dict:
        return {"kind": self.kind, "drawer": self.drawer, "color": self.color, "reason": self.reason,
                "success": self.success,
                "block": None if self.block is None else [round(float(v), 1) for v in self.block.center]}


def stop(reason: str, success: bool) -> Action:
    return Action("stop", reason=reason, success=success)


def decide(b: SortBelief) -> Action:
    if b.organizer is None:
        return stop("can't see the drawer organizer from the overview", False)
    drawers = [b.drawers[k] for k in sorted(b.drawers)]
    if any(d.state == "unknown" for d in drawers):
        return stop("can't tell which drawers are open", False)
    usable = [d for d in drawers if not b.drawer_stuck(d.index)]
    table = [blk for blk in b.table if not b.gave_up(blk)]
    counts = Counter(blk.color for blk in table)
    opened = [d for d in drawers if d.state in ("open", "partial")]

    spare = _spare_drawer(b, drawers, counts)

    def home(color: str, other_than: int | None = None) -> DrawerBelief | None:
        for d in drawers:
            if d is spare:
                continue                             # drawer 3 is never anyone's home
            if d.index != other_than and d.known and color in d.colors and (d.pure or d.keep_color(counts) == color):
                return d
        return None

    def has_work(d: DrawerBelief) -> bool:
        if d is spare:
            return d.known and not d.empty           # only taking blocks out of it
        if not d.known or d.mixed:
            return True
        if d.pure:
            return counts[d.color] > 0
        return any(home(c, d.index) is None for c in counts)

    def rest_sorted(skip: DrawerBelief) -> bool:
        """Every other drawer looked into and single-colored (or empty), nothing stuck."""
        return all(d.known and not d.mixed and not b.drawer_stuck(d.index) for d in drawers if d is not skip)

    if len(opened) > 1:
        keep = next((d for d in opened if has_work(d) and not b.drawer_stuck(d.index)), opened[0])
        other = next(d for d in opened if d is not keep)
        return Action("close", other.index, reason=f"one drawer open at a time; drawer {keep.index} has the work")

    if opened:
        d = opened[0]
        if b.drawer_stuck(d.index):
            return stop(f"drawer {d.index} is open and keeps failing to move", False)
        if d.state == "partial" or (d.opening_mm or 0.0) < b.cfg.organizer.usable_above_mm:
            return Action("open", d.index, reason=f"partly open ({d.opening_mm or 0:.0f} mm): pull it out to look in and drop")
        if not d.known:
            return stop(f"drawer {d.index} is open but the camera can't see inside it", False)
        if spare is not None and d is spare and not d.empty:
            out = [blk for blk in d.blocks if not b.gave_up(blk)]
            if out:
                blk = max(out, key=lambda x: _toward_front(x, b))
                return Action("unload", d.index, blk, blk.color,
                              f"drawer {d.index} stays free: taking {blk.color} out (it goes into drawer 1 or 2)")
            return Action("close", d.index, reason=f"drawer {d.index}'s blocks can't be picked out")
        if spare is not None and d is spare and d.empty:
            if not counts and rest_sorted(d):
                return stop(f"sorted into drawers 1 and 2; drawer {d.index} is left open", True)
            if counts or not rest_sorted(d):
                return Action("close", d.index, reason=f"drawer {d.index} stays free: the colors go into the first "
                                                       "drawers; it is opened again at the end")
        if d.mixed:
            keep = d.keep_color(counts)
            out = [blk for blk in d.blocks if blk.color != keep and not b.gave_up(blk)]
            if not out:  # its other blocks can't be taken out: keep one of those instead, if that can be fixed
                for alt in sorted(d.colors - {keep}):
                    alt_out = [blk for blk in d.blocks if blk.color != alt and not b.gave_up(blk)]
                    if alt_out:
                        keep, out = alt, alt_out
                        break
            if out:
                blk = max(out, key=lambda x: _toward_front(x, b))
                return Action("unload", d.index, blk, blk.color,
                              f"drawer {d.index} has {_fmt(d.contents)}: keeping {keep}, taking {blk.color} out")
            if counts[keep]:  # the others can't be picked out: still its {keep} drawer
                return Action("transfer", d.index, _best(table, keep, d, b), keep,
                              f"drawer {d.index} keeps {keep} (its other blocks can't be reached)")
            return Action("close", d.index, reason=f"drawer {d.index} is still mixed but its other blocks can't be picked")
        if d.pure:
            c = d.color
            if counts[c]:
                return Action("transfer", d.index, _best(table, c, d, b), c, f"drawer {d.index} holds {c}")
            return Action("close", d.index, reason=f"no more {c} on the table")
        homeless = sorted((c for c in counts if home(c, d.index) is None), key=lambda c: (-counts[c], c))
        if homeless:
            c = homeless[0]
            return Action("transfer", d.index, _best(table, c, d, b), c,
                          f"drawer {d.index} is empty; {c} has no drawer yet ({counts[c]} on the table)")
        return Action("close", d.index, reason="empty, and every color on the table already has a drawer")

    if spare is not None and spare.known and not spare.empty and any(not b.gave_up(x) for x in spare.blocks):
        return Action("open", spare.index, reason=f"drawer {spare.index} stays free: its {_fmt(spare.contents)} "
                                                  "come out first")
    mixed = [d for d in usable if d.mixed and d is not spare and any(not b.gave_up(x) for x in d.blocks)]
    if mixed:
        return Action("open", mixed[0].index, reason=f"it has mixed colors ({_fmt(mixed[0].contents)}): fix it")
    for c in sorted(counts, key=lambda c: (-counts[c], c)):
        h = home(c)
        if h is not None and h in usable:
            return Action("open", h.index, reason=f"{counts[c]} {c} on the table; drawer {h.index} holds {c}")
    unseen = [d for d in usable if not d.known]
    if spare is not None and counts and unseen and unseen[0] is spare and len(unseen) == 1:
        unseen = []                      # the spare can wait: it's opened (and looked into) at the end
    if unseen:
        return Action("open", unseen[0].index, reason="haven't looked inside it yet")
    if counts:
        empty = [d for d in usable if d.empty and d is not spare]
        if empty:
            c = sorted(counts, key=lambda c: (-counts[c], c))[0]
            return Action("open", empty[0].index, reason=f"{c} has no drawer; drawer {empty[0].index} is empty")
        return stop(f"no free drawer for {', '.join(sorted(counts))}", False)
    left = [f"{blk.color}" for blk in b.table if b.gave_up(blk)]
    still_mixed = [d.index for d in drawers if d.mixed]
    stuck = [d.index for d in drawers if b.drawer_stuck(d.index)]
    if left or still_mixed or stuck:
        bits = ([f"couldn't pick {', '.join(left)}"] if left else []) + \
               ([f"drawer(s) {still_mixed} still mixed"] if still_mixed else []) + \
               ([f"drawer(s) {stuck} won't move"] if stuck else [])
        return stop("; ".join(bits), False)
    if spare is not None and spare.state == "closed":
        return Action("open", spare.index, reason=f"sorted into drawers 1 and 2; drawer {spare.index} is left open")
    return stop("sorted: nothing left on the table and every drawer holds one color", True)


def _colors_in_play(b: SortBelief, drawers: list[DrawerBelief], counts: Counter) -> set[str]:
    """Every color the task has to give a drawer: on the table, or already in a drawer."""
    colors = set(counts)
    for d in drawers:
        if d.known and not d.empty:
            colors |= d.colors
    return colors


def _spare_drawer(b: SortBelief, drawers: list[DrawerBelief], counts: Counter) -> DrawerBelief | None:
    """The last drawer (drawer 3), kept free: never given a color, emptied if it holds anything, and left
    open at the end. None when the colors need every drawer (three colors), or the other drawers can't
    take them (one stuck)."""
    spare = drawers[-1]
    colors = _colors_in_play(b, drawers, counts)
    others = [d for d in drawers[:-1] if not b.drawer_stuck(d.index)]
    if b.drawer_stuck(spare.index) or len(colors) > len(others):
        return None
    return spare


def _fmt(c: Counter | None) -> str:
    return ", ".join(f"{n} {k}" for k, n in sorted((c or {}).items())) or "nothing"


def _best(table: list[Block], color: str, d: DrawerBelief, b: SortBelief) -> Block:
    """The block of `color` to move next: the most isolated (easiest to grip), then the nearest."""
    cands = [x for x in table if x.color == color]
    knob = None
    if b.organizer is not None:
        knob = b.organizer.drawer(d.index).knob[:2]

    def key(x: Block):
        others = [float(np.linalg.norm(x.center[:2] - o.center[:2])) for o in b.table if o is not x]
        isolation = min(others) if others else 1e9
        dist = float(np.linalg.norm(x.center[:2] - knob)) if knob is not None else 0.0
        return (-min(isolation, 150.0) // 25, dist)
    return min(cands, key=key)


def _toward_front(x: Block, b: SortBelief) -> float:
    """How far out (toward the drawer front) a block in a drawer sits: those are easiest to reach."""
    return float(x.center[:2] @ b.organizer.pull) if b.organizer is not None else 0.0
