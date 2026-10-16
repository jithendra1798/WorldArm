"""Belief state over block locations and drawer states: predict, observe, correct.

v0 is deterministic: each block has exactly one believed location (table, gripper, drawer:<k> or
unknown). Measurements always win over predictions, and every disagreement between the two is
recorded as a Surprise. A block whose location is unknown never causes a surprise; finding it is
a discovery, not a contradiction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .schema import GRIPPER, TABLE, UNKNOWN, Act, Action, Observation, Surprise, drawer


@dataclass
class Belief:
    blocks: dict[str, str]   # color -> location
    drawers: dict[int, bool]  # drawer index -> open?

    @classmethod
    def initial(cls, colors: Iterable[str], n_drawers: int) -> Belief:
        """Nothing known about blocks; drawers assumed closed."""
        return cls({c: UNKNOWN for c in colors}, {k: False for k in range(1, n_drawers + 1)})

    def copy(self) -> Belief:
        return Belief(dict(self.blocks), dict(self.drawers))

    def holding(self) -> str | None:
        held = self.at(GRIPPER)
        return held[0] if held else None

    def at(self, location: str) -> list[str]:
        return sorted(c for c, loc in self.blocks.items() if loc == location)

    def open_drawers(self) -> list[int]:
        return sorted(k for k, is_open in self.drawers.items() if is_open)

    def to_json(self) -> dict:
        return {"blocks": dict(self.blocks), "drawers": {str(k): v for k, v in self.drawers.items()}}

    @classmethod
    def from_json(cls, d: dict) -> Belief:
        return cls(dict(d["blocks"]), {int(k): bool(v) for k, v in d["drawers"].items()})


def predict(belief: Belief, action: Action) -> Belief:
    """The belief we expect after `action` succeeds."""
    b = belief.copy()
    if action.kind is Act.OPEN:
        b.drawers[action.drawer] = True
    elif action.kind is Act.CLOSE:
        b.drawers[action.drawer] = False
    elif action.kind is Act.PICK:
        b.blocks[action.color] = GRIPPER
    elif action.kind is Act.PLACE:
        held = b.holding()
        if held is not None:
            b.blocks[held] = drawer(action.drawer)
    return b


def preconditions(belief: Belief, action: Action) -> list[str]:
    """Reasons `action` can't run from this belief; empty means it can."""
    held = belief.holding()
    if action.kind in (Act.OPEN, Act.CLOSE, Act.PLACE) and action.drawer not in belief.drawers:
        return [f"there is no drawer {action.drawer}"]
    problems = []
    if action.kind in (Act.OPEN, Act.CLOSE) and held is not None:
        problems.append(f"gripper holds {held}; drawer moves need an empty gripper")
    if action.kind is Act.OPEN:
        others = [k for k in belief.open_drawers() if k != action.drawer]
        if others:
            problems.append(f"drawer {others[0]} is open; only one drawer may be open at a time")
    elif action.kind is Act.PICK:
        if held is not None:
            problems.append(f"gripper already holds {held}")
        if belief.blocks.get(action.color) != TABLE:
            problems.append(f"{action.color} is not believed to be on the table")
    elif action.kind is Act.PLACE:
        if held is None:
            problems.append("gripper is empty")
        if not belief.drawers[action.drawer]:
            problems.append(f"drawer {action.drawer} is believed closed")
    return problems


def update(prior: Belief, action: Action, obs: Observation) -> tuple[Belief, list[Surprise]]:
    """Correct the prior with pre-action facts, predict the action, then correct with post-action facts.

    If the pre-action facts show the action can't work (e.g. the block left the table before the
    pick), we don't expect its effect, so one event isn't counted as two surprises.
    """
    corrected, pre_surprises = correct(prior, prior, obs.pre_facts)
    expected = corrected if preconditions(corrected, action) else predict(corrected, action)
    post, post_surprises = correct(expected, corrected, obs.facts)
    return post, pre_surprises + post_surprises


def correct(expected: Belief, before: Belief, facts: dict[str, Any]) -> tuple[Belief, list[Surprise]]:
    """Reconcile `expected` with measured `facts`.

    `before` is the belief from before the action. When a measurement shows the action didn't
    happen (e.g. a failed pick), the block falls back to where it was before.
    """
    b = expected.copy()
    surprises: list[Surprise] = []

    if "holding" in facts:
        seen = facts["holding"] if facts["holding"] in b.blocks else None
        if expected.holding() != seen:
            surprises.append(Surprise("holding", expected.holding(), seen))
        for c, loc in expected.blocks.items():
            if loc == GRIPPER and c != seen:
                b.blocks[c] = before.blocks[c] if before.blocks[c] != GRIPPER else UNKNOWN
        if seen is not None:
            b.blocks[seen] = GRIPPER

    for key, value in facts.items():
        if key.startswith("drawer_open:"):
            k, is_open = int(key.split(":", 1)[1]), bool(value)
            if b.drawers.get(k) != is_open:
                surprises.append(Surprise(key, b.drawers.get(k), is_open))
            b.drawers[k] = is_open
        elif key == "on_table" or key.startswith("contents:"):
            place = TABLE if key == "on_table" else drawer(int(key.split(":", 1)[1]))
            surprise = _reconcile_place(b, key, place, set(value))
            if surprise is not None:
                surprises.append(surprise)
    return b, surprises


def _reconcile_place(b: Belief, key: str, place: str, seen: set[str]) -> Surprise | None:
    """Make `b` agree with the set of blocks seen at `place`; return a Surprise if it didn't already."""
    seen &= set(b.blocks)
    believed_here = {c for c, loc in b.blocks.items() if loc == place}
    believed_elsewhere = {c for c, loc in b.blocks.items() if loc not in (place, UNKNOWN)}
    missing = believed_here - seen
    unexpected = seen & believed_elsewhere
    notes = [f"{c} expected at {b.blocks[c]}" for c in sorted(unexpected)]
    notes += [f"{c} missing" for c in sorted(missing)]
    for c in seen:
        b.blocks[c] = place
    for c in missing:
        b.blocks[c] = UNKNOWN
    if not notes:
        return None
    return Surprise(key, sorted(believed_here), sorted(seen), "; ".join(notes))
