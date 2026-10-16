"""Choose the next action from the belief.

v0 is a fixed policy: sort each table block into its color's drawer (open → pick → place →
close), and search for unknown blocks (table first, then one drawer at a time).
"""

from __future__ import annotations

from dataclasses import dataclass

from .schema import GRIPPER, TABLE, UNKNOWN, Act, Action, drawer, drawer_index
from .world_model import Belief


@dataclass(frozen=True)
class Goal:
    kind: str                # "sort" | "find"
    color: str | None = None  # for "find"

    @classmethod
    def parse(cls, text: str) -> Goal:
        if text == "sort":
            return cls("sort")
        if text.startswith("find:"):
            return cls("find", text.split(":", 1)[1])
        raise ValueError(f"unknown goal {text!r}; use 'sort' or 'find:<color>'")

    def __str__(self) -> str:
        return self.kind if self.color is None else f"{self.kind}:{self.color}"


def satisfied(belief: Belief, goal: Goal, assignment: dict[str, int]) -> bool:
    """Whether the *belief* says the goal holds. Judge real success against ground truth."""
    if goal.kind == "sort":
        return (all(loc == drawer(assignment[c]) for c, loc in belief.blocks.items())
                and not belief.open_drawers())
    loc = belief.blocks[goal.color]
    k = drawer_index(loc)
    return loc == TABLE or (k is not None and belief.drawers[k])


def next_action(belief: Belief, goal: Goal, assignment: dict[str, int], searched: set[str]) -> Action | None:
    """The next action toward `goal`, or None when there's nothing left to do.

    `searched` holds the places already looked at for unknown blocks in this episode.
    """
    if goal.kind == "sort":
        return _sort(belief, assignment, searched)
    return _find(belief, goal.color, searched)


def _sort(b: Belief, assignment: dict[str, int], searched: set[str]) -> Action | None:
    held = b.holding()
    if held is not None:
        k = assignment[held]
        return Action(Act.PLACE, drawer=k) if b.drawers[k] else None  # None: stuck holding a block
    on_table = b.at(TABLE)
    for k in b.open_drawers():
        waiting = [c for c in on_table if assignment[c] == k]
        return Action(Act.PICK, color=waiting[0]) if waiting else Action(Act.CLOSE, drawer=k)
    if on_table:
        first = min(on_table, key=lambda c: (assignment[c], c))
        return Action(Act.OPEN, drawer=assignment[first])
    return _search(b, searched)


def _find(b: Belief, color: str, searched: set[str]) -> Action | None:
    loc = b.blocks[color]
    k = drawer_index(loc)
    if k is not None:
        if b.drawers[k]:
            return None
        others = b.open_drawers()
        return Action(Act.CLOSE, drawer=others[0]) if others else Action(Act.OPEN, drawer=k)
    if loc in (TABLE, GRIPPER):
        return None
    return _search(b, searched)


def _search(b: Belief, searched: set[str]) -> Action | None:
    """Look for unknown blocks: the table first, then each drawer, one open at a time."""
    if not b.at(UNKNOWN):
        return None
    if TABLE not in searched:
        return Action(Act.OBSERVE_TABLE)
    if b.open_drawers():
        return Action(Act.CLOSE, drawer=b.open_drawers()[0])
    for k in sorted(b.drawers):
        if drawer(k) not in searched:
            return Action(Act.OPEN, drawer=k)
    return None
