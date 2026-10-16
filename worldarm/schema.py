"""Shared vocabulary between skills, the world model, the planner and the logs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

COLORS = ("blue", "yellow", "green")

# Block locations
TABLE = "table"
GRIPPER = "gripper"
UNKNOWN = "unknown"


def drawer(k: int) -> str:
    """Location string for drawer k (1 = top)."""
    return f"drawer:{k}"


def drawer_index(location: str) -> int | None:
    return int(location.split(":", 1)[1]) if location.startswith("drawer:") else None


class Act(str, Enum):
    OBSERVE_TABLE = "observe_table"
    OPEN = "open"
    CLOSE = "close"
    PICK = "pick"
    PLACE = "place"


@dataclass(frozen=True)
class Action:
    kind: Act
    color: str | None = None   # PICK
    drawer: int | None = None  # OPEN, CLOSE, PLACE

    def __str__(self) -> str:
        arg = self.color if self.color is not None else self.drawer
        return f"{self.kind.value}({arg})" if arg is not None else self.kind.value

    def to_json(self) -> dict:
        return {"kind": self.kind.value, "color": self.color, "drawer": self.drawer}


@dataclass
class Observation:
    """What a skill measured, in world-model fact keys.

    `pre_facts` were measured before the action changed anything and are checked against the
    prior belief; `facts` were measured after it and are checked against the prediction. A key
    that wasn't measured is simply absent, so it can never cause a surprise.

    Fact keys:
      holding          color in the gripper, or None
      drawer_open:<k>  bool
      contents:<k>     colors seen inside open drawer k
      on_table         colors seen on the table
    """

    facts: dict[str, Any] = field(default_factory=dict)
    pre_facts: dict[str, Any] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)  # free-form, logged (positions, reasons)


@dataclass(frozen=True)
class Surprise:
    """A measured fact that contradicts what the world model expected."""

    key: str
    expected: Any
    observed: Any
    note: str = ""
