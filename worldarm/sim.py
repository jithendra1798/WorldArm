"""A simulated robot with the same interface as the real one, for working without an arm.

It keeps the true world separately from the robot's belief, injects failures with seeded
probabilities, and lets a "hidden hand" move blocks behind the robot's back (the surprise demo).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .schema import GRIPPER, TABLE, Act, Action, Observation, drawer

OFFSTAGE = "offstage"  # a block that exists but can never be seen (e.g. not detected in a dry run)


@dataclass(frozen=True)
class SimConfig:
    p_pick_fail: float = 0.0
    p_place_miss: float = 0.0       # the block lands on the table instead of in the drawer
    p_open_fail: float = 0.0
    p_close_fail: float = 0.0
    sees_contents: bool = True      # looks into a drawer after opening it or placing into it
    sees_drawer_state: bool = True  # measures whether a drawer actually opened/closed
    seed: int = 0

    @classmethod
    def like_real_v0(cls, **kw) -> SimConfig:
        """Only the checks the real v0 skills have: the gripper's holding sensor and table detections."""
        return cls(sees_contents=False, sees_drawer_state=False, **kw)


class FakeRobot:
    def __init__(self, world: dict[str, str], n_drawers: int, cfg: SimConfig = SimConfig()):
        self.blocks = dict(world)
        self.drawers = {k: False for k in range(1, n_drawers + 1)}
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)

    def move_block(self, color: str, location: str) -> None:
        """The hidden hand: change the world without telling the robot."""
        self.blocks[color] = location

    def truth(self) -> dict:
        return {"blocks": {c: loc for c, loc in self.blocks.items() if loc != OFFSTAGE},
                "drawers": {str(k): v for k, v in self.drawers.items()}}

    def _fails(self, p: float) -> bool:
        return self.rng.random() < p

    def _at(self, location: str) -> list[str]:
        return sorted(c for c, loc in self.blocks.items() if loc == location)

    def _held(self) -> str | None:
        held = self._at(GRIPPER)
        return held[0] if held else None

    def _drawer_facts(self, k: int) -> dict:
        facts = {}
        if self.cfg.sees_drawer_state:
            facts[f"drawer_open:{k}"] = self.drawers[k]
        if self.cfg.sees_contents and self.drawers[k]:
            facts[f"contents:{k}"] = self._at(drawer(k))
        return facts

    async def execute(self, action: Action) -> Observation:
        k, color, cfg = action.drawer, action.color, self.cfg
        if action.kind is Act.OBSERVE_TABLE:
            return Observation(facts={"on_table": self._at(TABLE)})

        if action.kind is Act.PICK:
            pre = {"on_table": self._at(TABLE)}
            if self._held() is None and self.blocks.get(color) == TABLE and not self._fails(cfg.p_pick_fail):
                self.blocks[color] = GRIPPER
            return Observation(pre_facts=pre, facts={"holding": self._held()})

        if action.kind is Act.PLACE:
            held = self._held()
            if held is not None:
                into_drawer = self.drawers[k] and not self._fails(cfg.p_place_miss)
                self.blocks[held] = drawer(k) if into_drawer else TABLE
            facts = {"holding": None}
            if cfg.sees_contents and self.drawers[k]:
                facts[f"contents:{k}"] = self._at(drawer(k))
            return Observation(facts=facts)

        if action.kind is Act.OPEN:
            blocked = self._held() is not None or any(o for j, o in self.drawers.items() if j != k)
            if not blocked and not self._fails(cfg.p_open_fail):
                self.drawers[k] = True
            return Observation(facts=self._drawer_facts(k))

        if action.kind is Act.CLOSE:
            if not self._fails(cfg.p_close_fail):
                self.drawers[k] = False
            return Observation(facts=self._drawer_facts(k))

        raise ValueError(f"unknown action {action}")
