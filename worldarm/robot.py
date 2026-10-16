"""The real arm: wires perception (vision.py) and the skills (pick.py, drawers.py, placing.py)
into the Robot interface the episode loop uses. Owner: Jithendra.

New measurements are wired in here: a skill's Observation can be extended with more facts (for
example `contents:<k>` from looking into a drawer) without touching the world model.
"""

from __future__ import annotations

from viam.robot.client import RobotClient
from viam.services.vision import VisionClient

from . import drawers, pick, placing
from .config import Config
from .manipulation import Manipulator
from .schema import Act, Action, Observation
from .vision import detect_blocks


class ViamRobot:
    def __init__(self, machine: RobotClient, cfg: Config):
        self.machine, self.cfg = machine, cfg
        self.vision = VisionClient.from_robot(machine, cfg.vision)  # fails early if it isn't configured
        self.skills = Manipulator(machine, cfg)

    def truth(self) -> None:
        return None  # the real world doesn't tell us; ask the operator (--ask-truth)

    async def detect_blocks(self, move: bool = True) -> list[dict]:
        if move:
            await self.skills.go(self.cfg.pose_table_view)
        return await detect_blocks(self.machine, self.cfg, self.vision)

    async def execute(self, action: Action) -> Observation:
        if action.kind is Act.OBSERVE_TABLE:
            blocks = await self.detect_blocks()
            return Observation(facts={"on_table": sorted({b["color"] for b in blocks})},
                               detail={"blocks": blocks})

        if action.kind is Act.PICK:
            blocks = await self.detect_blocks()
            seen = sorted({b["color"] for b in blocks})
            candidates = [b for b in blocks if b["color"] == action.color]
            if not candidates:
                return Observation(pre_facts={"on_table": seen}, facts={"holding": None},
                                   detail={"reason": f"{action.color} not detected"})
            obs = await pick.pick_block(self.skills, max(candidates, key=lambda b: b["points_bytes"]))
            obs.pre_facts["on_table"] = seen
            return obs

        if action.kind is Act.PLACE:
            return await placing.place(self.skills, action.drawer)

        if action.kind in (Act.OPEN, Act.CLOSE):
            if self.cfg.containers == "bins":  # open-top bins: nothing to open or close
                return Observation(detail={"bins": True})
            if action.kind is Act.OPEN:
                return await drawers.open_drawer(self.skills, action.drawer)
            return await drawers.close_drawer(self.skills, action.drawer)

        raise ValueError(f"unknown action {action}")
