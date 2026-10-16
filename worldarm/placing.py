"""Put the held block into its drawer. Owner: Sandeep (sort the blocks into the drawers by color).

The color → drawer assignment is WORLDARM_ASSIGNMENT (default blue:1, yellow:2, green:3).
v0 goes to `drop-k` and opens the gripper, so it only knows the block left the gripper. Looking
into the drawer to confirm it landed there (fact `contents:<k>`) is this track's first goal.
"""

from __future__ import annotations

from .manipulation import Manipulator
from .schema import Observation


async def place(m: Manipulator, k: int) -> Observation:
    await m.go(f"drop-{k}")
    await m.gripper.open()
    await m.settle()
    if not await m.holding():
        m.held = None
    return Observation(facts={"holding": m.held})
