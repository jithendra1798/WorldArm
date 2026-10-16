"""Open and close the drawers. Owner: Riyaj.

Pull each drawer to 75% of its maximum travel and never further: the drawers detach, and the arm
can't put a detached drawer back. v0 replays recordings taught by hand (docs/viam-config.md §3):
  open   `knob-k-pre` → recording grasp-k → grab → recording pull-k (to the 75% mark) → open → back off
  close  close the fingers → recording push-k (starts at drop-k)
"""

from __future__ import annotations

from .manipulation import Manipulator
from .schema import Observation


async def open_drawer(m: Manipulator, k: int) -> Observation:
    await m.go(f"knob-{k}-pre")
    await m.gripper.open()
    await m.settle()
    await m.play(f"grasp-{k}")
    await m.gripper.grab()
    await m.settle()
    if not await m.holding():
        await m.gripper.open()
        await m.back_off()
        return Observation(facts={f"drawer_open:{k}": False},
                           detail={"reason": "fingers did not close on the knob"})
    await m.play(f"pull-{k}")
    await m.gripper.open()
    await m.settle()
    await m.back_off()
    return Observation(detail={"knob_grabbed": True})


async def close_drawer(m: Manipulator, k: int) -> Observation:
    await m.gripper.grab()  # close the fingers to push with
    await m.settle()
    await m.play(f"push-{k}")
    return Observation(detail={"pushed": True})
