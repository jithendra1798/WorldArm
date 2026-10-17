"""sort_bench's episode loop against a scripted table (no arm): retries, close checks, aborts."""

import asyncio
import os
from types import SimpleNamespace

import numpy as np
import pytest

ROW = [[0.0, 0.0], [110.0, 0.0], [220.0, 0.0]]       # every drawer shut
SLID = [[0.0, 0.0], [110.0, 120.0], [220.0, 0.0]]    # drawer 2 still out after its "close"


@pytest.fixture
def sort_bench():
    saved = dict(os.environ)  # importing sort_bench sets armfarm03 defaults; keep them out of other tests
    import sort_bench
    yield sort_bench
    os.environ.clear()
    os.environ.update(saved)


def view(colors, knobs):
    blocks = [SimpleNamespace(color=c, center=np.array([300.0 + 60 * i, -200.0, 30.0]), height_mm=60.0,
                              short_mm=33.0, long_mm=41.0, short_axis_deg=0.0) for i, c in enumerate(colors)]
    return SimpleNamespace(blocks=blocks, report={"knobs": knobs})


class Table:
    """What each overview look sees and what each transfer does, in order; records the arm's actions."""

    def __init__(self, looks, transfers):
        self.looks, self.transfers, self.open, self.calls = iter(looks), iter(transfers), {}, []

    async def scan(self, assignment, save_to=None):
        self.calls.append("look")
        return next(self.looks)

    async def open_drawers(self):
        return dict(self.open)

    async def ensure_open(self, k):
        reused = str(k) in self.open
        for other in [d for d in self.open if d != str(k)]:
            await self.close(int(other))
        self.open[str(k)] = {}
        self.calls.append(f"open {k}" + (" (reused)" if reused else ""))
        return True, {"drop_point": {"x": 0.0, "y": 0.0, "z": 40.0}, "open_dir": [1.0, 0.0]}, "ok"

    async def transfer(self, frame, block, info):
        result = next(self.transfers)
        self.calls.append(f"{block.color}: {result}")
        return result, {"contact_z": None}

    async def check_in_drawer(self, color, info):
        return True, 0.5

    async def close(self, k):
        self.open.pop(str(k), None)
        self.calls.append(f"close {k}")
        return {"success": True}


def run(sort_bench, table, retries=1):
    bench = sort_bench.Bench.__new__(sort_bench.Bench)
    bench.args = SimpleNamespace(retries=retries)
    bench.wrist_for = lambda block: (0.0, 180.0)
    for name in ("scan", "open_drawers", "ensure_open", "transfer", "check_in_drawer", "close"):
        setattr(bench, name, getattr(table, name))
    steps = []
    logger = SimpleNamespace(dir="unused", step=steps.append)
    status, reason, _, truth = asyncio.run(bench.episode({"yellow": 2}, logger))
    return status, reason, truth, steps


def test_a_missed_pick_is_retried_with_the_drawer_left_open(sort_bench):
    table = Table([view(["yellow"], ROW), view(["yellow"], SLID), view([], ROW), view([], ROW)],
                  ["missed_grasp", "ok"])
    status, _, truth, steps = run(sort_bench, table)
    assert table.calls == ["look", "open 2", "yellow: missed_grasp", "look", "open 2 (reused)", "yellow: ok",
                           "close 2", "look", "look"]
    assert status == "done" and truth["blocks"] == {"yellow": "drawer:2"} and truth["all_closed"]
    assert [s["action"]["kind"] for s in steps].count("close-check") == 1


def test_a_close_that_left_the_drawer_out_stops_the_episode(sort_bench):
    table = Table([view(["yellow"], ROW), view([], SLID)], ["ok"])
    status, reason, truth, _ = run(sort_bench, table)
    assert table.calls == ["look", "open 2", "yellow: ok", "close 2", "look"]
    assert status == "aborted" and "out of line" in reason and not truth["all_closed"]


def test_a_drawer_open_at_the_start_stops_before_any_motion(sort_bench):
    table = Table([view(["yellow"], SLID)], [])
    status, reason, _, _ = run(sort_bench, table)
    assert table.calls == ["look"] and status == "aborted" and "Close it by hand" in reason


def test_out_of_retries_the_drawer_is_closed_and_the_block_counted_on_the_table(sort_bench):
    table = Table([view(["yellow"], ROW), view(["yellow"], SLID), view(["yellow"], ROW), view(["yellow"], ROW)],
                  ["missed_grasp", "missed_grasp"])
    status, _, truth, _ = run(sort_bench, table)
    assert table.calls == ["look", "open 2", "yellow: missed_grasp", "look", "open 2 (reused)",
                           "yellow: missed_grasp", "close 2", "look", "look"]
    assert status == "done" and truth["blocks"] == {"yellow": "table"} and truth["all_closed"]
