"""Logic of the drawer service's open/close split, with fake hardware (no arm needed)."""

import asyncio

import pytest
from viam.proto.common import Pose

from modules import drawer_service as ds


class FakeGripper:
    def __init__(self):
        self.commands = []

    async def do_command(self, cmd):
        self.commands.append(cmd)
        return {}


class FakeLogger:
    def warning(self, msg):
        pass

    error = warning


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    """The service waits for real hardware to settle; with fakes there's nothing to wait for. Still yield
    to the event loop, as the service's polling loops rely on it."""
    real = asyncio.sleep

    async def fast(delay=0, result=None):
        return await real(0, result)
    monkeypatch.setattr(ds.asyncio, "sleep", fast)


def make_service(tmp_path):
    svc = object.__new__(ds.MyGenericService)
    svc.logger = FakeLogger()
    svc.gripper = FakeGripper()
    svc.moves = []
    svc.pos = (0.0, 0.0, 0.0)
    svc.open_state_file = str(tmp_path / "open_state.json")
    svc.open_drawers = svc._load_open_state()

    async def gripper_to(x, y, z, theta, straight=True):
        svc.moves.append((round(x, 1), round(y, 1), round(z, 1)))
        svc.pos = (x, y, z)

    async def end_position():
        return Pose(x=svc.pos[0], y=svc.pos[1], z=svc.pos[2])

    async def prepare(drawer, log):
        fk = {"open_dir": [1.0, 0.0], "knobs": [{"drawer": d, "x": 400.0, "y": -200.0 + 80 * (d - 1), "z": 40.0}
                                               for d in (1, 2, 3)]}
        k = fk["knobs"][drawer - 1]
        return fk, {"ok": True, "x": k["x"], "y": k["y"]}, 0.0, 90.0

    async def nothing(*a, **k):
        return None

    async def watched(x, y, z, t):
        svc.pos = (x, y, z)
        return {"contact": False}

    async def holding_pos():
        return 214.0

    async def loads():
        return [1.0] * 6

    svc._gripper_to = gripper_to
    svc.arm = type("A", (), {"get_end_position": staticmethod(end_position), "do_command": staticmethod(nothing)})()
    svc._prepare = prepare
    svc._speed = nothing
    svc._watched_move = watched
    svc._gpos = holding_pos
    svc._loads = loads
    return svc


def run(coro):
    return asyncio.run(coro)


def test_open_leaves_the_drawer_open_and_returns_a_drop_point_inside_it(tmp_path):
    svc = make_service(tmp_path)
    r = run(svc.open_drawer(1))
    assert r["success"]
    # closed knob at x=400; grasp 8 mm out; pulled 120 mm -> fingers let go at x=528, knob at 520.
    assert r["open_knob"] == [520.0, -200.0]
    # organizer face = closed knob (400) - protrusion (20) = 380; drop 70 mm out from it = 450.
    assert r["drop_point"] == {"x": 450.0, "y": -200.0, "z": 40.0}
    assert 1 in svc.open_drawers


def test_only_one_drawer_open_at_a_time(tmp_path):
    svc = make_service(tmp_path)
    run(svc.open_drawer(1))
    moves = len(svc.moves)
    r = run(svc.open_drawer(2))
    assert not r["success"] and "close it first" in r["error"]
    assert len(svc.moves) == moves  # refused before moving


def test_close_pushes_back_to_the_closed_position_and_forgets_the_drawer(tmp_path):
    svc = make_service(tmp_path)
    run(svc.open_drawer(1))
    r = run(svc.close_drawer(1))
    assert r["success"] and r["fully_closed"] is True
    assert (408.0, -200.0, ds.GRASP_Z_MM) in svc.moves  # straight back to the closed grasp point
    assert svc.open_drawers == {}


def test_close_without_open_refuses_without_moving(tmp_path):
    svc = make_service(tmp_path)
    r = run(svc.close_drawer(2))
    assert not r["success"] and svc.moves == []


def test_open_state_survives_a_restart(tmp_path):
    run(make_service(tmp_path).open_drawer(3))
    assert 3 in make_service(tmp_path).open_drawers


@pytest.mark.parametrize("action", ["status", "forget"])
def test_status_and_forget_never_move(tmp_path, action):
    svc = make_service(tmp_path)
    run(svc.do_command({"action": action}))
    assert svc.moves == [] and svc.gripper.commands == []
