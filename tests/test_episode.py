import asyncio

from worldarm.episode import run_episode
from worldarm.planner import Goal, satisfied
from worldarm.schema import TABLE, drawer
from worldarm.sim import FakeRobot, SimConfig
from worldarm.world_model import Belief

ASSIGN = {"blue": 1, "yellow": 2, "green": 3}
ON_TABLE = {c: TABLE for c in ASSIGN}
SORTED = {c: drawer(k) for c, k in ASSIGN.items()}
SEEDS = range(30)


def episode(world: dict, goal: str = "sort", cfg: SimConfig = SimConfig(), blind: bool = False,
            belief: Belief | None = None, moves: dict | None = None):
    robot = FakeRobot(world, 3, cfg)
    for color, location in (moves or {}).items():
        robot.move_block(color, location)
    b = belief or Belief.initial(ASSIGN, 3)
    result = asyncio.run(run_episode(robot, b, Goal.parse(goal), ASSIGN, blind=blind, say=lambda _: None))
    truth_ok = satisfied(Belief.from_json(result.truth), Goal.parse(goal), ASSIGN)
    return result, truth_ok


def test_sort_without_failures_takes_13_steps():
    result, truth_ok = episode(ON_TABLE)
    assert result.status == "done" and truth_ok and result.steps == 13 and result.surprises == 0


def test_sort_recovers_from_injected_failures():
    outcomes = [episode(ON_TABLE, cfg=SimConfig(p_pick_fail=0.3, p_open_fail=0.2, seed=s))[1] for s in SEEDS]
    assert sum(outcomes) >= 0.9 * len(SEEDS)


def test_verification_beats_blind_when_places_miss():
    """H1 in simulation: with missed places, checking the drawer beats trusting the prediction."""
    cfg = lambda s: SimConfig(p_place_miss=0.3, seed=s)  # noqa: E731
    checked = sum(episode(ON_TABLE, cfg=cfg(s))[1] for s in SEEDS)
    blind = sum(episode(ON_TABLE, cfg=cfg(s), blind=True)[1] for s in SEEDS)
    assert checked > blind


def test_blind_robot_believes_it_succeeded_when_it_did_not():
    for s in SEEDS:
        result, truth_ok = episode(ON_TABLE, cfg=SimConfig(p_place_miss=0.3, seed=s), blind=True)
        if not truth_ok:
            assert result.status == "done"
            return
    raise AssertionError("expected at least one blind failure across seeds")


def test_hidden_hand_is_noticed_and_the_block_is_found():
    remembered = Belief(dict(SORTED), {1: False, 2: False, 3: False})
    result, truth_ok = episode(SORTED, goal="find:blue", belief=remembered, moves={"blue": drawer(3)})
    assert result.status == "done" and truth_ok and result.surprises >= 1
    assert result.belief.blocks["blue"] == drawer(3)
