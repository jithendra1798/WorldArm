from worldarm.planner import Goal, next_action, satisfied
from worldarm.schema import TABLE, UNKNOWN, Act, Action, drawer
from worldarm.world_model import Belief, predict

ASSIGN = {"blue": 1, "yellow": 2, "green": 3}


def plan_until_done(b: Belief, goal: Goal, searched: set[str]) -> list[str]:
    """Plan assuming every prediction comes true."""
    actions = []
    while (a := next_action(b, goal, ASSIGN, searched)) is not None and len(actions) < 50:
        actions.append(str(a))
        b = predict(b, a)
    assert satisfied(b, goal, ASSIGN)
    return actions


def test_sort_starts_by_looking_at_the_table():
    assert next_action(Belief.initial(ASSIGN, 3), Goal("sort"), ASSIGN, set()) == Action(Act.OBSERVE_TABLE)


def test_sort_plan_opens_picks_places_closes_per_drawer():
    b = Belief({c: TABLE for c in ASSIGN}, {1: False, 2: False, 3: False})
    assert plan_until_done(b, Goal("sort"), {TABLE}) == [
        "open(1)", "pick(blue)", "place(1)", "close(1)",
        "open(2)", "pick(yellow)", "place(2)", "close(2)",
        "open(3)", "pick(green)", "place(3)", "close(3)",
    ]


def test_find_opens_the_remembered_drawer():
    b = Belief({c: drawer(k) for c, k in ASSIGN.items()}, {1: False, 2: False, 3: False})
    assert plan_until_done(b, Goal.parse("find:green"), set()) == ["open(3)"]


def test_find_unknown_searches_table_then_one_drawer_at_a_time():
    b = Belief({"blue": UNKNOWN, "yellow": drawer(2), "green": drawer(3)}, {1: False, 2: False, 3: False})
    goal, searched = Goal.parse("find:blue"), set()
    assert next_action(b, goal, ASSIGN, searched) == Action(Act.OBSERVE_TABLE)
    searched.add(TABLE)
    assert next_action(b, goal, ASSIGN, searched) == Action(Act.OPEN, drawer=1)
    searched.add(drawer(1))
    b.drawers[1] = True
    assert next_action(b, goal, ASSIGN, searched) == Action(Act.CLOSE, drawer=1)
