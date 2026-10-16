from worldarm.schema import GRIPPER, TABLE, UNKNOWN, Act, Action, Observation, drawer
from worldarm.world_model import Belief, preconditions, update

COLORS = ["blue", "yellow", "green"]


def belief(blocks: dict, open_drawers=()) -> Belief:
    return Belief(dict(blocks), {k: k in open_drawers for k in (1, 2, 3)})


def test_first_look_at_table_is_a_discovery_not_a_surprise():
    b = Belief.initial(COLORS, 3)
    post, surprises = update(b, Action(Act.OBSERVE_TABLE), Observation(facts={"on_table": ["blue", "green"]}))
    assert surprises == []
    assert post.blocks == {"blue": TABLE, "yellow": UNKNOWN, "green": TABLE}


def test_successful_pick_matches_prediction():
    b = belief({"blue": TABLE}, open_drawers=[1])
    obs = Observation(pre_facts={"on_table": ["blue"]}, facts={"holding": "blue"})
    post, surprises = update(b, Action(Act.PICK, color="blue"), obs)
    assert surprises == [] and post.blocks["blue"] == GRIPPER


def test_failed_pick_is_one_surprise_and_block_stays_on_table():
    b = belief({"blue": TABLE}, open_drawers=[1])
    obs = Observation(pre_facts={"on_table": ["blue"]}, facts={"holding": None})
    post, surprises = update(b, Action(Act.PICK, color="blue"), obs)
    assert [s.key for s in surprises] == ["holding"]
    assert post.blocks["blue"] == TABLE


def test_block_gone_before_pick_is_one_surprise_and_becomes_unknown():
    b = belief({"blue": TABLE, "yellow": TABLE}, open_drawers=[1])
    obs = Observation(pre_facts={"on_table": ["yellow"]}, facts={"holding": None})
    post, surprises = update(b, Action(Act.PICK, color="blue"), obs)
    assert [s.key for s in surprises] == ["on_table"]
    assert post.blocks == {"blue": UNKNOWN, "yellow": TABLE}


def test_place_confirmed_by_looking_into_the_drawer():
    b = belief({"blue": GRIPPER}, open_drawers=[1])
    obs = Observation(facts={"holding": None, "contents:1": ["blue"]})
    post, surprises = update(b, Action(Act.PLACE, drawer=1), obs)
    assert surprises == [] and post.blocks["blue"] == drawer(1)


def test_missed_place_is_caught_by_looking_into_the_drawer():
    b = belief({"blue": GRIPPER}, open_drawers=[1])
    obs = Observation(facts={"holding": None, "contents:1": []})
    post, surprises = update(b, Action(Act.PLACE, drawer=1), obs)
    assert [s.key for s in surprises] == ["contents:1"]
    assert post.blocks["blue"] == UNKNOWN


def test_hidden_hand_is_noticed_when_the_drawer_opens():
    b = belief({"blue": drawer(1), "yellow": drawer(2), "green": drawer(3)})
    obs = Observation(facts={"drawer_open:2": True, "contents:2": ["blue", "yellow"]})
    post, surprises = update(b, Action(Act.OPEN, drawer=2), obs)
    assert len(surprises) == 1 and "blue expected at drawer:1" in surprises[0].note
    assert post.blocks["blue"] == drawer(2) and post.drawers[2]


def test_drawer_that_did_not_open_is_a_surprise():
    b = belief({"blue": TABLE})
    post, surprises = update(b, Action(Act.OPEN, drawer=1), Observation(facts={"drawer_open:1": False}))
    assert [s.key for s in surprises] == ["drawer_open:1"] and not post.drawers[1]


def test_unmeasured_outcome_trusts_the_prediction():
    b = belief({"blue": TABLE}, open_drawers=[2])
    post, surprises = update(b, Action(Act.CLOSE, drawer=2), Observation())
    assert surprises == [] and not post.drawers[2]


def test_only_one_drawer_open_at_a_time():
    assert preconditions(belief({"blue": TABLE}, open_drawers=[1]), Action(Act.OPEN, drawer=2))


def test_drawer_moves_need_an_empty_gripper():
    assert preconditions(belief({"blue": GRIPPER}), Action(Act.OPEN, drawer=1))
    assert preconditions(belief({"blue": GRIPPER}, open_drawers=[1]), Action(Act.CLOSE, drawer=1))
