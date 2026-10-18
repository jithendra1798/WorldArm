"""The sorting rules, one scenario each (no arm, no camera)."""

from collections import Counter

import numpy as np

from worldarm.blocks import Block
from worldarm.cell.config import CellConfig
from worldarm.see.organizer import DrawerView, Organizer
from worldarm.task.belief import SortBelief
from worldarm.task.policy import decide

CFG = CellConfig()
PULL = np.array([-0.866, 0.5])
ROW = np.array([-0.5, -0.866])


def blk(color, x=380.0, y=-260.0, base=2.0):
    return Block(color, (0, 0), np.zeros((2, 2), np.uint8), np.array([x, y, base + 30.0]), 60.0, 33.0, 41.0, 0.0, base)


def belief(states=("closed", "closed", "closed"), contents=(None, None, None), table=(), openings=None):
    b = SortBelief(CFG)
    views = []
    for k in (1, 2, 3):
        st = states[k - 1]
        o = (openings or {}).get(k, 120.0 if st == "open" else 50.0 if st == "partial" else 0.0)
        knob = np.array([200.0, -400.0, 44.0]) + np.append((k - 2) * 75.0 * ROW + o * PULL, 0.0)
        views.append(DrawerView(k, knob, o, st, float(knob[:2] @ ROW)))
    b.organizer = Organizer(PULL, ROW, 0.0, views, np.zeros((4, 2)), 97.0, "test")
    for k in (1, 2, 3):
        d = b.drawers[k]
        d.state, d.opening_mm = views[k - 1].state, views[k - 1].opening_mm
        c = contents[k - 1]
        if c is not None:
            d.contents = Counter(c)
            d.blocks = [blk(col, 150.0 + 10 * i, -450.0, 12.0) for i, col in enumerate(c)]
    b.table = [blk(c, 380.0 + 60 * i, -260.0) if isinstance(c, str) else c for i, c in enumerate(table)]
    return b


def test_no_organizer_in_view_stops():
    b = belief()
    b.organizer = None
    a = decide(b)
    assert a.kind == "stop" and a.success is False


def test_start_with_everything_unknown_opens_drawer_1_to_look():
    a = decide(belief(table=["yellow", "green"]))
    assert a.kind == "open" and a.drawer == 1


def test_open_empty_drawer_takes_the_most_common_color_without_a_drawer():
    a = decide(belief(states=("open", "closed", "closed"), contents=([], None, None),
                      table=["yellow", "green", "yellow"]))
    assert a.kind == "transfer" and a.drawer == 1 and a.color == "yellow"


def test_open_empty_drawer_skips_colors_that_already_have_a_drawer():
    a = decide(belief(states=("open", "closed", "closed"), contents=([], ["yellow"], None),
                      table=["yellow", "green", "yellow"]))
    assert a.kind == "transfer" and a.color == "green"


def test_open_drawer_with_one_color_takes_that_color():
    a = decide(belief(states=("closed", "open", "closed"), contents=(None, ["green"], None),
                      table=["yellow", "green"]))
    assert a.kind == "transfer" and a.drawer == 2 and a.color == "green"


def test_open_drawer_with_one_color_and_none_left_closes():
    a = decide(belief(states=("closed", "open", "closed"), contents=(None, ["green"], None), table=["yellow"]))
    assert a.kind == "close" and a.drawer == 2


def test_mixed_drawer_keeps_the_majority_and_takes_the_rest_out():
    a = decide(belief(states=("open", "closed", "closed"), contents=(["yellow", "green", "yellow"], None, None)))
    assert a.kind == "unload" and a.drawer == 1 and a.color == "green"


def test_mixed_drawer_tie_keeps_the_color_it_held():
    b = belief(states=("open", "closed", "closed"), contents=(["yellow", "green"], None, None))
    b.drawers[1].held_color = "green"
    a = decide(b)
    assert a.kind == "unload" and a.color == "yellow"


def test_partly_open_drawer_is_pulled_out_first():
    a = decide(belief(states=("closed", "closed", "partial"), contents=(None, None, None), table=["yellow"]))
    assert a.kind == "open" and a.drawer == 3


def test_two_open_closes_the_one_without_work():
    a = decide(belief(states=("open", "open", "closed"), contents=(["blue"], ["green"], None), table=["green"]))
    assert a.kind == "close" and a.drawer == 1


def test_none_open_opens_the_drawer_that_holds_a_table_color():
    a = decide(belief(contents=(["yellow"], [], ["green"]), table=["green"]))
    assert a.kind == "open" and a.drawer == 3


def test_none_open_fixes_a_known_mixed_drawer_first():
    a = decide(belief(contents=(["yellow"], ["green", "yellow"], []), table=["yellow"]))
    assert a.kind == "open" and a.drawer == 2


def test_none_open_new_color_goes_to_an_empty_drawer():
    a = decide(belief(contents=(["yellow"], [], ["green"]), table=["blue"]))
    assert a.kind == "open" and a.drawer == 2


def test_no_free_drawer_for_a_color_stops_unsorted():
    a = decide(belief(contents=(["yellow"], ["green"], ["black"]), table=["blue"]))
    assert a.kind == "stop" and a.success is False and "blue" in a.reason


def test_all_seen_single_colored_and_table_clear_is_sorted():
    # three colors use every drawer (two colors leave drawer 3 open: see the tests at the end)
    a = decide(belief(contents=(["yellow", "yellow"], ["green"], ["black"])))
    assert a.kind == "stop" and a.success is True


def test_unseen_drawers_are_checked_before_calling_it_sorted():
    a = decide(belief(contents=(["yellow"], None, [])))
    assert a.kind == "open" and a.drawer == 2


def test_a_block_that_keeps_failing_is_given_up_and_reported():
    b = belief(contents=(["yellow"], ["green"], []), table=["yellow"])
    for _ in range(CFG.task.max_tries_per_block):
        b._failed(b.table[0])
    a = decide(b)
    assert a.kind == "stop" and a.success is False and "couldn't pick yellow" in a.reason


def test_a_drawer_that_keeps_failing_is_skipped():
    b = belief(contents=(["yellow"], None, []), table=["green"])
    b.drawer_fails[2] = CFG.task.max_tries_per_drawer_move
    a = decide(b)
    assert a.kind == "open" and a.drawer == 3   # empty drawer 3 instead of the stuck, unseen drawer 2


# ---- two colors, three drawers: the colors fill drawers 1 and 2, drawer 3 ends open ----
def test_two_colors_first_color_goes_to_drawer_1():
    a = decide(belief(table=["yellow", "green", "yellow"]))
    assert a.kind == "open" and a.drawer == 1


def test_two_colors_second_color_goes_to_drawer_2_not_3():
    a = decide(belief(contents=(["yellow"], [], []), table=["green"]))
    assert a.kind == "open" and a.drawer == 2


def test_two_colors_open_empty_drawer_3_is_closed_while_blocks_remain():
    a = decide(belief(states=("closed", "closed", "open"), contents=(["yellow"], [], []), table=["green"]))
    assert a.kind == "close" and a.drawer == 3


def test_two_colors_sorted_opens_drawer_3_and_leaves_it_open():
    a = decide(belief(contents=(["yellow", "yellow"], ["green"], [])))
    assert a.kind == "open" and a.drawer == 3
    a = decide(belief(states=("closed", "closed", "open"), contents=(["yellow", "yellow"], ["green"], [])))
    assert a.kind == "stop" and a.success is True and "drawer 3 is left open" in a.reason


def test_two_colors_unseen_drawer_3_is_looked_into_last():
    a = decide(belief(contents=(["yellow"], ["green"], None), table=["green"]))
    assert a.kind == "open" and a.drawer == 2
    a = decide(belief(contents=(["yellow"], ["green"], None)))
    assert a.kind == "open" and a.drawer == 3


def test_three_colors_use_every_drawer_and_end_shut():
    a = decide(belief(contents=(["yellow"], ["green"], ["black"])))
    assert a.kind == "stop" and a.success is True


# ---- drawer 3 always stays free (up to two colors) ----
def test_one_color_also_leaves_drawer_3_open_at_the_end():
    a = decide(belief(contents=(["black"], [], [])))
    assert a.kind == "open" and a.drawer == 3
    a = decide(belief(states=("closed", "closed", "open"), contents=(["black"], [], [])))
    assert a.kind == "stop" and a.success is True and "drawer 3 is left open" in a.reason


def test_blocks_already_in_drawer_3_are_taken_out():
    # the 15:55 run: drawer 3 held a black block from before, so it was treated as black's drawer
    a = decide(belief(contents=(["black"], [], ["black"])))
    assert a.kind == "open" and a.drawer == 3 and "come out" in a.reason
    a = decide(belief(states=("closed", "closed", "open"), contents=(["black"], [], ["black"])))
    assert a.kind == "unload" and a.drawer == 3 and a.color == "black"


def test_a_color_only_in_drawer_3_still_gets_drawer_1_or_2():
    a = decide(belief(states=("open", "closed", "closed"), contents=([], [], ["yellow"]), table=["black"]))
    assert a.kind == "transfer" and a.drawer == 1 and a.color == "black"


def test_drawer_3_is_never_a_home_for_the_table_color():
    a = decide(belief(contents=([], [], ["black"]), table=["black"]))
    assert not (a.kind == "open" and a.drawer == 3 and "holds black" in a.reason)
