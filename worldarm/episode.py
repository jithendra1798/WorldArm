"""One episode: plan → predict → act → observe → correct, logged step by step."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable, Protocol

from .logger import EpisodeLogger
from .planner import Goal, next_action, satisfied
from .schema import TABLE, Act, Action, Observation, drawer
from .world_model import Belief, predict, preconditions, update


class Robot(Protocol):
    async def execute(self, action: Action) -> Observation: ...

    def truth(self) -> dict | None: ...


@dataclass
class EpisodeResult:
    status: str  # done | stuck | aborted | max_steps
    reason: str
    steps: int
    surprises: int
    belief: Belief
    truth: dict | None


async def run_episode(robot: Robot, belief: Belief, goal: Goal, assignment: dict[str, int],
                      logger: EpisodeLogger | None = None, *, blind: bool = False,
                      max_steps: int = 40, max_repeats: int = 3,
                      say: Callable[[str], None] = print) -> EpisodeResult:
    """Run until the planner has nothing left to do.

    blind=True is the ablation for H1: post-action measurements are ignored (the robot trusts its
    predictions). What verification would have caught is still logged as `ignored_surprises`.
    """
    searched: set[str] = set()
    n_surprises, repeats, last_failed = 0, 0, ""
    status, reason, steps = "max_steps", f"reached {max_steps} steps", 0

    for step in range(1, max_steps + 1):
        action = next_action(belief, goal, assignment, searched)
        if action is None:
            done = satisfied(belief, goal, assignment)
            status, reason = ("done", "goal satisfied") if done else ("stuck", "no action reaches the goal")
            break
        problems = preconditions(belief, action)
        if problems:
            status, reason = "stuck", f"{action}: {'; '.join(problems)}"
            break

        expected = predict(belief, action)
        t0 = time.time()
        try:
            obs = await robot.execute(action)
        except Exception as e:  # a failed skill ends the episode; the log records why
            status, reason = "aborted", f"{action} raised {type(e).__name__}: {e}"
            if logger:
                logger.step({"step": step, "action": action, "belief_before": belief, "error": reason})
            steps = step
            break

        ignored = []
        if blind and action.kind is not Act.OBSERVE_TABLE:
            _, ignored = update(belief, action, obs)
            post, surprises = update(belief, action, Observation(pre_facts=obs.pre_facts, detail=obs.detail))
        else:
            post, surprises = update(belief, action, obs)

        looked_at = {Act.OBSERVE_TABLE: TABLE, Act.OPEN: drawer(action.drawer or 0)}.get(action.kind)
        if surprises:
            searched.clear()  # the world changed under us; earlier searches are stale
        if looked_at:
            searched.add(looked_at)

        n_surprises += len(surprises)
        if surprises:
            repeats = repeats + 1 if str(action) == last_failed else 1
            last_failed = str(action)
        else:
            repeats, last_failed = 0, ""

        mark = "ok" if not surprises else "SURPRISE " + "; ".join(
            f"{s.key}: expected {s.expected}, saw {s.observed}" + (f" ({s.note})" if s.note else "")
            for s in surprises)
        say(f"{step:>3}  {str(action):<16} {mark}")
        if logger:
            logger.step({
                "step": step, "t_start": t0, "duration_s": round(time.time() - t0, 3),
                "action": action, "belief_before": belief, "predicted": expected,
                "observation": {"pre_facts": obs.pre_facts, "facts": obs.facts, "detail": obs.detail},
                "surprises": [asdict(s) for s in surprises],
                "ignored_surprises": [asdict(s) for s in ignored],
                "belief_after": post, "truth_after": robot.truth(),
            })
        belief, steps = post, step

        if repeats >= max_repeats:
            status, reason = "aborted", f"{action} failed {repeats} times in a row"
            break

    return EpisodeResult(status, reason, steps, n_surprises, belief, robot.truth())


def belief_accuracy(belief: Belief, truth: dict | None) -> float | None:
    """Fraction of blocks whose believed location matches the true one."""
    if not truth:
        return None
    blocks = truth["blocks"]
    return sum(belief.blocks.get(c) == loc for c, loc in blocks.items()) / len(blocks) if blocks else None
