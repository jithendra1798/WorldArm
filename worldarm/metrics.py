"""Scoreboard over logged episodes in runs/."""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate(k: int, n: int) -> str:
    if n == 0:
        return "—"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k / n:.0%}  [{lo:.0%}, {hi:.0%}]"


def load_runs(root: Path | str = "runs") -> list[dict]:
    runs = []
    for d in sorted(Path(root).glob("*/")):
        if not (d / "summary.json").exists():
            continue
        steps = [json.loads(line) for line in (d / "episode.jsonl").read_text().splitlines() if line]
        runs.append({"dir": d.name, "meta": json.loads((d / "meta.json").read_text()),
                     "summary": json.loads((d / "summary.json").read_text()), "steps": steps})
    return runs


def group_key(run: dict) -> str:
    m = run["meta"]
    return f"{m['mode']}  goal={m['goal']}" + ("  BLIND" if m.get("blind") else "")


def scoreboard(runs: list[dict]) -> str:
    if not runs:
        return "No finished episodes in runs/ yet."
    out = []
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        groups[group_key(r)].append(r)
    for key, rs in groups.items():
        sums = [r["summary"] for r in rs]
        if str(rs[0]["meta"].get("goal", "")).startswith("bench:"):  # skill benchmark: tries, not episodes
            out.append(f"\n== {key}  ({len(rs)} benchmark runs)")
        else:
            out.append(f"\n== {key}  ({len(rs)} episodes)")
            judged = [s for s in sums if s.get("truth_success") is not None]
            out.append(f"  success (truth)          {rate(sum(s['truth_success'] for s in judged), len(judged))}")
            out.append(f"  done (robot believes)    {rate(sum(s['status'] == 'done' for s in sums), len(sums))}")
            accs = [s["belief_accuracy"] for s in sums if s.get("belief_accuracy") is not None]
            if accs:
                out.append(f"  belief accuracy          mean {statistics.mean(accs):.0%} over {len(accs)} episodes")
            durations = [s["duration_s"] for s in sums]
            out.append(f"  episode time             median {statistics.median(durations):.1f} s")

        attempts, verified, ok = Counter(), Counter(), Counter()
        surprise_keys, ignored_keys = Counter(), Counter()
        for r in rs:
            for st in r["steps"]:
                if "action" not in st or "observation" not in st:
                    continue
                kind = st["action"]["kind"]
                attempts[kind] += 1
                if st["observation"]["facts"] or st["observation"]["pre_facts"]:
                    verified[kind] += 1
                    ok[kind] += not st["surprises"]
                surprise_keys.update(s["key"].split(":")[0] for s in st["surprises"])
                ignored_keys.update(s["key"].split(":")[0] for s in st.get("ignored_surprises", []))
        out.append("  per action (success among verified steps):")
        for kind in sorted(attempts):
            unverified = attempts[kind] - verified[kind]
            note = f"   ({unverified} unverified)" if unverified else ""
            out.append(f"    {kind:<14} {rate(ok[kind], verified[kind])}{note}")
        if surprise_keys:
            out.append("  surprises: " + ", ".join(f"{k} ×{n}" for k, n in surprise_keys.most_common()))
        if ignored_keys:
            out.append("  ignored (blind): " + ", ".join(f"{k} ×{n}" for k, n in ignored_keys.most_common()))
    return "\n".join(out)
