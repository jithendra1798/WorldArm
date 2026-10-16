"""Read-only: simulated H1 (drawer check vs. hand check only). Moves nothing."""

import subprocess
import sys

import matplotlib.pyplot as plt

RUNS, P_FAIL = 200, 0.15


def truly_sorted(extra: list[str]) -> int:
    ok = 0
    for seed in range(RUNS):
        out = subprocess.run([sys.executable, "run_episode.py", "--fake", "--p-fail", str(P_FAIL),
                              "--seed", str(seed), *extra], capture_output=True, text=True).stdout
        ok += "truth success: True" in out
    return ok


hand = truly_sorted(["--like-real-v0"])
drawer = truly_sorted([])
print(f"hand check only:   {hand}/{RUNS}")
print(f"with drawer check: {drawer}/{RUNS}")

plt.bar(["Hand check only", "With drawer check"], [100 * hand / RUNS, 100 * drawer / RUNS],
        color=["#C62828", "#2E7D32"])
plt.ylabel("Truly sorted (%)")
plt.ylim(0, 100)
plt.title(f"Simulated sorting, {RUNS} runs, {int(P_FAIL * 100)}% failure rate")
plt.savefig("captures/sim_h1.png", dpi=150, bbox_inches="tight")
print("chart saved to captures/sim_h1.png")