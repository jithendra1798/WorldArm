"""The task B simulation study at minimal size: every study writes its CSV with the right columns and summary.md
is built from them; and the study's own roll loop and error measures reproduce sim.py's intercept_rate,
prediction_error and stop_error exactly (same seed, same numbers)."""

import json

import numpy as np
import pytest

from worldarm.ball import study
from worldarm.ball.intercept import ArmModel
from worldarm.ball.sim import intercept_rate, prediction_error, stop_error

TINY = study.Plan("tiny", lane_speeds=(150.0, 400.0), latencies=(0.45,), half_widths=(60.0,),
                  ball_speeds=((350.0, 900.0),), h5_n=6, horizons=(0.5, 1.0), watched=(0.3, 0.8),
                  decel_errors=(-0.5, 0.5), h4_n=20, h4_rate_n=4,
                  robust={"noise_mm": (4.0, 10.0), "fps": (23.0, 10.0)}, robust_n=4)
COLUMNS = {"h5_sweep.csv": study.H5_COLS, "h4_arrival.csv": study.ARRIVAL_COLS, "h4_stop.csv": study.STOP_COLS,
           "h4_misdecel.csv": study.MIS_COLS, "robust.csv": study.ROBUST_COLS, "needs.csv": study.NEEDS_COLS}


def test_every_study_writes_its_csv_and_the_summary(tmp_path):
    out = study.run(TINY, study.Baseline(), study.STUDIES, tmp_path / "run", charts=False, say=lambda _: None)
    for name, cols in COLUMNS.items():
        assert (out / name).read_text().splitlines()[0].split(",") == cols, name
    rows = {name: study.read_rows(out / name) for name in COLUMNS}
    assert len(rows["h5_sweep.csv"]) == 2 * 3        # 2 lane speeds x (predict, chase, the stronger chase)
    assert len(rows["h4_arrival.csv"]) == 2 * 2      # 2 models x 2 horizons
    assert len(rows["h4_stop.csv"]) == 2
    assert len(rows["h4_misdecel.csv"]) == 3 + 1     # -50%, 0, +50%, and constant velocity
    assert len(rows["robust.csv"]) == 4 * 2          # 4 levels x (predict, chase)
    assert len(rows["needs.csv"]) == 1 + 2           # one per latency, one per lane speed
    assert all(0.0 <= r["lo95"] <= r["hi95"] <= 1.0 for r in rows["h5_sweep.csv"])
    summary = (out / "summary.md").read_text()
    assert all(s in summary for s in ("For the slide", "## H5", "## H4", "## Robustness", "## What the arm needs"))
    meta = json.loads((out / "meta.json").read_text())
    assert {s["status"] for s in meta["studies"].values()} == {"done"} and meta["seeds"]["h5"]
    assert "commit" in meta["invocations"][-1]["git"] and "sim.py" in meta["invocations"][-1]["sources"]


def test_the_study_measures_what_sim_py_measures():
    base = study.Baseline()
    lane, cfg, arm = base.lane(), base.sim(), ArmModel(latency_s=0.45, speed_mm_s=250.0)
    for mode in ("predict", "chase"):
        r = study.simulate_rolls(lane, mode, cfg.true_decel, arm, cfg, 12, 7, 60.0, per_roll=False)
        ref = intercept_rate(lane, mode, cfg.true_decel, arm, cfg, n=12, seed=7, half_width_mm=60.0)
        assert (r.k, r.n) == (ref["defended"], ref["reached"])
    horizons = (1.0, 0.5)
    err, _ = study.arrival_errors(cfg, lane, (cfg.true_decel, 0.0), 25, 4, horizons)
    for k, decel in enumerate((cfg.true_decel, 0.0)):
        ref = prediction_error(decel, cfg, lane, n=25, seed=4, horizons=horizons)
        for j, h in enumerate(horizons):
            e = err[k, :, j]
            e = e[~np.isnan(e)]                      # both keep "it will stop short" (wrongly) as inf
            assert len(e) == ref[h][1] and np.median(e) == pytest.approx(ref[h][0], abs=1e-12)
    err, moving, _ = study.stop_errors(cfg, (cfg.true_decel,), 25, 5, (0.5,))
    e = err[0, 0][moving[0] & ~np.isnan(err[0, 0])]  # stop_error counts only balls still rolling
    ref = stop_error(cfg.true_decel, cfg, n=25, seed=5, after_s=0.5)
    assert len(e) == ref[1] and np.median(e) == pytest.approx(ref[0], abs=1e-12)


def test_paired_interval_is_tighter_when_both_goalies_face_the_same_rolls():
    rng = np.random.default_rng(0)
    a = rng.random(200) < 0.9
    b = a & (rng.random(200) < 0.8)                  # b blocks a subset of what a blocks: strongly paired
    d, lo, hi = study.gain_ci(a, b)
    _, u_lo, u_hi = study.gain_ci(a, b, paired=False)
    assert lo <= d <= hi and lo > 0 and hi - lo < u_hi - u_lo
