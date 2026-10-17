"""Camera correction: recover a known camera-mount error from looks with the wrist turned, or from
blocks the arm placed itself."""

import numpy as np
import pytest

from worldarm.cell.calibration import CameraCorrection, Sighting, fit_correction, sightings_from_looks, world_of
from worldarm.cell.geometry import rot_from_ov

TRUE = CameraCorrection((0.012, -0.018, 0.006), (19.0, -11.0, 3.0))  # ~1 deg tilt + ~2 cm offset
BLOCKS = np.array([[380.0, -240.0, 30.0], [300.0, -120.0, 30.0], [460.0, -330.0, 15.0], [250.0, -300.0, 25.0]])


def camera_pose(x, y, z, wrist_deg):
    """Configured camera-to-world for a camera looking straight down, offset 80 mm from the wrist axis."""
    R = rot_from_ov(0, 0, -1, wrist_deg)                  # camera axes follow the wrist
    offset = R @ np.array([-80.0, 15.0, 0.0])               # camera sits off the wrist axis
    return R, np.array([x, y, z]) + offset


def sight(R_cfg, t_cfg, world, key, rng, truth=None):
    """What the camera reports for `world` when it really sits at the corrected pose."""
    R_true, t_true = TRUE.apply(R_cfg, t_cfg)
    p = R_true.T @ (world - t_true) + rng.normal(0, 0.7, 3)
    return Sighting(p, R_cfg, t_cfg, key, truth)


def looks(wrists=(0.0, 180.0), heights=(480.0, 330.0)):
    return [camera_pose(360.0, -250.0, h, w) for w in wrists for h in heights]


def xy_error_field(corr, R, t):
    """How far each block lands from the truth under `corr`, for one look."""
    errs = []
    for b in BLOCKS:
        R_true, t_true = TRUE.apply(R, t)
        p = R_true.T @ (b - t_true)
        errs.append(np.linalg.norm((world_of(Sighting(p, R, t, "x"), corr) - b)[:2]))
    return np.array(errs)


def test_uncorrected_error_is_large():
    R, t = camera_pose(360.0, -250.0, 480.0, 0.0)
    assert xy_error_field(CameraCorrection(), R, t).mean() > 15.0


def test_wrist_flips_alone_fix_the_xy_error_at_every_look():
    rng = np.random.default_rng(1)
    s = [sight(R, t, b, f"b{i}", rng) for R, t in looks() for i, b in enumerate(BLOCKS)]
    corr, report = fit_correction(s)
    assert report["rms_after_mm"] < 2.0 < report["rms_before_mm"]
    for R, t in looks(wrists=(0.0, 90.0, 180.0, -90.0), heights=(480.0, 400.0, 330.0)):
        assert xy_error_field(corr, R, t).max() < 3.0  # also at wrist angles and heights not used to fit


def test_blocks_placed_by_the_arm_fix_it_from_one_wrist_angle():
    rng = np.random.default_rng(2)
    s = [sight(R, t, b, f"b{i}", rng, truth=b) for R, t in looks(wrists=(0.0,)) for i, b in enumerate(BLOCKS)]
    corr, report = fit_correction(s)
    assert report["rms_after_mm"] < 2.0
    R, t = camera_pose(360.0, -250.0, 480.0, 0.0)
    assert xy_error_field(corr, R, t).max() < 2.5


def test_correction_round_trips_through_json():
    c = CameraCorrection.from_json(TRUE.to_json())
    R, t = camera_pose(300, -200, 400, 30)
    assert np.allclose(c.apply(R, t)[1], TRUE.apply(R, t)[1], atol=0.01)
    assert CameraCorrection.from_json(None).is_identity


def test_too_little_data_is_refused():
    rng = np.random.default_rng(3)
    R, t = camera_pose(360.0, -250.0, 480.0, 0.0)
    with pytest.raises(ValueError):
        fit_correction([sight(R, t, BLOCKS[0], "a", rng)])


def test_blocks_are_matched_across_looks_by_color_and_position():
    rng = np.random.default_rng(4)
    colors = ["green", "yellow", "black", "green"]
    raw = []
    for R, t in looks():
        pts = []
        for c, b in zip(colors, BLOCKS):
            R_true, t_true = TRUE.apply(R, t)
            pts.append((c, R_true.T @ (b - t_true) + rng.normal(0, 0.5, 3)))
        raw.append((pts, R, t))
    s = sightings_from_looks(raw)
    assert len(s) == len(colors) * len(raw)
    corr, report = fit_correction(s)
    assert report["rms_after_mm"] < 2.0
