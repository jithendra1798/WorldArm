"""Wrist-camera correction: where the camera really sits on the gripper vs. the frame config.

The drawer service found this camera's positions off by ~24 mm and hard-codes a fix per wrist angle
(OVERHEAD_CORRECTION); its two-wrist knob looks disagreed by 47 mm. An error in how the camera is
mounted turns with the wrist, so two looks at the same block with the wrist half a turn apart see it
in two places. Fitting a small rotation + translation of the camera (in its own frame) that makes
every look agree - and, when the arm placed the block itself, agree with where it put it - fixes
every look the arm takes, at any height or wrist angle.

    world = R_cfg @ (R_d @ p_cam + tau) + t_cfg      (R_d, tau: the correction)
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .geometry import rotvec_matrix


@dataclass
class CameraCorrection:
    rotvec: tuple[float, float, float] = (0.0, 0.0, 0.0)       # radians, camera frame
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)  # mm, camera frame

    def apply(self, R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Configured camera-to-world (R, t) -> corrected camera-to-world."""
        Rd = rotvec_matrix(np.array(self.rotvec))
        return R @ Rd, R @ np.array(self.translation, float) + t

    def to_json(self) -> dict:
        return {"rotvec": [round(float(v), 6) for v in self.rotvec],
                "translation": [round(float(v), 3) for v in self.translation]}

    @classmethod
    def from_json(cls, d: dict | None) -> CameraCorrection:
        if not d:
            return cls()
        return cls(tuple(d.get("rotvec", (0, 0, 0))), tuple(d.get("translation", (0, 0, 0))))

    @property
    def is_identity(self) -> bool:
        return not any(abs(v) > 1e-9 for v in (*self.rotvec, *self.translation))


@dataclass
class Sighting:
    """One point measured in one look: in the camera frame, with that look's configured camera pose."""
    p_cam: np.ndarray
    R: np.ndarray
    t: np.ndarray
    key: str                       # sightings with the same key are the same physical point
    truth: np.ndarray | None = None  # its true world position, when the arm put it there


def world_of(s: Sighting, corr: CameraCorrection) -> np.ndarray:
    R, t = corr.apply(s.R, s.t)
    return R @ s.p_cam + t


def _residuals(sightings: list[Sighting], corr: CameraCorrection, use_z: bool) -> np.ndarray:
    dims = 3 if use_z else 2
    out = []
    for group in _groups(sightings).values():
        ws = [world_of(s, corr) for s in group]
        for a, b in itertools.combinations(range(len(group)), 2):
            out.append((ws[a] - ws[b])[:dims])
        for s, w in zip(group, ws):
            if s.truth is not None:
                out.append((w - s.truth)[:dims])
    return np.concatenate(out) if out else np.zeros(0)


def _groups(sightings: list[Sighting]) -> dict[str, list[Sighting]]:
    groups: dict[str, list[Sighting]] = {}
    for s in sightings:
        groups.setdefault(s.key, []).append(s)
    return groups


def fit_correction(sightings: list[Sighting], *, use_z: bool = False, ridge: float = 1e-2,
                   iters: int = 12) -> tuple[CameraCorrection, dict]:
    """Least-squares correction from sightings. Returns it with a report (rms before/after, mm).

    Parameters are 3 rotation angles (milliradians while solving) and 3 translations (mm). Directions
    the data can't see (e.g. moving the camera along its own view axis, for top-down looks) are held
    near zero by the small ridge instead of drifting.
    """
    corr = CameraCorrection()
    before = _residuals(sightings, corr, use_z)
    if before.size < 6:
        raise ValueError(f"not enough constraints to fit ({before.size // (3 if use_z else 2)}); "
                         "need the same blocks seen in several looks, or placed by the arm")
    x = np.zeros(6)  # rotvec in mrad, translation in mm
    eps = np.array([1e-3] * 3 + [1e-2] * 3)
    for _ in range(iters):
        corr = CameraCorrection(tuple(x[:3] * 1e-3), tuple(x[3:]))
        r = _residuals(sightings, corr, use_z)
        J = np.zeros((r.size, 6))
        for k in range(6):  # numeric Jacobian: 6 parameters, cheap and convention-proof
            dx = x.copy()
            dx[k] += eps[k]
            rk = _residuals(sightings, CameraCorrection(tuple(dx[:3] * 1e-3), tuple(dx[3:])), use_z)
            J[:, k] = (rk - r) / eps[k]
        step = np.linalg.solve(J.T @ J + ridge * np.eye(6), -J.T @ r)
        x = x + step
        if float(np.max(np.abs(step))) < 1e-4:
            break
    corr = CameraCorrection(tuple(float(v) for v in x[:3] * 1e-3), tuple(float(v) for v in x[3:]))
    after = _residuals(sightings, corr, use_z)
    sv = np.linalg.svd(J, compute_uv=False) if J.size else np.zeros(6)
    report = {
        "constraints": int(before.size // (3 if use_z else 2)),
        "rms_before_mm": round(float(np.sqrt(np.mean(before ** 2))), 2),
        "rms_after_mm": round(float(np.sqrt(np.mean(after ** 2))), 2),
        "worst_after_mm": round(float(np.max(np.abs(after))), 2),
        "rotation_deg": [round(float(np.degrees(v)), 3) for v in corr.rotvec],
        "translation_mm": [round(float(v), 2) for v in corr.translation],
        "observability": [round(float(v / (sv[0] or 1.0)), 4) for v in sv],
    }
    return corr, report


def sightings_from_looks(looks: list[tuple[list, np.ndarray, np.ndarray]], match_mm: float = 45.0,
                         ref: int = 0) -> list[Sighting]:
    """Pair the same blocks across looks by nearest position (uncorrected), and key them.

    looks: per look, (points as [(color, p_cam)], R_cfg, t_cfg). A block is matched to the reference
    look's same-colored block nearest to it in world (they're off by the camera error, < match_mm).
    """
    out: list[Sighting] = []
    ref_pts, Rr, tr = looks[ref]
    ref_world = [(c, Rr @ p + tr) for c, p in ref_pts]
    for i, (pts, R, t) in enumerate(looks):
        for c, p in pts:
            w = R @ p + t
            best, best_d = None, match_mm
            for j, (rc, rw) in enumerate(ref_world):
                d = float(np.linalg.norm((w - rw)[:2]))
                if rc == c and d < best_d:
                    best, best_d = j, d
            if best is not None:
                out.append(Sighting(np.asarray(p, float), R, t, f"{ref_world[best][0]}#{best}"))
    groups = _groups(out)
    return [s for s in out if len(groups[s.key]) > 1]
