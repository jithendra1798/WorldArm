"""Overhead-camera pixels <-> positions on the table, in an arm's world frame.

The ball rolls on a plane, so a homography (8 numbers) maps the ball's pixel centre to its world x, y.
It is fitted from a few points where both are known: the arm puts the ball (or its fingertips) at a known
spot and the overhead camera sees it there. One homography per arm puts both arms in one shared picture.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _normalizer(pts: np.ndarray) -> np.ndarray:
    c = pts.mean(axis=0)
    s = np.sqrt(2.0) / (np.mean(np.linalg.norm(pts - c, axis=1)) or 1.0)
    return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])


@dataclass
class TableMap:
    H: np.ndarray                    # 3x3: pixel (u, v, 1) -> world (x, y, 1)
    rms_mm: float = 0.0
    n: int = 0

    def to_world(self, uv) -> np.ndarray:
        uv = np.atleast_2d(np.asarray(uv, float))
        w = np.c_[uv, np.ones(len(uv))] @ self.H.T
        out = w[:, :2] / w[:, 2:3]
        return out[0] if np.asarray(uv).ndim == 1 or len(out) == 1 else out

    def to_pixel(self, xy) -> np.ndarray:
        xy = np.atleast_2d(np.asarray(xy, float))
        w = np.c_[xy, np.ones(len(xy))] @ np.linalg.inv(self.H).T
        out = w[:, :2] / w[:, 2:3]
        return out[0] if len(out) == 1 else out

    def to_json(self) -> dict:
        return {"H": np.round(self.H, 10).tolist(), "rms_mm": round(self.rms_mm, 2), "n": self.n}

    @classmethod
    def from_json(cls, d: dict) -> TableMap:
        return cls(np.array(d["H"], float), float(d.get("rms_mm", 0.0)), int(d.get("n", 0)))

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_json(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path | str) -> TableMap:
        return cls.from_json(json.loads(Path(path).read_text()))


def fit_table_map(pixels, world) -> TableMap:
    """Normalized DLT homography from >= 4 pixel <-> world correspondences, with its residual (mm)."""
    uv, xy = np.asarray(pixels, float), np.asarray(world, float)
    if len(uv) < 4:
        raise ValueError("need at least 4 points")
    Tu, Tx = _normalizer(uv), _normalizer(xy)
    a = np.c_[uv, np.ones(len(uv))] @ Tu.T
    b = np.c_[xy, np.ones(len(xy))] @ Tx.T
    rows = []
    for (u, v, _), (x, y, _) in zip(a, b):
        rows.append([-u, -v, -1, 0, 0, 0, x * u, x * v, x])
        rows.append([0, 0, 0, -u, -v, -1, y * u, y * v, y])
    Hn = np.linalg.svd(np.array(rows))[2][-1].reshape(3, 3)
    H = np.linalg.inv(Tx) @ Hn @ Tu
    H /= H[2, 2]
    m = TableMap(H, 0.0, len(uv))
    err = np.linalg.norm(m.to_world(uv) - xy, axis=1) if len(uv) > 1 else np.zeros(1)
    m.rms_mm = float(np.sqrt(np.mean(err ** 2)))
    return m


def leave_one_out_mm(pixels, world) -> list[float]:
    """How far each point lands when the map is fitted without it: the honest error to expect."""
    uv, xy = np.asarray(pixels, float), np.asarray(world, float)
    out = []
    for i in range(len(uv)):
        keep = np.arange(len(uv)) != i
        m = fit_table_map(uv[keep], xy[keep])
        out.append(float(np.linalg.norm(m.to_world(uv[i]) - xy[i])))
    return out
