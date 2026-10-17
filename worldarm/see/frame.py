"""One look through the wrist camera: aligned color + depth, intrinsics, and where the camera was."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Intrinsics:
    focal_x_px: float
    focal_y_px: float
    center_x_px: float
    center_y_px: float
    width_px: int = 1280
    height_px: int = 720

    @classmethod
    def from_viam(cls, ip) -> Intrinsics:
        return cls(float(ip.focal_x_px), float(ip.focal_y_px), float(ip.center_x_px), float(ip.center_y_px),
                   int(getattr(ip, "width_px", 0) or 1280), int(getattr(ip, "height_px", 0) or 720))

    def as_list(self) -> list[float]:
        return [self.focal_x_px, self.focal_y_px, self.center_x_px, self.center_y_px, self.width_px, self.height_px]


@dataclass
class Frame:
    color: np.ndarray                  # BGR
    depth: np.ndarray                  # mm, aligned to color
    intr: Intrinsics
    rot: np.ndarray                    # camera -> world, corrected (what everything uses)
    trans: np.ndarray
    rot_cfg: np.ndarray | None = None  # camera -> world as configured, before the correction (calibration)
    trans_cfg: np.ndarray | None = None
    gripper: dict | None = None        # gripper pose at the look
    stamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.rot_cfg is None:
            self.rot_cfg, self.trans_cfg = self.rot, self.trans

    def to_pixel(self, xyz) -> tuple[int, int]:
        cam = self.rot.T @ (np.asarray(xyz, float) - self.trans)
        return (int(round(self.intr.focal_x_px * cam[0] / cam[2] + self.intr.center_x_px)),
                int(round(self.intr.focal_y_px * cam[1] / cam[2] + self.intr.center_y_px)))

    def in_camera(self, world_xyz) -> np.ndarray:
        """A world point (as measured with this frame's transform) back in the camera's own frame."""
        return self.rot.T @ (np.asarray(world_xyz, float) - self.trans)

    def save(self, path: Path | str) -> Path:
        path = Path(path).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, color=self.color, depth=self.depth, intr=np.array(self.intr.as_list()),
                            rot=self.rot, trans=self.trans, rot_cfg=self.rot_cfg, trans_cfg=self.trans_cfg,
                            gripper=json.dumps(self.gripper or {}), stamp=self.stamp)
        return path

    @classmethod
    def load(cls, path: Path | str) -> Frame:
        d = np.load(Path(path), allow_pickle=False)
        intr = [float(v) for v in d["intr"]]
        intr += [1280, 720][len(intr) - 4:] if len(intr) < 6 else []
        files = set(d.files)
        gripper = json.loads(str(d["gripper"])) if "gripper" in files else None
        return cls(d["color"], d["depth"], Intrinsics(*intr[:4], int(intr[4]), int(intr[5])), d["rot"], d["trans"],
                   d["rot_cfg"] if "rot_cfg" in files else None, d["trans_cfg"] if "trans_cfg" in files else None,
                   gripper or None, float(d["stamp"]) if "stamp" in files else 0.0)
