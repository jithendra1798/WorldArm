"""What the robot thinks it sees, drawn on the camera image: every block with where the fingers will close,
the organizer with each drawer's number, state and opening, the inside of open drawers and what's in them."""

from __future__ import annotations

import math

import cv2
import numpy as np

from worldarm.cell.config import CellConfig
from worldarm.see.frame import Frame
from worldarm.see.scene import Scene

STATE_BGR = {"closed": (60, 180, 60), "partial": (0, 165, 255), "open": (0, 220, 255)}


def draw_scene(frame: Frame, scene: Scene, cfg: CellConfig | None = None, title: str = "") -> np.ndarray:
    from worldarm.act.grasp_plan import plan_grasp  # local: drawing must not depend on skills at import

    img = frame.color.copy()

    def px(xyz):
        return frame.to_pixel(xyz)

    for b in scene.table_blocks:
        contours, _ = cv2.findContours(b.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (255, 255, 0), 2)
        if cfg is not None:
            p = plan_grasp(b, cfg, scene.table_blocks)
            a = math.radians(p.finger_axis_deg)
            top = b.table_z_mm + b.height_mm
            ends = [px((p.x + s * p.open_mm / 2 * math.cos(a), p.y + s * p.open_mm / 2 * math.sin(a), top)) for s in (-1, 1)]
            cv2.line(img, ends[0], ends[1], (0, 255, 0) if p.feasible else (0, 0, 255), 3)
        cv2.putText(img, f"{b.color} {b.height_mm:.0f}mm", (b.pixel[0] + 14, b.pixel[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    org = scene.organizer
    if org is not None:
        top = org.box_top_z
        box = np.array([px((x, y, top)) for x, y in org.box], np.int32)
        cv2.polylines(img, [box], True, (200, 200, 200), 2)
        for d in org.drawers:
            c = STATE_BGR.get(d.state, (200, 200, 200))
            u, v = px(d.knob)
            cv2.circle(img, (u, v), 16, c, 3)
            label = f"{d.index} {d.state}" + (f" {d.opening_mm:.0f}mm" if d.state != "closed" else "")
            cv2.putText(img, label, (u + 18, v + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
            if d.inside is not None:
                z = d.floor_z if d.floor_z is not None else top
                poly = np.array([px((x, y, z)) for x, y in d.inside], np.int32)
                cv2.polylines(img, [poly], True, c, 2)
                for b in scene.contents.get(d.index, []):
                    contours, _ = cv2.findContours(b.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(img, contours, -1, (255, 0, 255), 2)
                    cv2.putText(img, b.color, (b.pixel[0] + 8, b.pixel[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        mid = org.drawers[len(org.drawers) // 2].knob
        tip = mid[:2] + org.pull * 60
        cv2.arrowedLine(img, px(mid), px((tip[0], tip[1], mid[2])), (255, 255, 255), 2, tipLength=0.3)

    lines = ([title] if title else []) + scene.warnings[:3]
    for i, text in enumerate(lines):
        cv2.putText(img, text[:110], (12, 28 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return img
