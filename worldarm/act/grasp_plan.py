"""How to grip one block, from its measured geometry alone. Pure: nothing moves.

  finger axis   across the block's short side (the narrowest grip); across the long side instead if a
                neighbour would be in the way of an open finger and the long side still fits
  opening       the width across the fingers + clearance each side, no more than the gripper opens
  grip height   finger pads centred on the block's middle, but the fingertips never lower than a few
                mm above whatever the block stands on (table or drawer floor) - so short blocks are
                gripped as deep as is safe and tall ones around their middle
  checks        the expected finger width when holding it (a hold is verified against this), where to
                start judging contact, and how far the block hangs below the gripper while carried
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from worldarm.blocks import Block
from worldarm.cell.config import CellConfig
from worldarm.cell.geometry import other_wrist, point_in_polygon, theta_for_finger_axis, vec_of


@dataclass
class GraspPlan:
    x: float
    y: float
    finger_axis_deg: float           # world direction the fingers close along
    theta: float                     # wrist angle for that axis
    alt_theta: float                 # the same axis with the wrist half a turn round
    open_mm: float                   # finger opening on the way down
    frame_z_grip: float              # gripper frame height when closing
    frame_z_hover: float             # where the descent starts
    reach_z: float                   # contact is judged below this gripper frame height
    expected_width_mm: float         # finger width when holding this block
    jaw_extent_mm: float             # the block's size along the jaws (the other horizontal side)
    bottom_below_frame_mm: float     # gripper frame height minus the block's bottom, while held
    block_height_mm: float
    base_z: float                    # what the block stands on
    feasible: bool = True
    reason: str = ""

    def to_json(self) -> dict:
        return {k: (round(v, 1) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def _finger_hits(centre: np.ndarray, axis: np.ndarray, open_mm: float, cfg: CellConfig, nb: Block) -> bool:
    """Would an open finger (either side) overlap neighbour `nb` (its footprint rectangle + clearance)?"""
    g = cfg.gripper
    across = np.array([-axis[1], axis[0]])
    s_n, l_n = vec_of(nb.short_axis_deg), vec_of(nb.short_axis_deg + 90.0)
    hs = nb.short_mm / 2.0 + cfg.grasp.neighbor_clearance_mm
    hl = nb.long_mm / 2.0 + cfg.grasp.neighbor_clearance_mm
    c = np.asarray(nb.center[:2], float)
    for side in (-1.0, 1.0):
        fc = centre + side * (open_mm / 2.0 + g.finger_depth_mm / 2.0) * axis
        for da in np.linspace(-g.finger_depth_mm / 2.0, g.finger_depth_mm / 2.0, 3):
            for db in np.linspace(-g.finger_width_mm / 2.0, g.finger_width_mm / 2.0, 4):
                d = fc + da * axis + db * across - c
                if abs(float(d @ s_n)) <= hs and abs(float(d @ l_n)) <= hl:
                    return True
    return False


def _fingers_inside(centre: np.ndarray, axis: np.ndarray, open_mm: float, cfg: CellConfig, region: np.ndarray) -> bool:
    """Do both open fingers stay inside `region` (e.g. a drawer's walls)?"""
    g = cfg.gripper
    across = np.array([-axis[1], axis[0]])
    corners = [centre + side * (open_mm / 2.0 + da) * axis + sb * g.finger_width_mm / 2.0 * across
               for side in (-1.0, 1.0) for da in (0.0, g.finger_depth_mm) for sb in (-1.0, 1.0)]
    return bool(point_in_polygon(np.array(corners), region).all())


def plan_grasp(block: Block, cfg: CellConfig, neighbors: list[Block] | tuple = (), base_theta: float = 0.0,
               inside: np.ndarray | None = None, clear_z: float | None = None) -> GraspPlan:
    """inside: polygon the open fingers must stay in (a drawer's walls); clear_z: the top of anything
    around the block the fingers must clear before descending (a drawer's walls)."""
    g, gc = cfg.gripper, cfg.grasp
    centre = np.asarray(block.center[:2], float)
    near = [n for n in neighbors if n is not block and float(np.linalg.norm(n.center[:2] - centre)) < 200.0]
    base = float(block.table_z_mm)
    height = float(block.height_mm)
    top = base + height

    # grip height: pads centred on the middle, fingertips at least tip_clearance above the base
    tip_z = max(base + gc.tip_clearance_mm, base + 0.5 * height - g.pad_length_mm / 2.0,
                cfg.workcell.table_z_mm + cfg.workcell.min_tip_above_table_mm)   # never below the arm's own floor
    frame_z_grip = tip_z + g.tip_below_frame_mm
    tallest = max([top] + [n.table_z_mm + n.height_mm for n in near] + ([clear_z] if clear_z is not None else []))
    frame_z_hover = max(frame_z_grip + 60.0, tallest + gc.hover_above_top_mm + g.tip_below_frame_mm)
    reach_z = top + g.tip_below_frame_mm + cfg.contact.reach_margin_mm

    def plan(axis_deg: float, width: float, jaw: float, open_mm: float, ok: bool = True, why: str = "") -> GraspPlan:
        theta = theta_for_finger_axis(axis_deg % 180.0, base_theta)
        return GraspPlan(float(centre[0]), float(centre[1]), axis_deg % 180.0, theta, other_wrist(theta), open_mm,
                         frame_z_grip, frame_z_hover, reach_z, width, jaw, frame_z_grip - base, height, base, ok, why)

    reasons = []
    for axis_deg, width, jaw in ((block.short_axis_deg, block.short_mm, block.long_mm),
                                 (block.short_axis_deg + 90.0, block.long_mm, block.short_mm)):
        min_open = width + 2.0 * 5.0
        if min_open > g.max_open_mm:
            reasons.append(f"{width:.0f} mm across is wider than the gripper opens ({g.max_open_mm:.0f} mm)")
            continue
        axis = vec_of(axis_deg)
        open_mm = min(g.max_open_mm, width + 2.0 * gc.approach_clearance_mm)
        def blocked(o: float) -> bool:
            return (any(_finger_hits(centre, axis, o, cfg, n) for n in near)
                    or (inside is not None and not _fingers_inside(centre, axis, o, cfg, inside)))
        while blocked(open_mm):
            if open_mm <= min_open:
                break
            open_mm = max(min_open, open_mm - 4.0)  # open less, down to 5 mm clearance a side
        if blocked(open_mm):
            reasons.append(f"no room for the fingers gripping across {width:.0f} mm "
                           f"({'a wall' if inside is not None else 'a neighbour'} in the way)")
            continue
        return plan(axis_deg, width, jaw, open_mm)
    return plan(block.short_axis_deg, block.short_mm, block.long_mm, g.max_open_mm, False, "; ".join(reasons))
