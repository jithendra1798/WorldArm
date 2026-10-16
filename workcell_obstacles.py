"""Workcell collision volumes for motion planning (armfarm obstacles fragment).

These match the configured obstacle components on armfarm05
(``table``, ``wall-front``, ``wall-side`` — and ``ceiling`` when included):

- table:       box 3000×3000×200 at world (0, 0, -123) → top at z = -23 mm
- wall-front:  box 100×3000×1600 at world (740, 0, 300)
- wall-side:   box 3000×100×1600 at world (0, -500, 300)
- ceiling:     box 3000×3000×100 at world (0, 0, 1050)

Geometry labels are prefixed with ``safety-`` so they do not collide with the
same-named volumes already attached to those components in the frame system.

``table_keep_out_mm`` optionally extends a keep-out slab *above* the configured
table top so the planner treats near-table air as occupied (stops TCP/jaws from
driving through the table surface).
"""

from __future__ import annotations

import os

from viam.proto.common import (
    Geometry,
    GeometriesInFrame,
    Pose,
    RectangularPrism,
    Vector3,
    WorldState,
)

# Configured table: center z=-123, height 200 → top surface.
TABLE_TOP_Z_MM = -23.0
# Wall positions default to armfarm05. Other arms override them, e.g. armfarm03 has its
# side wall at y = +500 (measured with verify_camera.py).
WALL_FRONT_X_MM = float(os.getenv("VIAM_WALL_FRONT_X_MM", "740"))
WALL_SIDE_Y_MM = float(os.getenv("VIAM_WALL_SIDE_Y_MM", "-500"))

OBSTACLE_COMPONENT_NAMES = ("table", "wall-front", "wall-side", "ceiling")


def workcell_world_state(
    *,
    include_ceiling: bool = True,
    table_keep_out_mm: float = 0.0,
) -> WorldState:
    """Return a WorldState with uniquely labeled copies of the workcell obstacles."""
    if table_keep_out_mm < 0:
        raise ValueError("table_keep_out_mm must be >= 0")

    geoms = [
        Geometry(
            label="safety-table",
            center=Pose(x=0, y=0, z=-123, o_z=1),
            box=RectangularPrism(dims_mm=Vector3(x=3000, y=3000, z=200)),
        ),
        Geometry(
            label="safety-wall-front",
            center=Pose(x=WALL_FRONT_X_MM, y=0, z=300, o_z=1),
            box=RectangularPrism(dims_mm=Vector3(x=100, y=3000, z=1600)),
        ),
        Geometry(
            label="safety-wall-side",
            center=Pose(x=0, y=WALL_SIDE_Y_MM, z=300, o_z=1),
            box=RectangularPrism(dims_mm=Vector3(x=3000, y=100, z=1600)),
        ),
    ]
    # Explicit keep-out above the known table top (same XY footprint as the table).
    if table_keep_out_mm > 0:
        keep_center_z = TABLE_TOP_Z_MM + table_keep_out_mm / 2.0
        geoms.append(
            Geometry(
                label="safety-table-keepout",
                center=Pose(x=0, y=0, z=keep_center_z, o_z=1),
                box=RectangularPrism(
                    dims_mm=Vector3(x=3000, y=3000, z=table_keep_out_mm)
                ),
            )
        )
    if include_ceiling:
        geoms.append(
            Geometry(
                label="safety-ceiling",
                center=Pose(x=0, y=0, z=1050, o_z=1),
                box=RectangularPrism(dims_mm=Vector3(x=3000, y=3000, z=100)),
            )
        )
    return WorldState(
        obstacles=[GeometriesInFrame(reference_frame="world", geometries=geoms)]
    )


def assert_tcp_above_table(z_mm: float, *, margin_mm: float, context: str) -> None:
    """Refuse poses that would put the gripper TCP at or below the table top + margin."""
    floor = TABLE_TOP_Z_MM + margin_mm
    if z_mm < floor:
        raise RuntimeError(
            f"{context}: world Z={z_mm:.1f} mm is below table safety floor "
            f"{floor:.1f} mm (table top {TABLE_TOP_Z_MM:.1f} + margin {margin_mm:.1f}). "
            "Refusing to command motion through the table."
        )


def compute_grasp_tcp_z(
    detected_world_z: float,
    *,
    offset_above_detection_mm: float,
    table_margin_mm: float,
) -> float:
    """Grasp TCP height in world Z — never below the detected surface or table floor.

    ``offset_above_detection_mm`` must be >= 0: millimeters *above* the vision
    detection (toward +world Z / away from the table). Negative offsets are
    rejected because they drive the TCP down through the cube toward the table.
    """
    if offset_above_detection_mm < 0:
        raise RuntimeError(
            "VIAM_GRASP_TCP_Z_OFFSET_MM must be >= 0 (mm above the detected surface). "
            f"Got {offset_above_detection_mm}. Negative values lower the TCP toward "
            "the table and were causing penetration."
        )
    z_floor = TABLE_TOP_Z_MM + table_margin_mm
    grasp_z = detected_world_z + offset_above_detection_mm
    # Never command below the detected cloud (top of cube facing the camera).
    if grasp_z < detected_world_z:
        grasp_z = detected_world_z
    if grasp_z < z_floor:
        raise RuntimeError(
            f"Grasp TCP Z={grasp_z:.1f} mm is below table safety floor {z_floor:.1f} mm "
            f"(detection Z={detected_world_z:.1f}, offset={offset_above_detection_mm:.1f}). "
            "Increase VIAM_GRASP_TCP_Z_OFFSET_MM and/or VIAM_MIN_TCP_ABOVE_TABLE_MM."
        )
    return grasp_z
