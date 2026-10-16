"""Workcell collision volumes — mirrors armfarm obstacles fragment (table, walls)."""

from __future__ import annotations

from viam.proto.common import (
    Geometry,
    GeometriesInFrame,
    Pose,
    RectangularPrism,
    Vector3,
    WorldState,
)

TABLE_TOP_Z_MM = -23.0


def workcell_world_state(
    *,
    include_ceiling: bool = True,
    table_keep_out_mm: float = 0.0,
) -> WorldState:
    """Uniquely labeled copies of table / wall-front / wall-side (+ ceiling).

    Labels use the ``safety-`` prefix so they do not collide with component
    geometry names ``table``, ``wall-front``, ``wall-side`` already in the
    frame system. ``table_keep_out_mm`` adds a slab above the table top.
    """
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
            center=Pose(x=740, y=0, z=300, o_z=1),
            box=RectangularPrism(dims_mm=Vector3(x=100, y=3000, z=1600)),
        ),
        Geometry(
            label="safety-wall-side",
            center=Pose(x=0, y=-500, z=300, o_z=1),
            box=RectangularPrism(dims_mm=Vector3(x=3000, y=100, z=1600)),
        ),
    ]
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
