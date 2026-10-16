"""Workcell collision volumes for catch motion (armfarm obstacles fragment).

table top Z = -23 mm. Labels use safety-* so they do not collide with component names.
"""

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


def workcell_world_state(*, include_ceiling: bool = True) -> WorldState:
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
