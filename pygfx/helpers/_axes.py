import numpy as np

from .. import (
    cone_geometry,
    Geometry,
    Line,
    LineSegmentMaterial,
    Mesh,
    MeshBasicMaterial,
)
from ..linalg import Vector3
from ..utils import Color


class AxesHelper(Line):
    """An object indicating the axes directions.

    Parameters:
        size (float): The length of the lines (default 1).
        thickness (float): The thickness of the lines (default 2 px).
        arrow_start (float): Percentage where the arrow head should start (default 0.80).
        arrow_radius (float): Radius of the arrow heads' cone base (default 4).
    """

    def __init__(self, size=1.0, thickness=2, arrow_start=0.80, arrow_radius=4):
        positions = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
                [0, 0, 1],
            ],
            dtype="f4",
        )
        positions *= arrow_start * size

        colors = np.array(
            [
                [1, 0, 0, 1],
                [1, 0, 0, 1],  # x is red
                [0, 1, 0, 1],
                [0, 1, 0, 1],  # y is green
                [0, 0, 1, 1],
                [0, 0, 1, 1],  # z is blue
            ],
            dtype="f4",
        )

        geometry = Geometry(positions=positions, colors=colors)
        material = LineSegmentMaterial(vertex_colors=True, thickness=thickness, aa=True)

        super().__init__(geometry, material)

        height = (1 - arrow_start) * size
        cone = cone_geometry(radius=arrow_radius, height=height)

        for pos, color in zip(positions[1::2], colors[1::2]):
            material = MeshBasicMaterial(color=color)
            arrow_head = Mesh(cone, material)
            arrow_head.position = Vector3(*pos)
            # offset by half of height since the cones
            # are centered around the origin
            arrow_head.position.add_scaled_vector(Vector3(*pos).normalize(), height / 2)
            arrow_head.rotation.set_from_unit_vectors(
                Vector3(0, 0, 1),
                Vector3(*pos).normalize(),
            )
            self.add(arrow_head)

    def set_colors(self, x, y, z):
        x, y, z = Color(x), Color(y), Color(z)
        # update lines
        self._geometry.colors.data[0] = x
        self._geometry.colors.data[1] = x
        self._geometry.colors.data[2] = y
        self._geometry.colors.data[3] = y
        self._geometry.colors.data[4] = z
        self._geometry.colors.data[5] = z
        self._geometry.colors.update_range(0, self._geometry.colors.nitems)
        # update arrow heads
        for arrow, color in zip(self.children, [x, y, z]):
            arrow.material.color = color
