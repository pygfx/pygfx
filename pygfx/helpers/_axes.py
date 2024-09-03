import numpy as np
import pylinalg as la

from .. import (
    cone_geometry,
    Geometry,
    Line,
    LineSegmentMaterial,
    Mesh,
    MeshBasicMaterial,
)
from ..utils import Color


class AxesHelper(Line):
    """A WorldObject to indicate the scene's axes.

    Generates three arrows starting at the local origin and pointing into the
    direction of the local x, y, and z-axis respectively. Each arrow is colored
    to represent the respective axis. In particular, the x-axis arrow is blue,
    the y-axis arrow is green, and the z-axis arrow is red.

    Parameters
    ----------
    size : float
        The length of the lines in local space.
    thickness : float
        The thickness of the lines in (onscreen) pixels.

    """

    def __init__(self, size=1.0, thickness=2):
        line_positions = np.array(
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

        arrow_radius = size * 0.1
        # the radius of the cone is the thickness, so that the arrow is twice as wide
        # as the line it sits on.
        # we want the arrow head to maintain the proportions of a equilateral triangle
        # when viewed from the side, so the desired height can be computed
        # by multiplying the radius by sqrt(3)
        arrow_size = np.sqrt(3) * arrow_radius
        cone = cone_geometry(radius=arrow_radius, height=arrow_size)

        line_size = np.max([0.1, size - arrow_size])  # ensure > 0.0
        line_positions *= line_size

        geometry = Geometry(positions=line_positions, colors=colors)
        material = LineSegmentMaterial(
            color_mode="vertex", thickness=thickness, aa=True
        )

        super().__init__(geometry, material)

        for pos, color in zip(line_positions[1::2], colors[1::2]):
            material = MeshBasicMaterial(color=color)
            arrow_head = Mesh(cone, material)
            arrow_head.local.position = pos
            # offset by half of height since the cones
            # are centered around the origin
            # Do not use +=
            # see: https://github.com/pygfx/pygfx/issues/651
            arrow_head.local.position = (
                arrow_head.local.position + arrow_size / 2 * la.vec_normalize(pos)
            )
            arrow_head.local.rotation = la.quat_from_vecs(
                (0, 0, 1), la.vec_normalize(pos)
            )
            self.add(arrow_head)

    def set_colors(self, x, y, z):
        """Update arrow colors.

        Parameters
        ----------
        x : int, float, str, tuple
            The color of the x arrow. This is a either a single int or float
            (gray), a 4-tuple ``(r,g,b,a)`` of ints or floats, or a hex-coded
            color string in one of the following formats: ``#RGB``, ``#RGBA``,
            ``#RRGGBB``, ``#RRGGBBAA``.
        y : int, float, str, tuple
            The color of the x arrow. This is a either a single int or float
            (gray), a 4-tuple ``(r,g,b,a)`` of ints or floats, or a hex-coded
            color string in one of the following formats: ``#RGB``, ``#RGBA``,
            ``#RRGGBB``, ``#RRGGBBAA``.
        z : int, float, str, tuple
            The color of the x arrow. This is a either a single int or float
            (gray), a 4-tuple ``(r,g,b,a)`` of ints or floats, or a hex-coded
            color string in one of the following formats: ``#RGB``, ``#RGBA``,
            ``#RRGGBB``, ``#RRGGBBAA``.

        """

        x, y, z = Color(x), Color(y), Color(z)
        # update lines
        self._geometry.colors.data[0] = x
        self._geometry.colors.data[1] = x
        self._geometry.colors.data[2] = y
        self._geometry.colors.data[3] = y
        self._geometry.colors.data[4] = z
        self._geometry.colors.data[5] = z
        self._geometry.colors.update_full()
        # update arrow heads
        for arrow, color in zip(self.children, [x, y, z]):
            arrow.material.color = color
