import numpy as np

from .. import Geometry, Line, LineSegmentMaterial
from ..utils import Color

DTYPE = "f4"


class GridHelper(Line):
    """A WorldObject that shows a grid-shaped wireframe.

    The generated grid will be in the xz-plane centered at the origin. To
    position the grid, manipulate its parent's position, rotation, etc.

    Parameters
    ----------
    size : float
        The size of the wireframe in the direction of the x-, and y-axis.
    divisions : int
        The number of (evenly spaced) divisions to perform along each axis.
    color1 : int, float, str, tuple
        The color of the center lines. This is a either a single int or float
        (gray), a 4-tuple ``(r,g,b,a)`` of ints or floats, or a hex-coded color
        string in one of the following formats: ``#RGB``, ``#RGBA``,
        ``#RRGGBB``, ``#RRGGBBAA``.
    color2 : int, float, str, tuple
        The color of non-center lines. This is a either a single int or float
        (gray), a 4-tuple ``(r,g,b,a)`` of ints or floats, or a hex-coded color
        string in one of the following formats: ``#RGB``, ``#RGBA``,
        ``#RRGGBB``, ``#RRGGBBAA``.
    thickness : int
        The thickness in screen units (pixels).

    """

    def __init__(
        self,
        size=10.0,
        divisions=10,
        color1=(0.35, 0.35, 0.35, 1),
        color2=(0.1, 0.1, 0.1, 1),
        thickness=1,
    ):
        assert isinstance(divisions, int)
        assert size > 0.0

        half_size = size / 2
        n_lines = divisions + 1
        x = np.linspace(-half_size, half_size, num=n_lines, dtype=DTYPE)

        # the grid is made up of 2 * n_lines line segments
        # where each line has two endpoints (2, 3)
        positions = np.zeros((2, n_lines, 2, 3), dtype=DTYPE)
        positions[0, ..., 0] = x[:, None]
        positions[0, ..., 2] = [[-half_size, half_size]]
        positions[1, ..., 0] = [[-half_size, half_size]]
        positions[1, ..., 2] = x[:, None]

        # color1 for the center lines, color2 for the rest
        colors = np.empty((2, n_lines, 2, 4), dtype=DTYPE)
        colors[..., :] = Color(color2)
        colors[:, n_lines // 2, :, :] = Color(color1)

        geometry = Geometry(
            positions=positions.reshape((-1, 3)), colors=colors.reshape((-1, 4))
        )
        material = LineSegmentMaterial(
            color_mode="vertex", thickness=thickness, aa=True
        )

        super().__init__(geometry, material)
