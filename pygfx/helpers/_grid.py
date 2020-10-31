import numpy as np

from .. import Geometry, Line, LineThinSegmentMaterial


DTYPE = "f4"


class GridHelper(Line):
    def __init__(
        self,
        size=10.0,
        divisions=10,
        color1=(0.35, 0.35, 0.35, 1),
        color2=(0.1, 0.1, 0.1, 1),
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
        colors[..., :] = color2
        colors[:, n_lines // 2, :, :] = color1

        geometry = Geometry(
            positions=positions.reshape((-1, 3)), colors=colors.reshape((-1, 4))
        )
        material = LineThinSegmentMaterial(vertex_colors=True)

        super().__init__(geometry, material)
