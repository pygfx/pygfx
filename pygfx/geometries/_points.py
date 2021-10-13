import numpy as np

from ..resources import Buffer
from ._base import Geometry


class PointsGeometry(Geometry):
    """A geometry defining Points with optional individual sizes and colors."""

    def __init__(
        self,
        positions=[0, 0, 0],
        sizes=None,
        colors=None,
    ):
        super().__init__()
        self.sizes = None
        self.colors = None

        assert np.all(sizes > 0)

        sizes = np.atleast_1d(sizes)
        colors = np.atleast_2d(colors)
        if not sizes.ndim == 1:
            raise ValueError("Expected Nx1 data for sizes")
        if not (colors.ndim == 2 and colors.shape[-1] == 4):
            raise ValueError("Expected Nx4 data for colors")

        self.positions = Buffer(positions)
        if sizes is not None:
            self.sizes = Buffer(sizes)
        if colors is not None:
            self.colors = Buffer(colors)
