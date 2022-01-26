from ..resources import TextureView
from ._base import Material


class ImageBasicMaterial(Material):

    uniform_type = dict(
        clim="2xf4",
    )

    def __init__(self, clim=None, map=None, **kwargs):
        super().__init__(**kwargs)
        self.map = map
        self.clim = clim

    @property
    def map(self):
        """The colormap to turn the image values into its final color.
        If not given or None, the values themselves represents the color.
        The dimensionality of the map can be 1D, 2D or 3D, but should match the
        number of channels in the image.
        """
        return self._map

    @map.setter
    def map(self, map):
        assert map is None or isinstance(map, TextureView)
        self._map = map

    @property
    def clim(self):
        """The contrast limits to scale the data values with. Default (0, 1)."""
        v1, v2 = self.uniform_buffer.data["clim"]
        return float(v1), float(v2)

    @clim.setter
    def clim(self, clim):
        # Check and store given clim
        if clim is None:
            clim = 0, 1
        clim = float(clim[0]), float(clim[1])
        # Update uniform data
        self.uniform_buffer.data["clim"] = clim
        self.uniform_buffer.update_range(0, 1)
