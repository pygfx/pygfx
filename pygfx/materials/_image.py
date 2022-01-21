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
        """The colormap for the image."""
        # todo: not implemented in the renderer
        return self._map

    @map.setter
    def map(self, map):
        self._map = map

    @property
    def clim(self):
        """The contrast limits to scale the data values with before the colormap is applied."""
        return self.uniform_buffer.data["clim"]

    @clim.setter
    def clim(self, clim):
        # Check and store given clim
        if clim is not None:
            clim = float(clim[0]), float(clim[1])
        else:
            clim = 0.0, 1.0
        # Update uniform data
        self.uniform_buffer.data["clim"] = clim
        self.uniform_buffer.update_range(0, 1)
