from ..resources import Texture
from ._base import Material


class ImageBasicMaterial(Material):
    """Rasterized image material.

    Parameters
    ----------
    clim : tuple
        The contrast limits to scale the data values with. Default (0, 1).
    map : Texture
        The texture map to turn the image values into its final color. Optional.
    interpolation : str
        The method to interpolate the image data. Either 'nearest' or 'linear'. Default 'nearest'.
    map_interpolation: str
        The method to interpolate the color map. Either 'nearest' or 'linear'. Default 'linear'.
    kwargs : Any
        Additional kwargs will be passed to the :class:`material base class
        <pygfx.Material>`.

    """

    uniform_type = dict(
        Material.uniform_type,
        clim="2xf4",
    )

    def __init__(
        self,
        clim=None,
        map=None,
        interpolation="nearest",
        map_interpolation="linear",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.map = map
        self.clim = clim
        self.interpolation = interpolation
        self.map_interpolation = map_interpolation

    @property
    def map(self):
        """The texture map to turn the image values into its final color.
        If None, the values themselves represents the color. The
        dimensionality of the texture map can be 1D, 2D or 3D, but
        should match the number of channels in the image.
        """
        return self._store.map

    @map.setter
    def map(self, map):
        assert map is None or isinstance(map, Texture)
        self._store.map = map

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
        self.uniform_buffer.update_full()

    @property
    def interpolation(self):
        """The method to interpolate the image data. Either 'nearest' or 'linear'."""
        return self._store.interpolation

    @interpolation.setter
    def interpolation(self, value):
        assert value in ("nearest", "linear")
        self._store.interpolation = value

    @property
    def map_interpolation(self):
        """The method to interpolate the colormap. Either 'nearest' or 'linear'."""
        return self._store.map_interpolation

    @map_interpolation.setter
    def map_interpolation(self, value):
        assert value in ("nearest", "linear")
        self._store.map_interpolation = value
