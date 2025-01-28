from ..resources import Texture, TextureMap
from ._base import Material
from ..utils import assert_type


class ImageBasicMaterial(Material):
    """Rasterized image material.

    Parameters
    ----------
    clim : tuple
        The contrast limits to scale the data values with. Default (0, 1).
    map : TextureMap | Texture
        The texture map to turn the image values into its final color. Optional.
    gamma : float
        The gamma correction to apply to the image data. Default 1.
    interpolation : str
        The method to interpolate the image data. Either 'nearest' or 'linear'. Default 'nearest'.
    kwargs : Any
        Additional kwargs will be passed to the :class:`material base class
        <pygfx.Material>`.

    """

    uniform_type = dict(
        Material.uniform_type,
        clim="2xf4",
        gamma="f4",
    )

    def __init__(
        self,
        clim=None,
        map=None,
        gamma=1.0,
        interpolation="nearest",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.map = map
        self.clim = clim
        self.gamma = gamma
        self.interpolation = interpolation

    @property
    def map(self):
        """The texture map to turn the image values into its final color.
        If None, the values themselves represents the color. The
        dimensionality of the texture map can be 1D, 2D or 3D, but
        should match the number of channels in the image.

        Note: for scientific data, it is usually to set wrap mode to 'clamp'.
        """
        return self._store.map

    @map.setter
    def map(self, map):
        assert_type("map", map, None, Texture, TextureMap)
        if isinstance(map, Texture):
            map = TextureMap(map, wrap="clamp")
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
    def gamma(self):
        """The gamma correction to apply to the image data. Default 1."""
        return self.uniform_buffer.data["gamma"]

    @gamma.setter
    def gamma(self, value):
        value = float(value)
        if value <= 0:
            raise ValueError("gamma must be greater than 0")
        self.uniform_buffer.data["gamma"] = float(value)
        self.uniform_buffer.update_full()

    @property
    def interpolation(self):
        """The method to interpolate the image data. Either 'nearest' or 'linear'."""
        return self._store.interpolation

    @interpolation.setter
    def interpolation(self, value):
        assert value in ("nearest", "linear")
        self._store.interpolation = value
