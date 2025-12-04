from ..resources import Texture, TextureMap
from ._base import Material
from ..utils import assert_type


class ImageBasicMaterial(Material):
    """Rasterized image material.

    Parameters
    ----------
    maprange : tuple
        The range of the ``geometry.texcoords`` that is projected onto the (color) map. Default (0, 1).
    clim : tuple
        The contrast limits. Alias for maprange.
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
        maprange="2xf4",
        gamma="f4",
    )

    def __init__(
        self,
        maprange=None,
        clim=None,
        map=None,
        gamma=1.0,
        interpolation="nearest",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.map = map
        self.maprange = maprange if maprange is not None else clim
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
    def maprange(self):
        """The range of the ``geometry.texcoords`` that is projected onto the (color) map.

        By default this value is (0.0, 1.0), but if the ``texcoords`` represents some
        domain-specific value, e.g. temperature, then ``maprange`` can be set to e.g. (0, 100).
        """
        v1, v2 = self.uniform_buffer.data["maprange"]
        return float(v1), float(v2)

    @maprange.setter
    def maprange(self, maprange):
        # Check and store given value
        if maprange is None:
            maprange = 0, 1
        maprange = float(maprange[0]), float(maprange[1])
        # Update uniform data
        self.uniform_buffer.data["maprange"] = maprange
        self.uniform_buffer.update_full()

    @property
    def clim(self):
        """Alias for maprange; for images, clim (contrast limits) is a common term."""
        return self.maprange

    @clim.setter
    def clim(self, clim):
        self.maprange = clim

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
