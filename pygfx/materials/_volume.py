from ..resources import Texture
from ._base import Material


class VolumeBasicMaterial(Material):
    """Basic volume material.

    Parameters
    ----------
    clim : tuple
        The contrast limits to scale the data values with. Default (0, 1).
    map : Texture
        The colormap to turn the voxel values into their final color.
    interpolation : str
        The method to interpolate the image data. Either 'nearest' or 'linear'. Default 'linear'.
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
        interpolation="linear",
        map_interpolation="linear",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.map = map
        self.clim = clim
        # Note: the default volume interpolation is 'linear' while it's nearest
        # for images. The ability to spot the individual voxels simply results in
        # poor visual quality.
        self.interpolation = interpolation
        self.map_interpolation = map_interpolation

    @property
    def map(self):
        """The colormap to turn the voxel values into their final color.
        If not given or None, the values themselves represents the color.
        The dimensionality of the map can be 1D, 2D or 3D, but should match the
        number of channels in the volume.
        """
        return self._map

    @map.setter
    def map(self, map):
        assert map is None or isinstance(map, Texture)
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


class VolumeSliceMaterial(VolumeBasicMaterial):
    """A material for rendering a slice through a 3D texture at the surface of a mesh.
    This material is not affected by lights.
    """

    uniform_type = dict(
        VolumeBasicMaterial.uniform_type,
        plane="4xf4",
    )

    def __init__(self, plane=(0, 0, 1, 0), **kwargs):
        super().__init__(**kwargs)
        self.plane = plane

    @property
    def plane(self):
        """The plane to slice at, represented with 4 floats ``(a, b, c, d)``,
        which make up the equation: ``ax + by + cz + d = 0`` The plane
        definition applies to the world space (of the scene).
        """
        return self.uniform_buffer.data["plane"]

    @plane.setter
    def plane(self, plane):
        self.uniform_buffer.data["plane"] = plane
        self.uniform_buffer.update_range(0, 1)


class VolumeRayMaterial(VolumeBasicMaterial):
    """A material for rendering volumes using raycasting."""

    # todo: define render modes as material subclasses or using a `mode` or `style` property?
    render_mode = "mip"


class VolumeMipMaterial(VolumeRayMaterial):
    """A material rendering a volume using MIP rendering."""


class VolumeIsoMaterial(VolumeRayMaterial):
    """A material rendering a volume using isosurface rendering."""
