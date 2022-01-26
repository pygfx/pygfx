from ..resources import TextureView
from ._base import Material


class VolumeBasicMaterial(Material):
    """Base volume material."""

    uniform_type = dict(
        clim="2xf4",
    )

    def __init__(self, clim=None, map=None, **kwargs):
        super().__init__(**kwargs)
        self.map = map
        self.clim = clim

    @property
    def map(self):
        """The colormap to turn the volume values into its final color.
        If not given or None, the values themselves represents the color.
        The dimensionality of the map can be 1D, 2D or 3D, but should match the
        number of channels in the volume.
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


class VolumeSliceMaterial(VolumeBasicMaterial):
    """A material for rendering a slice through a 3D texture at the surface of a mesh.
    This material is not affected by lights.
    """

    uniform_type = dict(
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
