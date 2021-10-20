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
        """The colormap for the volume."""
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
