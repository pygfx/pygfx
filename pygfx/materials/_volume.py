from ._base import Material
from ._mesh import clim_from_format


class VolumeBasicMaterial(Material):
    """Base volume material."""

    uniform_type = dict(
        clipping_planes="0*4xf4",  # array<vec4<f32>,3>
        clim="2xf4",
        opacity="f4",
    )

    def __init__(self, clim=None, map=None, **kwargs):
        super().__init__(**kwargs)
        self._given_clim = None
        self.map = map
        self.clim = clim

    def _wgpu_get_pick_info(self, pick_value):
        size = self.map.size
        x, y, z = [(v / 1048576) * s - 0.5 for v, s in zip(pick_value[1:], size)]
        return {"instance_index": 0, "voxel_index": (x, y, z)}

    @property
    def map(self):
        """The 3D texture representing the volume."""
        return self._map

    @map.setter
    def map(self, map):
        self._map = map
        # If the clim is on "auto", update the clim when attaching a new map
        if self._given_clim is None:
            self.clim = None

    @property
    def clim(self):
        """The contrast limits to apply to the map. By default (and if
        `None` is given) the clim is based on the map's format."""
        return self.uniform_buffer.data["clim"]

    @clim.setter
    def clim(self, clim):
        # Check and store given clim
        if clim is not None:
            clim = float(clim[0]), float(clim[1])
        self._given_clim = clim
        # If a clim was not set, try to derive it from the map
        clim = clim or clim_from_format(self.map)
        # Update uniform data
        self.uniform_buffer.data["clim"] = clim
        self.uniform_buffer.update_range(0, 1)


class VolumeSliceMaterial(VolumeBasicMaterial):
    """A material for rendering a slice through a 3D texture at the surface of a mesh.
    This material is not affected by lights.
    """

    uniform_type = dict(
        plane="4xf4",
        clipping_planes="0*4xf4",  # array<vec4<f32>,3>
        clim="2xf4",
        opacity="f4",
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
