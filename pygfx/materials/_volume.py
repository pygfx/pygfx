from ..utils import array_from_shadertype
from ..resources import Buffer
from ._base import Material


class VolumeBasicMaterial(Material):
    """Base volume material."""

    uniform_type = dict(
        clim=("float32", 2),
    )

    def __init__(self, **kwargs):
        super().__init__()

        self.uniform_buffer = Buffer(
            array_from_shadertype(self.uniform_type), usage="UNIFORM"
        )

        self._map = None
        self.clim = 0, 1

        for argname, val in kwargs.items():
            if not hasattr(self, argname):
                raise AttributeError(f"No attribute '{argname}'")
            setattr(self, argname, val)

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

    @property
    def clim(self):
        """The contrast limits to apply to the map. Default (0, 1)"""
        return self.uniform_buffer.data["clim"]

    @clim.setter
    def clim(self, clim):
        self.uniform_buffer.data["clim"] = clim
        self.uniform_buffer.update_range(0, 1)


class VolumeSliceMaterial(VolumeBasicMaterial):
    """A material for rendering a slice through a 3D texture at the surface of a mesh.
    This material is not affected by lights.
    """

    uniform_type = dict(
        plane=("float32", 4),
        clim=("float32", 2),
    )

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
