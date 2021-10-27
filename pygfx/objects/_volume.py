import numpy as np

from ._base import WorldObject
from ..geometries import Geometry
from ..resources import Texture


class Volume(WorldObject):
    """A volume represents a 3D image in space.

    The geometry for this object consists only of `geometry.grid`: a texture with the 3D data.

    The picking info of a Volume (the result of ``renderer.get_pick_info()``)
    will for most materials include ``voxel_index`` (tuple of 3 floats).

    Parameters:
      data (ndarray, Texture): The 3D data of the volume, as gfx.Texture or a numpy array.
      material: The `Material` used to render the volume.
    """

    def __init__(self, data, material):
        super().__init__()

        if isinstance(data, Geometry):
            geometry = data
        elif isinstance(data, Texture):
            geometry = Geometry(grid=data)
        elif isinstance(data, np.ndarray):
            texture = Texture(data, dim=3)
            geometry = Geometry(grid=texture)
        else:
            raise TypeError("Volume data must be numpy np.ndarray or gfx.Texture.")

        # The texture provides a regular grid of data that represents this object's geometry.
        # The vertex positions for the corners are calculated in the shader.
        self._geometry = geometry
        self.material = material

    @property
    def geometry(self):
        """The geometry of the volume."""
        return self._geometry

    @property
    def material(self):
        """The material of the volume."""
        return self._material

    @material.setter
    def material(self, material):
        self._material = material

    def _wgpu_get_pick_info(self, pick_value):
        size = self.geometry.grid.size
        x, y, z = [(v / 1048576) * s - 0.5 for v, s in zip(pick_value[1:], size)]
        return {"instance_index": 0, "voxel_index": (x, y, z)}
