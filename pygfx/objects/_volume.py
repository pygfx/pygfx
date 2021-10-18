import numpy as np

from ._base import WorldObject
from ..geometries import BoxGeometry
from ..resources import Buffer, Texture


class Volume(WorldObject):
    """A volume represents a 3D image in space.

    The geometry for this object consists of:
    * `geometry.grid`: a texture with the 3D data.
    * `geometry.positions`: placing the volume in the scene, with the center of
      voxel (0,0,0) at the origin of the local coordinate frame. Positioning and dealing
      with anisotropy should be dealt with using the scale and position properties.
    * `geometry.texcoords`: the 3D texture coordinates used to sample the volume.

    The picking info of a Volume (the result of
    ``renderer.get_pick_info()``) will for most materials include
    ``voxel_index`` (tuple of 3 floats).

    Parameters:
      data (ndarray, Texture): The 3D data of the volume, as gfx.Texture or a numpy array.
      material: The `Material` used to render the volume.
    """

    def __init__(self, data, material):
        super().__init__()
        if isinstance(data, np.ndarray):
            texture = Texture(data, dim=3)
        elif isinstance(data, Texture):
            texture = data
        else:
            raise TypeError("Volume data must be numpy np.ndarray or gfx.Texture.")

        self.geometry = self._make_geometry(texture.size, texture)
        self.material = material

    @property
    def geometry(self):
        """The geometry of the volume."""
        return self._geometry

    @geometry.setter
    def geometry(self, geometry):
        self._geometry = geometry

    @property
    def material(self):
        """The material of the volume."""
        return self._material

    @material.setter
    def material(self, material):
        self._material = material

    def _make_geometry(self, size, texture):
        size = int(size[0]), int(size[1]), int(size[2])
        # Create box geometry, and map to 0..1
        geometry = BoxGeometry(1, 1, 1)
        geometry.positions.data[:, :3] += 0.5
        # This is our 3D texture coords
        geometry.texcoords = Buffer(geometry.positions.data[:, :3].copy())
        # Map to volume size
        for i in range(3):
            column = geometry.positions.data[:, i]
            column *= size[i]
            column -= 0.5  # Pixel centers are in origin and size

        # Apply
        geometry.grid = texture
        return geometry

    def _wgpu_get_pick_info(self, pick_value):
        size = self.geometry.grid.size
        x, y, z = [(v / 1048576) * s - 0.5 for v, s in zip(pick_value[1:], size)]
        return {"instance_index": 0, "voxel_index": (x, y, z)}
